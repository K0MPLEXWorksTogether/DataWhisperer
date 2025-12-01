"""
Module for determining model suitability for datasets.

For classification: Tests linear separability using SVM.
For regression: Tests linear vs polynomial model fit.
"""

from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from src.preprocess import DataPreProcessor
from src.utils import detect_problem_type

import dask.dataframe as dd


def return_splits(dataframe, target_index, problem_type=None) -> list:
    """
    Simplifies calling DataPreProcessor's preprocess() method.
    :param dataframe: The dataframe passed.
    :param target_index: The index of the target features in the dataframe.
    :param problem_type: The problem type ('regression', 'binary', 'multiclass').
    :return: List of [train, test, validation] DataFrames.
    """
    if problem_type is None:
        problem_type = detect_problem_type(dataframe, target_index)
    
    preprocess = DataPreProcessor(dataframe, target_index, problem_type)
    result = preprocess.preprocess()
    return result


def approx(value1: float, value2: float, tolerance: float = 0.01) -> bool:
    """
    Responsible for making approximations in training and test accuracy/score
    closeness. Returns a boolean.
    :param value1: A floating point value.
    :param value2: Another floating point value.
    :param tolerance: The amount which can be considered as approximate.
    :return: A boolean returning whether both the values are approximates.
    """
    return abs(value2 - value1) <= tolerance


class ModelSuitability:
    """
    The class can determine model suitability for the dataset.
    
    For Classification: Uses Support Vector Classifier (SVC) to test linear separability.
    For Regression: Compares linear vs polynomial regression models.
    """

    def __init__(self, test_data: dd.DataFrame, index_target: int, problem_type: str = None):
        # Detect problem type if not provided
        if problem_type is None:
            problem_type = detect_problem_type(test_data, index_target)
        
        self.problem_type = problem_type
        
        # Splitting into train, test and validation
        train, test, validation = return_splits(test_data, index_target, problem_type)

        # Finding the target column.
        target_column = train.columns[-1]

        # Separating features from targets.
        self.x_train = train.drop(columns=[target_column]).compute()
        self.y_train = train[target_column].compute()
        self.x_test = test.drop(columns=[target_column]).compute()
        self.y_test = test[target_column].compute()

        if problem_type == "regression":
            # For regression: Compare linear vs polynomial models
            self.linear_model = LinearRegression()
            self.poly_model = Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression())
            ])
            
            # Fitting models
            self.linear_model.fit(self.x_train, self.y_train)
            self.poly_model.fit(self.x_train, self.y_train)
            
            # Finding R² scores
            self.train_score_linear = self.linear_model.score(self.x_train, self.y_train)
            self.train_score_poly = self.poly_model.score(self.x_train, self.y_train)
            self.test_score_linear = self.linear_model.score(self.x_test, self.y_test)
            self.test_score_poly = self.poly_model.score(self.x_test, self.y_test)
        else:
            # For classification: Use SVC to test linear separability
            self.svm_linear = SVC(kernel="linear")
            self.svm_non_linear = SVC(kernel="rbf")

            # Fitting training data.
            self.svm_linear.fit(self.x_train, self.y_train)
            self.svm_non_linear.fit(self.x_train, self.y_train)

            # Finding training accuracy.
            self.train_score_linear = self.svm_linear.score(self.x_train, self.y_train)
            self.train_score_non_linear = self.svm_non_linear.score(self.x_train, self.y_train)

            # Finding test accuracy.
            self.test_score_linear = self.svm_linear.score(self.x_test, self.y_test)
            self.test_score_non_linear = self.svm_non_linear.score(self.x_test, self.y_test)

    def prediction(self) -> dict:
        """
        The function is responsible for displaying all the data
        to check for model suitability.
        :return: A dictionary representing the data.
        """
        final_verdict = dict()
        
        if self.problem_type == "regression":
            final_verdict["Training R² Score (Linear)"] = self.train_score_linear
            final_verdict["Training R² Score (Polynomial)"] = self.train_score_poly
            final_verdict["Test R² Score (Linear)"] = self.test_score_linear
            final_verdict["Test R² Score (Polynomial)"] = self.test_score_poly
            
            # Determine if polynomial provides significant improvement
            improvement = self.test_score_poly - self.test_score_linear
            if improvement > 0.05:
                final_verdict["Verdict"] = (
                    f"Polynomial model shows significant improvement (R² difference: {improvement:.4f}). "
                    "Consider using non-linear models or feature engineering."
                )
            elif approx(self.test_score_linear, self.train_score_linear, tolerance=0.1):
                final_verdict["Verdict"] = (
                    "Linear model performs well with consistent train-test scores. "
                    "Dataset appears suitable for linear regression."
                )
            else:
                final_verdict["Verdict"] = (
                    "Consider regularization or feature selection - possible overfitting detected."
                )
        else:
            # Classification verdict
            final_verdict["Training Accuracy Of Linear SVM"] = self.train_score_linear
            final_verdict["Training Accuracy Of Non-Linear SVM"] = self.train_score_non_linear
            final_verdict["Test Accuracy Of Linear SVM"] = self.test_score_linear
            final_verdict["Test Accuracy Of Non-Linear SVM"] = self.test_score_non_linear

            # Adding the final verdict.
            if approx(self.train_score_linear, self.test_score_linear):
                final_verdict["Verdict"] = (
                    "Dataset is linearly separable, as linear SVM has approximately equal train "
                    "and test accuracy"
                )
            else:
                final_verdict["Verdict"] = (
                    "Dataset is not linearly separable, as linear SVM has significant "
                    "difference in train and test accuracy."
                )

        return final_verdict


def main():
    """
    Testing function. Run the program to call main() for testing.
    :return: None.
    """
    data = dd.read_csv("../data/weather.csv")

    model_suit = ModelSuitability(data, 10)
    print(model_suit.prediction())


if __name__ == "__main__":
    main()
