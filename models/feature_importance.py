from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.preprocess import DataPreProcessor
from src.utils import detect_problem_type

import dask.dataframe as dd
import matplotlib.pyplot as plt
import io


class FeatureImportance:
    """
    The class utilizes RandomForest (Classifier or Regressor) from scikit-learn
    to find important features of the dataset.
    It provides two methods:
    - importance() - Finds the importance of all the features as a dictionary.
    - visualize_importance() - Visualizes the feature importance using
    a simple bar graph.
    """

    def __init__(self, dataframe: dd.DataFrame, target_index: int, problem_type: str = None) -> None:
        # Detect problem type if not provided
        if problem_type is None:
            problem_type = detect_problem_type(dataframe, target_index)
        
        self.problem_type = problem_type
        preprocessor = DataPreProcessor(dataframe, target_index, problem_type)
        self.train = preprocessor.preprocess()[0].compute()
        
        # Use appropriate model based on problem type
        if problem_type == "regression":
            self.model = RandomForestRegressor(random_state=42)
        else:
            self.model = RandomForestClassifier(random_state=42)

        target_column = self.train.columns[-1]
        self.features = self.train.drop(columns=[target_column])
        self.target = self.train[target_column]

        self.model.fit(self.features, self.target)

    def importance(self) -> dict:
        importance = self.model.feature_importances_

        feature_importance = dict(zip(self.features.columns, importance))
        return feature_importance

    def visualize_importance(self) -> io.BytesIO:
        important_features = self.importance()

        sorted_features = sorted(important_features.items(), key=lambda sort: sort[1], reverse=True)

        # Extracting feature names and importance.
        features_sorted = [feature[0] for feature in sorted_features]
        importance_sorted = [feature[1] for feature in sorted_features]

        # Plotting the bar graph.
        plt.figure(figsize=(10, 6))
        plt.bar(features_sorted, importance_sorted)
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.title("Feature Importance")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Saving the graph in BytesIO object.
        img = io.BytesIO()
        plt.savefig(img, format="png")
        plt.close()
        img.seek(0)

        return img


def main():
    """
    Testing function for feature importance. Run the file to view
    the functionality of the class.
    :return: None.
    """
    dataframe = dd.read_csv("../data/weather.csv")

    feature_importance = FeatureImportance(dataframe, 10)
    print(feature_importance.importance())
    print(feature_importance.visualize_importance())


if __name__ == "__main__":
    main()