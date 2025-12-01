import dask.dataframe as dd
import logging

from dask_ml.impute import SimpleImputer
from pandas.api.types import is_numeric_dtype
from dask_ml.model_selection import train_test_split
from sklearn.impute import KNNImputer
from dask_ml.preprocessing import LabelEncoder
from dask_ml.preprocessing import OneHotEncoder
from dask_ml.preprocessing import StandardScaler
from src.utils import detect_problem_type, is_regression

"""
This file contains a class that can perform the following data preprocessing
tasks:

- Separating the features from the target.
- Removing Outliers.
- Handling missing data.
- Handling duplicate data.
- Encoding feature categorical variables with OneHotEncoder.
- Encoding the target using LabelEncoder.
- Separating data into train, test and validation splits.
- Scaling the data to standardize.
"""


def write_logs(error: bool, message: str, error_reason: Exception = None) -> None:
    """
    The function automates the process of writing to the log file.
    :param error: If there has been an error, pass True.
    :param message: The message to written to the log file.
    :param error_reason: If there is an error, pass the exception to raise it.
    :return: None.
    """
    if error:
        logging.info(msg=message)
        raise error_reason
    else:
        logging.info(msg=message)


class DataPreProcessor:

    def __init__(self, input_df: dd.DataFrame, target_index: int, problem_type: str = None) -> None:
        """
        The constructor initializes the dataframe and the target index
        as attributes of the class.
        It also initializes logging for the class.
        
        :param input_df: The input Dask DataFrame
        :param target_index: The index of the target column
        :param problem_type: Optional problem type ('regression', 'binary', 'multiclass'). 
                            If None, will be auto-detected.
        """
        try:
            # Initializing dataframe and target index as attributes.
            self.df = input_df
            self.target = target_index
            self.missing(impute=True)

            # Detect or set problem type
            if problem_type is None:
                self.problem_type = detect_problem_type(self.df, self.target)
            else:
                self.problem_type = problem_type

            # Separating the target from the features.
            self.X = self.df.iloc[:, self.df.columns != self.df.columns[self.target]]
            self.y = self.df.iloc[:, self.target]

            # Initializing names for categories and numerical features.
            self.categories = None
            self.numerical = None

            # Initializing train-validation-test split.
            self.train = None
            self.test = None
            self.validation = None

            # Initializing the scaler and encoders to perform inverse transformation.
            self.scaler = None
            self.target_encoder = None
            self.feature_encoder = None

            # Initializing logging.
            logging.basicConfig(
                level=logging.INFO,
                filemode="a",
                filename="../reports/logs.log",
                format="%(asctime)s - %(levelname)s - %(message)s"
            )
            write_logs(message="Constructor Executed Successfully.", error=False)

        except Exception as ConstructorError:
            write_logs(
                error_reason=ConstructorError,
                error=True,
                message=f"Constructor Could Not Run Due To {ConstructorError}"
            )

    def categories_numerical(self) -> None:
        """
        This is an internal helper function.
        You can call the function to get categorical and numerical
        variables, under attributes categorical and numerical.

        The function does not return anything.
        :return: None
        """
        try:
            # Finding numerical variables.
            self.numerical = self.df.select_dtypes(include=["number"]).columns.tolist()

            # Finding categorical variables.
            self.categories = []
            for column in self.df.columns:
                if column not in self.numerical:
                    self.categories.append(column)

            # Removing target if added.
            target_name = self.df.columns[self.target]
            if target_name in self.categories:
                self.categories.remove(target_name)

            if target_name in self.numerical:
                self.numerical.remove(target_name)

            write_logs(
                error=False,
                message="Successfully Separated Numerical And Categorical Variables In The Dataset."
            )

        except Exception as SeparationError:
            write_logs(
                message=f"Separation Of Dataset Failed Due To {SeparationError}",
                error_reason=SeparationError,
                error=True
            )

    def missing_percentage(self) -> float:
        """
        The function is a helper function for missing().

        You can call it to know the total amount of missing data in the dataset,
        as it returns the percentage of total missing data in the dataset.
        :return: None.
        """
        try:
            # Finding total data items.
            data_items = self.df.compute().shape[0] * self.df.compute().shape[1]

            # Finding total missing data items.
            data_missing = self.df.compute().isnull().sum().sum()

            # Finding and returning percentage.
            percent = (data_missing / data_items) * 100

            write_logs(
                error=False,
                message="Successfully Found The Missing Percentage In The Dataset."
            )
            return percent

        except Exception as PercentError:
            write_logs(
                error_reason=PercentError,
                error=True,
                message=f"Could Not Find Total Missing Percentage Due To {PercentError}"
            )

    def missing(self, impute: bool) -> None:
        """
        The function is responsible for handling the missing values in the dataset.
        :param impute: Pass True to handle by imputation,
        else it will be handled by removal.
        :return: None.
        """
        try:
            # Finding categorical and numerical column names.
            self.categories_numerical()

            # Finding missing percentage.
            missing_percent = self.missing_percentage()

            # Converting to pandas dataframe.
            pandas = self.df.compute()

            if missing_percent > 5 or impute:
                # Create simple imputer for imputing numerical types
                numerical_imputer = KNNImputer(n_neighbors=5)
                pandas[self.numerical] = numerical_imputer.fit_transform(pandas[self.numerical])

                # Create simple imputer for imputing categorical types
                categorical_imputer = SimpleImputer(strategy="most_frequent")
                pandas[self.categories] = categorical_imputer.fit_transform(pandas[self.categories])

                # Store back as dask dataframe.
                self.df = dd.from_pandas(pandas)

            else:
                self.df = self.df.dropna()

            write_logs(
                error=False,
                message="Successfully Handled Missing Values."
            )

        except Exception as MissingError:
            write_logs(
                error_reason=MissingError,
                message=f"Could Not Deal With Missing Values Due To {MissingError}",
                error=True
            )

    def duplicates(self) -> None:
        """
        The function deletes the presence of exact dataframes in the dataset.
        :return: None.
        """
        try:
            # Dropping duplicates and storing in the dataframe itself.
            self.df = self.df.drop_duplicates()

            write_logs(
                error=False,
                message=f"Successfully Handled Duplicates"
            )

        except Exception as DuplicateError:
            write_logs(
                error_reason=DuplicateError,
                error=True,
                message=f"Could Not Handle Duplicates Due To {DuplicateError}"
            )

    def outliers(self, column: str) -> None:
        """
        The function handles all the outliers in the dataset using
        Inter-Quantile Range method and stores the results in self.df
        :param column: The name of the column to handle outliers.
        :return: None.
        """
        # Finding 25th And 75th Percentile.
        q1 = self.df.iloc[:, column].quantile(0.25)
        q3 = self.df.iloc[:, column].quantile(0.75)
        iqr = q3 - q1

        # Calculating upper and lower bounds.
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        self.df = self.df[(self.df.iloc[:, column] >= lower_bound) & (self.df.iloc[:, column]) <= upper_bound]

    def encode_target(self) -> None:
        """
        The function encodes the target variable using the LabelEncoder for classification,
        or keeps it as-is for regression.
        :return: None
        """
        try:
            # For regression, skip encoding - keep continuous values
            if self.problem_type == "regression":
                write_logs(
                    error=False,
                    message="Regression target - no encoding needed."
                )
                return

            # For classification, encode if not numeric
            # Convert for pandas.
            pandas = self.y.compute()

            # Checking for numeric type.
            if not is_numeric_dtype(pandas):
                # Initializing LabelEncoder.
                self.target_encoder = LabelEncoder()

                # Fitting and transforming the feature to the LabelEncoder.
                self.y = dd.from_dask_array(self.target_encoder.fit_transform(self.y))

                write_logs(
                    error=False,
                    message="Successfully Label Encoded Target."
                )
            else:
                write_logs(
                    error=False,
                    message="No Need To Encode Feature."
                )

        except Exception as LabelEncodeError:
            write_logs(
                error=True,
                error_reason=LabelEncodeError,
                message=f"Could Not Encode Target Due To {LabelEncodeError}"
            )

    def encode_features(self) -> None:
        """
        The function encodes the features using the OneHotEncoder.
        :return: None.
        """
        try:
            # Categorizing Dataset.
            self.X.categorize()

            # Finding the categorical and numerical columns.
            self.categories_numerical()

            # Changing column categories.
            for col in self.categories:
                self.X[col] = self.X[col].astype("category")

            # Instantiating ColumnTransformer
            self.feature_encoder = OneHotEncoder(sparse_output=False)

            # Fitting and transforming data.
            encoded = self.feature_encoder.fit_transform(self.X[self.categories].categorize())

            # Create the new encoded dataframe.
            encoded_df = dd.concat([self.X.drop(columns=self.categories), encoded], axis=1)

            # Storing in self.df
            self.X = encoded_df

            write_logs(
                error=False,
                message="Successfully Encoded Categorical Variables"
            )

            # Man-O-Man is this solved! Sat on this for 2 days.

        except Exception as EncodeFeatureError:
            write_logs(
                error=True,
                error_reason=EncodeFeatureError,
                message=f"Could Not Encode Features Due To {EncodeFeatureError}"
            )

    def train_test_validation(self, output_df: dd.DataFrame) -> None:
        """
        The function splits the data into three splits
        for training, testing and validation.
        :return: None.
        """
        # Getting the train and test split.
        self.train, self.test = train_test_split(output_df, test_size=0.2, random_state=42)

        # Getting the training data re-split, and validation data
        # 0.8 * 0.25 = 0.2
        self.train, self.validation = train_test_split(self.train, test_size=0.25, random_state=42)

    def scale(self, numerical_columns: list[str], type_of_data: str) -> None:
        """
        The function scales all the numerical values in the dataset.
        It uses the StandardScaler.
        :return: None.
        """
        try:
            if type_of_data == "train":
                # Finding numerical and categorical columns.
                self.categories_numerical()

                # Instantiating StandardScaler
                self.scaler = StandardScaler()

                # Fitting and transforming data.
                scaled = self.scaler.fit_transform(self.train[numerical_columns])

                # Replacing the original data.
                self.train[numerical_columns] = scaled

                write_logs(
                    error=False,
                    message="Successfully Scaled Training Data."
                )

            elif type_of_data == "test":
                # Finding numerical and categorical columns.
                self.categories_numerical()

                # Fitting and transforming data.
                scaled = self.scaler.transform(self.test[numerical_columns])

                # Replacing the original data.
                self.test[numerical_columns] = scaled

                write_logs(
                    error=False,
                    message="Successfully Scaled Testing Data."
                )

            else:
                # Finding numerical and categorical columns.
                self.categories_numerical()

                # Fitting and transforming data.
                scaled = self.scaler.transform(self.validation[numerical_columns])

                # Replacing the original data.
                self.validation[numerical_columns] = scaled

                write_logs(
                    error=False,
                    message="Successfully Scaled Validation Data."
                )

        except Exception as ScaleError:
            write_logs(
                message=f"Could Not Scale Numerical Data Due To {ScaleError}",
                error=True,
                error_reason=ScaleError
            )

    def preprocess(self, impute_missing: bool = True) -> list:
        """
        The function calls all the necessary preprocessing
        methods and returns the preprocessed dataframe.
        :param impute_missing: Whether to handle missing data by imputation or not.
        :return: A list of train, test and validation dataframes.
        """
        try:
            # Handle missing data
            self.missing(impute=impute_missing)

            # Remove duplicates
            self.duplicates()

            # Encode the target variable
            self.encode_target()

            # Encode the feature variables
            self.encode_features()

            # Combining preprocessed data.
            output = dd.concat([self.X, self.y.to_frame()], axis=1)

            # Perform train, test and validation split.
            self.train_test_validation(output_df=output)

            # Scale numerical features
            self.scale(numerical_columns=self.numerical, type_of_data="train")
            self.scale(numerical_columns=self.numerical, type_of_data="test")
            self.scale(numerical_columns=self.numerical, type_of_data="validation")

            write_logs(
                error=False,
                message="Data Preprocessing Completed Successfully."
            )
            return [self.train, self.test, self.validation]

        except Exception as PreprocessError:
            write_logs(
                error=True,
                error_reason=PreprocessError,
                message=f"Data Preprocessing Failed Due To {PreprocessError}"
            )

    def inverse_transform(self, data: dd.DataFrame) -> dd.DataFrame:
        try:
            # Inverse transform the numerical features
            numerical_cols = [col for col in data.columns if col in self.numerical]
            data[numerical_cols] = self.scaler.inverse_transform(data[numerical_cols])

            # Inverse transform the categorical features
            if self.feature_encoder is not None:
                categorical_cols = self.feature_encoder.get_feature_names_out(self.categories)
                encoded_df = data[categorical_cols].compute()
                decoded = self.feature_encoder.inverse_transform(encoded_df)
                decoded_df = dd.DataFrame(decoded)
                data = data.drop(columns=categorical_cols)
                data = dd.concat([data, dd.from_pandas(decoded_df, npartitions=data.npartitions)], axis=1)

            # Inverse transform the target
            if self.feature_encoder is not None:
                data[self.df.columns[self.target]] = self.feature_encoder.inverse_transform(
                    data[self.df.columns[self.target]].compute())

            write_logs(
                error=False,
                message="Successfully Inverse Transformed Data."
            )

            return data

        except Exception as InverseTransformError:
            write_logs(
                error=True,
                error_reason=InverseTransformError,
                message=f"Could Not Inverse Transform Data Due To {InverseTransformError}"
            )
