import io

from src.preprocess import DataPreProcessor
from sklearn.cluster import DBSCAN
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score

import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DBScan:
    """
    The class performs DBSCAN clustering of the dataset, but first it performs
    a cross validation using GridSearchCV, to find the best eps and min_samples.
    It provides 3 methods:
    - search_hyperparameters(): To find the best hyperparameters for DBSCAN, as
    well as best possible validation score and labels.
    - cluster(): Actually form the clusters.
    - visualize(): Visualize the clusters in the dataset using colormaps.
    """

    def __init__(self, dataframe: dd.DataFrame, target_index: int, problem_type: str = None) -> None:
        """
        Preprocesses the data for training, testing and validation.
        :param dataframe: The dataframe to cluster.
        :param target_index: The index of the target column.
        :param problem_type: Optional problem type specification.
        """
        preprocess = DataPreProcessor(dataframe, target_index, problem_type)
        train, test, validation = preprocess.preprocess()

        self.train = train.compute()
        self.test = test.compute()
        self.validation = validation.compute()

        self.model = None
        self.predictions = None

    def search_hyperparameters(self) -> list:
        """
        The function finds the best hyperparameters, labels and
        silhouette score for the model.
        :return: A list of hyperparameters, score and labels.
        """
        best_score = -1
        best_params = None
        best_labels = None

        params_grid = {
            "eps": np.arange(0.1, 1.1, 0.1),
            "min_samples": range(2, 10)
        }

        self.validation.columns = self.validation.columns.astype(str)

        for params in ParameterGrid(params_grid):
            dbscan = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
            labels = dbscan.fit_predict(self.validation)

            # Need at least 2 clusters (excluding noise) for silhouette
            unique_labels = set(labels)
            if len(unique_labels) > 1:
                try:
                    score = silhouette_score(self.validation, labels)
                except Exception:
                    score = -1
                if score > best_score:
                    best_params = params
                    best_labels = labels
                    best_score = score

        # Fallback if no params produced >1 cluster
        if best_params is None:
            best_params = {"eps": 0.5, "min_samples": 5}
            dbscan = DBSCAN(eps=best_params["eps"], min_samples=best_params["min_samples"])
            best_labels = dbscan.fit_predict(self.validation)
            # Silhouette not meaningful with single cluster; keep score as None
            best_score = None

        return [best_params, best_score, best_labels]

    def cluster(self) -> None:
        """
        The function finds the best hyperparameters and initializes a model
        for training, which finds clusters.
        :return: None
        """

        test_results = self.search_hyperparameters()
        self.train.columns = self.train.columns.astype(str)

        params, _, labels = test_results
        # Ensure train columns are strings for consistency
        self.model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
        self.predictions = self.model.fit_predict(self.train)
        self.train["clusters"] = self.predictions

    def visualize(self) -> io.BytesIO:
        """
        The function visualizes the clusters using a colormap.
        :return: A BytesIO object of the image.
        """
        plt.figure(figsize=(10, 12))
        sns.scatterplot(x=self.train.iloc[:, 0], y=self.train.iloc[:, 1], hue=self.train["clusters"], palette="viridis")
        plt.title('DBSCAN Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

        # Convert plot to bytes stream
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return buf

    def score(self) -> float:
        """
        The function finds and returns the silhouette score
        of the clustering.
        :return: The silhouette score.
        """
        # Guard against cases with <2 clusters
        if self.predictions is None:
            return None
        unique_labels = set(self.predictions)
        if len(unique_labels) <= 1:
            return None
        try:
            return silhouette_score(self.train, self.predictions)
        except Exception:
            return None


def main():
    """
    Test function. Run the file to view the functionality.
    :return: None.
    """
    read = dd.read_csv("../data/weather.csv")

    dbscan_test = DBScan(read, 10)
    dbscan_test.cluster()
    dbscan_test.visualize()


if __name__ == "__main__":
    main()
