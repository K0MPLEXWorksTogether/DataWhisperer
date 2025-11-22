import io
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import dask.dataframe as dd
from src.preprocess import DataPreProcessor


class Hierarchies(DataPreProcessor):
    """
    The class performs hierarchical clustering on a passed dataset, using
    scipy's cluster.
    It returns a dendrogram displaying the hierarchies in the dataset.
    It also inherits from DataPreProcessor
    """

    def __init__(self, dataframe: dd.DataFrame, index_of_target: int, problem_type: str = None):
        super().__init__(dataframe, index_of_target, problem_type)

        # Assuming preprocess returns a tuple where the first element is the processed data
        self.train = self.preprocess(impute_missing=True)[0].compute()

    def dendrogram(self) -> io.BytesIO:
        hierarchical_clusters = linkage(self.train, method="ward", metric="euclidean")

        plt.figure(figsize=(12, 8))
        dendrogram(hierarchical_clusters)
        plt.xlabel("Features")
        plt.ylabel("Distance")
        plt.title("Dendrogram Of Hierarchies")

        dendrogram_image = io.BytesIO()
        plt.savefig(dendrogram_image, format="png")
        plt.close()
        dendrogram_image.seek(0)

        return dendrogram_image


def main():
    """
    Testing function for understanding the functionality. Run the
    program to call main().
    :return: None.
    """

    data = dd.read_csv("../data/weather.csv")

    hierarchies = Hierarchies(data, 10)
    print(hierarchies.dendrogram())


if __name__ == "__main__":
    main()
