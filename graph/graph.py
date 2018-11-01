import click
import pandas as pd
from sklearn.datasets import load_iris
from models.data_set import DataSet
from sklearn.model_selection import train_test_split
import mglearn
import matplotlib.pyplot as plt

@click.command()
def datasetscatter():
    iris_dataset = load_iris()
    myset = DataSet(
        *train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0)
    )
    print(iris_dataset.feature_names)
    # Plot test data
    iris_dataframe = pd.DataFrame(myset.X_train, columns=iris_dataset.feature_names)
    pd.plotting.scatter_matrix(
        iris_dataframe,
        c=myset.y_train,
        figsize=(10, 10),
        marker="o",
        hist_kwds={"bins": 20},
        s=60,
        alpha=.5,
        cmap=mglearn.cm3,
    )
    plt.show()

