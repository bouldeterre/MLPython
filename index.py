import click
from demo import demo

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn
import matplotlib.pyplot as plt


class DataSet:
    def __init__(self, _X_train, _X_test, _y_train, _y_test):
        self.X_train = _X_train
        self.y_train = _y_train
        self.X_test = _X_test
        self.y_test = _y_test

    def __str__(self):
        return f"Set Shape:\n X_train:{self.X_train.shape} y_train:{self.y_train.shape} \n \
X_test:{self.X_test.shape} y_test:{self.y_test.shape}"


@click.group()
def cli():
    pass


@click.command()
def datasetdesc():
    iris_dataset = load_iris()
    print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
    print(iris_dataset["DESCR"][:983])


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


@click.command()
def run():
    print("Run:")
    iris_dataset = load_iris()

    myset = DataSet(
        *train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0)
    )
    print(myset)


cli.add_command(demo.demo_numpy)
cli.add_command(demo.demo_scipy)
cli.add_command(demo.demo_plot)
cli.add_command(demo.demo_panda)
cli.add_command(run)
cli.add_command(datasetdesc)
cli.add_command(datasetscatter)

if __name__ == "__main__":
    cli()
