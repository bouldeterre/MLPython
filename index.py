from demo import demo
from graph import graph
from models.data_set import DataSet
from sklearn.model_selection import train_test_split

import click
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

@click.group()
def cli():
    pass


@click.command()
def datasetdesc():
    iris_dataset = load_iris()
    print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
    print(iris_dataset["DESCR"][:983])



@click.command()
def run():
    print("Run:")
    iris_dataset = load_iris()

    myset = DataSet(
        *train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0, test_size=0.25)
    )
    print(myset)

    #Model
    knn = KNeighborsClassifier(n_neighbors=1)
    #Train with data
    knn.fit(myset.X_train, myset.y_train)

    sepalLength = 5
    sepalWidth = 2.9
    petalLength = 1
    petalWidth = 0.2
    
    X_new = np.array([[sepalLength, sepalWidth, petalLength, petalWidth]])
    print("X_new.shape: {}".format(X_new.shape))
    
    prediction = knn.predict(X_new)
    print("Prediction: {}".format(prediction))
    print("Predicted target name: {}".format(
        iris_dataset['target_names'][prediction]))

    y_pred = knn.predict(myset.X_test)
    print("Test set predictions:\n {}".format(y_pred))
    score = np.mean(y_pred == myset.y_test)
    print("Test set score: {:.2f}".format(score))
    
cli.add_command(demo.demo_numpy)
cli.add_command(demo.demo_scipy)
cli.add_command(demo.demo_plot)
cli.add_command(demo.demo_panda)
cli.add_command(graph.datasetscatter)
cli.add_command(run)
cli.add_command(datasetdesc)

if __name__ == "__main__":
    cli()
