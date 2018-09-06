import click
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display


@click.command()
def demo_numpy():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    print("x:\n{}".format(x))


@click.command()
def demo_scipy():
    # Create a 2D NumPy array with a diagonal of ones, and zeros everywhere else
    eye = np.eye(4)
    print("NumPy array:\n{}".format(eye))

    # Convert the NumPy array to a SciPy sparse matrix in CSR format
    # Only the nonzero entries are stored
    sparse_matrix = sparse.csr_matrix(eye)
    print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))

    data = np.ones(4)
    print("NumPy ones:\n{}".format(data))
    row_indices = np.arange(4)
    print("row_indices:\n{}".format(row_indices))
    col_indices = np.arange(4)
    eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
    print("COO representation:\n{}".format(eye_coo))


@click.command()
def demo_plot():
    # Generate a sequence of numbers from -10 to 10 with 100 steps in between
    x = np.linspace(-10, 10, 100)
    # Create a second array using sine
    y = np.sin(x)
    # The plot function makes a line chart of one array against another
    plt.plot(x, y, marker="x")
    plt.show()


@click.command()
def demo_panda():
    # create a simple dataset of people
    data = {
        "Name": ["John", "Anna", "Peter", "Linda"],
        "Location": ["New York", "Paris", "Berlin", "London"],
        "Age": [24, 13, 53, 33],
    }
    data_pandas = pd.DataFrame(data)
    # IPython.display allows "pretty printing" of dataframes
    # in the Jupyter notebook
    display(data_pandas)
    print("filter Age > 30")
    display(data_pandas[data_pandas.Age > 30])
