import click
from demo import demo

from sklearn.datasets import load_iris


@click.group()
def cli():
    pass
    

@click.command()
def run ():
    print("Run:")
    iris_dataset = load_iris()
    print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))


cli.add_command(demo.demo_numpy)
cli.add_command(demo.demo_scipy)
cli.add_command(demo.demo_plot)
cli.add_command(demo.demo_panda)
cli.add_command(run)

if __name__ == "__main__":
    cli()
