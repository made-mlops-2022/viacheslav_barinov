import click
from entity import read_training_params, fix_path, fix_config
from models import run_train_pipeline


def train_model(config_path: str):
    config_path = fix_path(config_path)
    params = fix_config(read_training_params(config_path))

    run_train_pipeline(params)


@click.command(name="train_model")
@click.argument("config_path")
def train_model_command(config_path: str):
    train_model(config_path)


if __name__ == "__main__":
    train_model_command()
