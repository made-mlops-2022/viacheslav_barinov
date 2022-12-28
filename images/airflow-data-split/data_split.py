import os
from sklearn.model_selection import train_test_split
import pandas as pd
import click


FEATURES_FILENAME = "features.csv"
TARGETS_FILENAME = "targets.csv"

TRAIN_FEATURES_FILENAME = "train_features.csv"
TRAIN_TARGETS_FILENAME = "train_targets.csv"

VAL_FEATURES_FILENAME = "val_features.csv"
VAL_TARGETS_FILENAME = "val_targets.csv"


@click.command("split")
@click.option("--dir_from", default="../data/processed")
@click.option("--val_size", default=0.3)
def split(dir_from: str, val_size: float) -> None:
    os.makedirs(dir_from, exist_ok=True)

    X = pd.read_csv(os.path.join(dir_from, FEATURES_FILENAME))
    y = pd.read_csv(os.path.join(dir_from, TARGETS_FILENAME))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size)

    train_features = pd.DataFrame(X_train)
    train_targets = pd.DataFrame(y_train)
    train_features.to_csv(os.path.join(dir_from, TRAIN_FEATURES_FILENAME), index=False)
    train_targets.to_csv(os.path.join(dir_from, TRAIN_TARGETS_FILENAME), index=False)

    val_features = pd.DataFrame(X_val)
    val_targets = pd.DataFrame(y_val)
    val_features.to_csv(os.path.join(dir_from, VAL_FEATURES_FILENAME), index=False)
    val_targets.to_csv(os.path.join(dir_from, VAL_TARGETS_FILENAME), index=False)


if __name__ == "__main__":
    split()
