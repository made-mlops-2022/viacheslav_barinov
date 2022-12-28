import os
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
import click

AVG_DATASET_SIZE = 100
FEATURES_FILENAME = "features.csv"
TARGETS_FILENAME = "targets.csv"


@click.command("generate")
@click.option("--dir_in", default="../data/raw")
def generate(dir_in: str) -> None:
    os.makedirs(dir_in, exist_ok=True)

    wine_dataset = load_wine()
    real_data = wine_dataset['data']
    real_targets = wine_dataset['target']

    fake_features = []
    fake_targets = []

    for _ in range(round(np.random.normal(1.0, 0.1) * AVG_DATASET_SIZE)):
        row_number = np.random.randint(0, len(real_data) - 1)
        fake_features.append(real_data[row_number])
        fake_targets.append(real_targets[row_number])

    features = pd.DataFrame(fake_features)
    targets = pd.DataFrame(fake_targets)

    features.to_csv(os.path.join(dir_in, FEATURES_FILENAME), index=False)
    targets.to_csv(os.path.join(dir_in, TARGETS_FILENAME), index=False)


if __name__ == "__main__":
    generate()
