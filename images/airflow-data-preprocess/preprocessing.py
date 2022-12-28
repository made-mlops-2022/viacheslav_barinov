import os
import pandas as pd
import numpy as np
import click
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pickle

AVG_DATASET_SIZE = 100
FEATURES_FILENAME = "features.csv"
TARGETS_FILENAME = "targets.csv"
TRANSFORM_MODEL_FILENAME = "transform.pkl"


@click.command("preprocess")
@click.option("--dir_from", default="../data/raw")
@click.option("--dir_in", default="../data/processed")
@click.option("--transform_dir", default="../data/transformer_model")
def preprocess(dir_from: str, dir_in: str, transform_dir: str) -> None:
    os.makedirs(dir_from, exist_ok=True)
    os.makedirs(dir_in, exist_ok=True)
    os.makedirs(transform_dir, exist_ok=True)

    features = pd.read_csv(os.path.join(dir_from, FEATURES_FILENAME))
    targets = pd.read_csv(os.path.join(dir_from, TARGETS_FILENAME))

    transformer = Pipeline([
        ("fill_na", SimpleImputer(missing_values=np.nan, strategy='mean')),
        ("scaler", StandardScaler())
    ])

    transformer.fit(features)

    features_prep = pd.DataFrame(transformer.fit_transform(features))

    features_prep.to_csv(os.path.join(dir_in, FEATURES_FILENAME), index=False)
    targets.to_csv(os.path.join(dir_in, TARGETS_FILENAME), index=False)

    with open(os.path.join(transform_dir, TRANSFORM_MODEL_FILENAME), 'wb') as fout:
        pickle.dump(transformer, fout)


if __name__ == "__main__":
    preprocess()
