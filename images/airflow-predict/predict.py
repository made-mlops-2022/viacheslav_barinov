import os
import pandas as pd
import pickle
import click

DATA_FILENAME = "features.csv"
PREDICT_FILENAME = "predict.csv"


@click.command("predict")
@click.option("--dir_from", default="../data/raw")
@click.option("--transformer_path", default="../data/transformer_model/transform.pkl")
@click.option("--model_path", default="../data/models/model.pkl")
@click.option("--dir_in", default="../data/predictions")
def predict(dir_from: str, dir_in: str, model_path: str, transformer_path: str) -> None:
    os.makedirs(dir_from, exist_ok=True)
    os.makedirs(dir_in, exist_ok=True)

    data = pd.read_csv(os.path.join(dir_from, DATA_FILENAME))

    with open(transformer_path, 'rb') as fin:
        transformer = pickle.load(fin)

    with open(model_path, 'rb') as fin:
        model = pickle.load(fin)

    transform_data = pd.DataFrame(transformer.fit_transform(data))

    pred = pd.DataFrame(model.predict(transform_data))

    pred.to_csv(os.path.join(dir_in, PREDICT_FILENAME), index=False)


if __name__ == '__main__':
    predict()
