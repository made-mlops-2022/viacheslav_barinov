import os
import pandas as pd
from ml_project import get_model, save_model, save_predict, extract_target
from sklearn.linear_model import LogisticRegression


def test_get_model(params):
    model = get_model(params)
    assert isinstance(model, LogisticRegression)


def test_save_model(params):
    model = get_model(params)
    save_model(model, params.model_path)
    assert os.path.exists(params.model_path)


def test_open_model(params):
    model = get_model(params)
    assert isinstance(model, LogisticRegression)


def test_save_predict(params):
    data = pd.read_csv(params.input_train_data_path)
    _, target = extract_target(data, params)
    save_predict(params.features.target_col, target, params.predict_path)
    assert os.path.exists(params.predict_path)
