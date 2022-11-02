import pandas as pd
import numpy as np
import pickle
import logging
from typing import Union, Dict

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

from ml_project.entity import TrainingParams
from ml_project.data import split_train_val_data
from ml_project.features import extract_target, CustomTransformer

Model = Union[LogisticRegression, GaussianNB, ColumnTransformer]
logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def get_model(params: TrainingParams) -> Union[LogisticRegression, GaussianNB]:
    logger.info(f"Model type:{params.model_type}")
    lr_random_state = params.logistic_regression_params.random_state
    lr_penalty = params.logistic_regression_params.penalty
    lr_c = params.logistic_regression_params.C
    if params.model_type == "LogisticRegression":
        model = LogisticRegression(
            random_state=lr_random_state, penalty=lr_penalty, C=lr_c
        )
    elif params.model_type == "GaussianNB":
        model = GaussianNB()
    else:
        logger.warning("No such model_type! Use LogisticRegression")
        model = LogisticRegression(
            random_state=lr_random_state, penalty=lr_penalty, C=lr_c
        )
    return model


def evaluate_model(predict: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    result_metrics = {
        "Accuracy": metrics.accuracy_score(target, predict),
        "ROC AUC score": metrics.roc_auc_score(target, predict),
        "F1-score": metrics.f1_score(target, predict),
    }
    for key in result_metrics:
        logger.info(f"{key}, : {result_metrics[key]}")
    return result_metrics


def save_model(model: Model, path: str):
    logger.info(f"Saving model to {path}")
    with open(path, "wb") as f:
        pickle.dump(model, f)


def open_model(path: str) -> Model:
    logger.info(f"Opening model {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def save_predict(target: str, predict: np.ndarray, path: str):
    logger.info(f"Saving predict to {path}")
    predict_df = pd.DataFrame({target: predict})
    predict_df.to_csv(path, index=False)


def run_train_pipeline(params: TrainingParams) -> Dict[str, float]:
    data = pd.read_csv(params.input_train_data_path)
    data, target = extract_target(data, params)

    preprocess_pipeline = ColumnTransformer(
        transformers=[
            (
                "numeric",
                CustomTransformer(params.features.numerical_features),
                params.features.numerical_features,
            ),
            ("categorical", OneHotEncoder(), params.features.categorical_features),
        ]
    )
    preprocess_pipeline.fit(data)
    save_model(preprocess_pipeline, params.preprocess_pipeline_path)

    train_data, val_data, train_target, val_target = split_train_val_data(
        data, target, params
    )

    preprocessed_train_data = preprocess_pipeline.transform(train_data)

    model = get_model(params)
    model.fit(preprocessed_train_data, train_target)
    save_model(model, params.model_path)

    preprocessed_val_data = preprocess_pipeline.transform(val_data)
    predict = model.predict(preprocessed_val_data)
    result_metrics = evaluate_model(predict, val_target)
    return result_metrics


def run_test_pipeline(params: TrainingParams):
    target_col = params.features.target_col
    data = pd.read_csv(params.input_test_data_path)

    preprocess_pipeline = open_model(params.preprocess_pipeline_path)
    model = open_model(params.model_path)

    preprocessed_data = preprocess_pipeline.transform(data)

    predict = model.predict(preprocessed_data)

    answer = pd.read_csv(params.input_train_data_path)[target_col]

    result_metrics = evaluate_model(predict, answer)

    save_predict(target_col, predict, params.predict_path)

    return result_metrics
