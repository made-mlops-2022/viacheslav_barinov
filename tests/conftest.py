import pytest
from typing import List
import numpy as np
import pandas as pd

from ml_project import (
    Features,
    TrainingParams,
    SplittingParams,
    LogisticRegressionParams,
    fix_path,
)


@pytest.fixture(scope="session")
def target_col():
    return "condition"


@pytest.fixture(scope="session")
def categorical_features() -> List[str]:
    return ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]


@pytest.fixture(scope="session")
def numerical_features() -> List[str]:
    return ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]


@pytest.fixture()
def params(
    categorical_features, numerical_features, target_col, tmpdir
) -> TrainingParams:
    np.random.seed(42)
    rows_number = 100
    data = pd.DataFrame()

    for col in categorical_features:
        values = [0, 1, 2, 3]
        column = np.random.choice(values, rows_number)
        data[col] = column

    for col in numerical_features:
        column = np.random.randint(200.0, size=rows_number)
        data[col] = column

    test_filename = tmpdir.mkdir("tmpdir").join("test_data.csv")
    train_filename = tmpdir.join("tmpdir/train_data.csv")
    preprocess_pipeline_p = tmpdir.join("tmpdir/preprocess_pipeline.pkl")
    model_p = tmpdir.join("tmpdir/model.pkl")
    predict_p = tmpdir.join("tmpdir/predict.csv")
    data.to_csv(test_filename, index_label=False)
    data[target_col] = np.random.choice([0, 1], rows_number)
    data.to_csv(train_filename, index_label=False)

    features = Features(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col,
    )
    params = TrainingParams(
        input_train_data_path=train_filename,
        input_test_data_path=test_filename,
        preprocess_pipeline_path=preprocess_pipeline_p,
        model_path=model_p,
        predict_path=predict_p,
        model_type="LogisticRegression",
        features=features,
        splitting_params=SplittingParams(val_size=0.1, random_state=42, stratify=True),
        logistic_regression_params=LogisticRegressionParams(
            random_state=0, penalty="l2", C=0.9
        ),
    )
    return params


@pytest.fixture()
def real_params(
    categorical_features, numerical_features, target_col, tmpdir
) -> TrainingParams:
    test_filename = tmpdir.mkdir("tmpdir_real_data").join("test_data.csv")
    train_filename = tmpdir.join("tmpdir_real_data/train_data.csv")
    preprocess_pipeline_p = tmpdir.join("tmpdir_real_data/preprocess_pipeline.pkl")
    model_p = tmpdir.join("tmpdir_real_data/model.pkl")
    predict_p = tmpdir.join("tmpdir_real_data/predict.csv")

    train_data = pd.read_csv(fix_path("data_csv/train_data.csv"))
    train_data.to_csv(train_filename, index_label=False)
    test_data = pd.read_csv(fix_path("data_csv/test_data.csv"))
    test_data.to_csv(test_filename, index_label=False)

    features = Features(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col,
    )
    real_params = TrainingParams(
        input_train_data_path=train_filename,
        input_test_data_path=test_filename,
        preprocess_pipeline_path=preprocess_pipeline_p,
        model_path=model_p,
        predict_path=predict_p,
        model_type="LogisticRegression",
        features=features,
        splitting_params=SplittingParams(val_size=0.1, random_state=42, stratify=True),
        logistic_regression_params=LogisticRegressionParams(
            random_state=0, penalty="l2", C=0.9
        ),
    )
    return real_params
