import os

from ml_project import (
    TrainingParams,
    run_train_pipeline,
    run_test_pipeline,
)


def test_train(params: TrainingParams):
    metrics = run_train_pipeline(params)
    assert metrics["Accuracy"] > 0
    assert metrics["ROC AUC score"] > 0
    assert metrics["F1-score"] > 0
    assert os.path.exists(params.model_path)
    assert os.path.exists(params.preprocess_pipeline_path)


def test_predict(params: TrainingParams):
    run_train_pipeline(params)
    metrics = run_test_pipeline(params)
    assert metrics["Accuracy"] > 0
    assert metrics["ROC AUC score"] > 0
    assert metrics["F1-score"] > 0
    assert os.path.exists(params.predict_path)


def test_accuracy_on_real_data(real_params: TrainingParams):
    metrics = run_train_pipeline(real_params)

    assert abs(metrics["Accuracy"] - 0.8) <= 0.1
    assert abs(metrics["ROC AUC score"] - 0.8) <= 0.1
    assert abs(metrics["F1-score"] - 0.8) <= 0.1
    assert os.path.exists(real_params.model_path)
    assert os.path.exists(real_params.preprocess_pipeline_path)

    metrics = run_test_pipeline(real_params)

    assert abs(metrics["Accuracy"] - 0.8) <= 0.1
    assert abs(metrics["ROC AUC score"] - 0.8) < 0.1
    assert abs(metrics["F1-score"] - 0.8) < 0.1
    assert os.path.exists(real_params.predict_path)
