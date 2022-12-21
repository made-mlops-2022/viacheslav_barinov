import pandas as pd
from ml_project import (
    extract_target,
    open_model,
    run_train_pipeline,
)


def test_extract_target(params):
    data = pd.read_csv(params.input_train_data_path)
    expected_target = data[params.features.target_col]
    got_data, got_target = extract_target(data, params)

    cat_features = params.features.categorical_features
    num_features = params.features.numerical_features
    feature_count = len(cat_features) + len(num_features)

    assert (expected_target == got_target).all()
    assert got_data.shape[0] == got_data.shape[0]
    assert got_data.shape[1] == feature_count


def test_extract_cat_num_features(params):
    cat_features = params.features.categorical_features
    num_features = params.features.numerical_features

    assert cat_features == params.features.categorical_features
    assert num_features == params.features.numerical_features


def test_preprocess_features(params):
    run_train_pipeline(params)
    preprocess_pipeline = open_model(params.preprocess_pipeline_path)
    # ohe = open_model(params.ohe_path)
    data = pd.read_csv(params.input_train_data_path)

    # preprocessed_data = preprocess_features(data, ohe, transformer, params)
    preprocessed_data = preprocess_pipeline.transform(data)

    cat_features = params.features.categorical_features
    num_features = params.features.numerical_features
    expected_feature_count = 4 * len(cat_features) + len(num_features)

    assert preprocessed_data.shape[0] == data.shape[0]
    assert preprocessed_data.shape[1] == expected_feature_count
