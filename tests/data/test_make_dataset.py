import pandas as pd

from ml_project import (
    TrainingParams,
    split_train_val_data,
)


def test_split_train_val_data(params: TrainingParams):
    data = pd.read_csv(params.input_train_data_path)
    target = data[params.features.target_col]
    data.drop(columns=[params.features.target_col], inplace=True)
    train_data, val_data, train_target, val_target = split_train_val_data(
        data, target, params
    )

    val_size = params.splitting_params.val_size
    expected_val_size = int(val_size * data.shape[0])
    expected_train_size = int((1 - val_size) * data.shape[0])

    cat_features = params.features.categorical_features
    num_features = params.features.numerical_features
    feature_count = len(cat_features) + len(num_features)

    assert train_data.shape[1] == feature_count
    assert val_data.shape[1] == feature_count
    assert train_data.shape[0] == expected_train_size
    assert val_data.shape[0] == expected_val_size
    assert len(val_target) == expected_val_size
    assert len(train_target) == expected_train_size
    val_proportion = len(val_target == 1) / len(val_target)
    tarain_proportion = len(train_target == 1) / len(train_target)
    assert val_proportion == tarain_proportion
