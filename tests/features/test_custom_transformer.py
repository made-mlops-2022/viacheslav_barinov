import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from ml_project import TrainingParams, CustomTransformer


def test_custom_transformer(params: TrainingParams):
    data = pd.read_csv(params.input_test_data_path)
    custom_transformer = CustomTransformer(params.features.numerical_features)

    with pytest.raises(NotFittedError):
        custom_transformer.transform(data)

    custom_transformer.fit(data)
    transformed_data = custom_transformer.transform(
        data[params.features.numerical_features]
    )

    for col_idx in range(len(params.features.numerical_features)):
        assert np.isclose(np.mean(transformed_data[:, col_idx]), 0)
        assert np.isclose(np.var(transformed_data[:, col_idx]), 1)
