from typing import NoReturn

import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features) -> NoReturn:
        self.scaler = StandardScaler()
        self.numerical_features = numerical_features
        self.fitted = False

    def check_is_fitted(self):
        if not self.fitted:
            raise NotFittedError("CustomTransformer not fitted")

    def fit(self, data: pd.DataFrame):
        self.scaler.fit(data[self.numerical_features])
        self.fitted = True
        return self

    def transform(self, data: pd.DataFrame):
        self.check_is_fitted()
        data_copy = data[self.numerical_features].copy()
        data_copy = self.scaler.transform(data_copy)
        return data_copy
