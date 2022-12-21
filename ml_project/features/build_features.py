import pandas as pd
import numpy as np
import logging
from typing import List, Tuple

from ml_project.entity import TrainingParams

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def extract_target(
    data: pd.DataFrame, params: TrainingParams
) -> Tuple[pd.DataFrame, np.ndarray]:
    target_col = params.features.target_col
    target = data[target_col].values
    data = data.drop(columns=[target_col])
    return data, target


def extract_cat_num_features(params: TrainingParams) -> Tuple[List[str], List[str]]:
    cat_features = params.features.categorical_features
    num_features = params.features.numerical_features

    logger.info(
        f"Preprocess train data parameters: "
        f"categorical parameters = {cat_features}, "
        f"numerical_features = {num_features}, "
        f"target_col = {params.features.target_col}"
    )
    return cat_features, num_features
