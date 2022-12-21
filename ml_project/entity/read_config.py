from marshmallow_dataclass import class_schema
from dataclasses import dataclass
import yaml

from .features import Features
from .split_params import SplittingParams
from .logistic_regression_params import LogisticRegressionParams


@dataclass()
class TrainingParams:
    input_train_data_path: str
    input_test_data_path: str
    preprocess_pipeline_path: str
    model_path: str
    predict_path: str
    model_type: str
    features: Features
    splitting_params: SplittingParams
    logistic_regression_params: LogisticRegressionParams


TrainingParamsSchema = class_schema(TrainingParams)


def read_training_params(path: str) -> TrainingParams:
    with open(path, "r") as input_stream:
        schema = TrainingParamsSchema()
        return schema.load(yaml.safe_load(input_stream))


def fix_path(path: str) -> str:
    # return "../" + path
    return path  # sometimes we need ('../' + path)


def fix_config(params: TrainingParams) -> TrainingParams:
    params.input_train_data_path = fix_path(params.input_train_data_path)
    params.input_test_data_path = fix_path(params.input_test_data_path)
    params.preprocess_pipeline_path = fix_path(params.preprocess_pipeline_path)
    params.model_path = fix_path(params.model_path)
    params.predict_path = fix_path(params.predict_path)
    return params
