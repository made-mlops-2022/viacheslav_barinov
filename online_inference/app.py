import logging
import os
import pickle
from typing import List, Union, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, conlist
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


class ConditionModel(BaseModel):
    data: List[conlist(Union[float, str, None], min_items=0, max_items=80)]
    features: List[str]


class ConditionResponse(BaseModel):
    condition: float


model: Optional[Pipeline] = None
preprocess_pipeline: Optional[Pipeline] = None


def make_predict(
    data: List, features: List[str], mod: Pipeline, pipe: Pipeline
) -> List[ConditionResponse]:
    data = pd.DataFrame(data, columns=features)

    preprocessed_data = pipe.transform(data)

    predicts = mod.predict(preprocessed_data)
    return [
        ConditionResponse(condition=int(pred)) for pred in predicts
    ]


app = FastAPI()


@app.get("/")
def main():
    return "Hello! Go to /docs to see methods :)"


@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("PATH_TO_MODEL", default='../model/model_19.pkl')
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_object(model_path)

    global preprocess_pipeline
    pipe_path = os.getenv("PATH_TO_PIPE", default='../model/preprocess_pipeline_19.pkl')
    if pipe_path is None:
        err = f"PATH_TO_PIPE {pipe_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    preprocess_pipeline = load_object(pipe_path)


@app.get("/health")
def health() -> int:
    return 200 if not (model is None) else 404


@app.get("/predict", response_model=List[ConditionResponse])
def predict(request: ConditionModel):
    return make_predict(request.data, request.features, model, preprocess_pipeline)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
