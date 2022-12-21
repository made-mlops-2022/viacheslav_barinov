FROM python:3.6-slim-stretch

COPY online_inference/requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY model/model_19.pkl /model.pkl
COPY model/preprocess_pipeline_19.pkl /preprocess_pipeline.pkl
COPY online_inference/app.py /app.py
COPY ml_project /ml_project

WORKDIR .

ENV PATH_TO_MODEL="/model.pkl"
ENV PATH_TO_PIPE="/preprocess_pipeline.pkl"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]