from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
from docker.types import Mount

VAL_SIZE = 0.3
METRICS_DIR_NAME = "/data/metrics/{{ ds }}"
GENERATE_DIR_NAME = "/data/raw/{{ ds }}"
PROCESSED_DIR_NAME = "/data/processed/{{ ds }}"
TRANSFORMER_DIR_NAME = "/data/transformer_model/{{ ds }}"
MODEL_DIR_NAME = "/data/models/{{ ds }}"
MOUNT_SOURCE = Mount(
    source="/home/slava/mlops/airflow-examples/data/",
    target="/data",
    type='bind'
    )

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
        "train",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(0),
) as dag:

    features_sensor = FileSensor(
        task_id="features_sensor",
        filepath="/opt/airflow/data/raw/{{ ds }}/features.csv"
    )

    targets_sensor = FileSensor(
        task_id="targets_sensor",
        filepath="/opt/airflow/data/raw/{{ ds }}/targets.csv"
    )

    preprocess_data = DockerOperator(
        image="airflow-data-preprocess",
        command=f"--dir_from {GENERATE_DIR_NAME} --dir_in {PROCESSED_DIR_NAME} --transform_dir {TRANSFORMER_DIR_NAME}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=[MOUNT_SOURCE]
    )

    split_data = DockerOperator(
        image="airflow-data-split",
        command=f"--dir_from {PROCESSED_DIR_NAME} --val_size {VAL_SIZE}",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=[MOUNT_SOURCE]
    )

    train_model = DockerOperator(
        image="airflow-train",
        command=f"--dir_from {PROCESSED_DIR_NAME} --dir_in {MODEL_DIR_NAME}",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=[MOUNT_SOURCE]
    )

    val_model = DockerOperator(
        image="airflow-validation",
        command=f"--model_dir_from {MODEL_DIR_NAME} --data_dir_from {PROCESSED_DIR_NAME} --metric_dir {METRICS_DIR_NAME}",
        task_id="docker-airflow-valid",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=[MOUNT_SOURCE]
    )

    [features_sensor, targets_sensor] >> preprocess_data
    preprocess_data >> split_data >> train_model >> val_model
