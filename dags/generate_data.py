from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount


GENERATE_DIR_NAME = "data/raw/{{ ds }}"
default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
        "data_generator",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(0),
) as dag:
    generate = DockerOperator(
        image="airflow-data-generate",
        command=f"--dir_in {GENERATE_DIR_NAME}",
        task_id="docker-airflow-generate",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=[Mount(
            source="/home/slava/mlops/airflow-examples/data/",
            target="/data",
            type='bind'
            )]
    )

    generate
