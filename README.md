# Домашнее задание №3
Для корректной работы с переменными, созданными из UI:
------------
    export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
------------
Для сборки и запуска airflow:
------------
    docker compose up --build
------------

Для правильной работы dag'ов:
- Запустить dag data_generator
- необходимо создать File (path) connection с названием 'fs_default' и в поле Extra ввести {"path": "/opt/airflow/data"}
- Запустить dag train
- для dag'а 'predict' необходимо создать airflow Variables с названием MODELPATH и TRANSFORMERPATH со значением /data/models/{{ds}}/model.pkl и /data/transformer_model/{{ds}}/transform.pkl, где {{ds}} - дата в формате YYYY-MM-DD
- Запустить dag predict