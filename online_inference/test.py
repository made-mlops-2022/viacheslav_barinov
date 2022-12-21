from fastapi.testclient import TestClient
from app import app


def test_main():
    with TestClient(app) as client:
        response = client.get("/")
        assert 200 == response.status_code
        assert response.json() == 'Hello! Go to /docs to see methods :)'


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200


def test_predict():
    with TestClient(app) as client:
        request_features = ['age', 'sex', 'cp', 'trestbps', 'chol',
                            'fbs', 'restecg', 'thalach', 'exang',
                            'oldpeak', 'slope', 'ca', 'thal']
        request_data = [69.0, 1.0, 0.0, 160.0, 234.0, 1.0,
                        2.0, 131.0, 0.0, 0.1, 1.0, 1.0, 0.0]

        response = client.get(
            "/predict",
            json={"data": [request_data], "features": request_features},
        )
        assert response.status_code == 200
        assert response.json() == [{'condition': 0.0}]

        request_data = [58.0, 1.0, 2.0, 105.0, 240.0, 0.0,
                        2.0, 154.0, 1.0, 0.6, 1.0, 0.0, 2.0]
        response = client.get(
            "/predict",
            json={"data": [request_data], "features": request_features},
        )
        assert response.status_code == 200
        assert response.json() == [{'condition': 1.0}]
