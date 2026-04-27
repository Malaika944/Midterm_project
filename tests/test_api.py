import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="module")
def client():
    with patch("noshow_iq.api.MongoClient"), \
         patch("noshow_iq.api.load_model") as mock_load:

        mock = MagicMock()
        mock.predict_proba.return_value = [[0.6, 0.4]]
        mock_load.return_value = mock

        from noshow_iq.api import app
        with TestClient(app) as c:
            yield c


SAMPLE = {
    "age": 30,
    "scholarship": 0,
    "hipertension": 0,
    "diabetes": 0,
    "alcoholism": 0,
    "handcap": 0,
    "sms_received": 1,
    "days_in_advance": 5,
    "hour_of_booking": 10,
}


def test_health_returns_200(client):
    assert client.get("/health").status_code == 200


def test_health_has_status_key(client):
    assert "status" in client.get("/health").json()


def test_predict_returns_200(client):
    assert client.post("/predict", json=SAMPLE).status_code == 200


def test_predict_has_risk_level(client):
    assert "risk_level" in client.post("/predict", json=SAMPLE).json()


def test_predict_risk_level_valid(client):
    assert client.post(
        "/predict", json=SAMPLE
    ).json()["risk_level"] in ["high", "low"]


def test_predict_has_recommendation(client):
    assert "recommendation" in client.post("/predict", json=SAMPLE).json()