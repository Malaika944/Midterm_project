import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


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


@pytest.fixture(scope="module")
def client():
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.6, 0.4]]

    with patch("noshow_iq.api.get_model", return_value=mock_model), \
         patch("noshow_iq.api.get_db") as mock_db:

        mock_col = MagicMock()
        mock_col.insert_one = MagicMock()
        mock_col.find.return_value.sort.return_value.limit.return_value = []
        mock_col.aggregate.return_value = []
        mock_db.return_value.__getitem__ = MagicMock(return_value=mock_col)

        from noshow_iq.api import app
        with TestClient(app) as c:
            yield c


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
    