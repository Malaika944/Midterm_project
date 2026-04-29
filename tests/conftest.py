from unittest.mock import MagicMock, patch
import numpy as np

mock_model = MagicMock()
mock_model.predict_proba.return_value = np.array([[0.6, 0.4]])

with patch("noshow_iq.api.get_model", return_value=mock_model), \
        patch("noshow_iq.api.get_db"):
    from fastapi.testclient import TestClient
    from noshow_iq.api import app

client = TestClient(app)

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


def test_health_returns_200():
    assert client.get("/health").status_code == 200


def test_health_has_status_key():
    assert "status" in client.get("/health").json()


def test_predict_returns_200():
    assert client.post("/predict", json=SAMPLE).status_code == 200


def test_predict_has_risk_level():
    assert "risk_level" in client.post("/predict", json=SAMPLE).json()


def test_predict_risk_level_valid():
    assert client.post(
        "/predict", json=SAMPLE
    ).json()["risk_level"] in ["high", "low"]


def test_predict_has_recommendation():
    assert "recommendation" in client.post("/predict", json=SAMPLE).json()