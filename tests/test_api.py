import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


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


def setup_module(module):
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.6, 0.4]])

    mock_col = MagicMock()
    mock_col.insert_one = MagicMock()
    mock_col.find.return_value.sort.return_value.limit.return_value = []
    mock_col.aggregate.return_value = []

    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_col)

    patch("noshow_iq.api.get_model", return_value=mock_model).start()
    patch("noshow_iq.api.get_db", return_value=mock_db).start()
    patch("noshow_iq.api.model", mock_model).start()


setup_module(None)

from fastapi.testclient import TestClient
from noshow_iq.api import app

client = TestClient(app)


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
