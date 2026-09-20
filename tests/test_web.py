"""Flask route tests. The Predictor is mocked so these do not load BERT."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FAKE_PREDICTION = {
    "text": "Breakfast was poor but the staff were excellent.",
    "sentiment": {
        "label": "negative",
        "confidence": 0.62,
        "scores": {"negative": 0.62, "neutral": 0.1, "positive": 0.28},
    },
    "aspects": [
        {"label": "Breakfast", "score": 0.9},
        {"label": "Staff", "score": 0.85},
    ],
}


class FakePredictor:
    """Stand-in for the real Predictor so web tests never load BERT."""

    def predict(self, text):
        return FAKE_PREDICTION


@pytest.fixture
def client():
    """Build a Flask test client with a fake predictor swapped in.

    The module's own startup logic already skips loading real BERT
    checkpoints when the configured artifact directories do not exist, so
    importing this module here never touches the network.
    """
    from src.web import app as app_module

    app_module.predictor = FakePredictor()
    app_module.predictor_error = None
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as test_client:
        yield test_client


def test_get_index_renders_the_form(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Review sentence" in response.data


def test_post_valid_review_shows_prediction(client):
    response = client.post("/", data={"review": "Breakfast was poor but the staff were excellent."})
    assert response.status_code == 200
    assert b"negative" in response.data
    assert b"Breakfast" in response.data


def test_post_blank_review_is_rejected(client):
    response = client.post("/", data={"review": "   "})
    assert response.status_code == 200
    assert b"Please enter a review sentence." in response.data


def test_post_oversized_review_is_rejected(client):
    response = client.post("/", data={"review": "x" * 2001})
    assert response.status_code == 200
    assert b"Please limit the review" in response.data


def test_api_predict_returns_json(client):
    response = client.post("/api/predict", json={"review": "Great stay."})
    assert response.status_code == 200
    assert response.get_json() == FAKE_PREDICTION


def test_api_predict_rejects_blank_review(client):
    response = client.post("/api/predict", json={"review": ""})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_health_reports_ok_when_models_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
