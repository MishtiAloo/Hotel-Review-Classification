"""Small Flask app that serves the sentiment and aspect predictor.

Keep this file focused on request handling. All model logic lives in
src/inference/predictor.py so the CLI, tests, and this app share one code
path for tokenization and thresholding.
"""

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import Predictor

MAX_REVIEW_LENGTH = 2000
EXAMPLE_SENTENCE = "Breakfast was poor but the staff were excellent."

SENTIMENT_MODEL_DIR = os.environ.get("SENTIMENT_MODEL_DIR", "artifacts/sentiment")
ASPECT_MODEL_DIR = os.environ.get("ASPECT_MODEL_DIR", "artifacts/aspects")
DEVICE = os.environ.get("DEVICE", "auto")

app = Flask(__name__)


def load_predictor():
    """Load both checkpoints once, or return (None, message) if that fails."""
    if not Path(SENTIMENT_MODEL_DIR).exists() or not Path(ASPECT_MODEL_DIR).exists():
        return None, "Model checkpoints not found. Train both models first."
    try:
        return Predictor(SENTIMENT_MODEL_DIR, ASPECT_MODEL_DIR, device=DEVICE), None
    except Exception as error:  # noqa: BLE001 - report any load failure on /health
        return None, str(error)


# Load both models once at startup, not on every request.
predictor, predictor_error = load_predictor()


def validate_review_text(text):
    """Return an error message for blank or oversized input, otherwise None."""
    if text is None or not text.strip():
        return "Please enter a review sentence."
    if len(text) > MAX_REVIEW_LENGTH:
        return f"Please limit the review to {MAX_REVIEW_LENGTH} characters."
    return None


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the review form and, on POST, show the prediction below it."""
    result = None
    error = None
    review_text = ""

    if request.method == "POST":
        review_text = request.form.get("review", "")
        error = validate_review_text(review_text)

        if error is None and predictor is None:
            error = "The models failed to load. Check the server logs."

        if error is None:
            result = predictor.predict(review_text.strip())

    return render_template(
        "index.html",
        example_sentence=EXAMPLE_SENTENCE,
        review_text=review_text,
        result=result,
        error=error,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Return the same prediction as the form, as JSON, for demos and tests."""
    payload = request.get_json(silent=True) or {}
    review_text = payload.get("review", "")
    error = validate_review_text(review_text)

    if error is None and predictor is None:
        error = "The models failed to load. Check the server logs."

    if error is not None:
        return jsonify({"error": error}), 400

    result = predictor.predict(review_text.strip())
    return jsonify(result)


@app.route("/health")
def health():
    """Report whether both models loaded successfully."""
    if predictor is None:
        return jsonify({"status": "error", "detail": predictor_error}), 503
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # The reloader can load the models twice, which wastes memory on a GPU box.
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
