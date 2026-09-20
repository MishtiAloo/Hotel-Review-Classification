"""Tests for the shared Predictor class in src/inference/predictor.py."""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.labels import ASPECT_COLUMNS
from src.inference.predictor import Predictor, load_thresholds

# Real training writes checkpoints here. The smoke-test checkpoints are a
# lightweight stand-in so this test can run without a full GPU training run.
FINAL_SENTIMENT_DIR = PROJECT_ROOT / "artifacts" / "sentiment"
FINAL_ASPECT_DIR = PROJECT_ROOT / "artifacts" / "aspects"
SMOKE_SENTIMENT_DIR = PROJECT_ROOT / "artifacts" / "smoke_sentiment"
SMOKE_ASPECT_DIR = PROJECT_ROOT / "artifacts" / "smoke_aspects"


def _pick_available_checkpoints():
    """Return (sentiment_dir, aspect_dir) for whichever checkpoints exist, or None."""
    if FINAL_SENTIMENT_DIR.exists() and FINAL_ASPECT_DIR.exists():
        return FINAL_SENTIMENT_DIR, FINAL_ASPECT_DIR
    if SMOKE_SENTIMENT_DIR.exists() and SMOKE_ASPECT_DIR.exists():
        return SMOKE_SENTIMENT_DIR, SMOKE_ASPECT_DIR
    return None


def test_load_thresholds_defaults_to_one_half_without_a_file(tmp_path):
    thresholds = load_thresholds(tmp_path)

    for aspect in ASPECT_COLUMNS:
        assert thresholds[aspect] == 0.5


def test_load_thresholds_reads_saved_per_label_values(tmp_path):
    saved = {"per_label_thresholds": {"Clean": 0.35, "Staff": 0.6}}
    (tmp_path / "thresholds.json").write_text(json.dumps(saved), encoding="utf-8")

    thresholds = load_thresholds(tmp_path)

    assert thresholds["Clean"] == 0.35
    assert thresholds["Staff"] == 0.6
    # Aspects not mentioned in the file still fall back to 0.5.
    assert thresholds["Room"] == 0.5


@pytest.mark.skipif(
    _pick_available_checkpoints() is None,
    reason="No trained checkpoints found under artifacts/; run training first.",
)
def test_predictor_outputs_are_well_formed():
    sentiment_dir, aspect_dir = _pick_available_checkpoints()
    predictor = Predictor(sentiment_dir, aspect_dir, device="cpu")

    result = predictor.predict("The staff were friendly and the room was spotless.")

    scores = result["sentiment"]["scores"]
    assert set(scores.keys()) == {"negative", "neutral", "positive"}
    assert abs(sum(scores.values()) - 1.0) < 0.01
    assert result["sentiment"]["label"] in scores

    for aspect in result["aspects"]:
        assert 0.0 <= aspect["score"] <= 1.0
        threshold = predictor.thresholds[aspect["label"]]
        assert aspect["score"] >= threshold
