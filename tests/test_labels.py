"""Tests that the canonical label order and name/index mappings are stable."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.labels import ASPECT_COLUMNS, SENTIMENT_LABELS, SENTIMENT_TO_ID


def test_sentiment_label_count_and_order():
    assert SENTIMENT_LABELS == ["negative", "neutral", "positive"]


def test_sentiment_to_id_round_trips():
    for label in SENTIMENT_LABELS:
        index = SENTIMENT_TO_ID[label]
        assert SENTIMENT_LABELS[index] == label


def test_there_are_21_aspect_columns_with_no_duplicates():
    assert len(ASPECT_COLUMNS) == 21
    assert len(set(ASPECT_COLUMNS)) == 21
