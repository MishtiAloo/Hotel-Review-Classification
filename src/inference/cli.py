"""Command-line smoke test for the shared Predictor.

Run this after both models are trained to check that the checkpoints load and
produce sensible sentiment and aspect predictions before wiring up Flask.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import Predictor

EXAMPLE_SENTENCES = [
    "The staff were friendly and the room was spotless.",
    "The Wi-Fi was slow and the room was noisy.",
    "Breakfast was poor but the staff were excellent.",
    "The hotel is located three kilometres from the airport.",
]


def parse_args():
    """Read the checkpoint locations and an optional custom sentence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sentiment-model-dir",
        type=Path,
        default=Path("artifacts/sentiment"),
    )
    parser.add_argument(
        "--aspect-model-dir",
        type=Path,
        default=Path("artifacts/aspects"),
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--text", default=None, help="Predict one custom sentence instead of the examples.")
    return parser.parse_args()


def print_prediction(result):
    """Print one prediction result in a readable form."""
    print(f"Text: {result['text']}")
    sentiment = result["sentiment"]
    print(f"  Sentiment: {sentiment['label']} (confidence {sentiment['confidence']})")
    print(f"    Scores: {sentiment['scores']}")
    if result["aspects"]:
        print("  Aspects:")
        for aspect in result["aspects"]:
            print(f"    {aspect['label']}: {aspect['score']}")
    else:
        print("  Aspects: none passed the configured threshold")


def main():
    """Load both models and print predictions for the example sentences."""
    args = parse_args()
    predictor = Predictor(
        args.sentiment_model_dir,
        args.aspect_model_dir,
        device=args.device,
    )

    if args.text:
        sentences = [args.text]
    else:
        sentences = EXAMPLE_SENTENCES

    for sentence in sentences:
        result = predictor.predict(sentence)
        print_prediction(result)
        print()


if __name__ == "__main__":
    main()
