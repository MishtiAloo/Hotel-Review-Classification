"""Load the frozen sentiment and aspect checkpoints and run inference on text.

Both the command-line smoke test and the Flask app use this one class so that
tokenization, thresholding, and output formatting never drift apart.
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.labels import ASPECT_COLUMNS, SENTIMENT_LABELS

MAX_LENGTH = 128


def choose_device(requested_device):
    """Choose CUDA when available unless the caller explicitly requests CPU."""
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_thresholds(aspect_model_dir):
    """Load per-aspect thresholds, falling back to 0.5 for any missing entry."""
    thresholds_path = Path(aspect_model_dir) / "thresholds.json"
    thresholds = {}
    for aspect in ASPECT_COLUMNS:
        thresholds[aspect] = 0.5

    if thresholds_path.exists():
        with thresholds_path.open("r", encoding="utf-8") as file_handle:
            saved = json.load(file_handle)
        saved_thresholds = saved.get("per_label_thresholds", {})
        for aspect, value in saved_thresholds.items():
            thresholds[aspect] = value

    return thresholds


class Predictor:
    """Load both BERT checkpoints once and predict sentiment plus aspects."""

    def __init__(self, sentiment_model_dir, aspect_model_dir, device="auto"):
        self.device = choose_device(device)
        self.thresholds = load_thresholds(aspect_model_dir)

        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_dir)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            sentiment_model_dir
        )
        self.sentiment_model.to(self.device)
        self.sentiment_model.eval()

        self.aspect_tokenizer = AutoTokenizer.from_pretrained(aspect_model_dir)
        self.aspect_model = AutoModelForSequenceClassification.from_pretrained(
            aspect_model_dir
        )
        self.aspect_model.to(self.device)
        self.aspect_model.eval()

        # Read the label order back from the saved checkpoint config so the
        # predictor never depends on assuming the training-time column order.
        self.sentiment_labels = self._ordered_labels(self.sentiment_model, SENTIMENT_LABELS)
        self.aspect_labels = self._ordered_labels(self.aspect_model, ASPECT_COLUMNS)

    @staticmethod
    def _ordered_labels(model, fallback_labels):
        """Return label names ordered by id, using the checkpoint's id2label map."""
        id_to_label = model.config.id2label
        ordered = []
        for index in range(len(id_to_label)):
            ordered.append(id_to_label[index])
        if ordered:
            return ordered
        return list(fallback_labels)

    def predict(self, text):
        """Return sentiment and aspect predictions for one review sentence."""
        sentiment_result = self._predict_sentiment(text)
        aspect_result = self._predict_aspects(text)
        return {
            "text": text,
            "sentiment": sentiment_result,
            "aspects": aspect_result,
        }

    def _predict_sentiment(self, text):
        """Run the sentiment model and softmax the logits into probabilities."""
        encoded = self.sentiment_tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}

        with torch.inference_mode():
            logits = self.sentiment_model(**encoded).logits
        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()

        scores = {}
        for label, probability in zip(self.sentiment_labels, probabilities):
            scores[label] = round(probability, 4)

        best_index = int(torch.argmax(logits, dim=1)[0])
        best_label = self.sentiment_labels[best_index]

        return {
            "label": best_label,
            "confidence": scores[best_label],
            "scores": scores,
        }

    def _predict_aspects(self, text):
        """Run the aspect model, sigmoid the logits, and apply saved thresholds."""
        encoded = self.aspect_tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}

        with torch.inference_mode():
            logits = self.aspect_model(**encoded).logits
        probabilities = torch.sigmoid(logits)[0].cpu().tolist()

        detected = []
        for label, probability in zip(self.aspect_labels, probabilities):
            threshold = self.thresholds.get(label, 0.5)
            if probability >= threshold:
                detected.append({"label": label, "score": round(probability, 4)})

        detected.sort(key=lambda item: item["score"], reverse=True)
        return detected
