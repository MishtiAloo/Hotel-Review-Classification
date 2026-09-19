"""Clear metric calculations for sentiment and multi-label aspects."""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_recall_fscore_support


def sentiment_metrics(true_labels, logits, label_names):
    """Calculate the required sentiment-classification metrics."""
    predicted_labels = np.argmax(logits, axis=1)
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(label_names))),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "macro_f1": f1_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        ),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            true_labels,
            predicted_labels,
            labels=list(range(len(label_names))),
        ),
    }


def sigmoid(values):
    """Convert raw logits to probabilities without requiring SciPy."""
    return 1.0 / (1.0 + np.exp(-values))


def aspect_metrics(true_labels, logits, aspect_names, threshold=0.5):
    """Calculate the required multi-label metrics at one threshold."""
    probabilities = sigmoid(logits)
    predicted_labels = (probabilities >= threshold).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average=None,
        zero_division=0,
    )

    per_aspect = {}
    for index, name in enumerate(aspect_names):
        per_aspect[name] = {
            "precision": precision[index],
            "recall": recall[index],
            "f1": f1[index],
            "support": support[index],
        }

    return {
        "threshold": threshold,
        "micro_f1": f1_score(
            true_labels,
            predicted_labels,
            average="micro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        ),
        "micro_average_precision": average_precision_score(
            true_labels,
            probabilities,
            average="micro",
        ),
        "macro_average_precision": average_precision_score(
            true_labels,
            probabilities,
            average="macro",
        ),
        "hamming_loss": hamming_loss(true_labels, predicted_labels),
        "exact_match_accuracy": accuracy_score(true_labels, predicted_labels),
        "per_aspect": per_aspect,
    }
