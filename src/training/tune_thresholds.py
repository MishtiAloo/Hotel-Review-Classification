"""Pick sigmoid decision thresholds for the aspect model on validation data.

This script never looks at the test split. It loads one already-trained aspect
checkpoint, scores the validation split, and searches a small fixed grid of
global thresholds for the one with the best validation macro-F1. The chosen
threshold is saved next to the checkpoint so the predictor and the training
report can both use it.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.labels import ASPECT_COLUMNS
from src.training.common import choose_device
from src.training.common import make_loader
from src.training.common import save_json
from src.training.metrics import aspect_metrics


def parse_args():
    """Read the checkpoint location and threshold grid from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--grid-start", type=float, default=0.20)
    parser.add_argument("--grid-stop", type=float, default=0.80)
    parser.add_argument("--grid-step", type=float, default=0.05)
    return parser.parse_args()


def get_validation_logits(model_dir, data_dir, max_length, batch_size, device):
    """Run the checkpoint once over validation and return labels and logits."""
    validation = pd.read_csv(data_dir / "validation.csv")
    labels = validation[ASPECT_COLUMNS].to_numpy(dtype=np.float32).tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    loader = make_loader(
        validation["review"],
        labels,
        tokenizer,
        max_length,
        batch_size,
        False,
        "aspects",
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    all_labels = []
    all_logits = []
    with torch.inference_mode():
        for batch in loader:
            batch_labels = batch.pop("labels")
            inputs = {}
            for name, value in batch.items():
                inputs[name] = value.to(device)
            logits = model(**inputs).logits
            all_labels.append(batch_labels)
            all_logits.append(logits.cpu())

    return torch.cat(all_labels).numpy(), torch.cat(all_logits).numpy()


def search_grid(true_labels, logits, start, stop, step):
    """Try each global threshold in the grid and return the best one by macro-F1."""
    results = []
    threshold = start
    # A small epsilon avoids skipping the last grid value due to float drift.
    while threshold <= stop + 1e-9:
        metrics = aspect_metrics(true_labels, logits, ASPECT_COLUMNS, threshold=round(threshold, 2))
        results.append({"threshold": round(threshold, 2), "macro_f1": metrics["macro_f1"]})
        threshold += step

    best = max(results, key=lambda item: item["macro_f1"])
    return best, results


def main():
    """Tune one global aspect threshold on validation data and save it."""
    args = parse_args()
    device = choose_device(args.device)

    true_labels, logits = get_validation_logits(
        args.model_dir,
        args.data_dir,
        args.max_length,
        args.eval_batch_size,
        device,
    )

    baseline_metrics = aspect_metrics(true_labels, logits, ASPECT_COLUMNS, threshold=0.5)
    best, grid_results = search_grid(
        true_labels,
        logits,
        args.grid_start,
        args.grid_stop,
        args.grid_step,
    )

    # Only keep the grid search result if it meaningfully beats the plain 0.5 baseline.
    improvement = best["macro_f1"] - baseline_metrics["macro_f1"]
    if improvement > 0.001:
        chosen_threshold = best["threshold"]
    else:
        chosen_threshold = 0.5

    thresholds = {}
    for column in ASPECT_COLUMNS:
        thresholds[column] = chosen_threshold

    output = {
        "global_threshold": chosen_threshold,
        "baseline_threshold_0_5_macro_f1": baseline_metrics["macro_f1"],
        "grid_search": grid_results,
        "per_label_thresholds": thresholds,
    }
    if chosen_threshold == best["threshold"]:
        chosen_macro_f1 = best["macro_f1"]
    else:
        chosen_macro_f1 = baseline_metrics["macro_f1"]

    save_json(output, args.model_dir / "thresholds.json")
    print(f"Chosen global threshold: {chosen_threshold}")
    print(f"Validation macro-F1 at 0.5: {baseline_metrics['macro_f1']:.4f}")
    print(f"Validation macro-F1 at {chosen_threshold}: {chosen_macro_f1:.4f}")
    print(f"Saved thresholds to {args.model_dir / 'thresholds.json'}")


if __name__ == "__main__":
    main()
