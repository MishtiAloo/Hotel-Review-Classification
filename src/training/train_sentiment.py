"""Fine-tune BERT for three-class sentence sentiment classification."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Allow this file to be run directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.labels import SENTIMENT_LABELS, SENTIMENT_TO_ID
from src.training.common import choose_device
from src.training.common import collect_predictions
from src.training.common import count_truncated
from src.training.common import describe_device
from src.training.common import load_config
from src.training.common import make_loader
from src.training.common import make_logger
from src.training.common import make_scheduler
from src.training.common import save_json
from src.training.common import set_seed
from src.training.common import train_one_epoch
from src.training.metrics import sentiment_metrics


def parse_args():
    """Read simple training options from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/sentiment_baseline"),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/train_sentiment_baseline.log"),
    )
    parser.add_argument("--loss", choices=["baseline", "weighted"], default="baseline")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def read_split(data_dir, name, smoke):
    """Read one prepared split, optionally taking a tiny smoke-test subset."""
    dataframe = pd.read_csv(data_dir / f"{name}.csv")
    if smoke:
        size = 64 if name == "train" else 32
        dataframe = dataframe.head(size).copy()
    return dataframe


def sentiment_ids(dataframe):
    """Convert prepared sentiment names to stable integer IDs."""
    labels = []
    for name in dataframe["sentiment"]:
        labels.append(SENTIMENT_TO_ID[name])
    return labels


def make_loss_function(train_labels, loss_name, device):
    """Create plain or class-weighted cross entropy."""
    if loss_name == "baseline":
        return torch.nn.CrossEntropyLoss(), None

    counts = np.bincount(train_labels, minlength=len(SENTIMENT_LABELS))
    weights = len(train_labels) / (len(SENTIMENT_LABELS) * counts)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return torch.nn.CrossEntropyLoss(weight=weight_tensor), weights.tolist()


def main():
    """Train, select by validation macro-F1, and evaluate once on test data."""
    args = parse_args()
    config = load_config(args.config)
    if args.smoke:
        config["epochs"] = 1
        config["train_batch_size"] = min(config["train_batch_size"], 4)
        config["eval_batch_size"] = min(config["eval_batch_size"], 8)

    logger = make_logger("train_sentiment", args.log_file)
    set_seed(config["seed"])
    device = choose_device(args.device)
    logger.info("Device: %s", describe_device(device))
    logger.info("Loss: %s", args.loss)
    logger.info("Smoke test: %s", args.smoke)

    train = read_split(args.data_dir, "train", args.smoke)
    validation = read_split(args.data_dir, "validation", args.smoke)
    test = read_split(args.data_dir, "test", args.smoke)
    logger.info(
        "Rows: train=%d validation=%d test=%d",
        len(train),
        len(validation),
        len(test),
    )

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    truncated = count_truncated(train["review"], tokenizer, config["max_length"])
    logger.info("Training reviews truncated at %d tokens: %d", config["max_length"], truncated)

    train_labels = sentiment_ids(train)
    validation_labels = sentiment_ids(validation)
    test_labels = sentiment_ids(test)

    train_loader = make_loader(
        train["review"],
        train_labels,
        tokenizer,
        config["max_length"],
        config["train_batch_size"],
        True,
        "sentiment",
    )
    validation_loader = make_loader(
        validation["review"],
        validation_labels,
        tokenizer,
        config["max_length"],
        config["eval_batch_size"],
        False,
        "sentiment",
    )
    test_loader = make_loader(
        test["review"],
        test_labels,
        tokenizer,
        config["max_length"],
        config["eval_batch_size"],
        False,
        "sentiment",
    )

    id_to_label = {}
    label_to_id = {}
    for index, label in enumerate(SENTIMENT_LABELS):
        id_to_label[index] = label
        label_to_id[label] = index

    model = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"],
        num_labels=len(SENTIMENT_LABELS),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    model.to(device)

    loss_function, class_weights = make_loss_function(
        train_labels,
        args.loss,
        device,
    )
    if class_weights is not None:
        logger.info("Class weights: %s", class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = make_scheduler(optimizer, len(train_loader), config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, config["epochs"] + 1):
        logger.info("Epoch %d/%d", epoch, config["epochs"])
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_function,
            device,
            config,
            logger,
        )
        validation_loss, labels, logits = collect_predictions(
            model,
            validation_loader,
            loss_function,
            device,
        )
        metrics = sentiment_metrics(labels, logits, SENTIMENT_LABELS)
        metrics["loss"] = validation_loss
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation": metrics,
            }
        )
        logger.info(
            "Epoch %d | train loss %.4f | validation loss %.4f | macro-F1 %.4f",
            epoch,
            train_loss,
            validation_loss,
            metrics["macro_f1"],
        )

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            epochs_without_improvement = 0
            model.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)
            logger.info("Saved new best checkpoint to %s", args.output_dir)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["early_stopping_patience"]:
                logger.info("Early stopping after %d epochs", epoch)
                break

    # Evaluate the selected checkpoint once on the held-out test split.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    best_model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
    best_model.to(device)
    test_loss, test_true, test_logits = collect_predictions(
        best_model,
        test_loader,
        loss_function,
        device,
    )
    test_metrics = sentiment_metrics(test_true, test_logits, SENTIMENT_LABELS)
    test_metrics["loss"] = test_loss
    logger.info(
        "Test | loss %.4f | accuracy %.4f | macro-F1 %.4f",
        test_loss,
        test_metrics["accuracy"],
        test_metrics["macro_f1"],
    )

    audit_path = args.data_dir / "audit.json"
    with audit_path.open("r", encoding="utf-8") as file_handle:
        data_audit = json.load(file_handle)

    run_summary = {
        "task": "sentiment",
        "loss": args.loss,
        "smoke_test": args.smoke,
        "base_model": config["base_model"],
        "seed": config["seed"],
        "device": describe_device(device),
        "source_sha256": data_audit["source_sha256"],
        "class_weights": class_weights,
        "best_validation_macro_f1": best_macro_f1,
        "history": history,
        "test": test_metrics,
    }
    save_json(run_summary, args.output_dir / "metrics.json")
    logger.info("Saved metrics to %s", args.output_dir / "metrics.json")
    logger.info("Sentiment training completed successfully")


if __name__ == "__main__":
    main()

