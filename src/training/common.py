"""Small shared helpers for the two straightforward training scripts."""

import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset


class ReviewDataset(Dataset):
    """Store review strings and their already prepared labels."""

    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]


def load_config(path):
    """Load the small YAML training configuration."""
    with path.open("r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def make_logger(name, log_path):
    """Log progress to both the terminal and a persistent UTF-8 file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def set_seed(seed):
    """Set the common random seeds used in this project."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested_device):
    """Choose CUDA when available unless the caller explicitly requests CPU."""
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available to PyTorch")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device):
    """Return a short, log-friendly description of the selected device."""
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        memory_gb = properties.total_memory / (1024 ** 3)
        return f"{properties.name} ({memory_gb:.2f} GB)"
    return "CPU"


def make_loader(texts, labels, tokenizer, max_length, batch_size, shuffle, task):
    """Create a dynamically padded data loader for one task."""
    dataset = ReviewDataset(texts, labels)

    def collate_batch(batch):
        batch_texts = []
        batch_labels = []
        for text, label in batch:
            batch_texts.append(text)
            batch_labels.append(label)

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        if task == "sentiment":
            encoded["labels"] = torch.tensor(batch_labels, dtype=torch.long)
        else:
            encoded["labels"] = torch.tensor(batch_labels, dtype=torch.float32)
        return encoded

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )


def count_truncated(texts, tokenizer, max_length):
    """Count reviews that are longer than the configured token limit."""
    truncated = 0
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
        if len(token_ids) > max_length:
            truncated += 1
    return truncated


def make_scheduler(optimizer, loader_length, config):
    """Create the linear warmup/decay schedule used by both models."""
    from transformers import get_linear_schedule_with_warmup

    accumulation = config["gradient_accumulation_steps"]
    updates_per_epoch = math.ceil(loader_length / accumulation)
    total_updates = updates_per_epoch * config["epochs"]
    warmup_updates = int(total_updates * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_updates,
        num_training_steps=total_updates,
    )
    return scheduler


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    loss_function,
    device,
    config,
    logger,
):
    """Train for one epoch and return the average unscaled batch loss."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    accumulation = config["gradient_accumulation_steps"]
    use_fp16 = bool(config["fp16"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    for batch_number, batch in enumerate(loader, start=1):
        labels = batch.pop("labels").to(device)
        inputs = {}
        for name, value in batch.items():
            inputs[name] = value.to(device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_fp16,
        ):
            logits = model(**inputs).logits
            loss = loss_function(logits, labels)
            scaled_loss = loss / accumulation

        scaler.scale(scaled_loss).backward()
        total_loss += loss.item()

        is_update_step = batch_number % accumulation == 0
        is_last_batch = batch_number == len(loader)
        if is_update_step or is_last_batch:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["max_grad_norm"],
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if batch_number % 100 == 0 or is_last_batch:
            average_loss = total_loss / batch_number
            logger.info(
                "Training batch %d/%d | average loss %.4f",
                batch_number,
                len(loader),
                average_loss,
            )

    return total_loss / len(loader)


def collect_predictions(model, loader, loss_function, device):
    """Run evaluation and return average loss, labels, and logits."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_logits = []

    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {}
            for name, value in batch.items():
                inputs[name] = value.to(device)

            logits = model(**inputs).logits
            loss = loss_function(logits, labels)
            total_loss += loss.item()
            all_labels.append(labels.cpu())
            all_logits.append(logits.cpu())

    labels_array = torch.cat(all_labels).numpy()
    logits_array = torch.cat(all_logits).numpy()
    return total_loss / len(loader), labels_array, logits_array


def save_json(data, path):
    """Save readable UTF-8 JSON, including NumPy scalar values."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Cannot serialize {type(value)}")

    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=2, ensure_ascii=False, default=convert)

