"""Unified training loop for classification (CIFAR-10) and language modelling (PTB)."""

import json
import math
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm


def _is_schedulefree(optimizer):
    return optimizer.__class__.__name__ == "AdamWScheduleFree"


def train_epoch_classification(model, loader, optimizer, device):
    """Train one epoch for image classification. Returns (avg_loss, num_steps)."""
    model.train()
    if _is_schedulefree(optimizer):
        optimizer.train()

    total_loss = 0.0
    correct = 0
    total = 0
    steps = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == targets).sum().item()
        total += targets.size(0)
        steps += 1

    return total_loss / steps, 100.0 * correct / total, steps


@torch.no_grad()
def eval_classification(model, loader, optimizer, device):
    """Evaluate classification accuracy. Returns (avg_loss, accuracy%)."""
    model.eval()
    if _is_schedulefree(optimizer):
        optimizer.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    batches = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        total_loss += loss.item()
        correct += (outputs.argmax(1) == targets).sum().item()
        total += targets.size(0)
        batches += 1

    return total_loss / batches, 100.0 * correct / total


def train_steps_lm(model, loader, optimizer, device, max_steps, grad_clip=1.0,
                   eval_loader=None, eval_every=1000, log_callback=None,
                   scheduler=None, start_step=0):
    """Train language model by step count. Returns list of epoch-style log dicts."""
    model.train()
    if _is_schedulefree(optimizer):
        optimizer.train()

    logs = []
    total_loss = 0.0
    step_count = 0
    data_iter = iter(loader)
    t_start = time.perf_counter()

    for step in range(start_step, max_steps):
        # Get next batch, cycling through data
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        step_count += 1

        # Periodic evaluation
        if (step + 1) % eval_every == 0 or (step + 1) == max_steps:
            avg_train_loss = total_loss / step_count
            train_ppl = math.exp(min(avg_train_loss, 20))  # cap to avoid overflow

            eval_loss, eval_ppl = None, None
            if eval_loader is not None:
                eval_loss, eval_ppl = eval_lm(model, eval_loader, optimizer, device)
                model.train()
                if _is_schedulefree(optimizer):
                    optimizer.train()

            elapsed = time.perf_counter() - t_start
            log = {
                "step": step + 1,
                "train_loss": round(avg_train_loss, 4),
                "train_perplexity": round(train_ppl, 2),
                "wall_time": round(elapsed, 2),
            }
            if eval_ppl is not None:
                log["val_loss"] = round(eval_loss, 4)
                log["val_perplexity"] = round(eval_ppl, 2)

            logs.append(log)
            if log_callback:
                log_callback(log)

            total_loss = 0.0
            step_count = 0

    return logs


@torch.no_grad()
def eval_lm(model, loader, optimizer, device):
    """Evaluate language model. Returns (avg_loss, perplexity)."""
    model.eval()
    if _is_schedulefree(optimizer):
        optimizer.eval()

    total_loss = 0.0
    batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        total_loss += loss.item()
        batches += 1

    avg_loss = total_loss / batches
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


def train_classification(model, train_loader, test_loader, optimizer, device,
                         epochs, scheduler=None, log_callback=None):
    """Full classification training loop. Returns list of per-epoch log dicts."""
    logs = []
    t_start = time.perf_counter()
    total_steps = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, steps = train_epoch_classification(
            model, train_loader, optimizer, device
        )
        if scheduler is not None:
            scheduler.step()
        total_steps += steps

        test_loss, test_acc = eval_classification(
            model, test_loader, optimizer, device
        )
        # Switch back to train mode for next epoch
        model.train()
        if _is_schedulefree(optimizer):
            optimizer.train()

        elapsed = time.perf_counter() - t_start
        log = {
            "epoch": epoch,
            "steps": total_steps,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 2),
            "test_loss": round(test_loss, 4),
            "test_accuracy": round(test_acc, 2),
            "wall_time": round(elapsed, 2),
        }
        logs.append(log)
        if log_callback:
            log_callback(log)

    return logs


def save_results(results, output_path):
    """Save results dict to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[results] Saved to {output_path}")


def make_output_path(results_dir, task, optimizer_name, lr, wd):
    """Generate output file path: {task}_{optimizer}_lr{lr}_wd{wd}.json"""
    fname = f"{task}_{optimizer_name}_lr{lr}_wd{wd}.json"
    return os.path.join(results_dir, fname)
