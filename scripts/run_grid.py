"""Run the full hyperparameter grid with skip-if-exists logic."""

import argparse
import itertools
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from trainers.device import get_device, seed_everything
from trainers.optimizer_factory import get_optimizer
from trainers.trainer import (
    train_classification,
    train_steps_lm,
    save_results,
    make_output_path,
)
from models.resnet_cifar import resnet18_cifar
from models.gpt_small import gpt_small
from data.cifar10 import get_cifar10_loaders
from data.ptb import get_ptb_loaders


ALL_OPTIMIZERS = ["sgd", "adamw", "lion", "ademamix", "soap", "muon", "schedulefree_adamw"]


def find_best_hparams(results_dir, task, optimizer_name):
    """Find the best LR and WD for a given task/optimizer from completed grid results.

    For CIFAR-10: maximizes test_accuracy.
    For PTB: minimizes val_perplexity.

    Returns:
        dict with keys: lr, weight_decay, metric_value, file_path
        or None if no results found.
    """
    best = None
    for fname in os.listdir(results_dir):
        if not fname.startswith(f"{task}_{optimizer_name}_lr") or not fname.endswith(".json"):
            continue
        if "_FAILED" in fname:
            continue
        fpath = os.path.join(results_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)

        if task == "cifar10":
            metric = data.get("final_test_accuracy")
            if metric is None:
                continue
            if best is None or metric > best["metric_value"]:
                best = {"lr": data["lr"], "weight_decay": data["weight_decay"],
                        "metric_value": metric, "file_path": fpath}
        else:  # ptb
            metric = data.get("final_val_perplexity")
            if metric is None:
                continue
            if best is None or metric < best["metric_value"]:
                best = {"lr": data["lr"], "weight_decay": data["weight_decay"],
                        "metric_value": metric, "file_path": fpath}

    return best


def run_single_experiment(task, opt_name, lr, wd, extra_kwargs, task_config,
                          device, results_dir, seed, num_workers):
    """Run one experiment and save results. Returns True on success."""
    output_path = make_output_path(results_dir, task, opt_name, lr, wd)
    failed_path = output_path.replace(".json", "_FAILED.json")

    # Skip-if-exists
    if os.path.exists(output_path):
        print(f"  [SKIP] {os.path.basename(output_path)} already exists")
        return True
    if os.path.exists(failed_path):
        print(f"  [SKIP] {os.path.basename(failed_path)} (previously failed)")
        return False

    seed_everything(seed)

    try:
        if task == "cifar10":
            nw = num_workers if num_workers is not None else 4
            train_loader, test_loader = get_cifar10_loaders(
                batch_size=task_config["batch_size"], num_workers=nw,
            )
            model = resnet18_cifar().to(device)
            optimizer = get_optimizer(opt_name, model, lr, wd, extra_kwargs)

            scheduler = None
            epochs = task_config["epochs"]
            if opt_name != "schedulefree_adamw":
                scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

            def log_fn(log):
                print(f"    Epoch {log['epoch']:3d}/{epochs} | "
                      f"loss={log['train_loss']:.4f} | "
                      f"acc={log['test_accuracy']:.2f}%")

            logs = train_classification(
                model, train_loader, test_loader, optimizer, device,
                epochs=epochs, scheduler=scheduler, log_callback=log_fn,
            )

            results = {
                "task": task, "optimizer": opt_name, "lr": lr,
                "weight_decay": wd, "epochs": epochs,
                "batch_size": task_config["batch_size"], "seed": seed,
                "device": str(device),
                "final_test_accuracy": logs[-1]["test_accuracy"],
                "final_train_loss": logs[-1]["train_loss"],
                "total_wall_time": logs[-1]["wall_time"],
                "total_steps": logs[-1]["steps"],
                "history": logs,
            }

        else:  # ptb
            nw = num_workers if num_workers is not None else 2
            train_loader, val_loader, test_loader, vocab_size = get_ptb_loaders(
                batch_size=task_config["batch_size"], num_workers=nw,
            )
            model = gpt_small(vocab_size=vocab_size).to(device)
            optimizer = get_optimizer(opt_name, model, lr, wd, extra_kwargs)

            max_steps = task_config["max_steps"]
            scheduler = None
            if opt_name != "schedulefree_adamw":
                scheduler = CosineAnnealingLR(optimizer, T_max=max_steps)

            def log_fn(log):
                msg = f"    Step {log['step']:6d}/{max_steps} | train_ppl={log['train_perplexity']:.2f}"
                if "val_perplexity" in log:
                    msg += f" | val_ppl={log['val_perplexity']:.2f}"
                print(msg)

            logs = train_steps_lm(
                model, train_loader, optimizer, device,
                max_steps=max_steps, grad_clip=1.0,
                eval_loader=val_loader, eval_every=1000,
                log_callback=log_fn, scheduler=scheduler,
            )

            results = {
                "task": task, "optimizer": opt_name, "lr": lr,
                "weight_decay": wd, "max_steps": max_steps,
                "batch_size": task_config["batch_size"], "seed": seed,
                "device": str(device),
                "final_val_perplexity": logs[-1].get("val_perplexity"),
                "final_train_loss": logs[-1]["train_loss"],
                "total_wall_time": logs[-1]["wall_time"],
                "total_steps": logs[-1]["step"],
                "history": logs,
            }

        save_results(results, output_path)
        return True

    except Exception as e:
        print(f"    FAILED: {e}")
        traceback.print_exc()
        # Save failure record
        failure = {
            "task": task, "optimizer": opt_name, "lr": lr,
            "weight_decay": wd, "error": str(e),
        }
        save_results(failure, failed_path)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run full HP grid")
    parser.add_argument("--task", type=str, required=True, choices=["cifar10", "ptb"])
    parser.add_argument("--optimizers", type=str, nargs="+", default=None,
                        help=f"Subset of optimizers to run. Default: all. Choices: {ALL_OPTIMIZERS}")
    parser.add_argument("--config", type=str, default="configs/grid.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    device = get_device(args.device)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    task_config = config[args.task]
    opt_names = args.optimizers or ALL_OPTIMIZERS

    # Validate optimizer names
    for name in opt_names:
        if name not in task_config:
            print(f"WARNING: Optimizer '{name}' not found in config for task '{args.task}', skipping.")

    total_runs = 0
    completed = 0
    skipped = 0
    failed = 0

    for opt_name in opt_names:
        if opt_name not in task_config:
            continue

        opt_config = task_config[opt_name]
        lrs = opt_config["lr"]
        wds = opt_config["weight_decay"]

        # Collect extra kwargs (everything except lr and weight_decay)
        extra_kwargs = {k: v for k, v in opt_config.items()
                        if k not in ("lr", "weight_decay")}

        grid = list(itertools.product(lrs, wds))
        print(f"\n{'='*60}")
        print(f"Optimizer: {opt_name} | {len(grid)} runs")
        print(f"LRs: {lrs}")
        print(f"WDs: {wds}")
        if extra_kwargs:
            print(f"Extra: {extra_kwargs}")
        print(f"{'='*60}")

        for i, (lr, wd) in enumerate(grid, 1):
            total_runs += 1
            output_path = make_output_path(args.results_dir, args.task, opt_name, lr, wd)
            failed_path = output_path.replace(".json", "_FAILED.json")

            if os.path.exists(output_path) or os.path.exists(failed_path):
                skipped += 1

            print(f"\n  [{i}/{len(grid)}] {opt_name} lr={lr} wd={wd}")
            success = run_single_experiment(
                args.task, opt_name, lr, wd, extra_kwargs, task_config,
                device, args.results_dir, args.seed, args.num_workers,
            )
            if success:
                completed += 1
            else:
                failed += 1

    print(f"\n{'='*60}")
    print(f"Grid complete: {completed} succeeded, {failed} failed, {skipped} skipped (of {total_runs} total)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
