"""Diagnostic experiments for RQ2 (training horizon) and RQ3 (batch size sensitivity)."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from trainers.device import get_device, seed_everything
from trainers.optimizer_factory import get_optimizer
from trainers.trainer import (
    train_classification,
    train_steps_lm,
    save_results,
)
from models.resnet_cifar import resnet18_cifar
from models.gpt_small import gpt_small
from data.cifar10 import get_cifar10_loaders
from data.ptb import get_ptb_loaders
from scripts.run_grid import find_best_hparams


# ─── RQ2: Training Horizon ────────────────────────────────────────────────────

HORIZON_OPTIMIZERS = ["ademamix", "adamw", "sgd"]
HORIZON_FRACTIONS = [0.10, 0.25, 0.50, 1.00]


def run_horizon_diagnostic(task, device, results_dir, seed, num_workers):
    """RQ2: Record test metric at 10%/25%/50%/100% of training budget.

    Uses best hparams from main grid for each optimizer.
    Only implemented for CIFAR-10 (epoch-based fractions).
    """
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 1 — Training Horizon (RQ2)")
    print("=" * 60)

    full_epochs = 50  # full CIFAR-10 budget

    for opt_name in HORIZON_OPTIMIZERS:
        best = find_best_hparams(results_dir, task, opt_name)
        if best is None:
            print(f"\n  [SKIP] No grid results for {opt_name} on {task}. Run main grid first.")
            continue

        lr, wd = best["lr"], best["weight_decay"]
        print(f"\n  Optimizer: {opt_name} | Best LR={lr}, WD={wd} (metric={best['metric_value']})")

        for frac in HORIZON_FRACTIONS:
            epochs = max(1, int(full_epochs * frac))
            out_name = f"diag_horizon_{task}_{opt_name}_frac{frac}.json"
            out_path = os.path.join(results_dir, out_name)

            if os.path.exists(out_path):
                print(f"    [SKIP] {out_name} already exists")
                continue

            print(f"    Fraction={frac} ({epochs} epochs) ...")
            seed_everything(seed)

            nw = num_workers if num_workers is not None else 4
            train_loader, test_loader = get_cifar10_loaders(batch_size=128, num_workers=nw)
            model = resnet18_cifar().to(device)

            extra_kwargs = {}
            if opt_name == "sgd":
                extra_kwargs["momentum"] = 0.9
            optimizer = get_optimizer(opt_name, model, lr, wd, extra_kwargs)

            scheduler = None
            if opt_name != "schedulefree_adamw":
                scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

            logs = train_classification(
                model, train_loader, test_loader, optimizer, device,
                epochs=epochs, scheduler=scheduler,
            )

            result = {
                "diagnostic": "horizon",
                "task": task,
                "optimizer": opt_name,
                "lr": lr,
                "weight_decay": wd,
                "fraction": frac,
                "epochs": epochs,
                "full_epochs": full_epochs,
                "seed": seed,
                "device": str(device),
                "final_test_accuracy": logs[-1]["test_accuracy"],
                "final_train_loss": logs[-1]["train_loss"],
                "total_wall_time": logs[-1]["wall_time"],
                "history": logs,
            }
            save_results(result, out_path)
            print(f"      -> test_acc={logs[-1]['test_accuracy']:.2f}%")


# ─── RQ3: Batch Size Sensitivity ──────────────────────────────────────────────

BATCH_SIZES = [32, 64, 128, 256]
BATCHSIZE_FIXED_STEPS = 20000

ALL_OPTIMIZERS = ["sgd", "adamw", "lion", "ademamix", "soap", "muon", "schedulefree_adamw"]


def run_batchsize_diagnostic(task, device, results_dir, seed, num_workers, optimizers=None):
    """RQ3: Run all optimizers at different batch sizes with fixed step count.

    Uses best hparams from main grid for each optimizer.
    """
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 2 — Batch Size Sensitivity (RQ3)")
    print("=" * 60)

    opt_names = optimizers or ALL_OPTIMIZERS

    for opt_name in opt_names:
        best = find_best_hparams(results_dir, task, opt_name)
        if best is None:
            print(f"\n  [SKIP] No grid results for {opt_name} on {task}. Run main grid first.")
            continue

        lr, wd = best["lr"], best["weight_decay"]
        print(f"\n  Optimizer: {opt_name} | Best LR={lr}, WD={wd}")

        for bs in BATCH_SIZES:
            out_name = f"diag_batchsize_{task}_{opt_name}_bs{bs}.json"
            out_path = os.path.join(results_dir, out_name)

            if os.path.exists(out_path):
                print(f"    [SKIP] {out_name} already exists")
                continue

            print(f"    Batch size={bs}, steps={BATCHSIZE_FIXED_STEPS} ...")
            seed_everything(seed)

            try:
                if task == "cifar10":
                    nw = num_workers if num_workers is not None else 4
                    train_loader, test_loader = get_cifar10_loaders(
                        batch_size=bs, num_workers=nw,
                    )
                    model = resnet18_cifar().to(device)

                    extra_kwargs = {}
                    if opt_name == "sgd":
                        extra_kwargs["momentum"] = 0.9
                    optimizer = get_optimizer(opt_name, model, lr, wd, extra_kwargs)

                    scheduler = None
                    # Convert fixed steps to epochs
                    steps_per_epoch = len(train_loader)
                    epochs = max(1, BATCHSIZE_FIXED_STEPS // steps_per_epoch)
                    if opt_name != "schedulefree_adamw":
                        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

                    logs = train_classification(
                        model, train_loader, test_loader, optimizer, device,
                        epochs=epochs, scheduler=scheduler,
                    )

                    result = {
                        "diagnostic": "batchsize",
                        "task": task,
                        "optimizer": opt_name,
                        "lr": lr,
                        "weight_decay": wd,
                        "batch_size": bs,
                        "fixed_steps": BATCHSIZE_FIXED_STEPS,
                        "actual_epochs": epochs,
                        "seed": seed,
                        "device": str(device),
                        "final_test_accuracy": logs[-1]["test_accuracy"],
                        "final_train_loss": logs[-1]["train_loss"],
                        "total_wall_time": logs[-1]["wall_time"],
                        "total_steps": logs[-1]["steps"],
                        "history": logs,
                    }

                else:  # ptb
                    nw = num_workers if num_workers is not None else 2
                    train_loader, val_loader, _, vocab_size = get_ptb_loaders(
                        batch_size=bs, num_workers=nw,
                    )
                    model = gpt_small(vocab_size=vocab_size).to(device)

                    extra_kwargs = {}
                    if opt_name == "sgd":
                        extra_kwargs["momentum"] = 0.9
                    optimizer = get_optimizer(opt_name, model, lr, wd, extra_kwargs)

                    scheduler = None
                    if opt_name != "schedulefree_adamw":
                        scheduler = CosineAnnealingLR(optimizer, T_max=BATCHSIZE_FIXED_STEPS)

                    logs = train_steps_lm(
                        model, train_loader, optimizer, device,
                        max_steps=BATCHSIZE_FIXED_STEPS, grad_clip=1.0,
                        eval_loader=val_loader, eval_every=1000,
                        scheduler=scheduler,
                    )

                    result = {
                        "diagnostic": "batchsize",
                        "task": task,
                        "optimizer": opt_name,
                        "lr": lr,
                        "weight_decay": wd,
                        "batch_size": bs,
                        "fixed_steps": BATCHSIZE_FIXED_STEPS,
                        "seed": seed,
                        "device": str(device),
                        "final_val_perplexity": logs[-1].get("val_perplexity"),
                        "final_train_loss": logs[-1]["train_loss"],
                        "total_wall_time": logs[-1]["wall_time"],
                        "total_steps": logs[-1]["step"],
                        "history": logs,
                    }

                save_results(result, out_path)
                metric = result.get("final_test_accuracy") or result.get("final_val_perplexity")
                print(f"      -> metric={metric}")

            except Exception as e:
                print(f"      FAILED: {e}")
                failure = {
                    "diagnostic": "batchsize",
                    "task": task, "optimizer": opt_name,
                    "batch_size": bs, "error": str(e),
                }
                failed_path = out_path.replace(".json", "_FAILED.json")
                save_results(failure, failed_path)


def main():
    parser = argparse.ArgumentParser(description="Run diagnostic experiments (RQ2 horizon, RQ3 batch size)")
    parser.add_argument("--task", type=str, default="cifar10", choices=["cifar10", "ptb"])
    parser.add_argument("--diagnostic", type=str, required=True,
                        choices=["horizon", "batchsize", "all"],
                        help="Which diagnostic to run")
    parser.add_argument("--optimizers", type=str, nargs="+", default=None,
                        help="Subset of optimizers (for batchsize diagnostic)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    device = get_device(args.device)

    if args.diagnostic in ("horizon", "all"):
        run_horizon_diagnostic(
            args.task, device, args.results_dir, args.seed, args.num_workers,
        )

    if args.diagnostic in ("batchsize", "all"):
        run_batchsize_diagnostic(
            args.task, device, args.results_dir, args.seed, args.num_workers,
            optimizers=args.optimizers,
        )


if __name__ == "__main__":
    main()
