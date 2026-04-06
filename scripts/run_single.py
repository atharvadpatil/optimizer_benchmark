"""Run a single experiment for debugging and verification."""

import argparse
import sys
import os

# Allow running from project root: python scripts/run_single.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def main():
    parser = argparse.ArgumentParser(description="Run a single optimizer experiment")
    parser.add_argument("--task", type=str, required=True, choices=["cifar10", "ptb"])
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for CIFAR-10")
    parser.add_argument("--max_steps", type=int, default=50000, help="Max steps for PTB")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/mps/cuda)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader workers (default: 4 for cifar10, 2 for ptb)")
    # Extra optimizer kwargs
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--alpha", type=float, default=5.0, help="AdEMAMix alpha")
    parser.add_argument("--beta3", type=float, default=0.9999, help="AdEMAMix beta3")
    parser.add_argument("--precondition_frequency", type=int, default=10, help="SOAP precond freq")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)

    # Build extra kwargs
    extra_kwargs = {
        "momentum": args.momentum,
        "alpha": args.alpha,
        "beta3": args.beta3,
        "precondition_frequency": args.precondition_frequency,
    }

    output_path = make_output_path(args.results_dir, args.task, args.optimizer, args.lr, args.wd)

    def log_fn(log):
        if args.task == "cifar10":
            print(f"  Epoch {log['epoch']:3d} | "
                  f"train_loss={log['train_loss']:.4f} | "
                  f"test_acc={log['test_accuracy']:.2f}% | "
                  f"time={log['wall_time']:.1f}s")
        else:
            msg = f"  Step {log['step']:6d} | train_ppl={log['train_perplexity']:.2f}"
            if "val_perplexity" in log:
                msg += f" | val_ppl={log['val_perplexity']:.2f}"
            msg += f" | time={log['wall_time']:.1f}s"
            print(msg)

    print(f"{'='*60}")
    print(f"Task: {args.task} | Optimizer: {args.optimizer} | LR: {args.lr} | WD: {args.wd}")
    print(f"Device: {device} | Seed: {args.seed}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    try:
        if args.task == "cifar10":
            nw = args.num_workers if args.num_workers is not None else 4
            train_loader, test_loader = get_cifar10_loaders(
                batch_size=args.batch_size, num_workers=nw,
            )
            model = resnet18_cifar().to(device)
            optimizer = get_optimizer(args.optimizer, model, args.lr, args.wd, extra_kwargs)

            # Cosine annealing for all except schedule-free
            scheduler = None
            if args.optimizer != "schedulefree_adamw":
                scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

            print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
            print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")
            print()

            logs = train_classification(
                model, train_loader, test_loader, optimizer, device,
                epochs=args.epochs, scheduler=scheduler, log_callback=log_fn,
            )

            results = {
                "task": args.task,
                "optimizer": args.optimizer,
                "lr": args.lr,
                "weight_decay": args.wd,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "device": str(device),
                "final_test_accuracy": logs[-1]["test_accuracy"],
                "final_train_loss": logs[-1]["train_loss"],
                "total_wall_time": logs[-1]["wall_time"],
                "total_steps": logs[-1]["steps"],
                "history": logs,
            }

        else:  # ptb
            nw = args.num_workers if args.num_workers is not None else 2
            train_loader, val_loader, test_loader, vocab_size = get_ptb_loaders(
                batch_size=args.batch_size, num_workers=nw,
            )
            model = gpt_small(vocab_size=vocab_size).to(device)
            optimizer = get_optimizer(args.optimizer, model, args.lr, args.wd, extra_kwargs)

            scheduler = None
            if args.optimizer != "schedulefree_adamw":
                scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps)

            print(f"Vocab size: {vocab_size} | Model params: {sum(p.numel() for p in model.parameters()):,}")
            print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
            print()

            logs = train_steps_lm(
                model, train_loader, optimizer, device,
                max_steps=args.max_steps, grad_clip=1.0,
                eval_loader=val_loader, eval_every=1000,
                log_callback=log_fn, scheduler=scheduler,
            )

            results = {
                "task": args.task,
                "optimizer": args.optimizer,
                "lr": args.lr,
                "weight_decay": args.wd,
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "device": str(device),
                "final_val_perplexity": logs[-1].get("val_perplexity"),
                "final_train_loss": logs[-1]["train_loss"],
                "total_wall_time": logs[-1]["wall_time"],
                "total_steps": logs[-1]["step"],
                "history": logs,
            }

        save_results(results, output_path)
        print(f"\nDone! Results saved to {output_path}")

    except Exception as e:
        print(f"\nFAILED: {e}")
        if "mps" in str(e).lower() or "MPS" in str(e):
            print("Hint: Try rerunning with --device cpu")
        raise


if __name__ == "__main__":
    main()
