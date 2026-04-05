"""Device detection and reproducibility utilities."""

import os
import random

import numpy as np
import torch


def get_device(force: str | None = None) -> torch.device:
    """Auto-detect best available device: MPS > CUDA > CPU.

    Args:
        force: If set, use this device directly (e.g. "cpu", "mps", "cuda").
    """
    if force:
        return torch.device(force)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        # Smoke test: run a small matmul to verify MPS works
        try:
            a = torch.randn(8, 8, device=device)
            _ = a @ a
            print(f"[device] Using MPS (Apple Silicon)")
            return device
        except Exception as e:
            print(f"[device] MPS available but smoke test failed: {e}")
            print("[device] Falling back to CPU")
            return torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[device] Using CUDA ({torch.cuda.get_device_name(0)})")
        return device

    print("[device] Using CPU")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # torch.manual_seed also seeds MPS on PyTorch 2.x+
    os.environ["PYTHONHASHSEED"] = str(seed)
