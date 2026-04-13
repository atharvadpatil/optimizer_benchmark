"""Unified optimizer factory for all 7 optimizers."""

import math

import torch
from torch.optim import SGD, AdamW
from torch.optim.optimizer import Optimizer


# ---------------------------------------------------------------------------
# Inline AdEMAMix implementation (no PyPI package available)
# Based on: Pagliardini et al., "The AdEMAMix Optimizer" (ICLR 2025)
# ---------------------------------------------------------------------------

class AdEMAMix(Optimizer):
    """AdEMAMix optimizer: mixes fast and slow EMA of gradients.

    Includes linear warmup for alpha and beta3 per Algorithm 1 of
    Pagliardini et al. (ICLR 2025).

    Args:
        params: Model parameters.
        lr: Learning rate.
        betas: Tuple of (beta1, beta2) for Adam-style moments.
        beta3: Target EMA coefficient for the slow gradient buffer (default 0.9999).
        alpha: Target mixing coefficient for the slow buffer (default 5.0).
        T_alpha: Steps to linearly warm up alpha from 0 to alpha (default 20000).
        T_beta3: Steps to linearly warm up beta3 from beta1 to beta3 (default 20000).
        eps: Term for numerical stability.
        weight_decay: Decoupled weight decay.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), beta3=0.9999,
                 alpha=5.0, T_alpha=20000, T_beta3=20000,
                 eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, beta3=beta3, alpha=alpha,
                        T_alpha=T_alpha, T_beta3=T_beta3,
                        eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            beta3 = group["beta3"]
            alpha = group["alpha"]
            T_alpha = group["T_alpha"]
            T_beta3 = group["T_beta3"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdEMAMix does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)      # fast EMA (m1)
                    state["exp_avg_sq"] = torch.zeros_like(p)    # second moment (v)
                    state["exp_avg_slow"] = torch.zeros_like(p)  # slow EMA (m2)

                state["step"] += 1
                step = state["step"]
                m1, v, m2 = state["exp_avg"], state["exp_avg_sq"], state["exp_avg_slow"]

                # Linear warmup schedules (Algorithm 1, Pagliardini et al.)
                alpha_t = min(step / T_alpha, 1.0) * alpha
                beta3_t = beta1 + min(step / T_beta3, 1.0) * (beta3 - beta1)

                # Decoupled weight decay
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                # Update moments
                m1.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                m2.mul_(beta3_t).add_(grad, alpha=1 - beta3_t)

                # Bias corrections
                m1_hat = m1 / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)
                # Approximate bias correction: uses current scheduled beta3_t.
                # Since beta3_t changes each step, 1 - beta3_t^step is not exact,
                # but matches common practice for scheduled EMA coefficients.
                m2_hat = m2 / (1 - beta3_t ** step)

                # AdEMAMix update: m1_hat + alpha_t * m2_hat
                denom = v_hat.sqrt().add_(group["eps"])
                update = (m1_hat + alpha_t * m2_hat) / denom

                p.add_(update, alpha=-group["lr"])

        return loss




# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_optimizer(name, model, lr, weight_decay, extra_kwargs=None):
    """Create an optimizer by name.

    Args:
        name: One of 'sgd', 'adamw', 'lion', 'ademamix', 'soap', 'muon',
              'schedulefree_adamw'.
        model: The nn.Module whose parameters to optimize.
        lr: Learning rate.
        weight_decay: Weight decay coefficient.
        extra_kwargs: Optional dict of extra arguments passed to the optimizer.

    Returns:
        optimizer: The constructed optimizer.
    """
    extra = extra_kwargs or {}
    name = name.lower().replace("-", "_")
    params = model.parameters()

    if name == "sgd":
        momentum = extra.get("momentum", 0.9)
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    elif name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)

    elif name == "lion":
        try:
            from lion_pytorch import Lion
        except ImportError:
            raise ImportError(
                "lion-pytorch not installed. Run: pip install lion-pytorch"
            )
        return Lion(params, lr=lr, weight_decay=weight_decay)

    elif name == "ademamix":
        alpha = extra.get("alpha", 5.0)
        beta3 = extra.get("beta3", 0.9999)
        T_alpha = extra.get("T_alpha", 20000)
        T_beta3 = extra.get("T_beta3", 20000)
        return AdEMAMix(params, lr=lr, weight_decay=weight_decay,
                        alpha=alpha, beta3=beta3,
                        T_alpha=T_alpha, T_beta3=T_beta3)

    elif name == "soap":
        try:
            from trainers.soap import SOAP as SOAPOptimizer
        except ImportError:
            raise ImportError(
                "SOAP not found. Ensure trainers/soap.py exists (vendored from "
                "https://github.com/nikhilvyas/SOAP)"
            )
        precond_freq = extra.get("precondition_frequency", 10)
        return SOAPOptimizer(params, lr=lr, weight_decay=weight_decay,
                             precondition_frequency=precond_freq)

    elif name == "muon":
        try:
            from muon import SingleDeviceMuonWithAuxAdam as MuonOptimizer
        except ImportError:
            raise ImportError(
                "muon not installed. Run: pip install muon-optimizer"
            )
        # Muon only works on 2D+ params (weight matrices); biases/norms go to aux Adam
        muon_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
        adam_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
        param_groups = [
            {"params": muon_params, "use_muon": True, "lr": lr, "weight_decay": weight_decay},
            {"params": adam_params, "use_muon": False, "lr": lr * 0.1, "weight_decay": weight_decay},
        ]
        return MuonOptimizer(param_groups)

    elif name == "schedulefree_adamw":
        try:
            from schedulefree import AdamWScheduleFree
        except ImportError:
            raise ImportError(
                "schedulefree not installed. Run: pip install schedulefree"
            )
        return AdamWScheduleFree(params, lr=lr, weight_decay=weight_decay)

    else:
        raise ValueError(
            f"Unknown optimizer '{name}'. Choose from: sgd, adamw, lion, "
            f"ademamix, soap, muon, schedulefree_adamw"
        )
