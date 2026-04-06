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

    Args:
        params: Model parameters.
        lr: Learning rate.
        betas: Tuple of (beta1, beta2) for Adam-style moments.
        beta3: EMA coefficient for the slow gradient buffer (default 0.9999).
        alpha: Mixing coefficient for the slow buffer (default 5.0).
        eps: Term for numerical stability.
        weight_decay: Decoupled weight decay.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), beta3=0.9999,
                 alpha=5.0, eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, beta3=beta3, alpha=alpha,
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
                m1, v, m2 = state["exp_avg"], state["exp_avg_sq"], state["exp_avg_slow"]

                # Decoupled weight decay
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                # Update moments
                m1.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                m2.mul_(beta3).add_(grad, alpha=1 - beta3)

                # Bias corrections
                step = state["step"]
                m1_hat = m1 / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)
                m2_hat = m2 / (1 - beta3 ** step)

                # AdEMAMix update: m1_hat + alpha * m2_hat
                denom = v_hat.sqrt().add_(group["eps"])
                update = (m1_hat + alpha * m2_hat) / denom

                p.add_(update, alpha=-group["lr"])

        return loss


# ---------------------------------------------------------------------------
# Inline SOAP implementation (no PyPI package available)
# Based on: Vyas et al., "SOAP: Improving and Stabilizing Shampoo using Adam"
# (ICLR 2025)
# ---------------------------------------------------------------------------

class SOAP(Optimizer):
    """SOAP optimizer: Shampoo + Adam hybrid with Kronecker-factored preconditioner.

    Args:
        params: Model parameters.
        lr: Learning rate.
        betas: Tuple of (beta1, beta2) for Adam moments.
        eps: Numerical stability term.
        weight_decay: Decoupled weight decay.
        precondition_frequency: How often to update the preconditioner.
        shampoo_beta: EMA decay for the Shampoo preconditioner statistics.
    """

    def __init__(self, params, lr=1e-3, betas=(0.95, 0.95), eps=1e-8,
                 weight_decay=0.0, precondition_frequency=10, shampoo_beta=0.95):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        precondition_frequency=precondition_frequency,
                        shampoo_beta=shampoo_beta)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            precond_freq = group["precondition_frequency"]
            shampoo_beta = group["shampoo_beta"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    # Only use Shampoo preconditioner for 2D params (weight matrices)
                    if p.dim() == 2:
                        m, n = p.shape
                        state["GG_left"] = torch.zeros(m, m, device=p.device, dtype=p.dtype)
                        state["GG_right"] = torch.zeros(n, n, device=p.device, dtype=p.dtype)
                        state["Q_left"] = torch.eye(m, device=p.device, dtype=p.dtype)
                        state["Q_right"] = torch.eye(n, device=p.device, dtype=p.dtype)

                state["step"] += 1
                step = state["step"]

                # Decoupled weight decay
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                # Standard Adam moments
                state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = state["exp_avg"] / (1 - beta1 ** step)
                v_hat = state["exp_avg_sq"] / (1 - beta2 ** step)

                if p.dim() == 2:
                    # Update Shampoo statistics
                    state["GG_left"].mul_(shampoo_beta).add_(
                        grad @ grad.T, alpha=1 - shampoo_beta
                    )
                    state["GG_right"].mul_(shampoo_beta).add_(
                        grad.T @ grad, alpha=1 - shampoo_beta
                    )

                    # Periodically update eigenbases
                    if step % precond_freq == 0:
                        try:
                            # Compute on CPU if MPS (eigendecomposition not always stable)
                            compute_device = p.device
                            if str(p.device).startswith("mps"):
                                compute_device = torch.device("cpu")

                            L = state["GG_left"].to(compute_device)
                            R = state["GG_right"].to(compute_device)

                            _, Q_l = torch.linalg.eigh(L + group["eps"] * torch.eye(
                                L.shape[0], device=compute_device, dtype=L.dtype))
                            _, Q_r = torch.linalg.eigh(R + group["eps"] * torch.eye(
                                R.shape[0], device=compute_device, dtype=R.dtype))

                            state["Q_left"] = Q_l.to(p.device)
                            state["Q_right"] = Q_r.to(p.device)
                        except Exception:
                            # If eigendecomp fails, keep identity (falls back to Adam)
                            pass

                    # Preconditioned update: project into eigenbasis, apply Adam, project back
                    Q_l = state["Q_left"]
                    Q_r = state["Q_right"]

                    m_projected = Q_l.T @ m_hat @ Q_r
                    v_projected = Q_l.T @ v_hat @ Q_r

                    update = m_projected / (v_projected.sqrt() + group["eps"])
                    update = Q_l @ update @ Q_r.T

                    p.add_(update, alpha=-group["lr"])
                else:
                    # For non-2D params, fall back to standard Adam
                    update = m_hat / (v_hat.sqrt() + group["eps"])
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
        return AdEMAMix(params, lr=lr, weight_decay=weight_decay,
                        alpha=alpha, beta3=beta3)

    elif name == "soap":
        precond_freq = extra.get("precondition_frequency", 10)
        return SOAP(params, lr=lr, weight_decay=weight_decay,
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
