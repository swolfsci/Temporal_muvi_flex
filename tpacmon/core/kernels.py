"""GP kernel functions for tpacmon."""

import math
import torch


def matern32(t1: torch.Tensor, t2: torch.Tensor,
             lengthscale: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
    """Matérn 3/2 kernel: k(r) = a² (1 + √3 r/l) exp(-√3 r/l)."""
    dist = torch.cdist(t1.unsqueeze(-1), t2.unsqueeze(-1), p=2)
    sqrt3 = math.sqrt(3.0)
    scaled = sqrt3 * dist / lengthscale.clamp(min=1e-6)
    return amplitude.square() * (1.0 + scaled) * torch.exp(-scaled)


def matern52(t1: torch.Tensor, t2: torch.Tensor,
             lengthscale: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
    """Matérn 5/2 kernel: k(r) = a² (1 + √5 r/l + 5r²/3l²) exp(-√5 r/l)."""
    dist = torch.cdist(t1.unsqueeze(-1), t2.unsqueeze(-1), p=2)
    sqrt5 = math.sqrt(5.0)
    scaled = sqrt5 * dist / lengthscale.clamp(min=1e-6)
    return amplitude.square() * (1.0 + scaled + scaled.square() / 3.0) * torch.exp(-scaled)


def rbf(t1: torch.Tensor, t2: torch.Tensor,
        lengthscale: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
    """RBF (squared exponential) kernel: k(r) = a² exp(-r²/2l²)."""
    dist_sq = torch.cdist(t1.unsqueeze(-1), t2.unsqueeze(-1), p=2).square()
    return amplitude.square() * torch.exp(-0.5 * dist_sq / lengthscale.clamp(min=1e-6).square())


KERNEL_REGISTRY = {
    "matern32": matern32,
    "matern52": matern52,
    "rbf": rbf,
}


def build_kernel(name: str, t1: torch.Tensor,
                 lengthscale: torch.Tensor, amplitude: torch.Tensor,
                 jitter: float = 1e-5, t_obs2: torch.Tensor = None) -> torch.Tensor:
    """Build a kernel matrix with optional jitter.

    Args:
        name: Kernel name from KERNEL_REGISTRY.
        t1: Time points (first set).
        lengthscale: Kernel lengthscale.
        amplitude: Kernel amplitude.
        jitter: Diagonal jitter for numerical stability.
        t_obs2: Second time points (for cross-covariance). If None, uses t1.
    """
    if name not in KERNEL_REGISTRY:
        raise ValueError(f"Unknown kernel '{name}'. Available: {list(KERNEL_REGISTRY.keys())}")

    t2 = t_obs2 if t_obs2 is not None else t1
    K = KERNEL_REGISTRY[name](t1, t2, lengthscale, amplitude)

    if jitter > 0 and t_obs2 is None:
        K = K + jitter * torch.eye(t1.shape[0], device=t1.device)

    return K
