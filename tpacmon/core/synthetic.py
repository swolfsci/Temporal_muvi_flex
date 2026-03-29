"""Synthetic data generation for testing tpacmon."""

from typing import Optional

import numpy as np
from tpacmon.core.kernels import matern32
import torch


def simulate_longitudinal(
    n_patients: int = 30,
    n_timepoints: int = 8,
    n_factors: int = 3,
    n_features: Optional[list] = None,
    n_views: int = 2,
    n_covariates: int = 2,
    lengthscales: Optional[np.ndarray] = None,
    noise_std: float = 0.5,
    sparsity: float = 0.7,
    missing_rate: float = 0.2,
    seed: int = 42,
) -> dict:
    """Generate synthetic longitudinal multi-omics data from a GP latent factor model.

    Returns:
        dict with keys: observations, time_points, patient_masks, covariates,
        true_z, true_w, true_lengthscales, factor_names, view_names
    """
    rng = np.random.default_rng(seed)

    if n_features is None:
        n_features = [100] * n_views

    # Time grid (irregular-ish)
    time_points = np.sort(rng.uniform(0, 10, size=n_timepoints)).astype(np.float32)

    # Patient masks (each patient observes at least 3 timepoints)
    patient_masks = np.ones((n_patients, n_timepoints), dtype=bool)
    for p in range(n_patients):
        n_missing = rng.binomial(n_timepoints, missing_rate)
        n_missing = min(n_missing, n_timepoints - 3)
        if n_missing > 0:
            drop_idx = rng.choice(n_timepoints, size=n_missing, replace=False)
            patient_masks[p, drop_idx] = False

    # Covariates
    covariates = rng.standard_normal((n_patients, n_covariates)).astype(np.float32)

    # Lengthscales
    if lengthscales is None:
        lengthscales = rng.uniform(1.0, 4.0, size=n_factors).astype(np.float32)

    # Generate true z from GP
    t_tensor = torch.tensor(time_points)
    true_z = np.zeros((n_patients, n_timepoints, n_factors), dtype=np.float32)

    for k in range(n_factors):
        ls = torch.tensor(lengthscales[k])
        amp = torch.tensor(1.0)
        K = matern32(t_tensor, t_tensor, ls, amp).numpy()
        K += 1e-5 * np.eye(n_timepoints)
        L = np.linalg.cholesky(K)
        for p in range(n_patients):
            true_z[p, :, k] = L @ rng.standard_normal(n_timepoints)

    # Generate true w (sparse loadings)
    view_names = [f"view_{m}" for m in range(n_views)]
    true_w = {}
    for m, vn in enumerate(view_names):
        w = rng.standard_normal((n_factors, n_features[m])).astype(np.float32)
        mask = rng.random((n_factors, n_features[m])) < sparsity
        w[mask] = 0.0
        true_w[vn] = w

    # Generate observations
    observations = {}
    for m, vn in enumerate(view_names):
        y = np.zeros((n_patients, n_timepoints, n_features[m]), dtype=np.float32)
        for p in range(n_patients):
            y[p] = true_z[p] @ true_w[vn] + noise_std * rng.standard_normal(
                (n_timepoints, n_features[m])
            ).astype(np.float32)
        # Mask unobserved timepoints with NaN
        for p in range(n_patients):
            y[p, ~patient_masks[p], :] = np.nan
        observations[vn] = y

    return {
        "observations": observations,
        "time_points": time_points,
        "patient_masks": patient_masks,
        "covariates": covariates,
        "true_z": true_z,
        "true_w": true_w,
        "true_lengthscales": lengthscales,
        "factor_names": [f"factor_{k}" for k in range(n_factors)],
        "view_names": view_names,
    }
