"""Enhanced synthetic data generator for tpacmon evaluation.

Extends the core simulate_longitudinal() to support all 5 evaluation levels:
  Level 1: dense i.i.d. factors only
  Level 2: GP temporal + i.i.d. factors
  Level 3: informed priors (noisy gene-set masks)
  Level 4: dual-level covariates
  Level 5: full model (dynamic covariates, missingness, irregular time)

Returns all ground truth needed for metrics computation.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from tpacmon.core.kernels import matern32


def generate_evaluation_data(
    # Dimensions
    n_patients: int = 50,
    n_timepoints: int = 6,
    n_views: int = 2,
    n_features: Optional[List[int]] = None,

    # Factor structure
    n_temporal_factors: int = 0,
    n_iid_factors: int = 3,
    lengthscales: Optional[List[float]] = None,

    # Prior masks
    use_prior_masks: bool = False,
    prior_noise_fraction: float = 0.0,
    n_active_features_per_factor: int = 30,

    # Covariates
    n_covariates: int = 0,
    covariate_types: Optional[List[str]] = None,
    covariate_names: Optional[List[str]] = None,
    gamma_true: Optional[np.ndarray] = None,
    beta_true: Optional[Dict[str, np.ndarray]] = None,
    beta_sparsity: float = 0.8,

    # Dynamic covariates (Level 5)
    dynamic_covariates: Optional[List[int]] = None,
    dynamic_effect_magnitude: float = 1.5,

    # Observation noise
    noise_std: float = 0.5,

    # Missingness
    missing_rate: float = 0.0,
    min_observations: int = 3,

    # Time grid
    time_points: Optional[np.ndarray] = None,
    time_spacing: str = "regular",

    # Reproducibility
    seed: int = 42,
) -> Dict:
    """Generate synthetic longitudinal multi-omics data for evaluation.

    Returns dict with two groups of keys:
      Data for model fitting:
        observations, time_points, patient_masks, covariates, prior_masks

      Ground truth for evaluation:
        true_z, true_w, true_w_mask, true_lengthscales, true_gamma,
        true_beta, true_is_temporal, true_prior_masks (clean),
        factor_names, view_names, covariate_names, covariate_types,
        dynamic_covariates
    """
    rng = np.random.default_rng(seed)
    K = n_temporal_factors + n_iid_factors

    if n_features is None:
        n_features = [100] * n_views

    view_names = [f"view_{m}" for m in range(n_views)]

    # -----------------------------------------------------------------------
    # Time grid
    # -----------------------------------------------------------------------
    if time_points is not None:
        t = np.asarray(time_points, dtype=np.float32)
        n_timepoints = len(t)
    elif time_spacing == "regular":
        t = np.linspace(0, 10, n_timepoints).astype(np.float32)
    elif time_spacing == "irregular":
        t = np.sort(rng.uniform(0, 10, size=n_timepoints)).astype(np.float32)
    else:
        t = np.linspace(0, 10, n_timepoints).astype(np.float32)

    # -----------------------------------------------------------------------
    # Patient masks
    # -----------------------------------------------------------------------
    patient_masks = np.ones((n_patients, n_timepoints), dtype=bool)
    if missing_rate > 0:
        for p in range(n_patients):
            n_missing = rng.binomial(n_timepoints, missing_rate)
            n_missing = min(n_missing, n_timepoints - min_observations)
            n_missing = max(n_missing, 0)
            if n_missing > 0:
                drop_idx = rng.choice(n_timepoints, size=n_missing, replace=False)
                patient_masks[p, drop_idx] = False

    # -----------------------------------------------------------------------
    # Covariates
    # -----------------------------------------------------------------------
    if n_covariates > 0:
        if covariate_types is None:
            covariate_types = ["continuous"] * n_covariates
        if covariate_names is None:
            covariate_names = [f"cov_{c}" for c in range(n_covariates)]

        covariates = np.zeros((n_patients, n_covariates), dtype=np.float32)
        for c in range(n_covariates):
            if covariate_types[c] == "binary":
                covariates[:, c] = rng.choice([0.0, 1.0], size=n_patients).astype(np.float32)
            else:
                covariates[:, c] = rng.standard_normal(n_patients).astype(np.float32)
    else:
        covariates = None
        covariate_types = []
        covariate_names = []

    if dynamic_covariates is None:
        dynamic_covariates = []

    # -----------------------------------------------------------------------
    # Lengthscales for temporal factors
    # -----------------------------------------------------------------------
    if lengthscales is not None:
        ls = np.asarray(lengthscales, dtype=np.float32)[:n_temporal_factors]
    elif n_temporal_factors > 0:
        ls = rng.uniform(1.5, 6.0, size=n_temporal_factors).astype(np.float32)
    else:
        ls = np.array([], dtype=np.float32)

    # -----------------------------------------------------------------------
    # True factor scores z: (P, T, K)
    # -----------------------------------------------------------------------
    true_z = np.zeros((n_patients, n_timepoints, K), dtype=np.float32)
    true_is_temporal = np.zeros(K, dtype=bool)
    t_tensor = torch.tensor(t)

    # Temporal factors (GP-drawn)
    for k in range(n_temporal_factors):
        true_is_temporal[k] = True
        ls_k = torch.tensor(float(ls[k]))
        amp = torch.tensor(1.0)
        K_mat = matern32(t_tensor, t_tensor, ls_k, amp).numpy()
        K_mat += 1e-5 * np.eye(n_timepoints)
        L = np.linalg.cholesky(K_mat)

        for p in range(n_patients):
            base = L @ rng.standard_normal(n_timepoints).astype(np.float32)

            # Dynamic covariate interaction: shift trajectory based on covariate
            if covariates is not None:
                for c_idx in dynamic_covariates:
                    if c_idx < n_covariates:
                        # Create time-varying effect: stronger at later timepoints
                        time_weight = (t - t[0]) / (t[-1] - t[0] + 1e-8)
                        effect = covariates[p, c_idx] * dynamic_effect_magnitude * time_weight
                        # Only affect some factors (first n_temporal//2 + 1)
                        if k <= n_temporal_factors // 2:
                            base += effect.astype(np.float32)

            true_z[p, :, k] = base

    # i.i.d. factors
    for k in range(n_temporal_factors, K):
        for p in range(n_patients):
            true_z[p, :, k] = rng.standard_normal(n_timepoints).astype(np.float32)

    # Static covariate effect on GP mean (gamma)
    if covariates is not None and n_covariates > 0:
        if gamma_true is not None:
            gamma = np.asarray(gamma_true, dtype=np.float32)
        else:
            gamma = rng.standard_normal((n_covariates, K)).astype(np.float32) * 0.5
            # Zero out some entries for sparsity
            gamma_mask = rng.random((n_covariates, K)) < 0.5
            gamma[gamma_mask] = 0.0

        # Apply gamma: shift factor means by covariate * gamma
        # gamma[c, k] shifts factor k mean for patients with covariate c
        for c in range(n_covariates):
            if c not in dynamic_covariates:
                # Static covariate: constant shift across time
                for k in range(K):
                    if gamma[c, k] != 0:
                        shift = covariates[:, c] * gamma[c, k]  # (P,)
                        true_z[:, :, k] += shift[:, np.newaxis]
    else:
        gamma = None

    # -----------------------------------------------------------------------
    # True loadings w: {view: (K, D_m)}
    # -----------------------------------------------------------------------
    true_w = {}
    true_w_mask = {}

    for m, vn in enumerate(view_names):
        D = n_features[m]
        w = np.zeros((K, D), dtype=np.float32)
        w_mask = np.zeros((K, D), dtype=bool)

        for k in range(K):
            n_active = min(n_active_features_per_factor, D)
            active_idx = rng.choice(D, size=n_active, replace=False)
            w_mask[k, active_idx] = True
            w[k, active_idx] = rng.standard_normal(n_active).astype(np.float32)

        true_w[vn] = w
        true_w_mask[vn] = w_mask

    # -----------------------------------------------------------------------
    # Prior masks (noisy version of true_w_mask for informed factors)
    # -----------------------------------------------------------------------
    if use_prior_masks:
        true_prior_masks = {}
        noisy_prior_masks = {}

        for vn in view_names:
            # Only sparse (temporal) factors get priors by default;
            # if no temporal, use all factors
            n_prior_factors = n_temporal_factors if n_temporal_factors > 0 else K
            clean_mask = true_w_mask[vn][:n_prior_factors].copy()
            true_prior_masks[vn] = clean_mask

            if prior_noise_fraction > 0:
                noisy = clean_mask.copy()
                n_entries = noisy.size
                n_flip = int(n_entries * prior_noise_fraction)
                flip_idx = rng.choice(n_entries, size=n_flip, replace=False)
                flat = noisy.ravel()
                flat[flip_idx] = ~flat[flip_idx]
                noisy = flat.reshape(clean_mask.shape)
                noisy_prior_masks[vn] = noisy
            else:
                noisy_prior_masks[vn] = clean_mask.copy()
    else:
        true_prior_masks = None
        noisy_prior_masks = None

    # -----------------------------------------------------------------------
    # Feature-level covariate effects beta: {view: (C, D_m)}
    # -----------------------------------------------------------------------
    if covariates is not None and n_covariates > 0:
        true_beta = {}
        for vn in view_names:
            D = n_features[view_names.index(vn)]
            if beta_true is not None and vn in beta_true:
                b = np.asarray(beta_true[vn], dtype=np.float32)
            else:
                b = rng.standard_normal((n_covariates, D)).astype(np.float32) * 0.3
                b_mask = rng.random((n_covariates, D)) < beta_sparsity
                b[b_mask] = 0.0
            true_beta[vn] = b
    else:
        true_beta = None

    # -----------------------------------------------------------------------
    # Generate observations y = z @ w + covariates @ beta + noise
    # -----------------------------------------------------------------------
    observations = {}
    for m, vn in enumerate(view_names):
        D = n_features[m]
        y = np.zeros((n_patients, n_timepoints, D), dtype=np.float32)

        for p in range(n_patients):
            # Factor contribution: z[p] @ w  →  (T, K) @ (K, D) = (T, D)
            y[p] = true_z[p] @ true_w[vn]

            # Feature-level covariate contribution
            if true_beta is not None:
                # covariates[p] @ beta  →  (C,) @ (C, D) = (D,)
                cov_effect = covariates[p] @ true_beta[vn]
                y[p] += cov_effect[np.newaxis, :]  # broadcast across time

            # Noise
            y[p] += noise_std * rng.standard_normal((n_timepoints, D)).astype(np.float32)

        # Mask unobserved timepoints
        for p in range(n_patients):
            y[p, ~patient_masks[p], :] = np.nan

        observations[vn] = y

    # -----------------------------------------------------------------------
    # Factor names
    # -----------------------------------------------------------------------
    factor_names = []
    for k in range(n_temporal_factors):
        factor_names.append(f"temporal_{k}")
    for k in range(n_iid_factors):
        factor_names.append(f"dense_{k}")

    return {
        # Data for model fitting
        "observations": observations,
        "time_points": t,
        "patient_masks": patient_masks,
        "covariates": covariates,
        "prior_masks": noisy_prior_masks,

        # Ground truth
        "true_z": true_z,
        "true_w": true_w,
        "true_w_mask": true_w_mask,
        "true_lengthscales": ls,
        "true_gamma": gamma,
        "true_beta": true_beta,
        "true_is_temporal": true_is_temporal,
        "true_prior_masks": true_prior_masks,
        "factor_names": factor_names,
        "view_names": view_names,
        "covariate_names": covariate_names,
        "covariate_types": covariate_types,
        "dynamic_covariates": dynamic_covariates,
    }
