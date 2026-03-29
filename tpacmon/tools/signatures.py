"""Predictive signatures via ElasticNet on learned factor scores."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV

logger = logging.getLogger(__name__)


def compute_predictive_signatures(
    model,
    view_name: Optional[str] = None,
    aggregation: str = "mean",
    n_alphas: int = 50,
    l1_ratio: Optional[list] = None,
    cv: int = 5,
    max_features: int = 50,
) -> dict:
    """Compute predictive gene signatures per factor using ElasticNet.

    For each factor, fits ElasticNet to predict the factor score from the
    original features, yielding a sparse set of predictive features.

    Args:
        model: A trained TemporalPACMON instance.
        view_name: Which view to use. If None, uses first view.
        aggregation: How to aggregate factor scores over time per patient.
            Options: 'mean', 'last', 'max', 'slope'.
        n_alphas: Number of alpha values for ElasticNetCV.
        l1_ratio: L1 ratio values to try. Default [0.5, 0.7, 0.9, 0.95, 1.0].
        cv: Number of cross-validation folds.
        max_features: Maximum features to report per factor.

    Returns:
        dict with:
            - 'signatures': {factor_name: DataFrame with feature, coefficient, abs_coefficient}
            - 'models': {factor_name: fitted ElasticNetCV}
            - 'scores': {factor_name: cv R² score}
    """
    if not model._trained:
        raise RuntimeError("Model must be trained first.")

    if l1_ratio is None:
        l1_ratio = [0.5, 0.7, 0.9, 0.95, 1.0]

    if view_name is None:
        view_name = model.view_names[0]

    # Get factor scores and aggregate over time
    z = model.get_factors()  # (P, T, K)
    masks = model.patient_masks_np

    z_agg = _aggregate_factors(z, masks, aggregation)  # (P, K)

    # Get features for the selected view
    obs = model.observations[view_name]  # (P, T, D)
    # Aggregate features the same way
    X = _aggregate_features(obs, masks, aggregation)  # (P, D)

    feature_names = model.feature_names[view_name]

    signatures = {}
    fitted_models = {}
    scores = {}

    for k, fname in enumerate(model.factor_names):
        y = z_agg[:, k]

        # Skip if no variance
        if np.nanstd(y) < 1e-8:
            logger.warning(f"Factor '{fname}' has near-zero variance, skipping.")
            continue

        # Remove NaN patients
        valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
        if valid.sum() < cv + 2:
            logger.warning(f"Factor '{fname}': too few valid patients ({valid.sum()}), skipping.")
            continue

        enet = ElasticNetCV(
            l1_ratio=l1_ratio,
            n_alphas=n_alphas,
            cv=cv,
            max_iter=10000,
        )
        enet.fit(X[valid], y[valid])

        coefs = enet.coef_
        nonzero = np.abs(coefs) > 1e-10
        n_selected = nonzero.sum()

        # Build signature DataFrame
        idx = np.argsort(-np.abs(coefs))[:max_features]
        sig_df = pd.DataFrame({
            "feature": [feature_names[i] for i in idx],
            "coefficient": coefs[idx],
            "abs_coefficient": np.abs(coefs[idx]),
        })
        sig_df = sig_df[sig_df["abs_coefficient"] > 1e-10]

        signatures[fname] = sig_df
        fitted_models[fname] = enet
        scores[fname] = enet.score(X[valid], y[valid])

        logger.info(f"Factor '{fname}': {n_selected} features selected, R²={scores[fname]:.3f}")

    return {
        "signatures": signatures,
        "models": fitted_models,
        "scores": scores,
    }


def _aggregate_factors(z: np.ndarray, masks: np.ndarray, method: str) -> np.ndarray:
    """Aggregate (P, T, K) factor scores to (P, K)."""
    P, T, K = z.shape
    result = np.full((P, K), np.nan)

    for p in range(P):
        obs_idx = np.where(masks[p])[0]
        if len(obs_idx) == 0:
            continue
        z_p = z[p, obs_idx, :]  # (n_obs, K)

        if method == "mean":
            result[p] = np.nanmean(z_p, axis=0)
        elif method == "last":
            result[p] = z_p[-1]
        elif method == "max":
            result[p] = np.nanmax(np.abs(z_p), axis=0) * np.sign(z_p[np.nanargmax(np.abs(z_p), axis=0), range(K)])
        elif method == "slope":
            t_obs = np.arange(len(obs_idx), dtype=np.float32)
            if len(obs_idx) < 2:
                result[p] = 0.0
            else:
                for k in range(K):
                    result[p, k] = np.polyfit(t_obs, z_p[:, k], 1)[0]
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    return result


def _aggregate_features(obs: np.ndarray, masks: np.ndarray, method: str) -> np.ndarray:
    """Aggregate (P, T, D) features to (P, D) using the same method."""
    P, T, D = obs.shape
    result = np.full((P, D), np.nan)

    for p in range(P):
        obs_idx = np.where(masks[p])[0]
        if len(obs_idx) == 0:
            continue
        x_p = obs[p, obs_idx, :]

        if method == "mean":
            result[p] = np.nanmean(x_p, axis=0)
        elif method == "last":
            result[p] = x_p[-1]
        elif method == "max":
            result[p] = np.nanmax(np.abs(x_p), axis=0)
        elif method == "slope":
            t_obs = np.arange(len(obs_idx), dtype=np.float32)
            if len(obs_idx) < 2:
                result[p] = 0.0
            else:
                for d in range(D):
                    result[p, d] = np.polyfit(t_obs, x_p[:, d], 1)[0]

    return result
