"""Shared evaluation metrics for tpacmon synthetic benchmarks.

All metrics operate on numpy arrays. Factor alignment (Hungarian algorithm)
must be computed first — all other metrics expect aligned arrays.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score


# ---------------------------------------------------------------------------
# Factor alignment
# ---------------------------------------------------------------------------

def align_factors(
    true_z: np.ndarray,
    learned_z: np.ndarray,
) -> Dict:
    """Align learned factors to true factors via optimal permutation.

    Uses absolute Pearson correlation as similarity + Hungarian algorithm.
    Follows PACMon's optim_perm pattern.

    Args:
        true_z: (P, K_true) or (P, T, K_true)
        learned_z: (P, K_learned) or (P, T, K_learned)

    Returns:
        dict with keys:
            permutation: index array mapping learned → true
            sign_flips: +1/-1 per aligned factor
            correlations: per-factor correlation after alignment
            cost_matrix: full correlation matrix (K_true x K_learned)
    """
    if true_z.ndim == 3:
        P, T, K_true = true_z.shape
        K_learned = learned_z.shape[2]
        zt = true_z.reshape(-1, K_true)
        zl = learned_z.reshape(-1, K_learned)
    else:
        zt = true_z
        zl = learned_z
        K_true = zt.shape[1]
        K_learned = zl.shape[1]

    # Drop NaN rows (from masked timepoints)
    valid = ~(np.isnan(zt).any(axis=1) | np.isnan(zl).any(axis=1))
    zt = zt[valid]
    zl = zl[valid]

    # Correlation matrix (K_true x K_learned)
    corr = np.zeros((K_true, K_learned))
    for i in range(K_true):
        for j in range(K_learned):
            r = np.corrcoef(zt[:, i], zl[:, j])[0, 1]
            corr[i, j] = r if np.isfinite(r) else 0.0

    # Hungarian on negative absolute correlation (minimize cost = maximize similarity)
    cost = -np.abs(corr)
    row_ind, col_ind = linear_sum_assignment(cost)

    # Build permutation: for each true factor i, col_ind[i] is the best learned factor
    permutation = col_ind
    sign_flips = np.sign(corr[row_ind, col_ind])
    sign_flips[sign_flips == 0] = 1.0
    correlations = corr[row_ind, col_ind]

    return {
        "permutation": permutation,
        "sign_flips": sign_flips,
        "correlations": correlations,
        "cost_matrix": corr,
    }


def apply_alignment(
    arr: np.ndarray,
    alignment: Dict,
    axis: int = -1,
) -> np.ndarray:
    """Reorder and sign-flip an array along the factor axis.

    Args:
        arr: array with factor dimension at `axis`
        alignment: output of align_factors
        axis: which axis is the factor dimension
    """
    perm = alignment["permutation"]
    signs = alignment["sign_flips"]
    out = np.take(arr, perm, axis=axis)
    # Broadcast sign flips
    shape = [1] * out.ndim
    shape[axis] = len(signs)
    out = out * signs.reshape(shape)
    return out


# ---------------------------------------------------------------------------
# Factor score metrics
# ---------------------------------------------------------------------------

def factor_score_correlation(
    true_z: np.ndarray,
    learned_z: np.ndarray,
    alignment: Dict,
) -> Dict:
    """Per-factor Pearson r between true and learned z (after alignment).

    For temporal data: average correlation across patients per factor.

    Returns:
        dict with per_factor (K,), mean, and per_patient_factor (P, K) if temporal
    """
    learned_aligned = apply_alignment(learned_z, alignment, axis=-1)

    if true_z.ndim == 3:
        P, T, K = true_z.shape
        per_patient_factor = np.full((P, K), np.nan)
        for p in range(P):
            for k in range(K):
                valid = ~(np.isnan(true_z[p, :, k]) | np.isnan(learned_aligned[p, :, k]))
                if valid.sum() >= 3:
                    r = np.corrcoef(true_z[p, valid, k], learned_aligned[p, valid, k])[0, 1]
                    per_patient_factor[p, k] = r if np.isfinite(r) else 0.0
        per_factor = np.nanmean(per_patient_factor, axis=0)
    else:
        K = true_z.shape[1]
        per_factor = np.array([
            np.corrcoef(true_z[:, k], learned_aligned[:, k])[0, 1]
            for k in range(K)
        ])
        per_patient_factor = None

    return {
        "per_factor": per_factor,
        "mean": float(np.nanmean(per_factor)),
        "per_patient_factor": per_patient_factor,
    }


# ---------------------------------------------------------------------------
# Loading metrics
# ---------------------------------------------------------------------------

def loading_aupr(
    true_w_mask: np.ndarray,
    learned_w: np.ndarray,
    alignment: Dict,
) -> Dict:
    """AUPR for loading sparsity recovery.

    Uses |learned_w| as score, true_w_mask as binary label.
    Follows PACMon's approach (runtime.py line 249).

    Args:
        true_w_mask: binary (K, D) ground truth active features
        learned_w: continuous (K, D) learned loadings
        alignment: factor alignment dict

    Returns:
        dict with overall AUPR and per_factor (K,) AUPR
    """
    learned_aligned = apply_alignment(learned_w, alignment, axis=0)
    scores = np.abs(learned_aligned)

    K = true_w_mask.shape[0]
    per_factor = np.zeros(K)
    for k in range(K):
        if true_w_mask[k].sum() == 0 or true_w_mask[k].all():
            per_factor[k] = np.nan
        else:
            per_factor[k] = average_precision_score(true_w_mask[k], scores[k])

    overall = average_precision_score(true_w_mask.ravel(), scores.ravel())

    return {
        "overall": float(overall),
        "per_factor": per_factor,
        "mean": float(np.nanmean(per_factor)),
    }


def loading_correlation(
    true_w: np.ndarray,
    learned_w: np.ndarray,
    alignment: Dict,
) -> Dict:
    """Per-factor Pearson r between true and learned loadings."""
    learned_aligned = apply_alignment(learned_w, alignment, axis=0)
    K = true_w.shape[0]
    per_factor = np.array([
        np.corrcoef(true_w[k], learned_aligned[k])[0, 1]
        for k in range(K)
    ])
    per_factor = np.where(np.isfinite(per_factor), per_factor, 0.0)
    return {
        "per_factor": per_factor,
        "mean": float(np.mean(per_factor)),
    }


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def reconstruction_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """RMSE of reconstruction, optionally masked to observed entries."""
    if mask is not None:
        # mask is (P, T) boolean — expand to (P, T, D)
        if mask.ndim == 2 and y_true.ndim == 3:
            mask_3d = mask[:, :, np.newaxis]
            valid = ~np.isnan(y_true) & mask_3d
        else:
            valid = ~np.isnan(y_true) & mask
    else:
        valid = ~np.isnan(y_true)

    diff = (y_true - y_pred)[valid]
    return float(np.sqrt(np.mean(diff ** 2)))


# ---------------------------------------------------------------------------
# GP-specific metrics
# ---------------------------------------------------------------------------

def lengthscale_recovery(
    true_ls: np.ndarray,
    learned_ls: np.ndarray,
    alignment: Dict,
) -> Dict:
    """Correlation and MAE between true and learned lengthscales.

    Only compares the temporal factors (first len(true_ls) in alignment order).
    """
    n_temporal = len(true_ls)
    perm = alignment["permutation"][:n_temporal]
    learned_aligned = learned_ls[perm]

    r = np.corrcoef(true_ls, learned_aligned)[0, 1] if n_temporal > 1 else np.nan
    mae = float(np.mean(np.abs(true_ls - learned_aligned)))

    return {
        "correlation": float(r) if np.isfinite(r) else np.nan,
        "mae": mae,
        "true": true_ls.tolist(),
        "learned": learned_aligned.tolist(),
    }


def zeta_discrimination(
    true_is_temporal: np.ndarray,
    learned_zeta: np.ndarray,
    alignment: Dict,
) -> Dict:
    """AUROC of learned zeta for classifying GP vs. i.i.d. factors.

    Low zeta = temporal, so use (1 - zeta) as score for 'is temporal'.

    Returns:
        dict with auroc and per-factor zeta values after alignment
    """
    from sklearn.metrics import roc_auc_score

    zeta_aligned = learned_zeta[alignment["permutation"]]
    scores = 1.0 - zeta_aligned  # high score = temporal

    n_pos = true_is_temporal.sum()
    n_neg = (~true_is_temporal).sum()

    if n_pos == 0 or n_neg == 0:
        auroc = np.nan
    else:
        auroc = roc_auc_score(true_is_temporal.astype(int), scores)

    return {
        "auroc": float(auroc) if np.isfinite(auroc) else np.nan,
        "zeta_temporal_mean": float(zeta_aligned[true_is_temporal].mean()) if n_pos > 0 else np.nan,
        "zeta_iid_mean": float(zeta_aligned[~true_is_temporal].mean()) if n_neg > 0 else np.nan,
        "zeta_per_factor": zeta_aligned.tolist(),
    }


# ---------------------------------------------------------------------------
# Covariate metrics
# ---------------------------------------------------------------------------

def gamma_recovery(
    true_gamma: np.ndarray,
    learned_gamma: np.ndarray,
    alignment: Dict,
) -> Dict:
    """Correlation between true and learned gamma (covariate → factor effect).

    Args:
        true_gamma: (C, K)
        learned_gamma: (C, K)
    """
    learned_aligned = apply_alignment(learned_gamma, alignment, axis=-1)
    C, K = true_gamma.shape

    # Overall correlation
    r_overall = np.corrcoef(true_gamma.ravel(), learned_aligned.ravel())[0, 1]

    # Per-covariate correlation
    per_cov = np.array([
        np.corrcoef(true_gamma[c], learned_aligned[c])[0, 1]
        if np.std(true_gamma[c]) > 0 else np.nan
        for c in range(C)
    ])

    # MAE
    mae = float(np.mean(np.abs(true_gamma - learned_aligned)))

    return {
        "overall_correlation": float(r_overall) if np.isfinite(r_overall) else np.nan,
        "per_covariate": per_cov.tolist(),
        "mae": mae,
        "true": true_gamma.tolist(),
        "learned": learned_aligned.tolist(),
    }


def beta_recovery(
    true_beta: np.ndarray,
    learned_beta: np.ndarray,
) -> Dict:
    """Correlation and AUPR for feature-level covariate effects.

    Args:
        true_beta: (C, D) — may contain zeros (sparse)
        learned_beta: (C, D)
    """
    r = np.corrcoef(true_beta.ravel(), learned_beta.ravel())[0, 1]

    # AUPR for nonzero detection
    mask = (np.abs(true_beta) > 0).ravel()
    if mask.any() and not mask.all():
        aupr = average_precision_score(mask, np.abs(learned_beta).ravel())
    else:
        aupr = np.nan

    return {
        "correlation": float(r) if np.isfinite(r) else np.nan,
        "aupr": float(aupr) if np.isfinite(aupr) else np.nan,
    }


# ---------------------------------------------------------------------------
# Variance explained
# ---------------------------------------------------------------------------

def variance_explained(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """R² = 1 - SS_res / SS_tot, matching MuVI/PACMon/MOFA2 formula."""
    if mask is not None:
        if mask.ndim == 2 and y_true.ndim == 3:
            mask_3d = mask[:, :, np.newaxis]
            valid = ~np.isnan(y_true) & mask_3d
        else:
            valid = ~np.isnan(y_true) & mask
    else:
        valid = ~np.isnan(y_true)

    y_t = y_true[valid]
    y_p = y_pred[valid]
    ss_tot = np.sum((y_t - y_t.mean()) ** 2)
    ss_res = np.sum((y_t - y_p) ** 2)

    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

def collect_result(
    level: int,
    seed: int,
    config: dict,
    metrics: dict,
    runtime_sec: float,
) -> dict:
    """Build a structured result dict for JSON serialization.

    This is the canonical output format. Every evaluation run produces one of these.
    """
    def _sanitize(obj):
        """Convert numpy types to Python types for JSON."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            v = float(obj)
            return None if np.isnan(v) else v
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        return obj

    return _sanitize({
        "level": level,
        "seed": seed,
        "config": config,
        "metrics": metrics,
        "runtime_sec": runtime_sec,
    })


def results_to_csv_row(result: dict) -> dict:
    """Flatten a result dict into a single-level dict suitable for CSV.

    Extracts the most important scalar metrics. Nested per-factor arrays
    are stored as their mean.
    """
    m = result["metrics"]
    row = {
        "level": result["level"],
        "seed": result["seed"],
        "runtime_sec": result["runtime_sec"],
    }

    # Config scalars
    for k, v in result["config"].items():
        if isinstance(v, (int, float, str, bool)):
            row[f"cfg_{k}"] = v

    # Factor scores
    if "factor_score_correlation" in m:
        row["z_corr_mean"] = m["factor_score_correlation"]["mean"]

    # Loadings
    if "loading_aupr" in m:
        row["w_aupr"] = m["loading_aupr"]["overall"]
        row["w_aupr_mean"] = m["loading_aupr"]["mean"]
    if "loading_correlation" in m:
        row["w_corr_mean"] = m["loading_correlation"]["mean"]

    # Reconstruction
    if "r2" in m:
        row["r2"] = m["r2"]
    if "rmse" in m:
        row["rmse"] = m["rmse"]

    # GP
    if "lengthscale_recovery" in m:
        row["ls_corr"] = m["lengthscale_recovery"]["correlation"]
        row["ls_mae"] = m["lengthscale_recovery"]["mae"]
    if "zeta_discrimination" in m:
        row["zeta_auroc"] = m["zeta_discrimination"]["auroc"]
        row["zeta_temporal_mean"] = m["zeta_discrimination"]["zeta_temporal_mean"]
        row["zeta_iid_mean"] = m["zeta_discrimination"]["zeta_iid_mean"]

    # Covariates
    if "gamma_recovery" in m:
        row["gamma_corr"] = m["gamma_recovery"]["overall_correlation"]
        row["gamma_mae"] = m["gamma_recovery"]["mae"]
    if "beta_recovery" in m:
        row["beta_corr"] = m["beta_recovery"]["correlation"]
        row["beta_aupr"] = m["beta_recovery"]["aupr"]

    # Alignment quality
    if "alignment" in m:
        row["alignment_corr_mean"] = float(np.mean(m["alignment"]["correlations"]))

    return row
