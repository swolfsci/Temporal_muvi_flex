"""Post-hoc enrichment analysis for learned latent factors."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentResult:
    """Container for enrichment analysis results.

    Attributes:
        fidelity: Prior fidelity metrics for sparse factors.
            Index: sparse factor names.
            Columns: fidelity, deviation, gini, retained, pruned, recruited,
            retained_frac, pruned_frac, recruited_frac.
        enrichment: PCGSE enrichment results.
            Columns: factor, gene_set, pvalue, padj, auroc, prior_rank.
        temporal: Temporal metrics for all factors.
            Index: factor names.
            Columns: zeta, temporal_strength, TVR, lengthscale.
        covariate: Covariate association metrics.
            Columns: factor, covariate, gamma, TD.
        covariate_summary: Covariate-level summary.
            Index: covariate names. Columns: TIR.
        signatures: Predictive signatures per factor (ElasticNet).
            Dict of {factor_name: DataFrame with feature, coefficient, abs_coefficient}.
            None unless compute_signatures=True.
        signature_scores: CV R² per factor for signature models.
            Dict of {factor_name: float}. None unless compute_signatures=True.
    """
    fidelity: Optional[pd.DataFrame] = None
    enrichment: Optional[pd.DataFrame] = None
    temporal: Optional[pd.DataFrame] = None
    covariate: Optional[pd.DataFrame] = None
    covariate_summary: Optional[pd.DataFrame] = None
    signatures: Optional[Dict[str, pd.DataFrame]] = None
    signature_scores: Optional[Dict[str, float]] = None

    _FIELDS = ("fidelity", "enrichment", "temporal", "covariate", "covariate_summary")

    def to_dict(self) -> dict:
        """Return non-None results as a dict of DataFrames/dicts."""
        out = {k: getattr(self, k) for k in self._FIELDS if getattr(self, k) is not None}
        if self.signatures is not None:
            out["signatures"] = self.signatures
        if self.signature_scores is not None:
            out["signature_scores"] = self.signature_scores
        return out

    def save(self, path: str) -> None:
        """Save enrichment results to disk.

        Args:
            path: File path (e.g., "enrichment.pkl").
        """
        import pickle
        data = {}
        for k in self._FIELDS:
            v = getattr(self, k)
            if v is not None:
                data[k] = v.to_dict(orient="split")
        if self.signatures is not None:
            data["signatures"] = {
                name: df.to_dict(orient="split") for name, df in self.signatures.items()
            }
        if self.signature_scores is not None:
            data["signature_scores"] = self.signature_scores
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Enrichment results saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "EnrichmentResult":
        """Load enrichment results from disk.

        Args:
            path: Path to saved enrichment file.

        Returns:
            EnrichmentResult with restored DataFrames.
        """
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        kwargs = {}
        for key, val in data.items():
            if key == "signatures":
                kwargs["signatures"] = {
                    name: pd.DataFrame(**split_dict)
                    for name, split_dict in val.items()
                }
            elif key == "signature_scores":
                kwargs["signature_scores"] = val
            else:
                kwargs[key] = pd.DataFrame(**val)
        return cls(**kwargs)

    def to_json(self, path: str) -> None:
        """Export enrichment results as JSON (for web consumption).

        Args:
            path: File path (e.g., "enrichment.json").
        """
        import json
        data = {}
        for key in self._FIELDS:
            df = getattr(self, key)
            if df is None:
                continue
            df_reset = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df
            data[key] = json.loads(df_reset.to_json(orient="records", double_precision=6))
        if self.signatures is not None:
            data["signatures"] = {
                name: json.loads(df.to_json(orient="records", double_precision=6))
                for name, df in self.signatures.items()
            }
        if self.signature_scores is not None:
            data["signature_scores"] = self.signature_scores
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Enrichment results exported to %s", path)


def enrich_factors(
    model,
    gene_sets=None,
    view_name: Optional[str] = None,
    covariate_names: Optional[List[str]] = None,
    covariate_types: Optional[Dict[str, str]] = None,
    compute_signatures: bool = False,
    signature_kwargs: Optional[dict] = None,
) -> EnrichmentResult:
    """Compute post-hoc enrichment analysis for learned latent factors.

    Args:
        model: A trained TemporalPACMON instance.
        gene_sets: Optional FeatureSets object for PCGSE enrichment.
        view_name: Which view to use for enrichment. If None, uses first view.
        covariate_names: Optional list of covariate names for labeling.
        covariate_types: Optional dict mapping covariate name to
            "categorical" or "continuous". Defaults to "continuous" for all.
        compute_signatures: If True, run ElasticNet predictive signatures
            (slow — fits one CV model per factor). Default False.
        signature_kwargs: Optional kwargs passed to compute_predictive_signatures
            (e.g., aggregation, n_alphas, cv, max_features).

    Returns:
        EnrichmentResult with populated DataFrames.
    """
    if not model._trained:
        raise RuntimeError("Model must be trained first.")

    if view_name is None:
        view_name = model.view_names[0]

    result = EnrichmentResult()

    # Block A: Prior fidelity (sparse factors only)
    if model.n_sparse_factors > 0 and model.prior_masks is not None:
        logger.info("Computing prior fidelity for %d sparse factors", model.n_sparse_factors)
        result.fidelity = _compute_fidelity(model)

    # Block C: Temporal metrics (always)
    logger.info("Computing temporal metrics for %d factors", model.n_factors)
    result.temporal = _compute_temporal(model)

    # Block B: PCGSE enrichment (if gene sets provided)
    if gene_sets is not None:
        logger.info("Computing PCGSE enrichment against %d gene sets", len(gene_sets))
        result.enrichment = _compute_pcgse(model, gene_sets, view_name)

    # Covariate association (if model has covariates)
    if model.n_covariates > 0 and model.covariates is not None:
        logger.info("Computing covariate associations for %d covariates", model.n_covariates)
        result.covariate, result.covariate_summary = _compute_covariate(
            model, covariate_names, covariate_types
        )

    # Predictive signatures (optional, slow)
    if compute_signatures:
        if model.observations is None:
            logger.warning("Signatures require observation data; skipping (loaded model).")
        else:
            from tpacmon.tools.signatures import compute_predictive_signatures
            logger.info("Computing predictive signatures (ElasticNet CV)...")
            sig_kwargs = {"view_name": view_name}
            if signature_kwargs is not None:
                sig_kwargs.update(signature_kwargs)
            sig_result = compute_predictive_signatures(model, **sig_kwargs)
            result.signatures = sig_result["signatures"]
            result.signature_scores = sig_result["scores"]

    return result


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _gini_coefficient(values: np.ndarray) -> float:
    """Compute Gini coefficient of a 1D array of non-negative values.

    Returns:
        Gini in [0, 1]. 0 = perfect equality, 1 = perfect inequality.
    """
    values = np.sort(np.abs(values))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2.0 * np.sum(index * values) / (n * np.sum(values))) - (n + 1) / n


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction.

    Args:
        pvalues: 1D array of p-values.

    Returns:
        Array of adjusted p-values, same length as input.
    """
    n = len(pvalues)
    if n == 0:
        return pvalues.copy()
    order = np.argsort(pvalues)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    adjusted = np.minimum(pvalues * n / ranks, 1.0)
    # Enforce monotonicity: walk backwards through sorted order
    adjusted_sorted = adjusted[order].copy()
    for i in range(n - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
    adjusted[order] = adjusted_sorted
    return adjusted


# ---------------------------------------------------------------------------
# Block A: Prior Fidelity
# ---------------------------------------------------------------------------

def _compute_fidelity(model) -> pd.DataFrame:
    """Compute prior fidelity metrics for sparse factors.

    For each sparse factor, measures how well the learned loadings match
    the prior gene set using AUROC, refinement counts, and Gini coefficient.

    Args:
        model: Trained TemporalPACMON instance with prior_masks.

    Returns:
        DataFrame indexed by sparse factor names.
    """
    # Concatenate loadings and masks across views (matches prior_scales layout)
    w_all = model._guide.get_w()  # (K, D_total)
    mask_all = np.concatenate(
        [model.prior_masks[vn] for vn in model.view_names], axis=1
    )  # (K_sparse, D_total)

    rows = []
    for k in range(model.n_sparse_factors):
        abs_w = np.abs(w_all[k, :])
        labels = (mask_all[k, :] > 0).astype(int)

        # AUROC fidelity
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            fidelity = np.nan
        else:
            fidelity = roc_auc_score(labels, abs_w)

        # Adaptive threshold: median |w| among prior-active genes
        active_w = abs_w[labels == 1]
        threshold = np.median(active_w) if len(active_w) > 0 else 0.0

        # Refinement counts
        retained = int(np.sum((labels == 1) & (abs_w >= threshold)))
        pruned = int(np.sum((labels == 1) & (abs_w < threshold)))
        recruited = int(np.sum((labels == 0) & (abs_w >= threshold)))

        n_prior = int(n_pos)
        n_non_prior = int(n_neg)

        rows.append({
            "fidelity": fidelity,
            "deviation": 1.0 - fidelity if not np.isnan(fidelity) else np.nan,
            "gini": _gini_coefficient(abs_w),
            "retained": retained,
            "pruned": pruned,
            "recruited": recruited,
            "retained_frac": retained / n_prior if n_prior > 0 else 0.0,
            "pruned_frac": pruned / n_prior if n_prior > 0 else 0.0,
            "recruited_frac": recruited / n_non_prior if n_non_prior > 0 else 0.0,
        })

    return pd.DataFrame(rows, index=model.factor_names[:model.n_sparse_factors])


# ---------------------------------------------------------------------------
# Block B: PCGSE Enrichment
# ---------------------------------------------------------------------------

def _compute_pcgse(model, gene_sets, view_name: str) -> pd.DataFrame:
    """Competitive gene set enrichment via Wilcoxon rank-sum.

    For each factor and gene set, tests whether loadings of genes inside
    the set are significantly higher than those outside.

    Args:
        model: Trained TemporalPACMON instance.
        gene_sets: FeatureSets object.
        view_name: Which view to test against.

    Returns:
        DataFrame with columns: factor, gene_set, pvalue, padj, auroc, prior_rank.
    """
    loadings = model.get_loadings()[view_name]  # (K, D_m)
    feature_names = model.feature_names[view_name]
    mask_df = gene_sets.to_mask(feature_names)  # (n_sets, D_m) DataFrame
    set_names = mask_df.index.tolist()

    rows = []
    for k in range(model.n_factors):
        abs_w = np.abs(loadings[k, :])
        pvalues_k = []

        for s_name in set_names:
            in_set = mask_df.loc[s_name].values.astype(bool)
            scores_in = abs_w[in_set]
            scores_out = abs_w[~in_set]

            if len(scores_in) < 1 or len(scores_out) < 1:
                pvalues_k.append(1.0)
                rows.append({
                    "factor": model.factor_names[k],
                    "gene_set": s_name,
                    "pvalue": 1.0,
                    "auroc": 0.5,
                })
                continue

            try:
                stat, pvalue = mannwhitneyu(
                    scores_in, scores_out, alternative="greater"
                )
                auroc = stat / (len(scores_in) * len(scores_out))
            except ValueError:
                pvalue = 1.0
                auroc = 0.5

            pvalues_k.append(pvalue)
            rows.append({
                "factor": model.factor_names[k],
                "gene_set": s_name,
                "pvalue": pvalue,
                "auroc": auroc,
            })

        # BH correction within this factor
        adjusted = _benjamini_hochberg(np.array(pvalues_k))
        start_idx = len(rows) - len(set_names)
        for i, padj in enumerate(adjusted):
            rows[start_idx + i]["padj"] = padj

    df = pd.DataFrame(rows)

    # Prior rank: for sparse factors, where does the prior gene set rank?
    df["prior_rank"] = np.nan
    for k in range(model.n_sparse_factors):
        factor_name = model.factor_names[k]
        factor_rows = df[df["factor"] == factor_name].copy()
        if factor_name in set_names:
            ranked = factor_rows.sort_values("pvalue")
            rank_position = (
                ranked["gene_set"].tolist().index(factor_name) + 1
            )
            df.loc[
                (df["factor"] == factor_name) & (df["gene_set"] == factor_name),
                "prior_rank",
            ] = rank_position

    return df


# ---------------------------------------------------------------------------
# Block C: Temporal Metrics
# ---------------------------------------------------------------------------

def _compute_temporal(model) -> pd.DataFrame:
    """Compute temporal metrics for all factors.

    Args:
        model: Trained TemporalPACMON instance.

    Returns:
        DataFrame indexed by factor names with columns:
        zeta, temporal_strength, TVR, lengthscale.
    """
    zeta = model.get_smoothness()
    lengthscale = model.get_lengthscales()
    z = model.get_factors()  # (P, T, K)

    tvr = np.zeros(model.n_factors)
    for k in range(model.n_factors):
        z_k = z[:, :, k]  # (P, T)
        # Within-patient temporal variance
        patient_vars = []
        for p in range(model.n_patients):
            obs_mask = model.patient_masks_np[p]
            z_pt = z_k[p, obs_mask]
            if len(z_pt) > 1:
                patient_vars.append(np.var(z_pt))
        within_var = np.mean(patient_vars) if patient_vars else 0.0
        # Total variance over all non-NaN entries
        all_vals = z_k[~np.isnan(z_k)]
        total_var = np.var(all_vals) if len(all_vals) > 1 else 0.0
        tvr[k] = within_var / total_var if total_var > 0 else 0.0

    return pd.DataFrame(
        {
            "zeta": zeta,
            "temporal_strength": 1.0 - zeta,
            "TVR": tvr,
            "lengthscale": lengthscale,
        },
        index=model.factor_names,
    )


# ---------------------------------------------------------------------------
# Covariate Association
# ---------------------------------------------------------------------------

def _compute_covariate(
    model,
    covariate_names: Optional[List[str]] = None,
    covariate_types: Optional[Dict[str, str]] = None,
) -> tuple:
    """Compute covariate association metrics.

    Args:
        model: Trained TemporalPACMON instance with covariates.
        covariate_names: Names for covariates. Default: cov_0, cov_1, ...
        covariate_types: Dict of covariate name -> "categorical"|"continuous".
            Defaults to "continuous" for all.

    Returns:
        Tuple of (covariate_df, covariate_summary_df).
    """
    if covariate_names is None:
        covariate_names = [f"cov_{c}" for c in range(model.n_covariates)]
    if covariate_types is None:
        covariate_types = {name: "continuous" for name in covariate_names}

    gamma = model._guide.mode("gamma")  # (C, K)
    z = model.get_factors()  # (P, T, K)
    covs = model.covariates  # (P, C)

    rows = []
    for c_idx, cov_name in enumerate(covariate_names):
        cov_type = covariate_types.get(cov_name, "continuous")
        x_c = covs[:, c_idx]

        for k in range(model.n_factors):
            td = _compute_td(
                z[:, :, k], x_c, model.patient_masks_np, cov_type
            )
            rows.append({
                "factor": model.factor_names[k],
                "covariate": cov_name,
                "gamma": gamma[c_idx, k],
                "TD": td,
            })

    cov_df = pd.DataFrame(rows)

    # TIR: gamma-weighted average of TD per covariate
    tir_rows = []
    for c_idx, cov_name in enumerate(covariate_names):
        mask = cov_df["covariate"] == cov_name
        td_vals = cov_df.loc[mask, "TD"].values
        gamma_vals = np.abs(cov_df.loc[mask, "gamma"].values)
        denom = gamma_vals.sum()
        tir = np.sum(td_vals * gamma_vals) / denom if denom > 0 else 0.0
        tir_rows.append({"TIR": tir})

    summary_df = pd.DataFrame(tir_rows, index=covariate_names)

    return cov_df, summary_df


def _compute_td(
    z_k: np.ndarray,
    x_c: np.ndarray,
    patient_masks: np.ndarray,
    cov_type: str,
) -> float:
    """Compute temporal divergence for one factor-covariate pair.

    Args:
        z_k: (P, T) factor scores (NaN for unobserved).
        x_c: (P,) covariate values.
        patient_masks: (P, T) boolean mask.
        cov_type: "categorical" or "continuous".

    Returns:
        TD value. 0 if insufficient data.
    """
    n_timepoints = z_k.shape[1]

    if cov_type == "categorical":
        # Binary split: group0 (x <= median), group1 (x > median)
        median_val = np.median(x_c)
        group0 = x_c <= median_val
        group1 = x_c > median_val
        # Handle case where median split puts all in one group
        if not np.any(group0) or not np.any(group1):
            return 0.0

        diffs = []
        for t in range(n_timepoints):
            obs_t = patient_masks[:, t]
            g0_mask = obs_t & group0
            g1_mask = obs_t & group1
            if np.sum(g0_mask) < 1 or np.sum(g1_mask) < 1:
                continue
            mean0 = np.nanmean(z_k[g0_mask, t])
            mean1 = np.nanmean(z_k[g1_mask, t])
            diffs.append(mean1 - mean0)

        return np.var(diffs) if len(diffs) > 1 else 0.0

    else:  # continuous
        cors = []
        for t in range(n_timepoints):
            obs_t = patient_masks[:, t]
            z_t = z_k[obs_t, t]
            x_t = x_c[obs_t]
            # Filter NaN
            valid = ~np.isnan(z_t)
            z_t = z_t[valid]
            x_t = x_t[valid]
            if len(z_t) < 3:
                continue
            # Pearson correlation
            std_z = np.std(z_t)
            std_x = np.std(x_t)
            if std_z == 0 or std_x == 0:
                cors.append(0.0)
            else:
                cor = np.corrcoef(z_t, x_t)[0, 1]
                cors.append(cor if not np.isnan(cor) else 0.0)

        return np.var(cors) if len(cors) > 1 else 0.0
