# Plan: TemporalPACMon — Temporal Bayesian Latent Factor Model

## Context

This builds a new Python package called **`tpacmon`** (Temporal PACMON) in the `Temporal_muvi_flex` repo. It extends the PACMon model (gene-set informed priors + covariate regression) by replacing the i.i.d. Normal prior on latent factors with a **Gaussian Process prior using a Matérn 3/2 kernel**, enabling flexible modeling of dependent repeated measurements without assuming equal temporal spacing.

**Why this is needed:** MEFISTO (MOFA2) handles temporal data but (a) uses only the RBF kernel (`exp(-d²/2l²)`, confirmed in `mefisto.R:926`), which can over-smooth biological dynamics, (b) optimizes lengthscales via grid search (`n_grid=20`) rather than learning them jointly, (c) requires a Kronecker structure `K = s_k * K_group ⊗ K_covariate` that forces all groups to share the same covariate grid (equal temporal spacing), and (d) provides no gene-set priors or covariate regression. MuVI and PACMon ignore temporal dependency entirely. **mofaflex** (`bioFAM/mofaflex`) supports both GP priors on factors and `InformedHorseshoe` on weights, but these operate independently — there is no covariate-informed GP mean, no factor-level covariate regression (`gamma`), and no explicit per-patient irregular masking. No existing tool combines (1) gene-set priors, (2) dual-level covariate regression (feature-level `beta` + factor-level `gamma` shifting the GP trajectory), and (3) flexible irregular-time GP structure with per-patient masking.

---

## Mathematical Model

### Notation
- P: patients, T: max timepoints (common grid), K: factors, D_m: features in view m
- `mask_p ∈ {0,1}^T`: which visits patient p attended
- `t ∈ R^T`: actual time values (common grid, e.g., [0, 1, 7, 30] days)
- `t_{p,obs}`: observed time values for patient p = `t[mask_p]`

### Factor Loadings (from PACMon — unchanged)
```
For each factor k, feature j, view m:
  local_scale_{k,j,m}  ~ HalfCauchy(1)
  factor_scale_{k,m}   ~ HalfCauchy(1)
  view_scale_m         ~ HalfCauchy(1)
  caux_{k,j,m}        ~ InverseGamma(0.5, 0.5)          [reg. horseshoe]
  c = sqrt(caux) * prior_scale[k,j,m]                   [gene-set prior modulation]
  w_scale = (global_scale * c * local * factor * view) / sqrt(c² + (local*factor*view)²)
  w_{k,j,m}            ~ Normal(0, w_scale)
```
Dense factors use `w_{k,j,m} ~ Normal(0, 1)`.

### Temporal Latent Factors — Covariate-informed GP prior (NON-LINEAR EXTENSION)
```
New learnable parameters:
  gamma_k ~ Normal(0, 1)    shape: (n_covariates, n_factors)
  zeta_k  ~ Beta(1, 1)      shape: (n_factors,)              # smoothness mixing weight

For each patient p, factor k:
  mu_k(x_p)  = x_p @ gamma[:, k]            # covariate-shifted mean (n_timepoints_obs,)
  t_{p,obs}  = time_points[mask_p]           # actual observed times
  K_k        = Matérn32(t_{p,obs}, t_{p,obs}; l_k, a_k)

  # Decomposed into GP (structured) + i.i.d. (unstructured) components:
  f_{p,k}    ~ MVN(mu_k(x_p) * ones, K_k)   # GP component (temporal structure)
  eta_{p,k}  ~ Normal(0, 1)                  # i.i.d. noise component (T_p_obs,)
  z_{p,k}    = sqrt(1 - zeta_k) * f_{p,k} + sqrt(zeta_k) * eta_{p,k}
```

**The `zeta_k` smoothness parameter** (inspired by mofaflex's approach):
- `zeta_k → 0`: factor k is **fully temporal** — trajectories are smooth GP curves
- `zeta_k → 1`: factor k is **i.i.d.** — behaves like standard PACMon (no temporal structure)
- Learned per factor via SVI, so the model **automatically discovers which factors benefit from temporal structure**
- This is critical because not all biological programs are temporal — some may be patient-specific constants

**Interpretation:**
- `gamma_{c,k}` = how much covariate `c` displaces patient p's expected trajectory for factor k
- GP residual = individual deviation from the group-level trajectory
- `zeta_k` = how much of factor k's variation is temporally structured vs. i.i.d.
- This is a **linear mixed effects** model at the factor level: fixed effects (covariates) + random effects (GP + noise)
- Still fully linear in `y`; non-linearity is only in the prior structure on `z`
- `gamma` is identifiable because covariates are observed (unlike `z`)

**Changes to `beta_m`**: The `beta_m` regression module still exists (for the observation-level covariate effect on features), but now `gamma` handles the covariate effect at the *latent factor level*. Both are simultaneously identifiable with appropriate priors.

**Relationship to PACMon's `conditions`**: PACMon already has a `conditions` mechanism (`models.py:3007-3013`) where `z_loc = conditions @ w_condition + b_condition`. Our `gamma` **subsumes and replaces** this: instead of a flat shift to the factor mean, `gamma` shifts the GP mean, producing a covariate-informed *trajectory* baseline. The `conditions` parameter is therefore not carried forward; users should use `covariates` for both the factor-level (`gamma`) and feature-level (`beta`) effects.

### Patient-level Covariate Regression (from PACMon — adapted)
```
beta_m ~ Normal(0, 1)    shape: (n_covariates, n_features_m)
# Broadcast to all timepoints:
covariate_effect_{p,t,m} = x_p @ beta_m     (same for all t of patient p)
```

### Likelihood
```
For each observed visit (patient p, time t where mask_p[t]=1), view m:
  y_{p,t,m} ~ Normal(z_{p,t} @ w_m + x_p @ beta_m, sigma_m)
Missing visits (mask_p[t]=0) are masked out via pyro.poutine.mask()
```

### Variational Posterior (NEW — structured VI for z)
```
For each patient p, factor k:
  q(z_{p,:,k}) = MVN(mu_{p,k}, L_{p,k} @ L_{p,k}^T)    [Cholesky parameterization]
  # Only over OBSERVED time points of patient p
  mu_{p,k}: learned mean vector (T_p_obs,)
  L_{p,k}:  learned lower-triangular Cholesky (T_p_obs x T_p_obs)
```
All other sites (w, beta, sigma, lengthscale, amplitude) use diagonal Normal guide (as in PACMon).

**Scalability note — Low-rank guide fallback**: The full Cholesky MVN stores `L_{p,k}` of size `(T_p_obs × T_p_obs)` per patient per factor. For moderate T (≤30) this is fine, but for large T (50+), offer a `LowRankMultivariateNormal` alternative (rank-r + diagonal), reducing parameters from O(T²) to O(rT). Implement as a `guide_type` option: `"cholesky"` (default) or `"lowrank"`.

**Batching note**: Since guide parameters for `z` are per-patient, mini-batching over patients requires indexing into a stored parameter tensor (not amortization). Store `mu_z` and `L_z` as `PyroParam` tensors indexed by patient ID within each mini-batch. This is consistent with PACMon's existing batching pattern.

---

## Comparison with MEFISTO (verified from codebase)

| Aspect | MEFISTO | mofaflex (`bioFAM/mofaflex`) | tpacmon (ours) |
|--------|---------|------|----------------|
| Kernel | RBF only | RBF + Matérn (via GPyTorch) | Matérn 3/2 (default), 5/2, RBF (pure torch) |
| Lengthscale optimization | Grid search (`n_grid=20`) | Learned via SVI (GPyTorch constraints) | Learned via SVI (LogNormal prior) |
| Temporal structure | Kronecker `K_group ⊗ K_covariate` — forces shared grid | Per-group covariates, sparse variational (inducing points) | Per-patient boolean mask on common grid |
| Gene-set priors | None | `InformedHorseshoe` on weights (independent from GP) | Regularized horseshoe on weights (from PACMon) |
| Factor-level covariate effect | None | None (GP mean is always zero) | `gamma` shifts GP mean per covariate — trajectory baseline |
| Feature-level covariate regression | None | "Guiding variables" (auxiliary loss, regresses *on* factors) | `beta_m` (generative, regresses covariates *on* features) |
| Smoothness discovery | Scale param `s_k` (0=independent, 1=smooth) | `zeta_k` (smoothness mixing) | `zeta_k ~ Beta(1,1)` per factor (GP vs. i.i.d. mixing) |
| Sparse GP | Optional (`frac_inducing=0.75`) | Default 100 inducing points | Low-rank guide fallback for large T |
| Interpolation | GP conditional (mean only, R-side) | Not documented | `predict()` returns mean + variance |
| Framework | R wrapper → mofapy2 Python | Pyro + GPyTorch | Pyro (no GPyTorch dependency) |

---

## Alternative Temporal Priors (not implemented but worth noting)

| Prior | Formula | Tradeoff |
|-------|---------|----------|
| **Matérn 3/2** ✓ | `(1+√3d/l)exp(-√3d/l)` | Good for biological data, C¹ smooth |
| Matérn 5/2 | `(1+√5d/l+5d²/3l²)exp(-√5d/l)` | C² smooth, slightly more regular |
| RBF | `exp(-d²/2l²)` | Infinitely smooth, may over-smooth |
| OU (Matérn 1/2) | `exp(-d/l)` | C⁰ only, good for noisy/rapid changes |
| Sparse GP (SVGP) | Inducing points M<<T | Scales to many patients/timepoints |
| Linear State Space | Kalman-like recursion | Only equal spacing; fast O(T) |

---

## Numerical Stability & Training Robustness

GP kernels combined with horseshoe priors create a challenging optimization landscape. The following defaults should be applied:

1. **Jitter**: 1e-5 on kernel diagonal (already specified). Increase adaptively to 1e-4 if Cholesky decomposition fails during training.
2. **Double precision**: Offer a `double_precision: bool = False` option on `TemporalPACMON`. When enabled, kernel computations run in float64 to avoid numerical issues with near-singular covariance matrices.
3. **Gradient clipping**: Apply `torch.nn.utils.clip_grad_norm_` with `max_norm=10.0` in the SVI loop. GP log-det gradients and horseshoe scale gradients can spike.
4. **Learning rate schedule**: Use `ClipLR` or `ReduceLROnPlateau` scheduler (configurable). Default: cosine annealing over the specified number of epochs.
5. **ELBO scaling**: Use `pyro.poutine.scale` to balance the GP prior log-likelihood against the observation likelihood when the number of observations per patient varies widely. Default scale factor = `1.0` (no scaling), but expose as `gp_scale: float` parameter.

---

## Module 2: Gene Set Preprocessing Pipeline (`tpacmon/tools/gene_set_prep.py`)

### Problem
Large ontologies (GO, Hallmarks, Reactome) contain hundreds of redundant, overlapping gene sets. Including them all inflates model complexity, reduces effective degrees of freedom, and degrades interpretability — most gene sets explain no variance.

### Component A — Size Filtering (via FeatureSets.filter())

Use the existing `FeatureSets.filter()` method (copied from `pacmon_flex/pacmon/tools/feature_sets.py`) to remove gene sets that are too small or too large to be informative:

```python
gene_sets.filter(
    features=all_feature_names,
    min_count=5,          # drop gene sets with fewer than 5 overlapping features
    max_count=500,        # drop gene sets larger than 500 (too broad to be specific)
    min_fraction=0.5,     # at least 50% of gene set members must be in the data
)
```

This is instant and removes the obvious junk (tiny sets, genome-wide sets). No learning step needed.

**The horseshoe prior handles the rest.** The regularized horseshoe with prior_scales modulation (`clip(mask + (1-confidence), 1e-8, 1.0)`) already shrinks uninformative gene set loadings to zero. Trying to pre-learn which gene sets matter is redundant — the model does this during training, and it does it better because it sees the full multi-view structure.

### Component B — Hierarchical Merging (reduce redundancy)

1. Compute pairwise Jaccard similarity: `J(A, B) = |A ∩ B| / |A ∪ B|`
2. Build distance matrix: `D = 1 - J`
3. Run Ward linkage hierarchical clustering (`scipy.cluster.hierarchy.linkage`)
4. Cut dendrogram at threshold (default `J ≥ 0.5` → `D ≤ 0.5`)
5. For each cluster of similar gene sets → merge into one consolidated "super-set":
   - **Union** of all member genes (preserves all biology)
   - Name = name of largest constituent set + `"_merged"` suffix
   - Store provenance: which original sets were merged

```python
def merge_gene_sets(
    gene_sets: dict[str, list[str]],
    similarity_threshold: float = 0.5,   # Jaccard threshold
    linkage_method: str = "ward",
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    # Returns: (merged_sets, provenance_map)
```

### Component C — Combined pipeline
```python
def prepare_gene_sets(
    gene_sets: FeatureSets,
    feature_names: list[str],
    min_count: int = 5,
    max_count: int = 500,
    similarity_threshold: float = 0.5,
) -> tuple[FeatureSets, dict[str, list[str]]]:
    filtered = gene_sets.filter(features=feature_names, min_count=min_count, max_count=max_count)
    merged, provenance = merge_gene_sets(filtered, similarity_threshold=similarity_threshold)
    return merged, provenance
```

---

## Module 3: Post-hoc Predictive Signature Module (`tpacmon/tools/signatures.py`)

### Goal
After model fitting, for each latent factor k, derive a **minimal feature signature** — the smallest set of molecular features that predicts factor k's scores across patients with high accuracy. This translates the learned factors into clinically actionable biomarker panels.

### Method: Cross-validated ElasticNet per factor
```
For factor k:
  z_k = factor scores, shape (P,) or (P×T,) averaged over time
  X_m = raw feature matrix for each view m, shape (P, D_m)
  X = concat([X_0, X_1, ..., X_{M-1}], axis=1)   # (P, sum(D_m))

  Fit: z_k ~ ElasticNet(X, l1_ratio=0.5)  via cross-validation
  Signature_k = {features j where coeff_j ≠ 0}
  Performance = CV R² (reported per factor)
```

```python
def compute_predictive_signatures(
    model: TemporalPACMON,
    data: dict[str, np.ndarray],          # raw features per view
    feature_names: dict[str, list[str]],  # names per view
    factor_selection: Optional[list[int]] = None,  # None = all factors
    l1_ratio: float = 0.5,               # 0=Ridge, 1=LASSO, 0.5=ElasticNet
    cv: int = 5,
    alpha_range: Optional[list[float]] = None,   # auto-tuned if None
) -> pd.DataFrame:
    # Returns DataFrame: factor x feature with non-zero weights
```

**Time aggregation options for z_k**:
- `"mean"`: Average factor scores across observed timepoints per patient
- `"last"`: Use last observed timepoint (e.g., endpoint outcome)
- `"max"`: Use peak factor activation
- `"slope"`: Fit a simple linear trend per patient per factor, use the slope as summary. Captures *trajectory direction*, which is often the clinically relevant signal (e.g., improving vs. declining patients).

**Output**:
- `signatures[k]`: dict with `features` (list), `weights` (dict), `cv_r2` (float)
- Summary DataFrame: factors × features, cell = regression coefficient (0 if not selected)

---

## File Structure

```
Temporal_muvi_flex/
├── tpacmon/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── models.py          # TemporalPACMON, TemporalModel, TemporalGuide
│   │   ├── kernels.py         # Matérn 3/2, RBF, Matérn 5/2 implementations
│   │   ├── callbacks.py       # Copied from pacmon_flex
│   │   ├── early_stopping.py  # Copied from pacmon_flex
│   │   └── synthetic.py       # Longitudinal data simulator
│   └── tools/
│       ├── __init__.py
│       ├── feature_sets.py    # Copied from pacmon_flex
│       ├── gene_set_prep.py   # NEW: screening + hierarchical merging
│       ├── signatures.py      # NEW: ElasticNet predictive signatures
│       ├── utils.py           # Post-hoc: variance explained, factor-time plots
│       └── plotting.py        # Temporal factor trajectory plots
├── pyproject.toml
├── tests/
│   ├── test_kernels.py
│   ├── test_model.py
│   ├── test_gene_set_prep.py
│   ├── test_signatures.py
│   ├── test_synthetic.py
│   ├── test_integration.py
│   └── test_predict.py
└── notebooks/
    └── example_longitudinal.ipynb
```

---

## Critical Files to Borrow From

| Source | File | What to reuse | Verified |
|--------|------|---------------|----------|
| `pacmon_flex` | `pacmon/core/models.py:54-118` | `PACMON.__init__` — constructor signature, parameter validation | Yes |
| `pacmon_flex` | `pacmon/core/models.py:529-758` | `_setup_prior_masks()` — prior scales computation: `clip(mask + (1-confidence), 1e-8, 1.0)` | Yes |
| `pacmon_flex` | `pacmon/core/models.py:2661-3048` | `PACMONModel` — horseshoe prior (2867-2908), beta regression (2940-2949), view scaling (2751-2756). Replace z sampling at 3007-3013 only | Yes |
| `pacmon_flex` | `pacmon/core/models.py:3051-3382` | `PACMONGuide` — site_to_shape setup (3100-3168), `_sample()` method, extraction methods. Replace z site with structured MVN | Yes |
| `pacmon_flex` | `pacmon/core/models.py:1616-1692` | Optimizer setup (ClippedAdam with LR decay) and SVI setup (TraceMeanField_ELBO) | Yes |
| `pacmon_flex` | `pacmon/core/models.py:1775-1937` | `fit()` — training loop with callbacks, early stopping, checkpoints | Yes |
| `pacmon_flex` | `pacmon/core/models.py:1460-1563` | `_decompose()` — SVD initialization (adapt for temporal: average over time first) | Yes |
| `pacmon_flex` | `pacmon/core/models.py:1938-2310` | `save()`/`load()` — serialize guide params as numpy. Extend for ragged Cholesky tensors | Yes |
| `pacmon_flex` | `pacmon/core/callbacks.py` | Copy directly (CheckpointCallback, LogCallback with sparsity/RMSE/R² tracking) | Yes |
| `pacmon_flex` | `pacmon/core/early_stopping.py` | Copy directly (min_epochs=100, tolerance=1e-5, patience=10) | Yes |
| `pacmon_flex` | `pacmon/tools/feature_sets.py` | Copy directly (FeatureSet/FeatureSets classes, GMT I/O, `to_mask()`) | Yes |
| `MuVI_flex` | `muvi/core/models.py:483-509` | `_setup_prior_confidence()` — maps "low"→0.99, "med"→0.995, "high"→0.999 | Yes |
| `MuVI_flex` | `muvi/core/models.py:511-713` | `_setup_prior_masks()` — mask alignment, feature matching, prior_scales computation | Yes |
| `MOFA2_flex` | `R/mefisto.R:206-242` | Conceptual: MEFISTO options (grid search for lengthscale, sparse GP, warping). We improve: learn lengthscales via SVI instead of grid search | Yes |
| `MOFA2_flex` | `R/mefisto.R:922-930` | Conceptual: GP conditional prediction `K*K⁻¹z`. MEFISTO uses RBF only (`exp(-d²/2l²)`); we use Matérn 3/2 | Yes |

**Note on PACMon's `conditions`**: PACMon has both `conditions` (affect z_loc at factor level, `models.py:3007-3013`) and `covariates` (affect y via beta at feature level, `models.py:2940-2949`). Our `gamma` parameter replaces `conditions` — it shifts the GP mean instead of z_loc. The `conditions` pathway is **not carried forward** into tpacmon.

**Note on save/load**: PACMon serializes guide parameters via `get_state()` (`models.py:1938-2022`) into `metadata.json`, `params.npz`, `structure.npz`. For tpacmon, we extend this to store: (a) per-patient Cholesky tensors (padded to max T with mask), (b) kernel hyperparameters (lengthscales, amplitudes), (c) temporal metadata (time_points, patient_masks).

---

## Implementation Steps (Ordered)

### Step 1 — Package scaffold
- Create `tpacmon/` directory structure
- Write `pyproject.toml` (based on pacmon's, Python ≥3.10, same deps + no extra)
- Write `tpacmon/__init__.py` and `tpacmon/core/__init__.py`

### Step 2 — `kernels.py`
Implement three kernel functions as pure torch functions:
```python
def matern32(t1, t2, lengthscale, amplitude):
    d = torch.cdist(t1[:,None], t2[:,None])   # pairwise distances
    sqrt3d = math.sqrt(3) * d / lengthscale
    return amplitude**2 * (1 + sqrt3d) * torch.exp(-sqrt3d)

def matern52(t1, t2, lengthscale, amplitude): ...
def rbf(t1, t2, lengthscale, amplitude): ...

def build_kernel(kernel: str, t_obs, lengthscale, amplitude, jitter=1e-5):
    """Build covariance matrix + jitter for numerical stability."""
```

### Step 3 — `TemporalModel` (generative model)
Subclass or heavily adapt `PACMONModel`:
- Add constructor args: `time_points: Tensor (T,)`, `patient_masks: Tensor (P,T)`, `kernel: str`
- Keep ALL loading-related plates and horseshoe prior unchanged
- Replace the `z` sampling site:
```python
# OLD (PACMon):
z ~ Normal(z_loc, 1)  shape: (n_samples, n_factors)

# NEW (TemporalModel):
with factor_plate:
    lengthscale_k ~ LogNormal(0, 1)
    amplitude_k   ~ LogNormal(0, 1)

with patient_plate:    # plate over P patients
    for each patient p:
        t_obs = time_points[patient_masks[p]]          # (T_p_obs,)
        K = build_kernel(kernel, t_obs, l_k, a_k)      # (T_p_obs, T_p_obs)
        z_{p,k} ~ MVN(zeros, K)                        # (T_p_obs, K)
```
- Observation loop: iterate over (patient, time) pairs; mask missing visits

### Step 4 — `TemporalGuide` (variational posterior)
Adapt `PACMONGuide`:
- Keep all sites EXCEPT `z`
- For `z`, parameterize per-patient per-factor Cholesky MVN:
```python
# Learned parameters (stored as PyroParam):
mu_z:  (P, T_p_obs, K)   # variational means
L_z:   (P, T_p_obs, T_p_obs, K)  # lower-triangular Cholesky per patient per factor

# Approximate posterior:
z_{p,k} ~ MVN(mu_z[p,:,k], L_z[p,:,:,k] @ L_z[p,:,:,k].T)
```
- For ragged structure: store as list of tensors or pad to max T with masking

### Step 5 — `TemporalPACMON` (main user class)
New main class with signature:
```python
class TemporalPACMON(PyroModule):
    def __init__(
        self,
        observations: dict[str, np.ndarray],    # view -> (P, T, D_m) or masked
        time_points: np.ndarray,                  # (T,) actual time values
        patient_masks: np.ndarray,               # (P, T) bool
        prior_masks: Optional[dict] = None,       # gene set priors
        covariates: Optional[np.ndarray] = None, # (P, C) patient-level
        prior_confidence: str = "low",
        n_sparse_factors: Optional[int] = None,
        n_dense_factors: Optional[int] = None,
        kernel: str = "matern32",               # 'matern32'|'matern52'|'rbf'
        likelihoods: Optional[dict] = None,
        normalize: bool = True,
        shared_lengthscale: bool = False,  # if True, one lengthscale shared across all factors
        guide_type: str = "cholesky",     # 'cholesky' or 'lowrank'
        gp_scale: float = 1.0,            # scale factor for GP prior log-likelihood
        double_precision: bool = False,    # float64 for kernel computations
        device: str = "auto",             # 'auto' detects CUDA, falls back to CPU gracefully
    )
```
- `_setup_observations()`: flatten `(P, T, D_m)` to `(P*T, D_m)` with mask tracking
- `_setup_temporal()`: validate time_points and patient_masks
- `_setup_model_guide()`: instantiate TemporalModel + TemporalGuide
- `_decompose()`: **Temporal SVD initialization** — average observations over time per patient to get `(P, D_total)`, run SVD as in PACMon (`models.py:1460-1563`) to get `Z_init (P, K)` and `W_init (K, D_total)`. Then tile `Z_init` across observed timepoints: `z_init[p, t, k] = Z_init[p, k]` for all observed `t`. This gives a reasonable starting point for the GP posterior means.
- `fit()`: standard SVI loop (copy from PACMon's `fit()` at `models.py:1775-1937`, using ClippedAdam with exponential LR decay). Add gradient clipping on top.
- `get_factors()`: return `(P, T, K)` factor scores
- `get_loadings()`: return `(K, D_m)` per view
- `get_lengthscales()`: return `(K,)` learned temporal lengthscales
- `predict(new_time_points)`: GP posterior prediction at unobserved times — returns `(P, T_new, K)` mean and variance. Uses standard GP conditional: `mu* = K*K⁻¹z`, `Sigma* = K** - K*K⁻¹K*ᵀ`. This is a key advantage over non-temporal models.
- `compute_elbo()`: evaluate ELBO on held-out data for model comparison (tpacmon vs. PACMon baseline)
- `score_holdout(held_out_timepoints)`: predict held-out timepoints and report MSE/log-likelihood, for temporal cross-validation

### Step 6 — `synthetic.py` — data simulator
```python
def simulate_longitudinal(
    n_patients, time_points, patient_masks,
    n_factors, n_features_per_view,
    prior_masks=None, covariates=None,
    kernel="matern32", lengthscales=None, seed=42
) -> dict
```
Generates ground-truth Z from GP, W from horseshoe, Y from normal likelihood. Used for unit tests and benchmarking.

### Step 7 — `gene_set_prep.py` — preprocessing pipeline
- `screen_gene_sets()`: RidgeCV R² filter, returns subset of gene set dict
- `merge_gene_sets()`: Jaccard matrix → Ward linkage → dendrogram cut → union merging
- `prepare_gene_sets()`: Combined one-call interface (size filter → merge → return)

Dependencies: `scipy.cluster.hierarchy` (linkage, fcluster) — already in PACMon deps. No sklearn needed for this module.

### Step 8 — `signatures.py` — post-hoc prediction
- `compute_predictive_signatures()`: ElasticNetCV per factor, returns DataFrame of weights
- Time aggregation: mean / last / max over timepoints before regression
- Reports CV R² as quality metric per factor

### Step 9 — Tests
- `test_kernels.py`: positive-definiteness, symmetry, correct shape
- `test_model.py`: model forward pass runs without error on synthetic data
- `test_synthetic.py`: simulator produces correct shapes and structure
- `test_gene_set_prep.py`: screening reduces gene sets; merging reduces redundancy
- `test_signatures.py`: ElasticNet returns correct shapes; zero weights for unrelated features
- `test_integration.py`: **End-to-end pipeline test** — `prepare_gene_sets() → TemporalPACMON.fit() → compute_predictive_signatures()` on synthetic data. Verifies the full workflow runs without error and produces sensible outputs.
- `test_predict.py`: Temporal interpolation/extrapolation via `predict()` — verify GP conditional mean/variance at held-out timepoints are close to ground truth on synthetic data.

---

## Key Technical Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Kernel | Matérn 3/2 | C¹ smooth, appropriate for biological dynamics, handles irregular spacing |
| Variational family for z | Structured MVN (Cholesky), with low-rank fallback | Captures temporal correlations in posterior; low-rank option for T>30 |
| Ragged structure | Common grid + boolean mask | Avoids per-patient variable-size tensors; consistent with PACMon batching |
| Covariates | Patient-level only | Broadcast across timepoints; simpler β shape (P×C instead of P×T×C) |
| Kernel parameters | Learnable via SVI (LogNormal prior) | Automatically adapts lengthscale per factor; no grid search needed |
| Lengthscale sharing | Per-factor (default), optional shared mode | Per-factor captures factor-specific dynamics; shared mode for sparse data |
| Base framework | Pyro SVI (same as PACMon) | Consistent with entire MuVI/PACMon codebase; no new dependencies |
| Jitter | 1e-5 adaptive to 1e-4 | Ensures positive-definiteness; adapts if Cholesky fails |
| Gradient clipping | max_norm=10.0 | GP + horseshoe produces spiky gradients; prevents divergence |
| Device handling | `"auto"` with graceful CPU fallback | No hard CUDA dependency; works on any machine |
| Smoothness parameter | `zeta_k ~ Beta(1,1)` per factor | Auto-discovers temporal vs. i.i.d. factors; inspired by mofaflex |
| GS screening | Size filter only; horseshoe does the rest | No learning step — horseshoe already shrinks uninformative sets to zero |
| GS merging | Jaccard + Ward linkage | Simple, interpretable, well-established in bioinformatics |
| Signature extraction | ElasticNetCV | L1 sparsity for feature selection + L2 for correlated features; no new deps |
| Temporal interpolation | GP conditional prediction via `predict()` | Key selling point; free with GP model, enables clinical forecasting |

---

## Verification

1. **Unit tests**: `pytest tests/` — all tests should pass
2. **Synthetic smoke test**: Run `simulate_longitudinal()` + `TemporalPACMON.fit()` on toy data; verify ELBO decreases
3. **Lengthscale recovery**: Simulate with known lengthscales; check learned values are close
4. **Factor trajectory plots**: Verify `get_factors()` returns smooth GP-shaped trajectories
5. **Compare to PACMon**: On non-temporal data (all patients same single timepoint), results should match PACMon baseline
6. **Gene set prep**: Run `prepare_gene_sets()` with a small GO subset; verify fewer sets returned post-screening/merging
7. **Signatures**: Run `compute_predictive_signatures()` on fitted model; verify non-zero weights only for relevant features
8. **Temporal interpolation**: Use `predict()` at held-out timepoints; verify GP conditional mean is close to ground truth on synthetic data
9. **End-to-end pipeline**: `prepare_gene_sets() → fit() → predict() → compute_predictive_signatures()` runs without error on synthetic data
10. **Model comparison**: On non-temporal data, compare `compute_elbo()` between tpacmon (single timepoint) and PACMon; should be equivalent
11. **Device fallback**: Verify model trains correctly on CPU when CUDA is unavailable
