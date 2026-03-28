# Plan: TemporalPACMon — Temporal Bayesian Latent Factor Model

## Context

This builds a new Python package called **`tpacmon`** (Temporal PACMON) in the `Temporal_muvi_flex` repo. It extends the PACMon model (gene-set informed priors + covariate regression) by replacing the i.i.d. Normal prior on latent factors with a **Gaussian Process prior using a Matérn 3/2 kernel**, enabling flexible modeling of dependent repeated measurements without assuming equal temporal spacing.

**Why this is needed:** MEFISTO (MOFA2) handles temporal data but assumes the same number of time points per subject and uses only RBF kernels. MuVI and PACMon ignore temporal dependency entirely. No existing tool combines (1) gene-set priors, (2) covariate regression, and (3) flexible irregular-time GP structure.

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
New learnable parameter:
  gamma_k ~ Normal(0, 1)    shape: (n_covariates, n_factors)

For each patient p, factor k:
  mu_k(x_p)  = x_p @ gamma[:, k]            # covariate-shifted mean (n_timepoints_obs,)
  t_{p,obs}  = time_points[mask_p]           # actual observed times
  K_k        = Matérn32(t_{p,obs}, t_{p,obs}; l_k, a_k)
  z_{p,k}    ~ MVN(mu_k(x_p) * ones, K_k)  # GP centered at covariate-informed mean
```

**Interpretation:**
- `gamma_{c,k}` = how much covariate `c` displaces patient p's expected trajectory for factor k
- GP residual = individual deviation from the group-level trajectory
- This is a **linear mixed effects** model at the factor level: fixed effects (covariates) + random effects (GP)
- Still fully linear in `y`; non-linearity is only in the prior structure on `z`
- `gamma` is identifiable because covariates are observed (unlike `z`)

**Changes to `beta_m`**: The `beta_m` regression module still exists (for the observation-level covariate effect on features), but now `gamma` handles the covariate effect at the *latent factor level*. Both are simultaneously identifiable with appropriate priors.

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

## Module 2: Gene Set Preprocessing Pipeline (`tpacmon/tools/gene_set_prep.py`)

### Problem
Large ontologies (GO, Hallmarks, Reactome) contain hundreds of redundant, overlapping gene sets. Including them all inflates model complexity, reduces effective degrees of freedom, and degrades interpretability — most gene sets explain no variance.

### Component A — Predictive Pre-Screening (soft down-weighting, biology-preserving)

**No gene sets are ever discarded.** Instead, each gene set's `prior_confidence` is scaled by its predictive R², so uninformative sets get looser priors while informative sets get tighter priors. The model still sees every gene set — it just trusts them proportionally.

For each gene set S_g with member genes G_g:
1. Compute gene set activity score per sample: `score_p = mean(X_p[G_g])` (simple mean aggregation)
2. Fit a cross-validated Ridge regression: `y_pv ~ score_p` for each omics view
3. Compute mean cross-validated R² across views → `r2_g ∈ [0, 1]`
4. **Scale prior confidence**: `conf_g = base_confidence * (1 + r2_g) / 2`
   - `r2_g = 0` (no signal) → `conf_g = base_conf / 2` (very loose prior, model barely constrained)
   - `r2_g = 0.5` → `conf_g = 0.75 * base_conf` (moderate prior)
   - `r2_g = 1.0` (perfect prediction) → `conf_g = base_conf` (full prior confidence)

```python
def compute_gene_set_weights(
    data: dict[str, np.ndarray],          # view -> (P, D_m) [averaged over time for temporal data]
    gene_sets: dict[str, list[str]],       # name -> gene list
    feature_names: dict[str, list[str]],   # view -> feature names
    base_confidence: float = 0.99,         # base prior confidence
    cv: int = 5,                           # cross-validation folds
) -> dict[str, float]:                    # name -> scaled prior_confidence
```

**Key design note**: Use `sklearn.linear_model.RidgeCV`. Fast on modern machines even for 500+ gene sets. The per-gene-set `prior_confidence` values are then passed directly into the `TemporalPACMON` constructor, overriding the single global `prior_confidence`.

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
def prepare_gene_sets(data, gene_sets, feature_names, ...):
    filtered = screen_gene_sets(data, gene_sets, ...)
    merged, provenance = merge_gene_sets(filtered, ...)
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
│   └── test_synthetic.py
└── notebooks/
    └── example_longitudinal.ipynb
```

---

## Critical Files to Borrow From

| Source | File | What to reuse |
|--------|------|---------------|
| `pacmon_flex` | `pacmon/core/models.py:54` | `PACMON.__init__`, prior setup, SVI loop, horseshoe prior (lines 2859–2900), covariate plates (lines 2940–2949) |
| `pacmon_flex` | `pacmon/core/models.py:2661` | `PACMONModel` — reuse all loading/covariate logic; replace `z` prior only |
| `pacmon_flex` | `pacmon/core/models.py:3051` | `PACMONGuide` — reuse all except `z` site; replace with structured MVN guide |
| `pacmon_flex` | `pacmon/core/callbacks.py` | Copy directly |
| `pacmon_flex` | `pacmon/core/early_stopping.py` | Copy directly |
| `pacmon_flex` | `pacmon/tools/feature_sets.py` | Copy directly |
| `muvi_flex` | `muvi/core/models.py:483` | `_setup_prior_confidence`, `_setup_prior_masks` — copy pattern |
| `MOFA2_flex` | `R/mefisto.R:926` | Conceptual: kernel formula, lengthscale grid search |

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
        device: str = "cuda",
    )
```
- `_setup_observations()`: flatten `(P, T, D_m)` to `(P*T, D_m)` with mask tracking
- `_setup_temporal()`: validate time_points and patient_masks
- `_setup_model_guide()`: instantiate TemporalModel + TemporalGuide
- `fit()`: standard SVI loop (copy from PACMon, patient-level batching)
- `get_factors()`: return `(P, T, K)` factor scores
- `get_loadings()`: return `(K, D_m)` per view
- `get_lengthscales()`: return `(K,)` learned temporal lengthscales

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
- `prepare_gene_sets()`: Combined one-call interface

Dependencies: `sklearn` (RidgeCV), `scipy.cluster.hierarchy` (linkage, fcluster) — already in PACMon deps.

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

---

## Key Technical Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Kernel | Matérn 3/2 | C¹ smooth, appropriate for biological dynamics, handles irregular spacing |
| Variational family for z | Structured MVN (Cholesky) | Captures temporal correlations in posterior; mean-field would ignore them |
| Ragged structure | Common grid + boolean mask | Avoids per-patient variable-size tensors; consistent with PACMon batching |
| Covariates | Patient-level only | Broadcast across timepoints; simpler β shape (P×C instead of P×T×C) |
| Kernel parameters | Learnable via SVI (LogNormal prior) | Automatically adapts lengthscale per factor; no grid search needed |
| Base framework | Pyro SVI (same as PACMon) | Consistent with entire MuVI/PACMon codebase; no new dependencies |
| Jitter | 1e-5 added to kernel diagonal | Ensures positive-definiteness for Cholesky |

---

---

## Key Technical Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Kernel | Matérn 3/2 | C¹ smooth, appropriate for biological dynamics, handles irregular spacing |
| Variational family for z | Structured MVN (Cholesky) | Captures temporal correlations in posterior; mean-field would ignore them |
| Ragged structure | Common grid + boolean mask | Avoids per-patient variable-size tensors; consistent with PACMon batching |
| Covariates | Patient-level only | Broadcast across timepoints; simpler β shape (P×C instead of P×T×C) |
| Kernel parameters | Learnable via SVI (LogNormal prior) | Automatically adapts lengthscale per factor; no grid search needed |
| Base framework | Pyro SVI (same as PACMon) | Consistent with entire MuVI/PACMon codebase; no new dependencies |
| Jitter | 1e-5 added to kernel diagonal | Ensures positive-definiteness for Cholesky |
| GS screening | RidgeCV R² filter | Fast, data-driven, no hyperparameter tuning required |
| GS merging | Jaccard + Ward linkage | Simple, interpretable, well-established in bioinformatics |
| Signature extraction | ElasticNetCV | L1 sparsity for feature selection + L2 for correlated features; no new deps |

---

## Verification

1. **Unit tests**: `pytest tests/` — all tests should pass
2. **Synthetic smoke test**: Run `simulate_longitudinal()` + `TemporalPACMON.fit()` on toy data; verify ELBO decreases
3. **Lengthscale recovery**: Simulate with known lengthscales; check learned values are close
4. **Factor trajectory plots**: Verify `get_factors()` returns smooth GP-shaped trajectories
5. **Compare to PACMon**: On non-temporal data (all patients same single timepoint), results should match PACMon baseline
6. **Gene set prep**: Run `prepare_gene_sets()` with a small GO subset; verify fewer sets returned post-screening/merging
7. **Signatures**: Run `compute_predictive_signatures()` on fitted model; verify non-zero weights only for relevant features
