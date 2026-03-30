# tpacmon — Temporal Bayesian Latent Factor Model

**tpacmon** (Temporal PACMON) extends the [PACMon](https://github.com/swolfsci/pacmon_flex) model with **Gaussian Process priors** on latent factors, enabling flexible modeling of longitudinal multi-omics data with irregular temporal spacing, gene-set informed priors, and dual-level covariate regression.

## Motivation

Existing methods for temporal multi-omics factor analysis have key limitations:

- **MEFISTO** (MOFA2) uses only the RBF kernel, optimizes lengthscales via grid search, and requires a Kronecker structure that forces equal temporal spacing across groups. No gene-set priors or covariate regression.
- **mofaflex** supports GP priors and informed horseshoe on weights, but these operate independently — no covariate-informed GP mean, no factor-level covariate regression, and no explicit per-patient irregular masking.
- **MuVI** and **PACMon** provide gene-set priors and covariate regression but ignore temporal dependency entirely.

**tpacmon** is the first tool to combine:
1. Gene-set informed priors (regularized horseshoe)
2. Dual-level covariate regression (feature-level `beta` + factor-level `gamma` shifting the GP trajectory)
3. Flexible irregular-time GP structure with per-patient masking

## Installation

```bash
pip install torch pyro-ppl scikit-learn scipy pandas numpy tqdm seaborn

# Then install tpacmon
pip install -e .
```

Requires Python >= 3.10.

## Quick Start

```python
from tpacmon import TemporalPACMON
from tpacmon.core.synthetic import simulate_longitudinal

# Generate synthetic longitudinal multi-omics data
data = simulate_longitudinal(
    n_patients=30, n_timepoints=8, n_factors=3,
    n_features=[200, 150], n_views=2, n_covariates=2,
)

# Fit the model
model = TemporalPACMON(
    observations=data["observations"],      # dict of (P, T, D_m) arrays
    time_points=data["time_points"],         # (T,) common time grid
    patient_masks=data["patient_masks"],     # (P, T) boolean mask
    covariates=data["covariates"],           # (P, C) patient-level covariates
    n_dense_factors=3,
    kernel="matern32",
)
model.fit(n_epochs=1000, learning_rate=0.01)

# Extract results
z = model.get_factors()           # (P, T, K) factor scores
w = model.get_loadings()          # dict of (K, D_m) per view
ls = model.get_lengthscales()     # (K,) learned temporal lengthscales
zeta = model.get_smoothness()     # (K,) temporal vs. i.i.d. per factor

# Predict at new time points (GP conditional)
pred = model.predict(new_time_points=[0.5, 2.0, 5.0])
# pred["mean"]: (P, T_new, K), pred["variance"]: (P, T_new, K)
```

### With Gene-Set Priors

```python
from tpacmon.tools.feature_sets import FeatureSets
from tpacmon.tools.gene_set_prep import prepare_gene_sets

# Load and preprocess gene sets
gene_sets = FeatureSets.from_gmt("c2.cp.v2023.gmt")
gene_sets = prepare_gene_sets(gene_sets, feature_names, min_size=10, max_size=500)

# Convert to prior masks and fit
prior_masks = {"rna": gene_sets.to_mask(feature_names)}
model = TemporalPACMON(
    observations=observations,
    time_points=time_points,
    patient_masks=patient_masks,
    prior_masks=prior_masks,
    prior_confidence="low",          # "low" | "med" | "high"
    n_dense_factors=2,               # additional uninformed factors
)
```

### Post-hoc Enrichment Analysis

```python
from tpacmon import enrich_factors

# Basic: prior fidelity + temporal metrics (fast)
result = enrich_factors(model)
print(result.fidelity)    # AUROC fidelity, refinement counts, Gini per sparse factor
print(result.temporal)    # zeta, temporal_strength, TVR, lengthscale per factor

# With external gene sets (PCGSE competitive enrichment)
result = enrich_factors(model, gene_sets=gene_sets, view_name="rna")
print(result.enrichment)  # Wilcoxon p-values, BH-adjusted, AUROC, prior_rank

# With covariate analysis (temporal divergence, TIR)
result = enrich_factors(
    model,
    covariate_names=["sex", "treatment"],
    covariate_types={"sex": "categorical", "treatment": "categorical"},
)
print(result.covariate)          # gamma, TD per (factor, covariate)
print(result.covariate_summary)  # TIR per covariate (static vs. dynamic classifier)

# Optionally include ElasticNet predictive signatures (slow)
result = enrich_factors(model, compute_signatures=True)
print(result.signatures)         # {factor_name: DataFrame of top features}
print(result.signature_scores)   # {factor_name: CV R²}

# Export for downstream tools / webapp
result.save("enrichment.pkl")            # pickle round-trip
result.to_json("enrichment.json")        # JSON for web consumption
loaded = EnrichmentResult.load("enrichment.pkl")
```

### Model Serialization

```python
# Save trained model (observations are NOT saved)
model.save("trained_model.pt")

# Load — all get_* methods work immediately
model = TemporalPACMON.load("trained_model.pt")
z = model.get_factors()
w = model.get_loadings()
pred = model.predict(new_time_points=[0.5, 2.0, 5.0])
```

### Predictive Signatures (standalone)

```python
from tpacmon.tools.signatures import compute_predictive_signatures

results = compute_predictive_signatures(model, aggregation="mean")
# results["signatures"]: {factor_name: DataFrame of top features}
# results["scores"]: {factor_name: CV R²}
```

---

## Mathematical Model

### Notation
- P: patients, T: max timepoints (common grid), K: factors, D_m: features in view m
- `mask_p`: which visits patient p attended
- `t`: actual time values (common grid, e.g., [0, 1, 7, 30] days)

### Factor Loadings — Regularized Horseshoe (from PACMon)
```
For each factor k, feature j, view m:
  local_scale  ~ HalfCauchy(1)
  factor_scale ~ HalfCauchy(1)
  view_scale   ~ HalfCauchy(1)
  caux         ~ InverseGamma(0.5, 0.5)
  c = sqrt(caux) * prior_scale[k,j,m]        # gene-set modulation
  w_scale = (global * c * local * factor * view) / sqrt(c² + (local*factor*view)²)
  w_{k,j,m}    ~ Normal(0, w_scale)
```

Prior scales encode gene-set membership: `clip(mask + (1 - confidence), 1e-8, 1.0)`, where confidence maps as `low=0.99, med=0.995, high=0.999`.

Dense (uninformed) factors use `w ~ Normal(0, 1)`.

### Temporal Latent Factors — GP Prior with Smoothness Mixing
```
Learnable parameters:
  gamma_k ~ Normal(0, 1)    shape: (n_covariates, n_factors)   # factor-level covariate effect
  zeta_k  ~ Beta(1, 1)      shape: (n_factors,)                # smoothness mixing weight

For each patient p, factor k:
  mu_k(x_p) = x_p @ gamma[:, k]               # covariate-shifted GP mean
  K_k        = Matern32(t_obs, t_obs; l_k, a_k)

  f_{p,k}   ~ MVN(mu_k(x_p), K_k)             # GP component (temporal structure)
  eta_{p,k} ~ Normal(0, 1)                     # i.i.d. component
  z_{p,k}   = sqrt(1 - zeta_k) * f + sqrt(zeta_k) * eta
```

**The `zeta_k` smoothness parameter** (inspired by mofaflex):
- `zeta_k -> 0`: factor k is **fully temporal** — smooth GP trajectories
- `zeta_k -> 1`: factor k is **i.i.d.** — standard PACMon behavior
- Learned per factor, so the model **automatically discovers which factors benefit from temporal structure**

### Dual-Level Covariate Regression
- **Factor level (`gamma`)**: shifts the GP mean trajectory per covariate — a covariate-informed baseline
- **Feature level (`beta_m`)**: direct covariate effect on observations, broadcast across timepoints

### Likelihood
```
For each observed visit (patient p, time t), view m:
  y_{p,t,m} ~ Normal(z_{p,t} @ w_m + x_p @ beta_m, sigma_m)
```
Missing visits are masked out via `pyro.poutine.mask()`.

### Variational Posterior
```
For each patient p, factor k:
  q(f_{p,k}) = MVN(mu_{p,k}, L_{p,k} L_{p,k}^T)    # Cholesky parameterization
```
All other sites use diagonal Normal/LogNormal/Beta guides.

---

## Comparison

| Aspect | MEFISTO | mofaflex | tpacmon |
|--------|---------|----------|---------|
| Kernel | RBF only | RBF + Matern (GPyTorch) | Matern 3/2 (default), 5/2, RBF |
| Lengthscale optimization | Grid search | Learned (GPyTorch) | Learned via SVI (LogNormal prior) |
| Temporal structure | Kronecker (forces shared grid) | Sparse variational (inducing points) | Per-patient boolean mask on common grid |
| Gene-set priors | None | InformedHorseshoe (independent from GP) | Regularized horseshoe (from PACMon) |
| Factor-level covariate effect | None | None | `gamma` shifts GP mean per covariate |
| Feature-level covariate regression | None | Guiding variables (auxiliary loss) | `beta_m` (generative) |
| Smoothness discovery | Scale param | `zeta_k` mixing | `zeta_k ~ Beta(1,1)` per factor |
| Interpolation/prediction | GP conditional (mean only) | Not documented | `predict()` returns mean + variance |
| Framework | R (mofapy2) | Pyro + GPyTorch | Pyro (no GPyTorch dependency) |

---

## Available Kernels

| Kernel | Formula | Properties |
|--------|---------|------------|
| **Matern 3/2** (default) | `(1+sqrt(3)d/l) exp(-sqrt(3)d/l)` | C^1 smooth, good for biological dynamics |
| Matern 5/2 | `(1+sqrt(5)d/l+5d^2/3l^2) exp(-sqrt(5)d/l)` | C^2 smooth |
| RBF | `exp(-d^2/2l^2)` | Infinitely smooth, may over-smooth |

---

## Training Details

- **Optimizer**: ClippedAdam with exponential learning rate decay
- **Loss**: Trace_ELBO
- **Gradient clipping**: `max_norm=10.0` (GP + horseshoe can produce spiky gradients)
- **Kernel jitter**: `1e-5` diagonal for numerical stability
- **Early stopping**: patience=10, tolerance=1e-5, min_epochs=100
- **View scaling**: ELBO contributions balanced across views with different feature counts

---

## Package Structure

```
temporal_pacmon/
├── tpacmon/
│   ├── __init__.py
│   ├── core/
│   │   ├── models.py          # TemporalModel, TemporalGuide, TemporalPACMON (+ save/load)
│   │   ├── kernels.py         # Matern 3/2, Matern 5/2, RBF kernels
│   │   ├── callbacks.py       # Checkpoint callback
│   │   ├── early_stopping.py  # Early stopping callback
│   │   └── synthetic.py       # Longitudinal data simulator
│   └── tools/
│       ├── feature_sets.py    # FeatureSet/FeatureSets classes, GMT I/O
│       ├── gene_set_prep.py   # Size filtering + Jaccard hierarchical merging
│       ├── signatures.py      # ElasticNet predictive signatures
│       └── enrichment.py      # Post-hoc enrichment analysis (enrich_factors)
├── tests/                     # 84 tests
├── pyproject.toml
└── README.md
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Kernel | Matern 3/2 | C^1 smooth, appropriate for biological dynamics |
| Variational family for z | Structured MVN (Cholesky) | Captures temporal correlations in posterior |
| Ragged structure | Common grid + boolean mask | Avoids variable-size tensors; handles irregular visits |
| Kernel parameters | Learned via SVI | Per-factor lengthscales without grid search |
| Smoothness parameter | `zeta_k ~ Beta(1,1)` | Auto-discovers temporal vs. i.i.d. factors |
| Gene set preprocessing | Size filter + Jaccard merge | Horseshoe handles the rest — no learning step needed |
| Signature extraction | ElasticNetCV | L1 sparsity + L2 for correlated features |
| Framework | Pyro (no GPyTorch) | Consistent with PACMon/MuVI ecosystem |

---

## Testing

```bash
# Run all tests (84 tests)
pytest tests/ -v

# Run fast tests only (exclude SVI smoke tests)
pytest tests/ -m "not slow" -v
```

---

## References

- **PACMon**: Gene-set informed Bayesian latent factor model ([pacmon_flex](https://github.com/swolfsci/pacmon_flex))
- **MuVI**: Multi-view latent factor model with informed priors ([MuVI_flex](https://github.com/swolfsci/MuVI_flex))
- **MEFISTO**: Temporal extension of MOFA2 ([Velten et al., 2022](https://doi.org/10.1038/s41592-021-01343-9))
- **mofaflex**: Flexible MOFA with GP priors and informed horseshoe ([bioFAM/mofaflex](https://github.com/bioFAM/mofaflex))
