"""Diagnostic: run one seed with no early stopping to check convergence."""

from evaluations.synthetic import generate_evaluation_data
from evaluations.metrics import align_factors, factor_score_correlation, variance_explained
from tpacmon.core.models import TemporalPACMON

data = generate_evaluation_data(
    n_patients=50, n_timepoints=6, n_views=2,
    n_features=[200, 150], n_iid_factors=3, seed=0,
)

model = TemporalPACMON(
    observations=data["observations"],
    time_points=data["time_points"],
    patient_masks=data["patient_masks"],
    n_dense_factors=3,
)

model.fit(n_epochs=3000, early_stopping=False, seed=0)

history = model.training_history
checkpoints = [500, 1000, 1500, 2000, 2500, 3000]
print("\n=== ELBO trajectory ===")
for cp in checkpoints:
    if cp <= len(history):
        print(f"  Epoch {cp}: {history[cp - 1]:.0f}")

print(f"\n=== Zeta (should be ~1.0 for i.i.d. factors) ===")
zeta = model.get_smoothness()
print(f"  {zeta}")

z = model.get_factors()[:, :6, :]
alignment = align_factors(data["true_z"], z)
corr = factor_score_correlation(data["true_z"], z, alignment)
print(f"\n=== Factor score correlation ===")
print(f"  Mean: {corr['mean']:.3f}")
print(f"  Per factor: {[f'{x:.3f}' for x in corr['per_factor']]}")

print(f"\n=== R² per view (model's own method, handles normalization) ===")
r2_result = model.get_variance_explained(per_factor=True)
for vn in data["view_names"]:
    print(f"  {vn}: {r2_result['total'][vn]:.1f}%")
    print(f"    per factor: {[f'{x:.1f}%' for x in r2_result['per_factor'][vn]]}")

import numpy as np
from evaluations.metrics import apply_alignment

print(f"\n=== Loading magnitude diagnosis ===")
learned_w = model.get_loadings()
for vn in data["view_names"]:
    true_w = data["true_w"][vn]
    lw = apply_alignment(learned_w[vn], alignment, axis=0)

    print(f"  {vn}:")
    print(f"    True W  — mean|w|: {np.mean(np.abs(true_w)):.4f}, max|w|: {np.max(np.abs(true_w)):.4f}, nonzero: {np.count_nonzero(true_w)}/{true_w.size}")
    print(f"    Learned W — mean|w|: {np.mean(np.abs(lw)):.4f}, max|w|: {np.max(np.abs(lw)):.4f}")
    print(f"    Ratio (learned/true mean|w|): {np.mean(np.abs(lw)) / np.mean(np.abs(true_w)):.3f}")

    # Check per-factor
    for k in range(true_w.shape[0]):
        true_active = np.abs(true_w[k]) > 0
        print(f"    Factor {k}: true active={true_active.sum()}, learned mean|w| on active={np.mean(np.abs(lw[k, true_active])):.4f}, true mean|w| on active={np.mean(np.abs(true_w[k, true_active])):.4f}")

print(f"\n=== Factor score magnitude ===")
print(f"  True z  — std: {np.nanstd(data['true_z']):.4f}")
print(f"  Learned z — std: {np.nanstd(z):.4f}")
print(f"  Ratio: {np.nanstd(z) / np.nanstd(data['true_z']):.3f}")

print(f"\n=== Guide internals ===")
guide = model._guide
print(f"  eta_mean range: [{guide.eta_mean.min().item():.4f}, {guide.eta_mean.max().item():.4f}]")
print(f"  eta_scale range: [{guide.eta_scale.min().item():.4f}, {guide.eta_scale.max().item():.4f}]")
print(f"  z_mean range: [{guide.z_mean.min().item():.4f}, {guide.z_mean.max().item():.4f}]")
print(f"  z_scale range: [{guide.z_scale.min().item():.4f}, {guide.z_scale.max().item():.4f}]")
print(f"  Learned amplitude: {guide.mode('amplitude')}")
print(f"  Learned zeta: {guide.mode('zeta')}")
