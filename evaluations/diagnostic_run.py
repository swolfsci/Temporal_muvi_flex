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
