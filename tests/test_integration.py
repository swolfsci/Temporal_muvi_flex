"""Integration test: synthetic data -> fit -> extract results."""

import numpy as np
import pytest
from tpacmon.core.synthetic import simulate_longitudinal
from tpacmon.core.models import TemporalPACMON


@pytest.mark.slow
class TestIntegration:
    """End-to-end tests (marked slow because they run SVI)."""

    def test_dense_only_smoke(self):
        """Smoke test: model runs without errors on synthetic data."""
        data = simulate_longitudinal(
            n_patients=8, n_timepoints=4, n_factors=2,
            n_features=[15, 10], n_views=2, seed=0,
        )
        model = TemporalPACMON(
            observations=data["observations"],
            time_points=data["time_points"],
            patient_masks=data["patient_masks"],
            n_dense_factors=2,
            device="cpu",
        )
        model.fit(n_epochs=5, learning_rate=0.05, early_stopping=False, seed=0)

        z = model.get_factors()
        assert z.shape == (8, 4, 2)
        assert not np.all(np.isnan(z))

        w = model.get_loadings()
        assert "view_0" in w
        assert w["view_0"].shape == (2, 15)

        ls = model.get_lengthscales()
        assert ls.shape == (2,)

        zeta = model.get_smoothness()
        assert zeta.shape == (2,)
        assert np.all((zeta >= 0) & (zeta <= 1))

    def test_with_covariates_smoke(self):
        """Smoke test with covariates."""
        data = simulate_longitudinal(
            n_patients=8, n_timepoints=4, n_factors=2,
            n_features=[15], n_views=1, n_covariates=2, seed=1,
        )
        model = TemporalPACMON(
            observations=data["observations"],
            time_points=data["time_points"],
            patient_masks=data["patient_masks"],
            covariates=data["covariates"],
            n_dense_factors=2,
            device="cpu",
        )
        model.fit(n_epochs=5, learning_rate=0.05, early_stopping=False, seed=1)

        betas = model.get_covariate_coefficients()
        assert "view_0" in betas
        assert betas["view_0"].shape == (2, 15)

    def test_predict_smoke(self):
        """Smoke test for GP prediction."""
        data = simulate_longitudinal(
            n_patients=5, n_timepoints=4, n_factors=2,
            n_features=[10], n_views=1, seed=2,
        )
        model = TemporalPACMON(
            observations=data["observations"],
            time_points=data["time_points"],
            patient_masks=data["patient_masks"],
            n_dense_factors=2,
            device="cpu",
        )
        model.fit(n_epochs=5, learning_rate=0.05, early_stopping=False, seed=2)

        new_t = np.array([0.5, 2.0, 5.0, 8.0])
        pred = model.predict(new_t)
        assert pred["mean"].shape == (5, 4, 2)
        assert pred["variance"].shape == (5, 4, 2)
        assert np.all(pred["variance"] >= 0)

    def test_training_history(self):
        data = simulate_longitudinal(
            n_patients=5, n_timepoints=3, n_factors=1,
            n_features=[10], n_views=1, seed=3,
        )
        model = TemporalPACMON(
            observations=data["observations"],
            time_points=data["time_points"],
            patient_masks=data["patient_masks"],
            n_dense_factors=1,
            device="cpu",
        )
        model.fit(n_epochs=10, early_stopping=False, seed=3)
        assert len(model.training_history) == 10
