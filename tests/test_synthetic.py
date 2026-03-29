"""Tests for synthetic data generation."""

import numpy as np
import pytest
from tpacmon.core.synthetic import simulate_longitudinal


class TestSimulateLongitudinal:
    def test_basic_shape(self):
        data = simulate_longitudinal(n_patients=10, n_timepoints=5, n_factors=2,
                                     n_features=[50, 30], n_views=2)
        assert data["time_points"].shape == (5,)
        assert data["patient_masks"].shape == (10, 5)
        assert data["covariates"].shape == (10, 2)
        assert data["true_z"].shape == (10, 5, 2)
        assert data["observations"]["view_0"].shape == (10, 5, 50)
        assert data["observations"]["view_1"].shape == (10, 5, 30)

    def test_masks_have_at_least_3_obs(self):
        data = simulate_longitudinal(n_patients=50, missing_rate=0.5)
        for p in range(50):
            assert data["patient_masks"][p].sum() >= 3

    def test_nan_at_unobserved(self):
        data = simulate_longitudinal(n_patients=10, missing_rate=0.3)
        for vn, obs in data["observations"].items():
            for p in range(10):
                unobserved = ~data["patient_masks"][p]
                if unobserved.any():
                    assert np.all(np.isnan(obs[p, unobserved, :]))

    def test_reproducibility(self):
        d1 = simulate_longitudinal(seed=123)
        d2 = simulate_longitudinal(seed=123)
        np.testing.assert_array_equal(d1["true_z"], d2["true_z"])
        np.testing.assert_array_equal(d1["time_points"], d2["time_points"])

    def test_default_features(self):
        data = simulate_longitudinal(n_views=3)
        assert len(data["observations"]) == 3
        for vn in data["observations"]:
            assert data["observations"][vn].shape[-1] == 100

    def test_lengthscales(self):
        ls = np.array([1.0, 5.0, 10.0])
        data = simulate_longitudinal(n_factors=3, lengthscales=ls)
        np.testing.assert_array_equal(data["true_lengthscales"], ls)
