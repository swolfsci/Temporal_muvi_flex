"""Tests for TemporalPACMON model construction and basic operations."""

import numpy as np
import pandas as pd
import pytest
import torch

from tpacmon.core.models import TemporalModel, TemporalGuide, TemporalPACMON


def make_simple_data(n_patients=5, n_timepoints=4, n_features=20, n_views=1):
    """Create minimal test data."""
    rng = np.random.default_rng(42)
    time_points = np.array([0.0, 1.0, 3.0, 7.0], dtype=np.float32)[:n_timepoints]
    patient_masks = np.ones((n_patients, n_timepoints), dtype=bool)
    # Make one patient miss a timepoint
    patient_masks[0, -1] = False

    obs = {}
    for m in range(n_views):
        y = rng.standard_normal((n_patients, n_timepoints, n_features)).astype(np.float32)
        y[0, -1, :] = np.nan
        obs[f"view_{m}"] = y

    return obs, time_points, patient_masks


class TestTemporalPACMONInit:
    def test_basic_dense_only(self):
        obs, tp, masks = make_simple_data()
        model = TemporalPACMON(
            observations=obs,
            time_points=tp,
            patient_masks=masks,
            n_dense_factors=3,
            device="cpu",
        )
        assert model.n_factors == 3
        assert model.n_sparse_factors == 0
        assert model.n_dense_factors == 3

    def test_with_prior_masks(self):
        obs, tp, masks = make_simple_data()
        # 2 gene sets, 20 features
        prior = {"view_0": np.eye(2, 20, dtype=np.float32)}
        model = TemporalPACMON(
            observations=obs,
            time_points=tp,
            patient_masks=masks,
            prior_masks=prior,
            n_dense_factors=1,
            device="cpu",
        )
        assert model.n_sparse_factors == 2
        assert model.n_dense_factors == 1
        assert model.n_factors == 3

    def test_with_covariates(self):
        obs, tp, masks = make_simple_data()
        covs = np.random.randn(5, 3).astype(np.float32)
        model = TemporalPACMON(
            observations=obs,
            time_points=tp,
            patient_masks=masks,
            n_dense_factors=2,
            covariates=covs,
            device="cpu",
        )
        assert model.n_covariates == 3

    def test_no_factors_raises(self):
        obs, tp, masks = make_simple_data()
        with pytest.raises(ValueError, match="Must specify at least one factor"):
            TemporalPACMON(
                observations=obs, time_points=tp, patient_masks=masks, device="cpu"
            )

    def test_multi_view(self):
        obs, tp, masks = make_simple_data(n_views=3)
        model = TemporalPACMON(
            observations=obs,
            time_points=tp,
            patient_masks=masks,
            n_dense_factors=2,
            device="cpu",
        )
        assert model.n_features == {"view_0": 20, "view_1": 20, "view_2": 20}

    def test_prior_confidence_mapping(self):
        obs, tp, masks = make_simple_data()
        for conf, expected in [("low", 0.99), ("med", 0.995), ("high", 0.999)]:
            model = TemporalPACMON(
                observations=obs, time_points=tp, patient_masks=masks,
                n_dense_factors=1, prior_confidence=conf, device="cpu",
            )
            assert model.prior_confidence == expected

    def test_flatten_observations(self):
        obs, tp, masks = make_simple_data(n_patients=3, n_timepoints=4)
        model = TemporalPACMON(
            observations=obs, time_points=tp, patient_masks=masks,
            n_dense_factors=1, device="cpu",
        )
        flat_obs, flat_mask, pat_idx, time_idx = model._flatten_observations()
        # Patient 0 has 3 obs, patients 1-2 have 4 each = 11 total
        assert flat_obs.shape[0] == 11
        assert pat_idx.shape[0] == 11

    def test_dataframe_input(self):
        tp = np.array([0.0, 1.0], dtype=np.float32)
        masks = np.ones((2, 2), dtype=bool)
        df = pd.DataFrame(np.random.randn(2, 10).astype(np.float32))
        # 2D input gets expanded to (P, 1, D)
        model = TemporalPACMON(
            observations={"rna": df},
            time_points=np.array([0.0], dtype=np.float32),
            patient_masks=np.ones((2, 1), dtype=bool),
            n_dense_factors=1,
            device="cpu",
        )
        assert model.n_features == {"rna": 10}


class TestSerialization:
    def _train_model(self, obs, tp, masks, **kwargs):
        model = TemporalPACMON(
            observations=obs, time_points=tp, patient_masks=masks,
            device="cpu", **kwargs,
        )
        model.fit(n_epochs=5, early_stopping=False, seed=0)
        return model

    def test_save_load_factors_match(self, tmp_path):
        obs, tp, masks = make_simple_data()
        model = self._train_model(obs, tp, masks, n_dense_factors=2)

        path = str(tmp_path / "model.pt")
        model.save(path)
        loaded = TemporalPACMON.load(path, map_location="cpu")

        np.testing.assert_allclose(
            model.get_factors(), loaded.get_factors(), atol=1e-6
        )

    def test_save_load_loadings_match(self, tmp_path):
        obs, tp, masks = make_simple_data()
        model = self._train_model(obs, tp, masks, n_dense_factors=2)

        path = str(tmp_path / "model.pt")
        model.save(path)
        loaded = TemporalPACMON.load(path, map_location="cpu")

        for vn in model.view_names:
            np.testing.assert_allclose(
                model.get_loadings()[vn], loaded.get_loadings()[vn], atol=1e-6
            )

    def test_save_load_metadata_preserved(self, tmp_path):
        obs, tp, masks = make_simple_data()
        model = self._train_model(obs, tp, masks, n_dense_factors=2)

        path = str(tmp_path / "model.pt")
        model.save(path)
        loaded = TemporalPACMON.load(path, map_location="cpu")

        assert loaded.factor_names == model.factor_names
        assert loaded.feature_names == model.feature_names
        assert loaded.view_names == model.view_names
        assert loaded._trained is True
        assert loaded.n_factors == model.n_factors

    def test_save_untrained_raises(self, tmp_path):
        obs, tp, masks = make_simple_data()
        model = TemporalPACMON(
            observations=obs, time_points=tp, patient_masks=masks,
            n_dense_factors=2, device="cpu",
        )
        with pytest.raises(RuntimeError, match="trained"):
            model.save(str(tmp_path / "model.pt"))

    def test_save_load_with_priors(self, tmp_path):
        obs, tp, masks = make_simple_data()
        prior = {"view_0": np.eye(2, 20, dtype=np.float32)}
        model = self._train_model(obs, tp, masks, prior_masks=prior, n_dense_factors=1)

        path = str(tmp_path / "model.pt")
        model.save(path)
        loaded = TemporalPACMON.load(path, map_location="cpu")

        assert loaded.n_sparse_factors == 2
        assert loaded.n_dense_factors == 1
        np.testing.assert_allclose(
            model.get_factors(), loaded.get_factors(), atol=1e-6
        )
        for vn in model.view_names:
            np.testing.assert_allclose(
                model.get_loadings()[vn], loaded.get_loadings()[vn], atol=1e-6
            )

    def test_save_load_smoothness_lengthscales(self, tmp_path):
        obs, tp, masks = make_simple_data()
        model = self._train_model(obs, tp, masks, n_dense_factors=2)

        path = str(tmp_path / "model.pt")
        model.save(path)
        loaded = TemporalPACMON.load(path, map_location="cpu")

        np.testing.assert_allclose(
            model.get_smoothness(), loaded.get_smoothness(), atol=1e-6
        )
        np.testing.assert_allclose(
            model.get_lengthscales(), loaded.get_lengthscales(), atol=1e-6
        )
