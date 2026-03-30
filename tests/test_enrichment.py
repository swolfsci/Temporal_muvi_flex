"""Tests for post-hoc enrichment analysis."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from tpacmon.tools.enrichment import (
    enrich_factors,
    _gini_coefficient,
    _benjamini_hochberg,
    EnrichmentResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(
    n_patients=10,
    n_timepoints=4,
    n_factors=3,
    n_sparse_factors=2,
    n_features=50,
    n_covariates=0,
    seed=42,
):
    """Create a mock TemporalPACMON with controllable parameters."""
    rng = np.random.default_rng(seed)
    model = MagicMock()
    model._trained = True
    model.n_patients = n_patients
    model.n_timepoints = n_timepoints
    model.n_factors = n_factors
    model.n_sparse_factors = n_sparse_factors
    model.n_dense_factors = n_factors - n_sparse_factors
    model.n_covariates = n_covariates
    model.view_names = ["view_0"]
    model.feature_names = {
        "view_0": [f"gene_{i}" for i in range(n_features)]
    }
    factor_names = []
    for k in range(n_sparse_factors):
        factor_names.append(f"sparse_{k}")
    for k in range(n_factors - n_sparse_factors):
        factor_names.append(f"dense_{k}")
    model.factor_names = factor_names
    model.patient_masks_np = np.ones((n_patients, n_timepoints), dtype=bool)
    model.time_points_np = np.linspace(0, 1, n_timepoints).astype(np.float32)
    model.covariates = None
    model.prior_masks = None

    # Default loadings: random
    w = rng.standard_normal((n_factors, n_features)).astype(np.float32)
    model.get_loadings.return_value = {"view_0": w}
    model._guide = MagicMock()
    model._guide.get_w.return_value = w

    # Default factor scores: random with some temporal structure
    z = rng.standard_normal((n_patients, n_timepoints, n_factors)).astype(np.float32)
    model.get_factors.return_value = z

    # Default smoothness and lengthscales
    model.get_smoothness.return_value = rng.uniform(0, 1, size=n_factors).astype(np.float32)
    model.get_lengthscales.return_value = rng.uniform(1, 5, size=n_factors).astype(np.float32)

    return model


# ---------------------------------------------------------------------------
# Test _gini_coefficient
# ---------------------------------------------------------------------------

class TestGiniCoefficient:
    def test_perfect_equality(self):
        values = np.ones(100)
        assert _gini_coefficient(values) == pytest.approx(0.0, abs=1e-10)

    def test_perfect_inequality(self):
        values = np.zeros(100)
        values[-1] = 1.0
        gini = _gini_coefficient(values)
        # For n=100, theoretical max Gini approaches 1.0
        assert gini > 0.95

    def test_empty(self):
        assert _gini_coefficient(np.array([])) == 0.0

    def test_all_zeros(self):
        assert _gini_coefficient(np.zeros(10)) == 0.0

    def test_known_value(self):
        # [1, 2, 3] -> Gini = 2/9 ≈ 0.222
        gini = _gini_coefficient(np.array([1.0, 2.0, 3.0]))
        assert gini == pytest.approx(2.0 / 9.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Test _benjamini_hochberg
# ---------------------------------------------------------------------------

class TestBenjaminiHochberg:
    def test_empty(self):
        result = _benjamini_hochberg(np.array([]))
        assert len(result) == 0

    def test_single(self):
        result = _benjamini_hochberg(np.array([0.05]))
        assert result[0] == pytest.approx(0.05)

    def test_monotonicity(self):
        pvals = np.array([0.01, 0.04, 0.03, 0.20])
        adjusted = _benjamini_hochberg(pvals)
        # Sorted adjusted p-values should be non-decreasing
        sorted_adj = adjusted[np.argsort(pvals)]
        for i in range(len(sorted_adj) - 1):
            assert sorted_adj[i] <= sorted_adj[i + 1] + 1e-15

    def test_clipped_to_one(self):
        pvals = np.array([0.5, 0.8, 0.9])
        adjusted = _benjamini_hochberg(pvals)
        assert np.all(adjusted <= 1.0)

    def test_preserves_order(self):
        pvals = np.array([0.001, 0.05, 0.5])
        adjusted = _benjamini_hochberg(pvals)
        assert adjusted[0] < adjusted[1] < adjusted[2]


# ---------------------------------------------------------------------------
# Test Prior Fidelity (Block A)
# ---------------------------------------------------------------------------

class TestFidelity:
    def test_perfect_match(self):
        """Loadings perfectly match prior -> fidelity = 1.0."""
        model = _make_mock_model(n_features=50, n_sparse_factors=1, n_factors=1)
        # Prior: first 10 genes active
        prior_mask = np.zeros((1, 50), dtype=np.float32)
        prior_mask[0, :10] = 1.0
        model.prior_masks = {"view_0": prior_mask}
        # Loadings: high for first 10, zero elsewhere
        w = np.zeros((1, 50), dtype=np.float32)
        w[0, :10] = 5.0
        model._guide.get_w.return_value = w
        model.get_loadings.return_value = {"view_0": w}

        result = enrich_factors(model)
        assert result.fidelity is not None
        assert result.fidelity.loc["sparse_0", "fidelity"] == pytest.approx(1.0)
        assert result.fidelity.loc["sparse_0", "deviation"] == pytest.approx(0.0)

    def test_random_loadings(self):
        """Random loadings w.r.t. prior -> fidelity ~ 0.5."""
        rng = np.random.default_rng(123)
        model = _make_mock_model(
            n_features=200, n_sparse_factors=1, n_factors=1, seed=123
        )
        prior_mask = np.zeros((1, 200), dtype=np.float32)
        prior_mask[0, :50] = 1.0
        model.prior_masks = {"view_0": prior_mask}
        # Random loadings
        w = rng.standard_normal((1, 200)).astype(np.float32)
        model._guide.get_w.return_value = w

        result = enrich_factors(model)
        fidelity = result.fidelity.loc["sparse_0", "fidelity"]
        assert abs(fidelity - 0.5) < 0.15  # within 0.15 of 0.5

    def test_refinement_counts(self):
        """Verify retained/pruned/recruited logic."""
        model = _make_mock_model(n_features=20, n_sparse_factors=1, n_factors=1)
        prior_mask = np.zeros((1, 20), dtype=np.float32)
        prior_mask[0, :10] = 1.0
        model.prior_masks = {"view_0": prior_mask}
        # Loadings: first 5 genes high (retained), next 5 low (pruned),
        # genes 10-14 high (recruited)
        w = np.zeros((1, 20), dtype=np.float32)
        w[0, :5] = 3.0      # retained (prior=1, |w| high)
        w[0, 5:10] = 0.01   # pruned (prior=1, |w| low)
        w[0, 10:15] = 3.0   # recruited (prior=0, |w| high)
        model._guide.get_w.return_value = w

        result = enrich_factors(model)
        row = result.fidelity.loc["sparse_0"]
        assert row["retained"] == 5
        assert row["pruned"] == 5
        assert row["recruited"] == 5

    def test_gini_in_fidelity(self):
        """Gini should be high for sparse loadings."""
        model = _make_mock_model(n_features=100, n_sparse_factors=1, n_factors=1)
        prior_mask = np.zeros((1, 100), dtype=np.float32)
        prior_mask[0, :5] = 1.0
        model.prior_masks = {"view_0": prior_mask}
        w = np.zeros((1, 100), dtype=np.float32)
        w[0, :5] = 10.0  # very sparse
        model._guide.get_w.return_value = w

        result = enrich_factors(model)
        assert result.fidelity.loc["sparse_0", "gini"] > 0.8


# ---------------------------------------------------------------------------
# Test PCGSE (Block B)
# ---------------------------------------------------------------------------

class TestPCGSE:
    def test_shape_and_range(self):
        """Output has expected shape and p-values in [0, 1]."""
        model = _make_mock_model(n_features=50, n_sparse_factors=0, n_factors=2)

        # Mock FeatureSets
        import pandas as pd
        gene_sets = MagicMock()
        mask_data = np.zeros((3, 50), dtype=bool)
        mask_data[0, :10] = True
        mask_data[1, 10:20] = True
        mask_data[2, 20:30] = True
        mask_df = pd.DataFrame(
            mask_data,
            index=["set_A", "set_B", "set_C"],
            columns=model.feature_names["view_0"],
        )
        gene_sets.to_mask.return_value = mask_df

        result = enrich_factors(model, gene_sets=gene_sets)
        assert result.enrichment is not None
        assert len(result.enrichment) == 2 * 3  # 2 factors * 3 sets
        assert set(result.enrichment.columns) == {
            "factor", "gene_set", "pvalue", "padj", "auroc", "prior_rank"
        }
        assert np.all(result.enrichment["pvalue"].between(0, 1))
        assert np.all(result.enrichment["padj"].between(0, 1))
        assert np.all(result.enrichment["auroc"].between(0, 1))

    def test_strong_signal_low_pvalue(self):
        """Gene set with strongly enriched loadings should get low p-value."""
        model = _make_mock_model(n_features=100, n_sparse_factors=0, n_factors=1)
        # Loadings: huge for first 10 genes
        w = np.zeros((1, 100), dtype=np.float32)
        w[0, :10] = 10.0
        w[0, 10:] = 0.01
        model.get_loadings.return_value = {"view_0": w}
        model._guide.get_w.return_value = w

        import pandas as pd
        gene_sets = MagicMock()
        mask_data = np.zeros((2, 100), dtype=bool)
        mask_data[0, :10] = True   # enriched set
        mask_data[1, 50:60] = True  # non-enriched set
        mask_df = pd.DataFrame(
            mask_data,
            index=["enriched", "control"],
            columns=model.feature_names["view_0"],
        )
        gene_sets.to_mask.return_value = mask_df

        result = enrich_factors(model, gene_sets=gene_sets)
        enriched_row = result.enrichment[
            (result.enrichment["gene_set"] == "enriched")
        ]
        control_row = result.enrichment[
            (result.enrichment["gene_set"] == "control")
        ]
        assert enriched_row["pvalue"].values[0] < 0.01
        assert enriched_row["auroc"].values[0] > 0.9
        assert control_row["pvalue"].values[0] > enriched_row["pvalue"].values[0]


# ---------------------------------------------------------------------------
# Test Temporal Metrics (Block C)
# ---------------------------------------------------------------------------

class TestTemporal:
    def test_basic_output(self):
        """Temporal DataFrame has correct shape and columns."""
        model = _make_mock_model(n_factors=3)
        result = enrich_factors(model)
        assert result.temporal is not None
        assert result.temporal.shape[0] == 3
        assert set(result.temporal.columns) == {
            "zeta", "temporal_strength", "TVR", "lengthscale"
        }

    def test_temporal_strength_complement(self):
        """temporal_strength = 1 - zeta."""
        model = _make_mock_model(n_factors=2)
        zeta = np.array([0.2, 0.8], dtype=np.float32)
        model.get_smoothness.return_value = zeta

        result = enrich_factors(model)
        np.testing.assert_allclose(
            result.temporal["temporal_strength"].values,
            1.0 - zeta,
        )

    def test_tvr_zero_temporal_variance(self):
        """TVR = 0 when factor scores are constant over time."""
        model = _make_mock_model(n_patients=5, n_timepoints=4, n_factors=1,
                                  n_sparse_factors=0)
        # Each patient has constant factor score over time
        z = np.zeros((5, 4, 1), dtype=np.float32)
        for p in range(5):
            z[p, :, 0] = float(p)  # constant per patient, different across patients
        model.get_factors.return_value = z

        result = enrich_factors(model)
        assert result.temporal.loc["dense_0", "TVR"] == pytest.approx(0.0)

    def test_tvr_all_temporal(self):
        """TVR approaches 1 when all variance is temporal."""
        model = _make_mock_model(n_patients=10, n_timepoints=8, n_factors=1,
                                  n_sparse_factors=0)
        # All patients have identical trajectory -> within-patient var = total var
        trajectory = np.sin(np.linspace(0, 2 * np.pi, 8)).astype(np.float32)
        z = np.tile(trajectory, (10, 1))[:, :, np.newaxis]
        model.get_factors.return_value = z

        result = enrich_factors(model)
        assert result.temporal.loc["dense_0", "TVR"] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Test Covariate Association
# ---------------------------------------------------------------------------

class TestCovariate:
    def test_td_identical_trajectories(self):
        """TD = 0 when two groups have identical trajectories."""
        model = _make_mock_model(n_patients=10, n_timepoints=4, n_factors=1,
                                  n_sparse_factors=0, n_covariates=1)
        # Binary covariate
        covs = np.zeros((10, 1), dtype=np.float32)
        covs[:5, 0] = 0.0
        covs[5:, 0] = 1.0
        model.covariates = covs
        # Same trajectory for both groups
        z = np.tile(np.array([1, 2, 3, 4], dtype=np.float32), (10, 1))[:, :, np.newaxis]
        model.get_factors.return_value = z
        # Gamma
        model._guide.mode.return_value = np.array([[0.5]], dtype=np.float32)

        result = enrich_factors(
            model,
            covariate_names=["treatment"],
            covariate_types={"treatment": "categorical"},
        )
        assert result.covariate is not None
        td = result.covariate.loc[
            result.covariate["covariate"] == "treatment", "TD"
        ].values[0]
        assert td == pytest.approx(0.0, abs=1e-10)

    def test_td_diverging_trajectories(self):
        """TD > 0 when groups have diverging trajectories."""
        model = _make_mock_model(n_patients=10, n_timepoints=4, n_factors=1,
                                  n_sparse_factors=0, n_covariates=1)
        covs = np.zeros((10, 1), dtype=np.float32)
        covs[:5, 0] = 0.0
        covs[5:, 0] = 1.0
        model.covariates = covs
        # Different trajectories: group0 flat, group1 increasing
        z = np.zeros((10, 4, 1), dtype=np.float32)
        z[:5, :, 0] = 0.0  # group 0: flat
        z[5:, :, 0] = np.array([0, 1, 2, 3])  # group 1: increasing
        model.get_factors.return_value = z
        model._guide.mode.return_value = np.array([[1.0]], dtype=np.float32)

        result = enrich_factors(
            model,
            covariate_names=["drug"],
            covariate_types={"drug": "categorical"},
        )
        td = result.covariate.loc[
            result.covariate["covariate"] == "drug", "TD"
        ].values[0]
        assert td > 0

    def test_tir_output(self):
        """TIR summary has correct shape."""
        model = _make_mock_model(n_patients=10, n_timepoints=4, n_factors=2,
                                  n_sparse_factors=0, n_covariates=2)
        covs = np.random.randn(10, 2).astype(np.float32)
        model.covariates = covs
        model._guide.mode.return_value = np.random.randn(2, 2).astype(np.float32)

        result = enrich_factors(
            model,
            covariate_names=["sex", "treatment"],
            covariate_types={"sex": "categorical", "treatment": "categorical"},
        )
        assert result.covariate_summary is not None
        assert result.covariate_summary.shape == (2, 1)
        assert "TIR" in result.covariate_summary.columns


# ---------------------------------------------------------------------------
# Test enrich_factors integration
# ---------------------------------------------------------------------------

class TestEnrichFactors:
    def test_untrained_raises(self):
        model = MagicMock()
        model._trained = False
        with pytest.raises(RuntimeError, match="trained"):
            enrich_factors(model)

    def test_dense_only_no_fidelity(self):
        """Dense-only model has no fidelity results."""
        model = _make_mock_model(n_sparse_factors=0, n_factors=2)
        result = enrich_factors(model)
        assert result.fidelity is None
        assert result.temporal is not None

    def test_no_covariates(self):
        """Model without covariates has no covariate results."""
        model = _make_mock_model(n_covariates=0)
        result = enrich_factors(model)
        assert result.covariate is None
        assert result.covariate_summary is None


# ---------------------------------------------------------------------------
# Test EnrichmentResult serialization
# ---------------------------------------------------------------------------

class TestEnrichmentResultSerialization:
    def _make_result(self):
        """Create an EnrichmentResult with all fields populated."""
        model = _make_mock_model(n_patients=10, n_timepoints=4, n_factors=2,
                                  n_sparse_factors=1, n_features=50, n_covariates=1)
        prior_mask = np.zeros((1, 50), dtype=np.float32)
        prior_mask[0, :10] = 1.0
        model.prior_masks = {"view_0": prior_mask}
        covs = np.zeros((10, 1), dtype=np.float32)
        covs[:5, 0] = 0.0
        covs[5:, 0] = 1.0
        model.covariates = covs
        model._guide.mode.return_value = np.array([[0.5, 0.3]], dtype=np.float32)

        return enrich_factors(
            model,
            covariate_names=["treatment"],
            covariate_types={"treatment": "categorical"},
        )

    def test_to_dict(self):
        result = self._make_result()
        d = result.to_dict()
        assert "temporal" in d
        assert "fidelity" in d
        assert "covariate" in d
        assert "covariate_summary" in d
        # enrichment not computed (no gene sets)
        assert "enrichment" not in d

    def test_save_load_roundtrip(self, tmp_path):
        result = self._make_result()
        path = str(tmp_path / "enrichment.pkl")
        result.save(path)
        loaded = EnrichmentResult.load(path)

        # Temporal
        pd.testing.assert_frame_equal(result.temporal, loaded.temporal, check_dtype=False)
        # Fidelity
        pd.testing.assert_frame_equal(result.fidelity, loaded.fidelity, check_dtype=False)
        # Covariate
        pd.testing.assert_frame_equal(result.covariate, loaded.covariate, check_dtype=False)
        pd.testing.assert_frame_equal(result.covariate_summary, loaded.covariate_summary, check_dtype=False)
        # Enrichment was None
        assert loaded.enrichment is None

    def test_to_json(self, tmp_path):
        import json
        result = self._make_result()
        path = str(tmp_path / "enrichment.json")
        result.to_json(path)

        with open(path) as f:
            data = json.load(f)
        assert "temporal" in data
        assert "fidelity" in data
        assert isinstance(data["temporal"], list)
        assert len(data["temporal"]) == 2  # 2 factors
