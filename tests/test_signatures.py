"""Tests for predictive signatures (requires a mock trained model)."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from tpacmon.tools.signatures import _aggregate_factors, _aggregate_features


class TestAggregateFactors:
    def test_mean_aggregation(self):
        z = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # (1, 3, 2)
        masks = np.array([[True, True, True]])
        result = _aggregate_factors(z, masks, "mean")
        np.testing.assert_allclose(result[0], [3.0, 4.0])

    def test_last_aggregation(self):
        z = np.array([[[1.0], [3.0], [5.0]]])
        masks = np.array([[True, True, True]])
        result = _aggregate_factors(z, masks, "last")
        np.testing.assert_allclose(result[0], [5.0])

    def test_respects_mask(self):
        z = np.array([[[1.0], [3.0], [999.0]]])  # last is unobserved
        masks = np.array([[True, True, False]])
        result = _aggregate_factors(z, masks, "mean")
        np.testing.assert_allclose(result[0], [2.0])

    def test_slope_aggregation(self):
        z = np.array([[[0.0], [1.0], [2.0]]])  # linear
        masks = np.array([[True, True, True]])
        result = _aggregate_factors(z, masks, "slope")
        assert abs(result[0, 0] - 1.0) < 0.01


class TestAggregateFeatures:
    def test_mean(self):
        obs = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        masks = np.array([[True, True]])
        result = _aggregate_features(obs, masks, "mean")
        np.testing.assert_allclose(result[0], [2.0, 3.0])
