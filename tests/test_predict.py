"""Tests for GP prediction logic (unit-level, no SVI)."""

import numpy as np
import pytest
import torch

from tpacmon.core.kernels import build_kernel


class TestGPConditional:
    """Test the GP conditional prediction math directly."""

    def test_interpolation_recovers_observed(self):
        """Predicting at observed points should recover the observations."""
        t_obs = torch.tensor([0.0, 2.0, 5.0, 8.0])
        ls = torch.tensor(3.0)
        amp = torch.tensor(1.0)

        K = build_kernel("matern32", t_obs, ls, amp, jitter=1e-5)
        z_obs = torch.tensor([1.0, -0.5, 0.3, 0.8])

        K_inv = torch.linalg.solve(K, torch.eye(4))
        K_star = build_kernel("matern32", t_obs, ls, amp, jitter=0.0, t_obs2=t_obs)

        mu_pred = K_star @ K_inv @ z_obs
        torch.testing.assert_close(mu_pred, z_obs, atol=1e-3, rtol=1e-3)

    def test_extrapolation_uncertainty_increases(self):
        """Variance should increase for points far from observations."""
        t_obs = torch.tensor([0.0, 1.0, 2.0])
        t_new = torch.tensor([3.0, 10.0, 50.0])
        ls = torch.tensor(2.0)
        amp = torch.tensor(1.0)

        K_obs = build_kernel("matern32", t_obs, ls, amp, jitter=1e-5)
        K_new = build_kernel("matern32", t_new, ls, amp, jitter=1e-5)
        K_cross = build_kernel("matern32", t_new, ls, amp, jitter=0.0, t_obs2=t_obs)

        K_inv = torch.linalg.solve(K_obs, torch.eye(3))
        var = torch.diag(K_new - K_cross @ K_inv @ K_cross.T).clamp(min=0)

        # Variance should generally increase with distance from obs
        assert var[2] > var[0]

    def test_cross_covariance_shape(self):
        t1 = torch.tensor([0.0, 1.0, 2.0])
        t2 = torch.tensor([0.5, 1.5])
        ls = torch.tensor(1.0)
        amp = torch.tensor(1.0)
        K = build_kernel("matern32", t1, ls, amp, t_obs2=t2)
        assert K.shape == (3, 2)
