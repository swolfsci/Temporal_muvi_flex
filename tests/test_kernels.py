"""Tests for GP kernel functions."""

import pytest
import torch
from tpacmon.core.kernels import matern32, matern52, rbf, build_kernel, KERNEL_REGISTRY


@pytest.fixture
def time_points():
    return torch.tensor([0.0, 1.0, 3.0, 7.0])


@pytest.fixture
def ls():
    return torch.tensor(2.0)


@pytest.fixture
def amp():
    return torch.tensor(1.0)


class TestKernelProperties:
    """Test basic kernel matrix properties."""

    @pytest.mark.parametrize("kernel_fn", [matern32, matern52, rbf])
    def test_shape(self, kernel_fn, time_points, ls, amp):
        K = kernel_fn(time_points, time_points, ls, amp)
        assert K.shape == (4, 4)

    @pytest.mark.parametrize("kernel_fn", [matern32, matern52, rbf])
    def test_symmetry(self, kernel_fn, time_points, ls, amp):
        K = kernel_fn(time_points, time_points, ls, amp)
        assert torch.allclose(K, K.T, atol=1e-6)

    @pytest.mark.parametrize("kernel_fn", [matern32, matern52, rbf])
    def test_positive_definite(self, kernel_fn, time_points, ls, amp):
        K = kernel_fn(time_points, time_points, ls, amp)
        K = K + 1e-5 * torch.eye(4)
        eigvals = torch.linalg.eigvalsh(K)
        assert (eigvals > 0).all()

    @pytest.mark.parametrize("kernel_fn", [matern32, matern52, rbf])
    def test_diagonal_is_amplitude_squared(self, kernel_fn, ls):
        t = torch.tensor([0.0, 1.0, 2.0])
        amp = torch.tensor(2.5)
        K = kernel_fn(t, t, ls, amp)
        expected_diag = amp.square()
        assert torch.allclose(torch.diag(K), expected_diag.expand(3), atol=1e-5)


class TestBuildKernel:
    def test_jitter(self, time_points, ls, amp):
        K = build_kernel("matern32", time_points, ls, amp, jitter=0.01)
        K_no_jitter = build_kernel("matern32", time_points, ls, amp, jitter=0.0)
        diff = torch.diag(K - K_no_jitter)
        assert torch.allclose(diff, torch.tensor(0.01).expand(4), atol=1e-6)

    def test_cross_covariance(self, ls, amp):
        t1 = torch.tensor([0.0, 1.0])
        t2 = torch.tensor([0.5, 1.5, 2.5])
        K = build_kernel("matern32", t1, ls, amp, jitter=0.0, t_obs2=t2)
        assert K.shape == (2, 3)

    def test_unknown_kernel(self, time_points, ls, amp):
        with pytest.raises(ValueError, match="Unknown kernel"):
            build_kernel("invalid", time_points, ls, amp)

    def test_registry_keys(self):
        assert set(KERNEL_REGISTRY.keys()) == {"matern32", "matern52", "rbf"}
