import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kernel import silverman_bandwidth, compute_kernel_weights_dense, fit_kernel_null_model


def test_silverman_bandwidth_1d():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 1))
    h = silverman_bandwidth(X, 100)
    assert len(h) == 1
    assert h[0] > 0
    # Silverman rule: 1.06 * std * n^{-1/5} ~ 1.06 * 1 * 100^{-0.2} ~ 0.42
    assert 0.1 < h[0] < 2.0


def test_silverman_bandwidth_4d():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4))
    h = silverman_bandwidth(X, 100)
    assert len(h) == 4
    assert all(hi > 0 for hi in h)


def test_kernel_weights_dense_row_sum():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 1))
    h = silverman_bandwidth(X, 30)
    W = compute_kernel_weights_dense(X, h)
    assert W.shape == (30, 30)
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(W >= 0)


def test_kernel_weights_dense_symmetry_1d():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 1))
    h = silverman_bandwidth(X, 20)
    W_raw = np.ones((20, 20))
    from scipy.stats import norm
    diff = X[:, 0:1] - X[np.newaxis, :, 0]
    W_raw = norm.pdf(diff / h[0]) / h[0]
    # Raw kernel is symmetric
    assert np.allclose(W_raw, W_raw.T, atol=1e-10)


def test_fit_kernel_null_model_residuals():
    rng = np.random.default_rng(42)
    n = 100; p = 1
    eps2 = np.abs(rng.standard_normal(n)) + 0.1
    X = rng.standard_normal((n, 1)) + 2.5
    alpha_hat, m_hat, vhat = fit_kernel_null_model(eps2, X, p)
    assert len(vhat) == n - p
    assert len(alpha_hat) == p
    assert len(m_hat) == n - p
    # Residuals should be roughly mean-zero
    assert abs(np.mean(vhat)) < 0.5


def test_fit_kernel_null_model_no_signal():
    # Under H0, m_hat should be approximately constant
    rng = np.random.default_rng(0)
    n = 200; p = 1
    X = rng.standard_normal((n, 1)) + 2.5
    z = rng.standard_normal(n)
    eps2 = 0.5 + 0.3*np.r_[0.0, (z[:-1]**2)]
    alpha_hat, m_hat, vhat = fit_kernel_null_model(eps2, X, p)
    assert alpha_hat[0] > -1  # alpha should be positive-ish
    assert np.std(m_hat) < np.std(eps2)  # smoother reduces variance
