import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.bspline import build_bspline_basis, fit_bspline_null_model


def test_bspline_basis_shape():
    x = np.linspace(1, 5, 100)
    Phi, knots = build_bspline_basis(x, n_internal_knots=4, order=4)
    assert Phi.shape == (100, 8)  # K = n_internal + order = 4 + 4 = 8
    assert len(knots) == 12  # K + order = 8 + 4 = 12


def test_bspline_basis_nonnegative():
    x = np.linspace(1, 5, 50)
    Phi, _ = build_bspline_basis(x)
    assert np.all(Phi >= 0)


def test_bspline_basis_partition_unity():
    x = np.linspace(1, 5, 50)
    Phi, _ = build_bspline_basis(x)
    row_sums = Phi.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_fit_bspline_null_model():
    rng = np.random.default_rng(42)
    n = 100; p = 1
    x1 = rng.uniform(1, 5, n)
    eps2 = 0.5 + 0.3*np.r_[0.0, (rng.standard_normal(n-1)**2)]
    alpha_hat, m_hat, vhat = fit_bspline_null_model(eps2, x1, p)
    assert len(alpha_hat) == p
    assert len(m_hat) == n - p
    assert len(vhat) == n - p
    assert all(a >= 0 for a in alpha_hat)
    assert all(m >= 0 for m in m_hat)


def test_fit_bspline_alpha_positive():
    rng = np.random.default_rng(0)
    n = 150; p = 1
    x1 = rng.standard_normal(n) + 2.5
    eps2 = 0.2 + 0.4*np.r_[0.0, rng.standard_normal(n-1)**2]
    alpha_hat, _, _ = fit_bspline_null_model(eps2, x1, p)
    assert alpha_hat[0] >= 0  # positivity constraint
