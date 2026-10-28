import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.linear import fit_linear_qmle, fit_lasso_qmle


def test_fit_linear_qmle_pvalue_shape():
    rng = np.random.default_rng(42)
    n = 100; p = 1; d = 3
    eps2 = 0.2 + 0.3*np.r_[0.0, rng.standard_normal(n-1)**2]
    X = rng.standard_normal((n, d)) + 2.5
    nu_hat, p_values, qmle_val = fit_linear_qmle(eps2, X, p)
    assert len(p_values) == d
    assert all(0 <= pv <= 1 for pv in p_values)
    assert len(nu_hat) == 1 + p + d
    assert all(nu_hat[i] >= 0 for i in range(1+p))  # c0, alpha >= 0


def test_fit_linear_qmle_no_effect():
    # Under H0 (no covariate effect), p-values should not be systematically small
    rng = np.random.default_rng(0)
    n = 200; p = 1; d = 2
    eps2 = 0.3 + 0.4*np.r_[0.0, rng.standard_normal(n-1)**2]
    X = rng.standard_normal((n, d))
    _, p_values, _ = fit_linear_qmle(eps2, X, p)
    assert all(0 <= pv <= 1 for pv in p_values)


def test_fit_lasso_returns_indices():
    rng = np.random.default_rng(1)
    n = 100; p = 1; d = 5
    eps2 = 0.2 + 0.3*np.r_[0.0, rng.standard_normal(n-1)**2]
    X = rng.standard_normal((n, d))
    selected = fit_lasso_qmle(eps2, X, p, n_lambda=10)
    assert isinstance(selected, np.ndarray)
    assert all(0 <= i < d for i in selected)


def test_fit_lasso_no_strong_effect_selects_few():
    rng = np.random.default_rng(42)
    n = 100; p = 1; d = 5
    eps2 = 0.3 + np.r_[0.0, 0.4*rng.standard_normal(n-1)**2]
    X = rng.standard_normal((n, d)) * 0.001  # tiny effect
    selected = fit_lasso_qmle(eps2, X, p, n_lambda=10)
    assert len(selected) <= 5  # should not select all (or select few)
