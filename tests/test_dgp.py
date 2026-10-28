import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dgp import (
    generate_innovations, generate_covariates, compute_sigma2_2cov,
    compute_sigma2_5cov, compute_sigma2_5cov_selection, compute_sigma2_10cov_selection,
    generate_arch_series
)


def test_innovations_normal_variance():
    rng = np.random.default_rng(42)
    z = generate_innovations(10000, 'normal', rng)
    assert abs(np.var(z) - 1.0) < 0.05
    assert abs(np.mean(z)) < 0.05


def test_innovations_laplace_variance():
    rng = np.random.default_rng(42)
    z = generate_innovations(10000, 'laplace', rng)
    assert abs(np.var(z) - 1.0) < 0.1


def test_innovations_t7_variance():
    rng = np.random.default_rng(42)
    z = generate_innovations(10000, 't7', rng)
    assert abs(np.var(z) - 1.0) < 0.1


def test_innovations_invalid():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        generate_innovations(10, 'unknown', rng)


def test_covariates_shape_and_mean():
    rng = np.random.default_rng(42)
    X = generate_covariates(500, d=5, rho=0.0, rng=rng)
    assert X.shape == (500, 5)
    assert abs(np.mean(X) - 2.5) < 0.2


def test_covariates_correlation_rho0():
    rng = np.random.default_rng(42)
    X = generate_covariates(10000, d=2, rho=0.0, rng=rng)
    corr = np.corrcoef(X[:, 0], X[:, 1])[0, 1]
    assert abs(corr) < 0.05


def test_covariates_correlation_rho05():
    rng = np.random.default_rng(42)
    X = generate_covariates(10000, d=2, rho=0.5, rng=rng)
    corr = np.corrcoef(X[:, 0], X[:, 1])[0, 1]
    assert abs(corr - 0.5) < 0.05


def test_sigma2_2cov_positive():
    for model in range(1, 9):
        for c in [0.0, 0.5, 1.0]:
            s2 = compute_sigma2_2cov(model, 0.5, 0.3, 2.5, 2.5, c)
            assert s2 > 0, f'Model {model}, c={c}: sigma2={s2}'


def test_sigma2_2cov_null_hypothesis():
    # Under H0 (c=0), X2 should not appear in models 1,3,5
    s2_x2_low = compute_sigma2_2cov(1, 0.0, 0.0, 2.5, 1.0, 0.0)
    s2_x2_high = compute_sigma2_2cov(1, 0.0, 0.0, 2.5, 5.0, 0.0)
    assert abs(s2_x2_low - s2_x2_high) < 1e-10  # X2 irrelevant under H0


def test_sigma2_5cov_positive():
    # Use x=1.0: sin(pi/2)=1 keeps model 5's un-squared sin formula positive
    X = np.full(5, 1.0)
    for model in range(1, 9):
        s2 = compute_sigma2_5cov(model, 0.3, 0.2, X, c=0.5)
        assert s2 > 0, f'Model {model}: sigma2={s2}'


def test_sigma2_5cov_selection_positive():
    X = np.full(5, 2.5)
    for model in range(1, 9):
        s2 = compute_sigma2_5cov_selection(model, 0.3, 0.2, X)
        assert s2 > 0


def test_sigma2_10cov_selection_positive():
    X = np.full(10, 2.5)
    for model in range(1, 9):
        s2 = compute_sigma2_10cov_selection(model, 0.3, 0.2, X)
        assert s2 > 0


def test_generate_arch_series_shape():
    rng = np.random.default_rng(0)
    n = 100; p = 1
    X_all = generate_covariates(n + 500, d=2, rho=0.0, rng=rng)
    z_all = generate_innovations(n + 500, dist='normal', rng=rng)
    from src.dgp import compute_sigma2_2cov
    def s2fn(t, el, X_t):
        return compute_sigma2_2cov(1, float(el[0]) if len(el) > 0 else 0.0, 0.0, X_t[0], X_t[1], 0.5)
    eps, X = generate_arch_series(n, p, X_all, z_all, s2fn)
    assert len(eps) == n
    assert X.shape == (n, 2)
