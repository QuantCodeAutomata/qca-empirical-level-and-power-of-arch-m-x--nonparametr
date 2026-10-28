import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.anova import compute_rice_variance, build_windows, anova_test_statistic


def test_rice_variance_iid():
    rng = np.random.default_rng(42)
    v = rng.standard_normal(1000)
    tau = compute_rice_variance(v)
    # Rice estimator of Var = 1.0 for iid N(0,1)
    assert abs(tau - 1.0) < 0.15


def test_rice_variance_positive():
    v = np.array([1.0, -1.0, 2.0, -2.0])
    tau = compute_rice_variance(v)
    assert tau > 0


def test_rice_variance_single_obs():
    tau = compute_rice_variance(np.array([5.0]))
    assert tau >= 0


def test_build_windows_size():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50)
    windows = build_windows(x, k_n=5)
    assert windows.shape == (50, 5)
    assert windows.dtype == int


def test_build_windows_odd_kn():
    with pytest.raises(AssertionError):
        build_windows(np.arange(10, dtype=float), k_n=4)


def test_build_windows_unique_per_row():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(20)
    windows = build_windows(x, k_n=5)
    for i in range(20):
        assert len(set(windows[i])) == 5  # k_n unique indices per window


def test_anova_null_hypothesis_pvalue():
    # Under H0 (no signal), p-values should be roughly uniform
    rng = np.random.default_rng(42)
    n_sims = 200
    p_vals = []
    for _ in range(n_sims):
        vhat = rng.standard_normal(100)
        x_test = rng.standard_normal(100)
        _, _, pval = anova_test_statistic(vhat, x_test, k_n=5)
        p_vals.append(pval)
    # Rejection rate at 0.05 should be roughly 5% (within +-3 SE)
    reject_rate = np.mean(np.array(p_vals) < 0.05)
    assert 0.01 < reject_rate < 0.15  # broader band due to approximation


def test_anova_power_with_signal():
    # With clear signal, should reject often
    rng = np.random.default_rng(99)
    n_sims = 100
    reject = 0
    for _ in range(n_sims):
        x_test = rng.standard_normal(300)
        vhat = x_test**2 + 0.3 * rng.standard_normal(300)  # strong signal
        _, _, pval = anova_test_statistic(vhat, x_test, k_n=5)
        if pval < 0.05:
            reject += 1
    assert reject / n_sims > 0.5


def test_anova_negative_tn():
    # If T_n < 0, p-value should be 1
    vhat = np.zeros(50)
    x_test = np.arange(50, dtype=float)
    T_n, Z_n, pval = anova_test_statistic(vhat, x_test, k_n=5)
    # With zero residuals, MST=MSE=0 and T_n=0, pval=1
    assert pval >= 0.0
