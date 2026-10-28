import numpy as np
from scipy.stats import norm, rankdata
from typing import Tuple


def compute_rice_variance(vhat: np.ndarray) -> float:
    """Rice variance estimator: tau = sum((diff(vhat))^2) / (2*(N-1))."""
    N = len(vhat)
    if N < 2:
        return max(float(np.var(vhat)), 1e-20)
    tau = np.sum(np.diff(vhat)**2) / (2.0 * (N - 1))
    return max(float(tau), 1e-20)


def build_windows(x_vals: np.ndarray, k_n: int) -> np.ndarray:
    """
    Build (N, k_n) window membership array indexed by rank of x_vals.
    Each row i contains k_n original indices with ranks nearest to rank(x_vals[i]).
    # Custom -- Context7 found no library equivalent (paper Eq. ANOVA)
    """
    N = len(x_vals)
    assert k_n % 2 == 1, 'k_n must be odd'
    half = k_n // 2
    ranks = rankdata(x_vals, method='ordinal').astype(int)  # 1-based
    sorted_orig_idx = np.argsort(ranks - 1)  # sorted_orig_idx[r-1] = original idx with rank r
    window_members = np.zeros((N, k_n), dtype=int)
    for i in range(N):
        r = ranks[i]
        if r <= half:
            win_ranks = np.arange(1, k_n + 1)
        elif r > N - half:
            win_ranks = np.arange(N - k_n + 1, N + 1)
        else:
            win_ranks = np.arange(r - half, r + half + 1)
        window_members[i] = sorted_orig_idx[win_ranks - 1]
    return window_members


def anova_test_statistic(vhat: np.ndarray, x_test: np.ndarray,
                         k_n: int = 5) -> Tuple[float, float, float]:
    """
    Compute T_n, Z_n, p_value for ANOVA test on x_test covariate.
    # Custom -- Context7 found no library equivalent (paper methodology)
    """
    N = len(vhat)
    tau_hat = compute_rice_variance(vhat)
    window_members = build_windows(x_test, k_n)
    vhat_windows = vhat[window_members]  # (N, k_n)
    vhat_dot = vhat_windows.mean(axis=1)  # (N,)
    vhat_dotdot = vhat_dot.mean()
    MST = (k_n / (N - 1)) * np.sum((vhat_dot - vhat_dotdot)**2)
    MSE = np.sum((vhat_windows - vhat_dot[:, np.newaxis])**2) / (N * (k_n - 1))
    T_n = MST - MSE
    if T_n <= 0:
        return float(T_n), -np.inf, 1.0
    normalizer = np.sqrt(4.0 * tau_hat**2 / 3.0)
    if normalizer < 1e-20:
        return float(T_n), np.inf, 0.0
    Z_n = np.sqrt(N / k_n) * T_n / normalizer
    p_value = 1.0 - norm.cdf(Z_n)
    return float(T_n), float(Z_n), float(p_value)
