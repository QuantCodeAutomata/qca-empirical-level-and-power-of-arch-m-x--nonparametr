import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from typing import Tuple


def build_bspline_basis(x: np.ndarray, n_internal_knots: int = 4,
                        order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build cubic B-spline design matrix with n_internal_knots at quantiles.
    # Using scipy.interpolate.BSpline -- Context7 confirmed
    """
    k = order - 1  # degree
    quantile_levels = np.linspace(0, 100, n_internal_knots + 2)[1:-1]
    internal_knots = np.percentile(x, quantile_levels)
    x_min, x_max = x.min(), x.max()
    knots = np.concatenate([np.repeat(x_min, k+1), internal_knots, np.repeat(x_max, k+1)])
    x_clipped = np.clip(x, x_min + 1e-10, x_max - 1e-10)
    Phi = BSpline.design_matrix(x_clipped, knots, k).toarray()
    return Phi, knots


def fit_bspline_null_model(eps2: np.ndarray, x1: np.ndarray,
                           p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit B-spline null model via QMLE.
    sigma^2_t = sum_j alpha_j*eps^2_{t-j} + theta^T * B_1(x_{1,t-1})
    Returns (alpha_hat, m_hat, vhat).
    """
    n = len(eps2)
    Y = eps2[p:]
    Z = np.column_stack([eps2[p-k:n-k] for k in range(1, p+1)])
    x1_lag = x1[p:n]  # Y[j]=eps2[p+j] uses X[p+j]; corrected alignment
    Phi, _ = build_bspline_basis(x1_lag, n_internal_knots=4, order=4)
    K = Phi.shape[1]
    mean_eps2 = np.mean(Y)
    nu0 = np.concatenate([np.full(p, 0.1), np.full(K, mean_eps2 / K)])
    bounds = [(1e-8, None)] * (p + K)

    def sigma2(nu):
        return np.clip(Z @ nu[:p] + Phi @ nu[p:], 1e-10, None)

    def obj(nu):
        s2 = sigma2(nu)
        return np.mean(Y / s2 + np.log(s2))

    res = minimize(obj, nu0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 500, 'ftol': 1e-10})
    if not res.success:
        rng = np.random.default_rng(42)
        best = res
        for _ in range(5):
            nu_p = np.clip(nu0 * np.exp(0.1 * rng.standard_normal(len(nu0))), 1e-8, None)
            r = minimize(obj, nu_p, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 500, 'ftol': 1e-10})
            if r.fun < best.fun:
                best = r
        res = best
    alpha_hat = res.x[:p]
    m_hat = Phi @ res.x[p:]
    vhat = Y - Z @ alpha_hat - m_hat
    return alpha_hat, m_hat, vhat
