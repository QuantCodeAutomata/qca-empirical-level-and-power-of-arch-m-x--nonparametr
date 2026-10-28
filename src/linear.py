import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from typing import Tuple, Optional


def fit_linear_qmle(eps2: np.ndarray, X: np.ndarray, p: int,
                    test_covariate_idx: Optional[int] = None
                    ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit linear ARCH-m(X) via QMLE.
    sigma^2_t = c0 + sum_j alpha_j*eps^2_{t-j} + sum_l beta_l*x_{l,t-1}
    Returns (nu_hat, p_values_for_betas, qmle_val).
    # Custom -- Context7 found no library equivalent for ARCH-m(X) QMLE
    """
    n = len(eps2)
    N = n - p
    d = X.shape[1]
    Y = eps2[p:]
    Z_arch = np.column_stack([eps2[p-k:n-k] for k in range(1, p+1)])
    X_lag = X[p:n]  # Y[j]=eps2[p+j] uses X[p+j]; corrected alignment
    n_params = 1 + p + d
    feat = np.column_stack([np.ones(N), Z_arch, X_lag])

    def s2(nu): return np.clip(feat @ nu, 1e-10, None)
    def obj(nu): s = s2(nu); return np.mean(Y/s + np.log(s))
    def grad(nu):
        s = s2(nu)
        g = (-Y/s**2 + 1.0/s)
        return (feat * g[:, np.newaxis]).mean(axis=0)

    mean_y = np.mean(Y)
    # c0 and alpha_j must be positive; beta_l are unconstrained (log returns can be negative)
    nu0 = np.array([mean_y * 0.5] + [0.1] * p + [0.0] * d)
    bounds = [(1e-8, None)] * (1 + p) + [(None, None)] * d
    best_val, best_nu = np.inf, nu0.copy()
    for seed in range(5):
        rng = np.random.default_rng(seed)
        ns = nu0 if seed == 0 else (
            np.concatenate([
                np.clip(nu0[:1+p] * np.exp(0.3 * rng.standard_normal(1+p)), 1e-8, None),
                rng.standard_normal(d) * 0.01
            ])
        )
        r = minimize(obj, ns, jac=grad, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 1000, 'ftol': 1e-14})
        if r.fun < best_val:
            best_val, best_nu = r.fun, r.x

    nu_hat = best_nu
    # Numerical Hessian for sandwich variance
    eps_fd = 1e-5
    A_hat = np.zeros((n_params, n_params))
    for i in range(n_params):
        np_i = nu_hat.copy(); nm_i = nu_hat.copy()
        np_i[i] += eps_fd; nm_i[i] -= eps_fd
        A_hat[i] = (grad(np_i) - grad(nm_i)) / (2*eps_fd)
    A_hat = 0.5*(A_hat + A_hat.T)
    scores = feat * ((-Y/s2(nu_hat)**2 + 1.0/s2(nu_hat)))[:, np.newaxis]
    B_hat = scores.T @ scores / N
    try:
        A_inv = np.linalg.inv(A_hat + 1e-10*np.eye(n_params))
        V_hat = A_inv @ B_hat @ A_inv / N
    except np.linalg.LinAlgError:
        V_hat = np.eye(n_params)
    p_values = np.ones(d)
    for l in range(d):
        idx = 1 + p + l
        var_l = V_hat[idx, idx]
        if var_l > 1e-15:
            p_values[l] = 1.0 - chi2.cdf(nu_hat[idx]**2 / var_l, df=1)
    return nu_hat, p_values, best_val


def fit_lasso_qmle(eps2: np.ndarray, X: np.ndarray, p: int,
                   n_lambda: int = 25) -> np.ndarray:
    """
    LASSO-penalized QMLE with BIC lambda selection.
    Returns selected covariate indices (0-based).
    # Custom -- Context7 found no library equivalent
    """
    n = len(eps2)
    N = n - p
    d = X.shape[1]
    Y = eps2[p:]
    Z_arch = np.column_stack([eps2[p-k:n-k] for k in range(1, p+1)])
    X_lag = X[p:n]  # Y[j]=eps2[p+j] uses X[p+j]; corrected alignment
    n_params = 1 + p + d
    feat_base = np.column_stack([np.ones(N), Z_arch, X_lag])

    def s2(nu): return np.clip(feat_base @ nu, 1e-10, None)
    def qobj(nu): s = s2(nu); return np.mean(Y/s + np.log(s))
    def qgrad(nu):
        s = s2(nu)
        return (feat_base * ((-Y/s**2 + 1.0/s))[:, np.newaxis]).mean(axis=0)

    mean_y = np.mean(Y)
    nu0 = np.array([mean_y * 0.5] + [0.1] * p + [0.0] * d)
    bounds_base = [(1e-8, None)] * (1 + p) + [(None, None)] * d
    res0 = minimize(qobj, nu0, jac=qgrad, method='L-BFGS-B', bounds=bounds_base,
                    options={'maxiter': 300})
    nu_unpen = res0.x
    nu_null = nu_unpen.copy(); nu_null[1+p:] = 0.0
    g_at_null = np.abs(qgrad(nu_null)[1+p:])
    lam_max = max(np.max(g_at_null), 1e-6)
    lambdas = np.exp(np.linspace(np.log(lam_max), np.log(1e-4*lam_max), n_lambda))
    best_bic, best_sel = np.inf, np.array([], dtype=int)
    nu_warm = nu_unpen.copy()
    smooth_eps = 1e-6
    for lam in lambdas:
        def pobj(nu):
            beta = nu[1+p:]
            return qobj(nu) + lam*np.sum(np.sqrt(beta**2 + smooth_eps**2) - smooth_eps)
        def pgrad(nu):
            g = qgrad(nu).copy()
            beta = nu[1+p:]
            g[1+p:] += lam * beta / np.sqrt(beta**2 + smooth_eps**2)
            return g
        res = minimize(pobj, nu_warm, jac=pgrad, method='L-BFGS-B',
                       bounds=bounds_base, options={'maxiter': 150})
        nu_warm = res.x.copy()
        beta_hat = res.x[1+p:]
        sel = np.where(np.abs(beta_hat) > 1e-4)[0]
        bic = qobj(res.x) + np.log(N)*len(sel)/N
        if bic < best_bic:
            best_bic, best_sel = bic, sel
    return best_sel
