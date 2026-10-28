import numpy as np
from scipy.stats import norm, laplace
from typing import Tuple, Optional


def generate_innovations(n: int, dist: str, rng: np.random.Generator) -> np.ndarray:
    """Generate n standardized innovations. dist in {'normal','laplace','t7'}."""
    if dist == 'normal':
        return rng.standard_normal(n)
    elif dist == 'laplace':
        return rng.laplace(0.0, 1.0, n) / np.sqrt(2.0)
    elif dist == 't7':
        return rng.standard_t(7, n) / np.sqrt(7.0 / 5.0)
    else:
        raise ValueError(f'Unknown distribution: {dist}')


def generate_covariates(n: int, d: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Generate n covariate vectors from MVN((2.5,...), Sigma) where Sigma_ij=rho^|i-j|."""
    mu = np.full(d, 2.5)
    i_idx, j_idx = np.meshgrid(np.arange(d), np.arange(d), indexing='ij')
    Sigma = rho ** np.abs(i_idx - j_idx).astype(float)
    return rng.multivariate_normal(mu, Sigma, size=n)


def compute_sigma2_2cov(model: int, eps_lag1: float, eps_lag2: float,
                        x1: float, x2: float, c: float) -> float:
    """Conditional variance for 2-covariate DGP models (exp_1). Testing H0: X2 not significant."""
    if model == 1:
        return 0.2 + 0.3*eps_lag1**2 + 0.5*(x1-2.5)**2 + 2.0*c*(x2-2.5)**2
    elif model == 2:
        return 0.2 + 0.3*eps_lag1**2 + 0.5*(x1 + 2.0*c*x2 - 10.0)**2
    elif model == 3:
        return 0.2 + 0.3*eps_lag1**2 + 0.2*eps_lag2**2 + 0.5*(x1-2.5)**2 + 2.0*c*(x2-2.5)**2
    elif model == 4:
        return 0.2 + 0.3*eps_lag1**2 + 0.2*eps_lag2**2 + 0.5*(x1 + 2.0*c*x2 - 10.0)**2
    elif model == 5:
        return 0.2 + 0.4*eps_lag1**2 + np.sin(np.pi*x1/2)**2 + 10.0*c*np.sin(np.pi*x2/2)**2
    elif model == 6:
        return 0.2 + 0.4*eps_lag1**2 + 10.0*c*np.sin(x1/2 + np.pi*c*x2/2)**2
    elif model == 7:
        return max(0.2 + 0.4*eps_lag1**2 + 0.25*x1 + c*x2, 1e-10)
    elif model == 8:
        return max(0.2 + 0.4*eps_lag1**2 + c*x1*x2, 1e-10)
    else:
        raise ValueError(f'Unknown model: {model}')


def compute_sigma2_5cov(model: int, eps_lag1: float, eps_lag2: float,
                        X: np.ndarray, c: float) -> float:
    """Conditional variance for 5-covariate DGP (exp_2). Testing H0: X3 not significant."""
    x1, x2, x3, x4, x5 = X
    quad_others = 0.5*((x1-2.5)**2 + (x2-2.5)**2 + (x4-2.5)**2 + (x5-2.5)**2)
    if model == 1:
        return 0.2 + 0.3*eps_lag1**2 + quad_others + 5.0*c*(x3-2.5)**2
    elif model == 2:
        return 0.2 + 0.3*eps_lag1**2 + 0.5*(x1+x2+5.0*c*x3+x4+x5-25.0)**2
    elif model == 3:
        return 0.2 + 0.3*eps_lag1**2 + 0.2*eps_lag2**2 + quad_others + 5.0*c*(x3-2.5)**2
    elif model == 4:
        return 0.2 + 0.3*eps_lag1**2 + 0.2*eps_lag2**2 + 0.5*(x1+x2+5.0*c*x3+x4+x5-25.0)**2
    elif model == 5:
        sin_others = np.sin(np.pi*x1/2) + np.sin(np.pi*x2/2) + np.sin(np.pi*x4/2) + np.sin(np.pi*x5/2)
        return 0.2 + 0.4*eps_lag1**2 + sin_others + 10.0*c*np.sin(np.pi*x3/2)
    elif model == 6:
        return (0.2 + 0.4*eps_lag1**2 + np.sin(np.pi*x1/2)**2*np.sin(np.pi*x2/2)**2
                + 10.0*c*np.sin(np.pi*x3/2)**2*np.sin(np.pi*x4/2)**2 + np.sin(np.pi*x5/2)**2)
    elif model == 7:
        return max(0.2 + 0.4*eps_lag1**2 + 0.25*(x1+x2+x4+x5) + 8.0*c*x3, 1e-10)
    elif model == 8:
        return max(0.2 + 0.4*eps_lag1**2 + c*x3*(x1+x2+x4+x5), 1e-10)
    else:
        raise ValueError(f'Unknown model: {model}')


def compute_sigma2_5cov_selection(model: int, eps_lag1: float, eps_lag2: float,
                                   X: np.ndarray) -> float:
    """Conditional variance for 5-cov variable selection DGP (exp_3). True active={1,3,4}."""
    x1, x2, x3, x4, x5 = X
    arch = 0.3*eps_lag1**2 + (0.2*eps_lag2**2 if model in [3, 4] else 0.0)
    if model == 1:
        m = 2*(x1-2.5)**2 + 2*(x3-2.5)**2 + 2*(x4-2.5)**2
    elif model == 2:
        m = 0.5*(2*x1 + 2*x3 + 2*x4 - 15)**2
    elif model == 3:
        m = 2*(x1-2.5)**2 + 2*(x3-2.5)**2 + 2*(x4-2.5)**2
    elif model == 4:
        m = 0.5*(2*x1 + 2*x3 + 2*x4 - 15)**2
    elif model == 5:
        m = 8*np.sin(np.pi*x1/2)**2 + 8*np.sin(np.pi*x3/2)**2 + 8*np.sin(np.pi*x4/2)**2
    elif model == 6:
        m = 10*np.sin(np.pi*x1/2)**2*np.sin(np.pi*x3/2)**2 + 10*np.sin(np.pi*x4/2)**2
    elif model == 7:
        m = 8*x1 + 8*x3 + 8*x4
    elif model == 8:
        m = x3*(x1 + x4)
    else:
        raise ValueError(f'Unknown model: {model}')
    return arch + max(m, 1e-6)


def compute_sigma2_10cov_selection(model: int, eps_lag1: float, eps_lag2: float,
                                    X: np.ndarray) -> float:
    """Conditional variance for 10-cov variable selection DGP (exp_4). True active={1,3,4,5,9}."""
    x1,x2,x3,x4,x5,x6,x7,x8,x9,x10 = X
    arch = 0.3*eps_lag1**2 + (0.2*eps_lag2**2 if model in [3, 4] else 0.0)
    if model == 1:
        m = 2*(x1-2.5)**2 + 2*(x3-2.5)**2 + 2*(x4-2.5)**2 + 2*(x5-2.5)**2 + 2*(x9-2.5)**2
    elif model == 2:
        m = 0.5*(2*x1 + 2*x3 + 2*x4 + 2*x5 + 2*x9 - 25)**2
    elif model == 3:
        m = 2*(x1-2.5)**2 + 2*(x3-2.5)**2 + 2*(x4-2.5)**2 + 2*(x5-2.5)**2 + 2*(x9-2.5)**2
    elif model == 4:
        m = 0.5*(2*x1 + 2*x3 + 2*x4 + 2*x5 + 2*x9 - 15)**2
    elif model == 5:
        m = (8*np.sin(np.pi*x1/2)**2 + 8*np.sin(np.pi*x3/2)**2 + 8*np.sin(np.pi*x4/2)**2
             + 8*np.sin(np.pi*x5/2)**2 + 8*np.sin(np.pi*x9/2)**2)
    elif model == 6:
        m = (10*np.sin(np.pi*x1/2)**2*np.sin(np.pi*x3/2)**2
             + 10*np.sin(np.pi*x4/2)**2*np.sin(np.pi*x5/2)**2 + 10*np.sin(np.pi*x9/2)**2)
    elif model == 7:
        m = 8*x1 + 8*x3 + 8*x4 + 8*x5 + 8*x9
    elif model == 8:
        m = (x1+x4)*x3 + x5*x9
    else:
        raise ValueError(f'Unknown model: {model}')
    return arch + max(m, 1e-6)


def generate_arch_series(n: int, p: int, X_all: np.ndarray, z_all: np.ndarray,
                         sigma2_fn, burn_in: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Generate ARCH-m(X) time series with burn-in."""
    total = n + burn_in
    eps_all = np.zeros(total + p)
    for t in range(total):
        eps_lags = eps_all[t+p-1:max(t+p-1-p, -1):-1]  # [eps_{t-1},...,eps_{t-p}]
        sigma2 = sigma2_fn(t, eps_lags, X_all[t])
        sigma2 = max(float(sigma2), 1e-10)
        eps_all[t + p] = z_all[t] * np.sqrt(sigma2)
    eps = eps_all[p + burn_in:]
    X = X_all[burn_in:]
    return eps, X
