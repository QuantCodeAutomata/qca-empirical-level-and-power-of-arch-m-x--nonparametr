import numpy as np
from scipy.stats import norm
from typing import Tuple


def silverman_bandwidth(X: np.ndarray, n_obs: int) -> np.ndarray:
    """Silverman bandwidth: h_j = 1.06*std_j*n^{-1/(d+4)} where d=X.shape[1]."""
    d = X.shape[1]
    exponent = -1.0 / (d + 4)
    stds = np.std(X, axis=0, ddof=1)
    stds = np.where(stds < 1e-10, 1.0, stds)
    return 1.06 * stds * (n_obs ** exponent)


def compute_kernel_weights_dense(X: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Row-normalized (N,N) product Gaussian kernel weight matrix."""
    N, d = X.shape
    W_raw = np.ones((N, N))
    for j in range(d):
        diff_j = X[:, j:j+1] - X[np.newaxis, :, j]  # (N,N)
        W_raw *= norm.pdf(diff_j / h[j]) / h[j]
    row_sums = W_raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-300, 1.0, row_sums)
    W = W_raw / row_sums
    return W


def compute_kernel_weights_sparse(X: np.ndarray, h: np.ndarray,
                                   k_nn: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Sparse kernel weights using k nearest neighbors."""
    from sklearn.neighbors import NearestNeighbors
    N = X.shape[0]
    k_nn = min(k_nn, N)
    X_std = X / h[np.newaxis, :]
    nbrs = NearestNeighbors(n_neighbors=k_nn, metric='euclidean', algorithm='auto')
    nbrs.fit(X_std)
    distances, indices = nbrs.kneighbors(X_std)
    weights_raw = np.exp(-0.5 * distances**2)
    row_sums = weights_raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-300, 1.0, row_sums)
    return indices, weights_raw / row_sums


def apply_smoother_sparse(indices: np.ndarray, weights: np.ndarray,
                          v: np.ndarray) -> np.ndarray:
    """Apply sparse smoother: W @ v using stored indices and weights."""
    return (weights * v[indices]).sum(axis=1)


def fit_kernel_null_model(eps2: np.ndarray, X_minus_l: np.ndarray, p: int,
                          use_sparse: bool = False,
                          k_nn: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit kernel null model for ARCH-m(X) test.
    Returns (alpha_hat, m_hat, vhat).
    """
    n = len(eps2)
    N = n - p
    Y = eps2[p:]
    Z = np.column_stack([eps2[p-k:n-k] for k in range(1, p+1)])
    X_lag = X_minus_l[p:n]  # Y[j]=eps2[p+j] was generated using X[p+j]; corrected alignment
    h = silverman_bandwidth(X_lag, N)
    if use_sparse or N > 2000:
        idx, wts = compute_kernel_weights_sparse(X_lag, h, k_nn=min(k_nn, N))
        def Wv(v): return apply_smoother_sparse(idx, wts, v)
    else:
        W = compute_kernel_weights_dense(X_lag, h)
        def Wv(v): return W @ v
    Ytilde = Y - Wv(Y)
    Ztilde = Z - np.column_stack([Wv(Z[:, k]) for k in range(p)])
    alpha_hat, _, _, _ = np.linalg.lstsq(Ztilde, Ytilde, rcond=None)
    partial_resid = Y - Z @ alpha_hat
    m_hat = Wv(partial_resid)
    vhat = Y - Z @ alpha_hat - m_hat
    return alpha_hat, m_hat, vhat
