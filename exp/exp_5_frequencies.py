"""
Experiment 5: Per-covariate selection frequencies for d=5 and d=10.
Model 1 (additive quadratic), Normal innovations, Tables 5 and 6 from paper.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

from src.dgp import generate_innovations, generate_covariates, generate_arch_series
from src.dgp import compute_sigma2_5cov_selection, compute_sigma2_10cov_selection
from src.kernel import fit_kernel_null_model
from src.linear import fit_linear_qmle, fit_lasso_qmle
from src.anova import anova_test_statistic
from src.selection import by_fdr

N_VALUES = [500, 1000, 5000, 10000]
N_VALUES_10COV = [500, 1000, 2000]   # d=10: cap at 2000 to avoid curse-of-dimensionality timeout
RHO_VALUES = [0.0, 0.5]
K_N = 5
Q_FDR = 0.05


def run_single_rep_5cov(args):
    seed, n, rho = args
    p = 1
    rng = np.random.default_rng(seed)
    D = 5
    X_all = generate_covariates(n + 500, d=D, rho=rho, rng=rng)
    z_all = generate_innovations(n + 500, dist='normal', rng=rng)

    def s2fn(t, el, X_t):
        el1 = float(el[0]) if len(el) > 0 else 0.0
        return compute_sigma2_5cov_selection(1, el1, 0.0, X_t)  # Model 1

    eps, X = generate_arch_series(n, p, X_all, z_all, s2fn, burn_in=500)
    eps2 = eps**2

    arm_pvals = np.ones(D)
    for l in range(D):
        try:
            excl = [j for j in range(D) if j != l]
            _, _, vhat = fit_kernel_null_model(eps2, X[:, excl], p, use_sparse=(n > 1000))
            _, _, pval = anova_test_statistic(vhat, X[p:n, l], k_n=K_N)  # corrected
            arm_pvals[l] = pval
        except Exception:
            pass
    arm_sel = by_fdr(arm_pvals, q=Q_FDR)

    lin_pvals = np.ones(D)
    try:
        _, lp, _ = fit_linear_qmle(eps2, X, p)
        lin_pvals = lp
    except Exception:
        pass
    lin_sel = by_fdr(lin_pvals, q=Q_FDR)

    lasso_sel = set()
    try:
        lasso_sel = set(fit_lasso_qmle(eps2, X, p).tolist())
    except Exception:
        pass

    arm_bin = [int(l in arm_sel) for l in range(D)]
    lin_bin = [int(l in lin_sel) for l in range(D)]
    lasso_bin = [int(l in lasso_sel) for l in range(D)]
    return arm_bin, lin_bin, lasso_bin


def run_single_rep_10cov(args):
    seed, n, rho = args
    p = 1
    rng = np.random.default_rng(seed)
    D = 10
    X_all = generate_covariates(n + 500, d=D, rho=rho, rng=rng)
    z_all = generate_innovations(n + 500, dist='normal', rng=rng)

    def s2fn(t, el, X_t):
        el1 = float(el[0]) if len(el) > 0 else 0.0
        return compute_sigma2_10cov_selection(1, el1, 0.0, X_t)  # Model 1

    eps, X = generate_arch_series(n, p, X_all, z_all, s2fn, burn_in=500)
    eps2 = eps**2

    # Adaptive k_nn: smaller for large n to limit computation at d=10
    k_nn = 100 if n >= 1000 else 200
    arm_pvals = np.ones(D)
    for l in range(D):
        try:
            excl = [j for j in range(D) if j != l]
            _, _, vhat = fit_kernel_null_model(eps2, X[:, excl], p, use_sparse=True, k_nn=k_nn)
            _, _, pval = anova_test_statistic(vhat, X[p:n, l], k_n=K_N)  # corrected
            arm_pvals[l] = pval
        except Exception:
            pass
    arm_sel = by_fdr(arm_pvals, q=Q_FDR)

    lin_pvals = np.ones(D)
    try:
        _, lp, _ = fit_linear_qmle(eps2, X, p)
        lin_pvals = lp
    except Exception:
        pass
    lin_sel = by_fdr(lin_pvals, q=Q_FDR)

    lasso_sel = set()
    try:
        lasso_sel = set(fit_lasso_qmle(eps2, X, p).tolist())
    except Exception:
        pass

    arm_bin = [int(l in arm_sel) for l in range(D)]
    lin_bin = [int(l in lin_sel) for l in range(D)]
    lasso_bin = [int(l in lasso_sel) for l in range(D)]
    return arm_bin, lin_bin, lasso_bin


def compute_frequencies(rep_results, D):
    arm_freq = np.mean([r[0] for r in rep_results], axis=0)
    lin_freq = np.mean([r[1] for r in rep_results], axis=0)
    lasso_freq = np.mean([r[2] for r in rep_results], axis=0)
    return arm_freq, lin_freq, lasso_freq


def run_setting(d, n_reps):
    """Run experiments for dimension d (5 or 10)."""
    fn = run_single_rep_5cov if d == 5 else run_single_rep_10cov
    n_values = N_VALUES if d == 5 else N_VALUES_10COV
    col_names = [f'X{l+1}' for l in range(d)]
    rows = []
    for rho in RHO_VALUES:
        for n in n_values:
            seeds = [400 + rep*997 + n//100 + int(rho*10) + d for rep in range(n_reps)]
            args_list = [(seed, n, rho) for seed in seeds]
            n_jobs = int(os.environ.get('N_JOBS', max(1, cpu_count()-1)))
            if n_jobs > 1 and n_reps > 2:
                with Pool(n_jobs) as pool:
                    rep_results = pool.map(fn, args_list)
            else:
                rep_results = [fn(a) for a in args_list]
            arm_freq, lin_freq, lasso_freq = compute_frequencies(rep_results, d)
            for method, freq in [('ARM', arm_freq), ('Linear', lin_freq), ('LASSO', lasso_freq)]:
                row = {'rho': rho, 'n': n, 'method': method}
                row.update({col_names[l]: round(freq[l], 3) for l in range(d)})
                rows.append(row)
            print(f'  d={d}, rho={rho}, n={n}: ARM={np.round(arm_freq, 2)}')
    return pd.DataFrame(rows, columns=['rho', 'n', 'method'] + col_names)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    n_reps = int(os.environ.get('N_REPS', 10))
    skip_5cov = os.environ.get('SKIP_5COV', '0') == '1'
    print(f'=== Experiment 5: Selection Frequencies (N_REPS={n_reps}) ===')

    if skip_5cov and os.path.exists('results/exp5_freq_5cov.csv'):
        print('\n--- Setting A: d=5 (loading saved results) ---')
        df5 = pd.read_csv('results/exp5_freq_5cov.csv')
    else:
        print('\n--- Setting A: d=5 ---')
        df5 = run_setting(d=5, n_reps=n_reps)
        df5.to_csv('results/exp5_freq_5cov.csv', index=False)
    print(df5.to_string())

    print(f'\n--- Setting B: d=10 (N_VALUES capped at {N_VALUES_10COV}) ---')
    df10 = run_setting(d=10, n_reps=n_reps)
    df10.to_csv('results/exp5_freq_10cov.csv', index=False)
    print(df10.to_string())
    print('Experiment 5 complete.')
