"""
Experiment 4: Variable selection in 10-covariate ARCH-m(X) setting.
True active set I_0 = {1,3,4,5,9} (0-indexed: {0,2,3,4,8}).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

from src.dgp import generate_innovations, generate_covariates, generate_arch_series, compute_sigma2_10cov_selection
from src.kernel import fit_kernel_null_model
from src.linear import fit_linear_qmle, fit_lasso_qmle
from src.anova import anova_test_statistic
from src.selection import by_fdr, compute_selection_metrics

MODELS = list(range(1, 9))
P_MAP = {1:1,2:1,3:2,4:2,5:1,6:1,7:1,8:1}
N_VALUES = [500, 1000, 5000, 10000]
RHO_VALUES = [0.0, 0.5]
DIST_VALUES = ['normal', 'laplace', 't7']
D = 10
ACTIVE_SET = {0, 2, 3, 4, 8}   # 0-indexed: X1,X3,X4,X5,X9
INACTIVE_SET = {1, 5, 6, 7, 9}  # 0-indexed: X2,X6,X7,X8,X10
K_N = 5
Q_FDR = 0.05


def run_single_replication(args):
    seed, model, n, rho, dist = args
    p = P_MAP[model]
    rng = np.random.default_rng(seed)
    burn_in = 500
    total = n + burn_in
    X_all = generate_covariates(total, d=D, rho=rho, rng=rng)
    z_all = generate_innovations(total, dist=dist, rng=rng)

    def sigma2_fn(t, eps_lags, X_t):
        el1 = float(eps_lags[0]) if len(eps_lags) > 0 else 0.0
        el2 = float(eps_lags[1]) if len(eps_lags) > 1 else 0.0
        return compute_sigma2_10cov_selection(model, el1, el2, X_t)

    eps, X = generate_arch_series(n, p, X_all, z_all, sigma2_fn, burn_in=burn_in)
    eps2 = eps**2

    arm_pvals = np.ones(D)
    for l in range(D):
        try:
            excl = [j for j in range(D) if j != l]
            X_minus_l = X[:, excl]  # (n, D-1)
            _, _, vhat = fit_kernel_null_model(eps2, X_minus_l, p, use_sparse=(n > 500))
            x_l_test = X[p:n, l]  # aligned to Y[j]=eps2[p+j]; corrected
            _, _, pval = anova_test_statistic(vhat, x_l_test, k_n=K_N)
            arm_pvals[l] = pval
        except Exception:
            arm_pvals[l] = 1.0
    arm_selected = by_fdr(arm_pvals, q=Q_FDR)

    lin_pvals = np.ones(D)
    try:
        _, p_values_lin, _ = fit_linear_qmle(eps2, X, p)
        lin_pvals = p_values_lin
    except Exception:
        pass
    lin_selected = by_fdr(lin_pvals, q=Q_FDR)

    try:
        lasso_sel_idx = fit_lasso_qmle(eps2, X, p)
        lasso_selected = set(lasso_sel_idx.tolist())
    except Exception:
        lasso_selected = set()

    return {
        'arm': compute_selection_metrics(arm_selected, ACTIVE_SET, INACTIVE_SET),
        'linear': compute_selection_metrics(lin_selected, ACTIVE_SET, INACTIVE_SET),
        'lasso': compute_selection_metrics(lasso_selected, ACTIVE_SET, INACTIVE_SET),
    }


def run_configuration(model, n, rho, dist, n_reps, base_seed=300):
    seeds = [base_seed + rep*997 + model*83 + n//100 + int(rho*10) + ord(dist[0])
             for rep in range(n_reps)]
    args_list = [(seed, model, n, rho, dist) for seed in seeds]
    n_jobs = int(os.environ.get('N_JOBS', max(1, cpu_count()-1)))
    if n_jobs > 1 and n_reps > 2:
        with Pool(n_jobs) as pool:
            rep_results = pool.map(run_single_replication, args_list)
    else:
        rep_results = [run_single_replication(a) for a in args_list]
    means = {}
    for method in ['arm', 'linear', 'lasso']:
        for metric in ['CS', 'IS', 'CE', 'IE']:
            means[f'{method}_{metric}'] = np.mean([r[method][metric] for r in rep_results])
    return means


def run_primary_table(n_reps):
    print(f'Running exp_4 primary: n=1000, rho=0, {n_reps} reps')
    rows = []
    for model in MODELS:
        for dist in DIST_VALUES:
            r = run_configuration(model, n=1000, rho=0.0, dist=dist, n_reps=n_reps)
            row = {'model': model, 'dist': dist}
            row.update(r)
            rows.append(row)
            print(f'  Model {model}, {dist}: ARM CS={r["arm_CS"]:.2f}/{ACTIVE_SET.__len__()}, '
                  f'Lin CS={r["linear_CS"]:.2f}, LASSO CS={r["lasso_CS"]:.2f}')
    return pd.DataFrame(rows)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    n_reps = int(os.environ.get('N_REPS', 10))
    print(f'=== Experiment 4: Variable Selection d=10 (N_REPS={n_reps}) ===')
    df_primary = run_primary_table(n_reps)
    df_primary.to_csv('results/exp4_selection_10cov.csv', index=False)
    print(df_primary.to_string())
    print('Experiment 4 complete. Results saved to results/exp4_selection_10cov.csv')
