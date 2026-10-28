"""
Experiment 1: 2-covariate ARCH-m(X) empirical level and power.
Tests H0: X2 not significant in 8 DGP models using Kernel, B-spline, and Linear methods.
Replicates Figure 1 and Table 1 from the paper.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

from src.dgp import generate_innovations, generate_covariates, generate_arch_series, compute_sigma2_2cov
from src.kernel import fit_kernel_null_model
from src.bspline import fit_bspline_null_model
from src.linear import fit_linear_qmle
from src.anova import anova_test_statistic

MODELS = list(range(1, 9))  # 1-8
P_MAP = {1: 1, 2: 1, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1}  # ARCH order per model
C_VALUES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
N_VALUES = [500, 1000]
RHO_VALUES = [0.0, 0.5]
DIST_VALUES = ['normal', 'laplace', 't7']
K_N = 5  # primary window size
ALPHA = 0.05


def run_single_replication(args):
    """
    Run one Monte Carlo replication for exp_1.
    Returns dict with rejection indicators for all methods.
    """
    seed, model, c, n, rho, dist = args
    p = P_MAP[model]
    rng = np.random.default_rng(seed)
    burn_in = 500
    total = n + burn_in

    X_all = generate_covariates(total, d=2, rho=rho, rng=rng)
    z_all = generate_innovations(total, dist=dist, rng=rng)

    def sigma2_fn(t, eps_lags, X_t):
        el1 = float(eps_lags[0]) if len(eps_lags) > 0 else 0.0
        el2 = float(eps_lags[1]) if len(eps_lags) > 1 else 0.0
        x1, x2 = float(X_t[0]), float(X_t[1])
        return compute_sigma2_2cov(model, el1, el2, x1, x2, c)

    eps, X = generate_arch_series(n, p, X_all, z_all, sigma2_fn, burn_in=burn_in)
    eps2 = eps**2

    results = {}

    # Kernel method: testing H0: X2 not significant, conditioning on X1
    try:
        X_minus2 = X[:, 0:1]  # only X1, shape (n, 1)
        alpha_hat, m_hat, vhat = fit_kernel_null_model(eps2, X_minus2, p, use_sparse=False)
        x2_test = X[p:n, 1]  # lagged X2 aligned to Y[j]=eps2[p+j]; corrected
        _, _, pval_kernel = anova_test_statistic(vhat, x2_test, k_n=K_N)
        results['kernel'] = int(pval_kernel < ALPHA)
    except Exception:
        results['kernel'] = 0

    # B-spline method: testing H0: X2 not significant
    try:
        x1_arr = X[:, 0]  # shape (n,)
        alpha_hat_bs, m_hat_bs, vhat_bs = fit_bspline_null_model(eps2, x1_arr, p)
        x2_test_bs = X[p:n, 1]  # corrected alignment
        _, _, pval_bs = anova_test_statistic(vhat_bs, x2_test_bs, k_n=K_N)
        results['bspline'] = int(pval_bs < ALPHA)
    except Exception:
        results['bspline'] = 0

    # Linear benchmark (Francq-Thieu): Wald test for beta_2=0
    try:
        _, p_values_lin, _ = fit_linear_qmle(eps2, X, p)
        results['linear'] = int(p_values_lin[1] < ALPHA)  # beta_2 (index 1)
    except Exception:
        results['linear'] = 0

    return results


def run_configuration(model, c, n, rho, dist, n_reps, base_seed=0):
    """Run n_reps MC replications for a given configuration."""
    seeds = [base_seed + rep * 997 + model * 83 + int(c*100) * 7 + n * 3 + int(rho*10) + ord(dist[0])
             for rep in range(n_reps)]
    args_list = [(seed, model, c, n, rho, dist) for seed in seeds]

    n_jobs = int(os.environ.get('N_JOBS', max(1, cpu_count() - 1)))
    if n_jobs > 1 and n_reps > 4:
        with Pool(n_jobs) as pool:
            rep_results = pool.map(run_single_replication, args_list)
    else:
        rep_results = [run_single_replication(a) for a in args_list]

    kernel_rate = np.mean([r.get('kernel', 0) for r in rep_results])
    bspline_rate = np.mean([r.get('bspline', 0) for r in rep_results])
    linear_rate = np.mean([r.get('linear', 0) for r in rep_results])
    return {'kernel': kernel_rate, 'bspline': bspline_rate, 'linear': linear_rate}


def run_primary_experiment(n_reps):
    """Run primary experiment: n=1000, rho=0, Normal, all 8 models, all c values."""
    print(f'Running exp_1 primary: n=1000, rho=0, Normal, {n_reps} reps')
    results = np.zeros((8, len(C_VALUES), 3))  # [model, c_idx, method]
    for mi, model in enumerate(MODELS):
        for ci, c in enumerate(C_VALUES):
            r = run_configuration(model, c, n=1000, rho=0.0, dist='normal', n_reps=n_reps)
            results[mi, ci, 0] = r['kernel']
            results[mi, ci, 1] = r['bspline']
            results[mi, ci, 2] = r['linear']
            print(f'  Model {model}, c={c:.1f}: kernel={r["kernel"]:.3f}, '
                  f'bspline={r["bspline"]:.3f}, linear={r["linear"]:.3f}')
    return results


def run_full_experiment(n_reps):
    """Run full experiment across all configurations."""
    print(f'Running exp_1 full: {n_reps} reps per config')
    # Shape: [8 models, 11 c_vals, 2 n_vals, 2 rho_vals, 3 dists, 3 methods]
    results = np.zeros((8, len(C_VALUES), 2, 2, 3, 3))
    for mi, model in enumerate(MODELS):
        for ci, c in enumerate(C_VALUES):
            for ni, n in enumerate(N_VALUES):
                for ri, rho in enumerate(RHO_VALUES):
                    for di, dist in enumerate(DIST_VALUES):
                        r = run_configuration(model, c, n, rho, dist, n_reps)
                        results[mi, ci, ni, ri, di, 0] = r['kernel']
                        results[mi, ci, ni, ri, di, 1] = r['bspline']
                        results[mi, ci, ni, ri, di, 2] = r['linear']
    return results


def plot_power_curves(results_primary, out_path):
    """Plot Figure 1: power curves for all 8 models."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    model_names = [f'Model {m}' for m in MODELS]
    colors = ['blue', 'red', 'green']
    linestyles = ['-', '--', ':']
    labels = ['Proposed-Kernel', 'Proposed-B-spline', 'Linear Benchmark']

    for mi, ax in enumerate(axes.flat):
        for method_idx in range(3):
            ax.plot(C_VALUES, results_primary[mi, :, method_idx],
                    color=colors[method_idx], linestyle=linestyles[method_idx],
                    linewidth=2, label=labels[method_idx])
        ax.axhline(0.05, color='black', linestyle='--', linewidth=1, alpha=0.7, label='Level=0.05')
        ax.set_title(model_names[mi], fontsize=12)
        ax.set_xlabel('Signal strength c')
        ax.set_ylabel('Rejection rate')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Figure 1: Empirical Power Curves (n=1000, rho=0, Normal)', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved power curves to {out_path}')


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    n_reps = int(os.environ.get('N_REPS', 30))
    print(f'=== Experiment 1: 2-Covariate Power (N_REPS={n_reps}) ===')

    # Primary results
    results_primary = run_primary_experiment(n_reps)
    np.save('results/exp1_primary.npy', results_primary)

    # Plot
    plot_power_curves(results_primary, 'results/fig1_power_2cov.png')

    # Summary
    print('\n=== Level check (c=0): ===')
    for mi, model in enumerate(MODELS):
        k_lev = results_primary[mi, 0, 0]
        b_lev = results_primary[mi, 0, 1]
        l_lev = results_primary[mi, 0, 2]
        print(f'  Model {model}: kernel={k_lev:.3f}, bspline={b_lev:.3f}, linear={l_lev:.3f}')

    print('\n=== Power at c=1.0: ===')
    for mi, model in enumerate(MODELS):
        k_pw = results_primary[mi, -1, 0]
        b_pw = results_primary[mi, -1, 1]
        l_pw = results_primary[mi, -1, 2]
        print(f'  Model {model}: kernel={k_pw:.3f}, bspline={b_pw:.3f}, linear={l_pw:.3f}')

    # Full run if requested
    full_run = os.environ.get('FULL_RUN', '0') == '1'
    if full_run:
        results_full = run_full_experiment(n_reps)
        np.save('results/exp1_full.npy', results_full)
        print('Full results saved to results/exp1_full.npy')

    print('Experiment 1 complete.')
