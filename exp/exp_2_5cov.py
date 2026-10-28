"""
Experiment 2: 5-covariate ARCH-m(X) empirical level and power.
Tests H0: X3 not significant in 8 DGP models using Kernel and Linear methods.
Replicates Figure 2 from the paper.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

from src.dgp import generate_innovations, generate_covariates, generate_arch_series, compute_sigma2_5cov
from src.kernel import fit_kernel_null_model
from src.linear import fit_linear_qmle
from src.anova import anova_test_statistic

MODELS = list(range(1, 9))
P_MAP = {1: 1, 2: 1, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1}
C_VALUES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
N_VALUES = [500, 1000]
RHO_VALUES = [0.0, 0.5]
DIST_VALUES = ['normal', 'laplace', 't7']
K_N = 5
ALPHA = 0.05
D = 5  # total number of covariates
TEST_COV = 3  # testing X3 (1-indexed), index 2 (0-indexed)


def run_single_replication(args):
    """Run one MC replication for exp_2."""
    seed, model, c, n, rho, dist = args
    p = P_MAP[model]
    rng = np.random.default_rng(seed)
    burn_in = 500
    total = n + burn_in

    X_all = generate_covariates(total, d=D, rho=rho, rng=rng)
    z_all = generate_innovations(total, dist=dist, rng=rng)

    def sigma2_fn(t, eps_lags, X_t):
        el1 = float(eps_lags[0]) if len(eps_lags) > 0 else 0.0
        el2 = float(eps_lags[1]) if len(eps_lags) > 1 else 0.0
        return compute_sigma2_5cov(model, el1, el2, X_t, c)

    eps, X = generate_arch_series(n, p, X_all, z_all, sigma2_fn, burn_in=burn_in)
    eps2 = eps**2
    results = {}

    # Kernel method: conditioning on X^{-3} = (X1, X2, X4, X5), 4-dimensional
    try:
        X_minus3 = np.column_stack([X[:, 0], X[:, 1], X[:, 3], X[:, 4]])  # (n, 4)
        alpha_hat, m_hat, vhat = fit_kernel_null_model(eps2, X_minus3, p, use_sparse=(n > 500))
        x3_test = X[p:n, 2]  # lagged X3 aligned to Y[j]=eps2[p+j]; corrected
        _, _, pval_kernel = anova_test_statistic(vhat, x3_test, k_n=K_N)
        results['kernel'] = int(pval_kernel < ALPHA)
    except Exception:
        results['kernel'] = 0

    # Linear benchmark: Wald test for beta_3=0 (index 2, 0-indexed)
    try:
        _, p_values_lin, _ = fit_linear_qmle(eps2, X, p)
        results['linear'] = int(p_values_lin[2] < ALPHA)  # beta_3
    except Exception:
        results['linear'] = 0

    return results


def run_configuration(model, c, n, rho, dist, n_reps, base_seed=100):
    """Run n_reps MC replications."""
    seeds = [base_seed + rep * 997 + model * 83 + int(c*100) * 7 + n * 3 + int(rho*10) + ord(dist[0])
             for rep in range(n_reps)]
    args_list = [(seed, model, c, n, rho, dist) for seed in seeds]
    n_jobs = int(os.environ.get('N_JOBS', max(1, cpu_count() - 1)))
    if n_jobs > 1 and n_reps > 4:
        with Pool(n_jobs) as pool:
            rep_results = pool.map(run_single_replication, args_list)
    else:
        rep_results = [run_single_replication(a) for a in args_list]
    return {
        'kernel': np.mean([r.get('kernel', 0) for r in rep_results]),
        'linear': np.mean([r.get('linear', 0) for r in rep_results]),
    }


def run_primary_experiment(n_reps):
    """Run primary: n=1000, rho=0, Normal."""
    print(f'Running exp_2 primary: n=1000, rho=0, Normal, {n_reps} reps')
    results = np.zeros((8, len(C_VALUES), 2))  # [model, c_idx, method(kernel/linear)]
    for mi, model in enumerate(MODELS):
        for ci, c in enumerate(C_VALUES):
            r = run_configuration(model, c, n=1000, rho=0.0, dist='normal', n_reps=n_reps)
            results[mi, ci, 0] = r['kernel']
            results[mi, ci, 1] = r['linear']
            print(f'  Model {model}, c={c:.1f}: kernel={r["kernel"]:.3f}, linear={r["linear"]:.3f}')
    return results


def plot_power_curves(results_primary, out_path):
    """Plot Figure 2: 5-covariate power curves."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    colors = ['blue', 'green']
    linestyles = ['-', ':']
    labels = ['Proposed-Kernel', 'Linear Benchmark']

    for mi, ax in enumerate(axes.flat):
        for method_idx in range(2):
            ax.plot(C_VALUES, results_primary[mi, :, method_idx],
                    color=colors[method_idx], linestyle=linestyles[method_idx],
                    linewidth=2, label=labels[method_idx])
        ax.axhline(0.05, color='black', linestyle='--', linewidth=1, alpha=0.7, label='Level=0.05')
        ax.set_title(f'Model {mi+1}', fontsize=12)
        ax.set_xlabel('Signal strength c')
        ax.set_ylabel('Rejection rate')
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Figure 2: 5-Covariate Power Curves (n=1000, rho=0, Normal)', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    n_reps = int(os.environ.get('N_REPS', 30))
    print(f'=== Experiment 2: 5-Covariate Power (N_REPS={n_reps}) ===')

    results_primary = run_primary_experiment(n_reps)
    np.save('results/exp2_primary.npy', results_primary)
    plot_power_curves(results_primary, 'results/fig2_power_5cov.png')

    print('\n=== Level check (c=0): ===')
    for mi, model in enumerate(MODELS):
        print(f'  Model {model}: kernel={results_primary[mi, 0, 0]:.3f}, linear={results_primary[mi, 0, 1]:.3f}')

    print('\n=== Power at c=1.0: ===')
    for mi, model in enumerate(MODELS):
        print(f'  Model {model}: kernel={results_primary[mi, -1, 0]:.3f}, linear={results_primary[mi, -1, 1]:.3f}')

    print('Experiment 2 complete.')
