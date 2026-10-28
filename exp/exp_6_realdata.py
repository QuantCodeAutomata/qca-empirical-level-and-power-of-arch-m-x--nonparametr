"""
Experiment 6: Real data application - S&P 500 volatility drivers.
Tests 11 global market and commodity covariates using ARM nonparametric test.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print('yfinance not available; using simulated data for demonstration')

from src.kernel import fit_kernel_null_model
from src.linear import fit_linear_qmle
from src.anova import anova_test_statistic
from src.selection import by_fdr, harmonic_number

COVARIATES = {
    'China_Market': '000001.SS',
    'Asia_Market': 'AAXJ',
    'Crude_Oil': 'CL=F',
    'USD_Index': 'DX-Y.NYB',
    'Gold': 'GC=F',
    'Copper': 'HG=F',
    'Silver': 'SI=F',
    'Steel': 'SLX',
    'Rice': 'ZR=F',
    'Wheat': 'ZW=F',
    'Europe_Market': '^STOXX50E',
}
SP500_TICKER = '^GSPC'
START_DATE = '2010-01-01'
END_DATE = '2023-12-31'
K_N = 5
Q_FDR = 0.05


def download_data():
    """Download S&P 500 and covariate price data from Yahoo Finance."""
    all_tickers = [SP500_TICKER] + list(COVARIATES.values())
    print(f'Downloading {len(all_tickers)} series from Yahoo Finance...')
    prices = {}
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True,
                             progress=False)
            if len(df) < 100:
                print(f'  {ticker}: insufficient data ({len(df)} obs), skipping')
                continue
            # Handle multi-level columns from newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                close = df[('Close', ticker)] if ('Close', ticker) in df.columns else df['Close'].iloc[:, 0]
            else:
                close = df['Close']
            # Ensure Series with DatetimeIndex
            close = close.squeeze()
            prices[ticker] = close
            print(f'  {ticker}: {len(df)} observations')
        except Exception as e:
            print(f'  {ticker}: download failed ({e})')
    return prices


def prepare_returns(prices):
    """Compute log returns and align series on common dates (inner join)."""
    returns_list = []
    for ticker, price_series in prices.items():
        lr = np.log(price_series / price_series.shift(1)).dropna()
        lr.name = ticker
        returns_list.append(lr)
    if not returns_list:
        raise ValueError('No valid return series available')
    df = pd.concat(returns_list, axis=1, join='inner')
    df = df.dropna()
    print(f'Aligned dataset: {len(df)} observations, {df.shape[1]} series')
    return df


def generate_synthetic_data(n=2500):
    """Generate synthetic S&P 500 + 11 covariates for demonstration."""
    print('Generating synthetic data for demonstration...')
    rng = np.random.default_rng(42)
    d = len(COVARIATES)
    # Simulate GARCH-like S&P 500
    sigma2 = np.zeros(n)
    eps = np.zeros(n)
    sigma2[0] = 0.0001
    for t in range(1, n):
        sigma2[t] = 0.00001 + 0.1*eps[t-1]**2 + 0.85*sigma2[t-1]
        eps[t] = rng.standard_normal() * np.sqrt(sigma2[t])
    sp500_r = eps[1:]
    # Simulate covariates
    cov_r = rng.standard_normal((n-1, d)) * 0.01
    cols = [SP500_TICKER] + list(COVARIATES.keys())
    df = pd.DataFrame(np.column_stack([sp500_r, cov_r]), columns=cols)
    return df


def run_arm_test(sp500_r, cov_returns):
    """Run ARM test for each covariate."""
    n = len(sp500_r)
    p = 1
    eps2 = sp500_r**2
    X = cov_returns.values  # (n, d)
    d = X.shape[1]
    cov_names = list(cov_returns.columns)

    # Winsorize eps2 at 99.9th percentile
    eps2 = np.clip(eps2, 0, np.percentile(eps2, 99.9))

    arm_pvals = np.ones(d)
    for l in range(d):
        try:
            excl = [j for j in range(d) if j != l]
            X_minus_l = X[:, excl]  # (n, d-1)
            _, _, vhat = fit_kernel_null_model(eps2, X_minus_l, p, use_sparse=True,
                                               k_nn=min(500, n-p))
            x_l_test = X[p:n, l]  # X[t-1] aligned to vhat at t; corrected
            _, _, pval = anova_test_statistic(vhat, x_l_test, k_n=K_N)
            arm_pvals[l] = pval
            print(f'  {cov_names[l]}: p-value = {pval:.6f}')
        except Exception as e:
            print(f'  {cov_names[l]}: failed ({e})')
            arm_pvals[l] = 1.0

    arm_selected_idx = by_fdr(arm_pvals, q=Q_FDR)
    arm_selected_names = [cov_names[i] for i in sorted(arm_selected_idx)]
    return arm_pvals, arm_selected_names, cov_names


def run_linear_test(sp500_r, cov_returns):
    """Run linear benchmark test."""
    n = len(sp500_r)
    p = 1
    eps2 = sp500_r**2
    X = cov_returns.values
    d = X.shape[1]
    cov_names = list(cov_returns.columns)
    try:
        _, p_values, _ = fit_linear_qmle(eps2, X, p)
        lin_selected_idx = by_fdr(p_values, q=Q_FDR)
        lin_selected_names = [cov_names[i] for i in sorted(lin_selected_idx)]
        return p_values, lin_selected_names
    except Exception as e:
        print(f'Linear benchmark failed: {e}')
        return np.ones(d), []


def create_results_table(pvals, cov_names, q=Q_FDR):
    """Create sorted p-value table with BY thresholds."""
    d = len(pvals)
    C_d = harmonic_number(d)
    sorted_idx = np.argsort(pvals)
    rows = []
    for rank, orig_idx in enumerate(sorted_idx):
        c_threshold = (rank + 1) / (d * C_d) * q
        reject = 'Reject' if pvals[orig_idx] <= c_threshold else 'Fail to Reject'
        rows.append({
            'Covariate': cov_names[orig_idx],
            'P-value': f'{pvals[orig_idx]:.6f}',
            'Rank': rank + 1,
            'BY_Threshold': f'{c_threshold:.6f}',
            'Decision': reject
        })
    return pd.DataFrame(rows)


def plot_pvalues(arm_pvals, cov_names, q=Q_FDR, out_path='results/fig6_realdata_pvalues.png'):
    """Bar chart of p-values vs BY thresholds."""
    d = len(arm_pvals)
    C_d = harmonic_number(d)
    sorted_idx = np.argsort(arm_pvals)
    sorted_p = arm_pvals[sorted_idx]
    sorted_names = [cov_names[i] for i in sorted_idx]
    thresholds = np.arange(1, d+1) / (d * C_d) * q

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['red' if p <= t else 'steelblue' for p, t in zip(sorted_p, thresholds)]
    x = np.arange(d)
    ax.bar(x, sorted_p, color=colors, alpha=0.7, label='ARM p-values')
    ax.plot(x, thresholds, 'k--', linewidth=2, marker='o', label='BY thresholds')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax.set_ylabel('P-value')
    ax.set_title('ARM Test: S&P 500 Volatility Driver Selection\n(Red bars = Rejected by BY-FDR)')
    ax.legend()
    ax.set_yscale('symlog', linthresh=1e-5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    print('=== Experiment 6: Real Data S&P 500 Volatility Drivers ===')

    use_real_data = HAS_YFINANCE and os.environ.get('USE_REAL_DATA', '0') == '1'

    if use_real_data:
        prices = download_data()
        if len(prices) < 3:
            print('Insufficient data downloaded, using synthetic demo.')
            use_real_data = False

    if use_real_data:
        returns_df = prepare_returns(prices)
        sp500_r = returns_df[SP500_TICKER].values
        cov_cols = [k for k in COVARIATES.keys() if COVARIATES[k] in returns_df.columns]
        cov_tickers = [COVARIATES[k] for k in cov_cols]
        cov_returns = returns_df[cov_tickers].copy()
        cov_returns.columns = cov_cols
    else:
        print('Using synthetic demonstration data (set USE_REAL_DATA=1 for real data).')
        demo_df = generate_synthetic_data(n=2501)
        sp500_r = demo_df[SP500_TICKER].values
        cov_returns = demo_df[list(COVARIATES.keys())]
        sp500_r = sp500_r

    print(f'\nSample size: n={len(sp500_r)}')
    print(f'Covariates: {list(cov_returns.columns)}')

    print('\n--- ARM Test (Kernel-based) ---')
    arm_pvals, arm_selected, cov_names = run_arm_test(sp500_r, cov_returns)

    print('\n--- Linear Benchmark ---')
    lin_pvals, lin_selected = run_linear_test(sp500_r, cov_returns)

    # Create results tables
    arm_table = create_results_table(arm_pvals, cov_names)
    lin_table = create_results_table(lin_pvals, cov_names)

    print('\n=== ARM Results Table ===')
    print(arm_table.to_string(index=False))
    print(f'\nARM Selected: {arm_selected}')

    print('\n=== Linear Benchmark Results Table ===')
    print(lin_table.to_string(index=False))
    print(f'\nLinear Selected: {lin_selected}')

    # Save
    arm_table.to_csv('results/exp6_arm_results.csv', index=False)
    lin_table.to_csv('results/exp6_linear_results.csv', index=False)
    plot_pvalues(arm_pvals, cov_names)

    summary = {
        'ARM_selected': str(arm_selected),
        'Linear_selected': str(lin_selected),
        'n_obs': len(sp500_r),
        'data_source': 'real' if use_real_data else 'synthetic_demo',
    }
    pd.DataFrame([summary]).to_csv('results/exp6_summary.csv', index=False)
    print('\nExperiment 6 complete.')
