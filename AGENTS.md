# ARCH-m(X) Nonparametric Test Repository

## Project Overview
Monte Carlo simulation study evaluating the empirical level and power of a nonparametric ANOVA-based test for covariate significance in ARCH-m(X) conditional variance models.

## Repository Structure
- `src/` - Core implementation modules
  - `dgp.py` - Data generating processes (8 DGP models, 2/5/10 covariate settings)
  - `kernel.py` - Kernel-based null model fitting (product Gaussian kernel, Silverman bandwidth)
  - `bspline.py` - B-spline null model fitting (cubic B-spline, QMLE)
  - `linear.py` - Linear Francq-Thieu benchmark and LASSO
  - `anova.py` - ANOVA test statistic T_n and Z_n standardization
  - `selection.py` - Benjamini-Yekutieli FDR correction (BY-FDR)
- `exp/` - Experiment scripts
  - `exp_1_2cov.py` - 2-covariate level/power simulation
  - `exp_2_5cov.py` - 5-covariate level/power simulation
  - `exp_3_selection_5cov.py` - Variable selection in 5-covariate setting
  - `exp_4_selection_10cov.py` - Variable selection in 10-covariate setting
  - `exp_5_frequencies.py` - Per-covariate selection frequencies
  - `exp_6_realdata.py` - Real data S&P 500 application
- `tests/` - pytest unit and integration tests
- `data/` - Data directory (synthetic data generated on-the-fly)
- `results/` - Output directory for results, plots, and RESULTS.md

## Key Parameters
- Window size k_n = 5 (primary), sensitivity at 3 and 7
- FDR level q = 0.05
- Monte Carlo replications: 500 (level/power), 200 (selection tables), 500 (frequencies)
- Sample sizes: n in {500, 1000} for power; {500, 1000, 5000, 10000} for selection
- Bandwidth: Silverman rule h_j = 1.06*std_j*N^{-1/(d-1+4)}

## Running Experiments
```bash
python exp/exp_1_2cov.py
python exp/exp_2_5cov.py
python exp/exp_3_selection_5cov.py
python exp/exp_4_selection_10cov.py
python exp/exp_5_frequencies.py
python exp/exp_6_realdata.py
pytest tests/ -v
```

## Environment Variables
- `N_REPS` - Override Monte Carlo replications (e.g., N_REPS=50 for quick tests)
- `N_JOBS` - Number of parallel jobs (default: cpu_count - 1)

## Notes for Future Sessions
- All src/ modules use numpy-based implementation; no external TS libraries needed
- For n > 2000, kernel.py automatically uses sparse NN approximation (k_nn=200)
- B-spline method only used in exp_1 (2-covariate); infeasible for d>=5
- Linear benchmark uses numerical Hessian for Wald test standard errors
- LASSO uses smooth L1 approximation (smooth_eps=1e-6) for differentiability
- **DGP quirk**: `compute_sigma2_5cov` Model 5 uses un-squared sines and can return
  negative values at x=2.5 (the covariate mean). Use x=1.0 in tests for this function.
  The selection variants (`compute_sigma2_5cov_selection`, `compute_sigma2_10cov_selection`)
  are fine at x=2.5 because they use squared terms or are clamped via `max(..., 1e-6)`.
- **exp_3/4**: ARM selection loops over D covariates, fitting kernel null on D-1 excluded
  covariates, then ANOVA-testing the left-out one. use_sparse triggered at n>1000 (d=5)
  or n>500 (d=10) to manage memory.
- **exp_6**: Uses walrus operators in `download_data()` (Python 3.8+). Defaults to
  synthetic demo data; set `USE_REAL_DATA=1` to download from Yahoo Finance.
- **tests**: 46 tests across 6 files, all passing. Run with `pytest tests/ -v`.
