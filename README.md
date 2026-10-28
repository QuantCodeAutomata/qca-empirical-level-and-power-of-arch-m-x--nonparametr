# ARCH-m(X) Nonparametric Test: Empirical Level and Power

Python implementation of Monte Carlo simulation study for nonparametric ANOVA-based testing in ARCH-m(X) conditional variance models.

## Methods
1. **Proposed-Kernel**: Product Gaussian kernel null model + ANOVA T_n
2. **Proposed-B-spline**: Cubic B-spline QMLE null model + ANOVA T_n (2-cov only)
3. **Linear Benchmark**: Parametric QMLE with Wald chi^2 test (Francq-Thieu)
4. **LASSO**: Penalized QMLE with BIC lambda selection
5. **BY-FDR**: Benjamini-Yekutieli correction for multiple testing

## Experiments
| ID | Description |
|----|-------------|
| exp_1 | 2-covariate power curves (8 models, 3 methods) |
| exp_2 | 5-covariate power curves (8 models, 2 methods) |
| exp_3 | Variable selection d=5, true active {1,3,4} |
| exp_4 | Variable selection d=10, true active {1,3,4,5,9} |
| exp_5 | Per-covariate selection frequencies |
| exp_6 | Real data: S&P 500 volatility drivers |

## Usage
```bash
pip install -r requirements.txt
N_REPS=20 python exp/exp_1_2cov.py  # quick test
pytest tests/ -v
```

## Results
Saved to `results/` directory including RESULTS.md summary.
