# Data Directory

This project uses fully synthetic data generated within each Monte Carlo replication.
No external datasets are stored here (exp_6 downloads market data at runtime).

## Data Generation
- Covariates: X_t ~ MVN((2.5,...,2.5), Sigma) where Sigma_ij = rho^|i-j|
- Innovations: Normal N(0,1), Laplace(0,1)/sqrt(2), t(7)/sqrt(7/5)
- All data generated with numpy random seeds for reproducibility.
