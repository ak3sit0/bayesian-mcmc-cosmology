"""Likelihood, goodness-of-fit and model selection for H(z) fits.

The observed data are treated as independent Gaussian measurements, so the
covariance is diagonal with entries ``sigma_i^2``.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def inverse_variance(errors):
    """Inverse covariance for independent Gaussian errors: ``diag(1/sigma^2)``.

    This is where the variance enters correctly: chi^2 must divide by
    ``sigma^2`` (not ``sigma``).
    """
    errors = np.asarray(errors, dtype=float)
    return np.diag(1.0 / errors ** 2)


def chi_squared(model, state, observed, inv_cov):
    """chi^2 = (model - data)^T C^{-1} (model - data).

    ``observed = (redshift, H_observed)``.
    """
    redshift, d_observed = observed
    diff = model(state)(redshift) - d_observed
    return diff @ inv_cov @ diff


def log_likelihood(chi2):
    """Gaussian log-likelihood up to a constant: ``-chi2 / 2``.

    Working in log space avoids the underflow (``exp(-chi2/2) -> 0``) that made
    the acceptance ratio divide by zero.
    """
    return -0.5 * chi2


def best_fit(model, initial, observed, inv_cov, bounds=None):
    """Minimize chi^2 to get the best-fit state and chi^2_min.

    Uses Nelder-Mead starting from ``initial`` (e.g. the posterior mean).
    Returns ``(state, chi2_min)``.
    """
    result = minimize(
        lambda state: chi_squared(model, state, observed, inv_cov),
        np.asarray(initial, dtype=float),
        method="Nelder-Mead",
        bounds=bounds,
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 10000},
    )
    return result.x, float(result.fun)


def information_criteria(chi2_min, k, n):
    """AIC and BIC from chi^2_min, number of parameters k, and sample size n.

    AIC = chi2_min + 2k;  BIC = chi2_min + k * ln(n). Lower is better; BIC
    penalizes extra parameters more strongly.
    """
    return {"aic": chi2_min + 2 * k, "bic": chi2_min + k * np.log(n)}


def model_selection_table(results, n):
    """Build a model-comparison table from fitted models.

    ``results``: mapping ``name -> (state, chi2_min, k)``.
    Returns a DataFrame with k, chi2_min, chi2/dof, AIC, BIC and the deltas
    (dAIC, dBIC) relative to the best (lowest-criterion) model.
    """
    rows = {}
    for name, (_state, chi2_min, k) in results.items():
        ic = information_criteria(chi2_min, k, n)
        rows[name] = {
            "k": k,
            "chi2_min": chi2_min,
            "chi2_dof": chi2_min / (n - k),
            "AIC": ic["aic"],
            "BIC": ic["bic"],
        }
    table = pd.DataFrame(rows).T
    table["k"] = table["k"].astype(int)
    table["dAIC"] = table["AIC"] - table["AIC"].min()
    table["dBIC"] = table["BIC"] - table["BIC"].min()
    return table
