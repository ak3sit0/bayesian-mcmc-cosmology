"""Likelihood and goodness-of-fit for H(z) fits.

The observed data are treated as independent Gaussian measurements, so the
covariance is diagonal with entries ``sigma_i^2``.
"""

import numpy as np


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
