"""Unit tests for the mcmc package."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcmc import chi_squared, inverse_variance, gelman_rubin, lcdm_model  # noqa: E402


def test_gelman_rubin_identical_chains():
    # Identical chains have zero between-chain variance, so R-hat reduces to
    # sqrt((n-1)/n) -> 1 as n grows.
    n = 500
    rng = np.random.default_rng(0)
    base = rng.normal(size=(n, 2))
    chains = [base.copy() for _ in range(4)]
    rhat = gelman_rubin(chains)
    assert np.allclose(rhat, np.sqrt((n - 1) / n), atol=1e-9)


def test_gelman_rubin_dispersed_chains():
    # Chains centered far apart have not converged -> R-hat > 1.1.
    rng = np.random.default_rng(1)
    chains = [rng.normal(loc=10 * j, scale=1, size=(500, 1)) for j in range(4)]
    assert (gelman_rubin(chains) > 1.1).all()


def test_inverse_variance_uses_sigma_squared():
    errors = np.array([2.0, 4.0])
    inv = inverse_variance(errors)
    assert np.allclose(np.diag(inv), [1 / 4.0, 1 / 16.0])


def test_chi_squared_known_value():
    # Perfect fit -> chi^2 = 0.
    z = np.array([0.0, 1.0])
    H_true = lcdm_model([70.0, 0.3])(z)
    inv = inverse_variance(np.array([1.0, 1.0]))
    assert chi_squared(lcdm_model, [70.0, 0.3], (z, H_true), inv) == 0.0
