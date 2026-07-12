"""Cosmological H(z) models.

Each model takes a parameter vector (``state``) and returns a vectorized
function ``H(z)``.  Public models expose an ASCII API (``lcdm_model``,
``cpl_model``); Greek symbols are kept only in plot labels.
"""

import numpy as np


def generalized_model(state, model_params, model_expr):
    """Build ``H(z)`` from a dimensionless expression ``E^2(z)``.

    ``state[0]`` is H0; ``model_expr(state, z, model_params)`` returns the
    dimensionless ``(H/H0)^2`` term.  Negative values are clipped to 0 before
    the square root to keep the model real.
    """
    def Hz(z):
        inner_expression = model_expr(state, z, model_params)
        return state[0] * np.sqrt(np.maximum(inner_expression, 0))
    return Hz


def _lcdm_expression(state, z, model_params):
    (Om,) = model_params
    return Om * (1 + z) ** 3 + (1 - Om)


def lcdm_model(state):
    """Flat ΛCDM. ``state = [H0, Om]``.

    ``H(z) = H0 * sqrt(Om (1+z)^3 + (1 - Om))``.
    """
    Om = state[1]
    return generalized_model(state, (Om,), _lcdm_expression)


def cpl_model(state):
    """CPL / w0-wa dynamical dark energy. ``state = [H0, Om, w0, wa]``.

    ``H(z)^2 = H0^2 [ Om (1+z)^3 +
                      (1 - Om)(1+z)^{3(1+w0+wa)} exp(-3 wa z/(1+z)) ]``.
    """
    H0, Om, w0, wa = state

    def Hz(z):
        inner_expression = (
            Om * (1 + z) ** 3
            + (1 - Om) * (1 + z) ** (3 * (1 + w0 + wa) * np.exp(-3 * wa * (z / (z + 1))))
        )
        return H0 * np.sqrt(np.maximum(inner_expression, 0))

    return Hz
