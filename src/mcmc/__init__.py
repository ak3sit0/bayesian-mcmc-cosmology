"""Monte Carlo methods for Bayesian inference, applied to H(z) cosmology."""

from .models import lcdm_model, cpl_model, generalized_model
from .inference import (
    chi_squared,
    log_likelihood,
    inverse_variance,
    best_fit,
    information_criteria,
    model_selection_table,
)
from .sampler import (
    metropolis_hastings,
    run_chains,
    gelman_rubin,
    running_gelman_rubin,
    summarize,
)
from .plotting import (
    plot_traces,
    plot_gelman_rubin,
    plot_hz_band,
    plot_model_comparison,
    create_marginal,
)

__all__ = [
    "lcdm_model",
    "cpl_model",
    "generalized_model",
    "chi_squared",
    "log_likelihood",
    "inverse_variance",
    "best_fit",
    "information_criteria",
    "model_selection_table",
    "metropolis_hastings",
    "run_chains",
    "gelman_rubin",
    "running_gelman_rubin",
    "summarize",
    "plot_traces",
    "plot_gelman_rubin",
    "plot_hz_band",
    "plot_model_comparison",
    "create_marginal",
]
