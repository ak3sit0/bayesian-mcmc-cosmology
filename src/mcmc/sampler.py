"""Metropolis-Hastings sampler and MCMC diagnostics.

The sampler works entirely in log-likelihood space for numerical stability and
supports burn-in, acceptance tracking, many-chain runs, and the Gelman-Rubin
convergence diagnostic.
"""

import numpy as np
import pandas as pd

from .inference import chi_squared, log_likelihood


def _within_bounds(state, bounds):
    """True if every parameter lies inside its (low, high) box."""
    if bounds is None:
        return True
    for value, (low, high) in zip(state, bounds):
        if value < low or value > high:
            return False
    return True


def metropolis_hastings(n_steps, initial, proposal_std, model, observed,
                        inv_cov, burn_in=0, bounds=None, rng=None):
    """Random-walk Metropolis-Hastings.

    ``bounds``: optional list of ``(low, high)`` per parameter — a uniform prior
    that makes the posterior proper (proposals outside the box are rejected).

    Returns ``(chain, acceptance_rate)`` where ``chain`` is a
    ``(n_steps - burn_in, n_params)`` array of post-burn-in samples.
    """
    rng = np.random.default_rng(rng)
    initial = np.asarray(initial, dtype=float)
    proposal_std = np.asarray(proposal_std, dtype=float)

    # A 1D proposal_std is an independent (diagonal) random walk; a 2D array is
    # a full proposal covariance (preconditioned walk) that mixes correlated
    # parameters, via its Cholesky factor.
    preconditioned = proposal_std.ndim == 2
    chol = np.linalg.cholesky(proposal_std) if preconditioned else None

    current = initial
    current_logl = log_likelihood(chi_squared(model, current, observed, inv_cov))

    chain = np.empty((n_steps, initial.size))
    n_accepted = 0
    for i in range(n_steps):
        if preconditioned:
            proposed = current + chol @ rng.standard_normal(initial.size)
        else:
            proposed = rng.normal(current, proposal_std)
        # Uniform prior: reject anything outside the box before touching chi^2.
        if _within_bounds(proposed, bounds):
            proposed_logl = log_likelihood(chi_squared(model, proposed, observed, inv_cov))
            # Accept in log space: exp is never formed, so no underflow.
            if proposed_logl >= current_logl or np.log(rng.random()) < proposed_logl - current_logl:
                current, current_logl = proposed, proposed_logl
                n_accepted += 1
        chain[i] = current

    acceptance_rate = n_accepted / n_steps
    return chain[burn_in:], acceptance_rate


def run_chains(n_chains, n_steps, init_fn, proposal_std, model, observed,
               inv_cov, burn_in=0, bounds=None, seed=0):
    """Run ``n_chains`` independent chains from over-dispersed starts.

    ``init_fn(rng)`` returns a (random) initial state for each chain.
    Returns ``(chains, acceptance_rates)``: a list of post-burn-in arrays and
    an array of per-chain acceptance rates.
    """
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(n_chains)
    chains, rates = [], []
    for s in child_seeds:
        rng = np.random.default_rng(s)
        chain, rate = metropolis_hastings(
            n_steps, init_fn(rng), proposal_std, model, observed,
            inv_cov, burn_in=burn_in, bounds=bounds, rng=rng,
        )
        chains.append(chain)
        rates.append(rate)
    return chains, np.array(rates)


def gelman_rubin(chains):
    """Gelman-Rubin potential scale reduction factor R-hat per parameter.

    ``chains``: list of ``(n, p)`` arrays (already burn-in trimmed, equal n).
    Converged chains give R-hat -> 1 (typically the target is < 1.1).
    """
    data = np.asarray(chains)              # (m, n, p)
    m, n, _ = data.shape
    chain_means = data.mean(axis=1)        # (m, p)
    grand_mean = chain_means.mean(axis=0)  # (p,)

    # Between-chain variance B and within-chain variance W.
    B = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2, axis=0)
    W = data.var(axis=1, ddof=1).mean(axis=0)

    var_plus = (n - 1) / n * W + B / n
    return np.sqrt(var_plus / W)


def running_gelman_rubin(chains, n_points=40):
    """R-hat computed on the first k samples, for k growing to full length.

    Returns ``(iters, rhat)`` where ``rhat`` has shape ``(len(iters), p)`` — the
    canonical Gelman-Rubin convergence plot.
    """
    n = min(len(c) for c in chains)
    iters = np.linspace(max(10, n // n_points), n, n_points, dtype=int)
    iters = np.unique(iters)
    rhat = np.array([gelman_rubin([c[:k] for c in chains]) for k in iters])
    return iters, rhat


def summarize(pooled, names):
    """Posterior summary table: mean, std, median, 16th/84th percentiles."""
    pooled = np.asarray(pooled)
    lo, med, hi = np.percentile(pooled, [16, 50, 84], axis=0)
    return pd.DataFrame(
        {
            "mean": pooled.mean(axis=0),
            "std": pooled.std(axis=0, ddof=1),
            "median": med,
            "p16": lo,
            "p84": hi,
        },
        index=names,
    )
