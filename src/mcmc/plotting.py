"""Plotting helpers: trace plots, Gelman-Rubin diagnostics, H(z) band, marginals."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_traces(chains, names, burn_in=None, out=None):
    """Overlay every chain's trajectory, one panel per parameter."""
    p = len(names)
    fig, axes = plt.subplots(p, 1, figsize=(9, 2.2 * p), sharex=True)
    axes = np.atleast_1d(axes)
    for j, (ax, name) in enumerate(zip(axes, names)):
        for chain in chains:
            ax.plot(chain[:, j], lw=0.5, alpha=0.5)
        if burn_in:
            ax.axvline(burn_in, color="k", ls="--", lw=1, label="burn-in")
        ax.set_ylabel(f"${name}$")
    axes[-1].set_xlabel("iteration (post burn-in)")
    axes[0].set_title("Trace plots")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120)
    plt.show()


def plot_gelman_rubin(iters, rhat, names, out=None):
    """R-hat vs iteration, with the 1.1 convergence threshold marked."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for j, name in enumerate(names):
        ax.plot(iters, rhat[:, j], marker="o", ms=3, label=f"${name}$")
    ax.axhline(1.1, color="red", ls="--", lw=1, label="threshold 1.1")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\hat{R}$")
    ax.set_title("Gelman-Rubin convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120)
    plt.show()


def plot_hz_band(z_obs, obs, errors, model, pooled, zvals=None,
                 max_samples=4000, out=None, title=None):
    """H(z) fit with 1sigma/2sigma posterior bands from the pooled samples."""
    if zvals is None:
        zvals = np.linspace(0, 2, 100)
    pooled = np.asarray(pooled)
    if len(pooled) > max_samples:      # thin for memory/speed
        idx = np.random.default_rng(0).choice(len(pooled), max_samples, replace=False)
        pooled = pooled[idx]

    preds = np.array([model(state)(zvals) for state in pooled])
    median = np.median(preds, axis=0)
    lo1, hi1 = np.percentile(preds, [16, 84], axis=0)
    lo2, hi2 = np.percentile(preds, [2.5, 97.5], axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(z_obs, obs, yerr=errors, fmt="o", capsize=5, color="navy",
                ms=5, label="Obs. data")
    ax.plot(zvals, median, color="black", label="posterior median")
    ax.fill_between(zvals, lo2, hi2, color="gray", alpha=0.2, label="2σ (95%)")
    ax.fill_between(zvals, lo1, hi1, color="gray", alpha=0.5, label="1σ (68%)")
    ax.set_xlabel("$z$")
    ax.set_ylabel("$H(z)$ [$km \\cdot s^{-1}/Mpc$]")
    ax.set_title(title or "Evolution of $H(z)$ with confidence intervals")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120)
    plt.show()


def create_marginal(x_data, y_data, x_name, y_name):
    """Joint + marginal posterior for two parameters (median reference lines)."""
    sns.set(style="whitegrid", context="paper")
    g = sns.JointGrid(x=x_data, y=y_data)
    g.plot_marginals(sns.histplot, kde=True, color="Blue", bins=30, discrete=False)
    g.plot_joint(sns.kdeplot, fill=True, cmap="Blues", levels=4, thresh=0.0005)
    g.ax_marg_x.axvline(np.median(x_data), color="#0C134F", linestyle="--")
    g.ax_marg_y.axhline(np.median(y_data), color="#0C134F", linestyle="--")
    g.set_axis_labels(f"${x_name}$", f"${y_name}$", fontsize=14)
    plt.show()
