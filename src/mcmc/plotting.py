"""Plotting helpers: trace plots, Gelman-Rubin diagnostics, H(z) band, marginals."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde


def plot_traces(chains, names, burn_in=None, out=None, max_points=1000):
    """Overlay every chain's trajectory, one panel per parameter.

    Chains are thinned to at most ``max_points`` points *for display only*, so
    the embedded PNG stays light (the diagnostic shape is unchanged).
    """
    p = len(names)
    step = max(1, len(chains[0]) // max_points)
    x = np.arange(0, len(chains[0]), step)
    fig, axes = plt.subplots(p, 1, figsize=(9, 2.2 * p), sharex=True, dpi=80)
    axes = np.atleast_1d(axes)
    for j, (ax, name) in enumerate(zip(axes, names)):
        for chain in chains:
            ax.plot(x, chain[::step, j], lw=0.5, alpha=0.5)
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


def plot_model_comparison(table, out=None):
    """Bar chart of dAIC / dBIC per model (lower is better; best sits at 0).

    Guide lines at Delta = 2, 6, 10 mark the usual evidence thresholds against
    the higher-criterion model.
    """
    models = list(table.index)
    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, table["dAIC"], width, label=r"$\Delta$AIC", color="#1D267D")
    ax.bar(x + width / 2, table["dBIC"], width, label=r"$\Delta$BIC", color="#D4ADFC")
    for thr in (2, 6, 10):
        ax.axhline(thr, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.text(len(models) - 0.5, thr, f" Δ={thr}", va="center", fontsize=8, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel(r"$\Delta$ (relative to best model)")
    ax.set_title("Model selection: ΛCDM vs CPL")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120)
    plt.show()


def _credible_contours(ax, x, y, grid=120):
    """Fill the 68% and 95% highest-density credible regions of a 2D posterior.

    Uses a Gaussian KDE evaluated on a grid; the contour levels are the density
    thresholds that enclose 68% and 95% of the total probability mass. This is
    the standard corner-plot look and is faithful to non-Gaussian posteriors
    (unlike a covariance ellipse).
    """
    kde = gaussian_kde(np.vstack([x, y]))
    xi = np.linspace(x.min(), x.max(), grid)
    yi = np.linspace(y.min(), y.max(), grid)
    X, Y = np.meshgrid(xi, yi)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    order = np.sort(Z.ravel())[::-1]
    cum = np.cumsum(order) / order.sum()
    l68 = order[np.searchsorted(cum, 0.68)]
    l95 = order[np.searchsorted(cum, 0.95)]

    ax.contourf(X, Y, Z, levels=[l95, l68, Z.max()],
                colors=["#bcd4f0", "#3b78c3"], alpha=0.9)
    ax.contour(X, Y, Z, levels=[l95, l68], colors="#0C134F", linewidths=1)


def create_marginal(x_data, y_data, x_name, y_name, out=None):
    """Joint + marginal posterior for two parameters.

    The joint panel shows the 68% / 95% credible regions; the marginals are
    histograms with a KDE and a dashed median line. If ``out`` is given, the
    figure is also saved there.
    """
    sns.set_theme(style="whitegrid", context="paper")
    x = np.asarray(x_data)
    y = np.asarray(y_data)
    g = sns.JointGrid(x=x, y=y)
    g.plot_marginals(sns.histplot, kde=True, color="Blue", bins=30, discrete=False)
    _credible_contours(g.ax_joint, x, y)
    g.ax_marg_x.axvline(np.median(x), color="#0C134F", linestyle="--")
    g.ax_marg_y.axhline(np.median(y), color="#0C134F", linestyle="--")
    g.set_axis_labels(f"${x_name}$", f"${y_name}$", fontsize=14)
    if out:
        g.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
