"""
Plot helpers for the loss reserving module. matplotlib is imported only when
a plot is requested.
"""

import numpy as np


def _axes(ax):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib. Install it with: pip install actuneo[viz]"
        ) from exc
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5.5))
    return ax


def plot_development(frame, title, ylabel, ax=None):
    """One line per origin period across development periods."""
    ax = _axes(ax)
    positions = np.arange(frame.shape[1])
    for origin, row in frame.iterrows():
        ax.plot(positions, row.to_numpy(), marker="o", markersize=3, label=str(origin))
    ax.set_xticks(positions)
    ax.set_xticklabels([str(c) for c in frame.columns])
    ax.set_xlabel("Development period")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Origin", fontsize="small", ncol=2)
    ax.grid(True, alpha=0.3)
    return ax


def plot_reserves(latest, ibnr, standard_error, title, ax=None):
    """Stacked bars of latest claims and reserve, with optional error bars."""
    ax = _axes(ax)
    positions = np.arange(len(latest))
    ax.bar(positions, latest.to_numpy(), label="Latest")
    errors = None if standard_error is None else standard_error.to_numpy()
    ax.bar(positions, ibnr.to_numpy(), bottom=latest.to_numpy(), label="Reserve",
           yerr=errors, capsize=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(o) for o in latest.index], rotation=45, ha="right")
    ax.set_xlabel("Origin period")
    ax.set_ylabel("Claims")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def plot_residuals(residuals, title, ax=None):
    """Standardised residuals against development step."""
    ax = _axes(ax)
    positions = np.arange(residuals.shape[1])
    for k in positions:
        column = residuals.iloc[:, k].dropna().to_numpy()
        ax.scatter(np.full(len(column), k), column, s=18, color="C0")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(c) for c in residuals.columns])
    ax.set_xlabel("Development step")
    ax.set_ylabel("Standardised residual")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_distribution(simulations, estimate, title, ax=None, bins=50):
    """Histogram of simulated reserves with the chain-ladder estimate marked."""
    ax = _axes(ax)
    ax.hist(simulations.to_numpy(), bins=bins, alpha=0.8)
    ax.axvline(estimate, color="black", linestyle="--", label="Chain-ladder estimate")
    ax.set_xlabel("Total reserve")
    ax.set_ylabel("Simulations")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return ax
