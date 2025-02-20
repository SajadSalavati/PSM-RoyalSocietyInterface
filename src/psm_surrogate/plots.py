"""Publication-style plots for POD energy and reconstruction error.

These reproduce the figures from the notebooks (cumulative energy vs. number of
modes, and per-snapshot reconstruction error), but as reusable functions that take
data and an optional ``save_path`` instead of inlining everything.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _style() -> list:
    plt.rc("font", family="DejaVu Serif", size=18)
    return sns.color_palette("deep", n_colors=100)


def plot_cumulative_energy(
    cumulative: np.ndarray,
    n_modes: int = 50,
    energy_line: float = 0.99,
    title: str = "Accumulated Energy",
    save_path: str | None = None,
):
    """Plot cumulative POD energy vs. number of retained modes."""
    colors = _style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        cumulative[:n_modes],
        color=colors[0],
        linewidth=3,
        marker="D",
        markersize=5,
        markevery=1,
    )
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.set_xlabel("# Modes", fontsize=20, weight="bold")
    ax.set_ylabel("Energy", fontsize=20, weight="bold")
    ax.axhline(y=energy_line, linewidth=3, color=colors[1])
    ax.grid(True)
    ax.set_title(title, fontsize=24, weight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=400, bbox_inches="tight")
    return fig, ax


def plot_reconstruction_error(
    error: np.ndarray,
    title: str = "Reconstruction Error",
    save_path: str | None = None,
):
    """Plot per-snapshot relative reconstruction error (log scale)."""
    colors = _style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(error, color=colors[3], linewidth=3)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.set_xlabel("Snapshot", fontsize=20, weight="bold")
    ax.set_ylabel("L2 Norm Difference %", fontsize=20, weight="bold")
    ax.set_yscale("log")
    ax.set_xlim(0, len(error))
    ax.set_ylim(0.01, 100)
    ax.grid(True)
    ax.set_title(title, fontsize=24, weight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=400, bbox_inches="tight")
    return fig, ax
