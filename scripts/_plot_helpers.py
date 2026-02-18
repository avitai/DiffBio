"""Shared utilities for documentation plot generation scripts.

Centralizes common plotting patterns, constants, and error handling
used across the example output generation scripts.
"""

import functools
import traceback
from collections.abc import Callable
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Matplotlib setup for headless rendering
matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
ASSETS_DIR = PROJECT_ROOT / "docs" / "assets" / "images" / "examples"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Plot constants
# ---------------------------------------------------------------------------
PLOT_DPI = 150
CMAP_SEQUENTIAL = "viridis"
CMAP_DIVERGING = "RdBu_r"

FIG_WIDE = (10, 5)
FIG_SQUARE = (8, 7)
FIG_LARGE = (10, 8)

# Reusable color palettes
PALETTE_PRIMARY = ["steelblue", "darkorange", "green", "red"]
PALETTE_BATCH = ["#1f77b4", "#ff7f0e", "#2ca02c"]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def save_plot(filename: str, fig: plt.Figure | None = None, dpi: int = PLOT_DPI) -> None:  # type: ignore[type-arg]
    """Save a plot to the assets directory."""
    if fig is None:
        fig = plt.gcf()
    filepath = ASSETS_DIR / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_generator(name: str) -> Callable:  # type: ignore[type-arg]
    """Decorator that wraps a plot-generation function with error handling."""

    def decorator(func: Callable) -> Callable:  # type: ignore[type-arg]
        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> None:
            print(f"\nGenerating {name} plots...")
            try:
                func(*args, **kwargs)
            except Exception as exc:
                print(f"  Error generating {name} plots: {exc}")
                traceback.print_exc()

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Simulated training curves
# ---------------------------------------------------------------------------
def simulated_loss(
    initial: float,
    decay: float,
    noise: float,
    epochs: int,
    floor: float,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Generate a simulated training-loss curve (exponential decay + noise).

    Args:
        initial: Starting loss value.
        decay: Per-epoch multiplicative decay factor (e.g. 0.95).
        noise: Standard deviation of additive Gaussian noise.
        epochs: Number of epochs.
        floor: Minimum loss value (clamp).
        seed: Random seed.

    Returns:
        Array of shape (epochs,) with simulated loss values.
    """
    rng = np.random.RandomState(seed)
    losses = [initial]
    for _ in range(1, epochs):
        losses.append(losses[-1] * decay + rng.randn() * noise)
    return np.maximum(np.array(losses), floor)


# ---------------------------------------------------------------------------
# Heatmap / bar-chart annotation helpers
# ---------------------------------------------------------------------------
def annotate_heatmap(
    ax: plt.Axes,  # type: ignore[type-arg]
    matrix: np.ndarray,
    *,
    threshold: float = 0.5,
    fmt: str = ".2f",
    fontsize: int = 9,
) -> None:
    """Add text annotations to a heatmap with adaptive text colour."""
    n_rows, n_cols = matrix.shape
    for i in range(n_rows):
        for j in range(n_cols):
            colour = "white" if matrix[i, j] > threshold else "black"
            ax.text(
                j,
                i,
                f"{matrix[i, j]:{fmt}}",
                ha="center",
                va="center",
                color=colour,
                fontsize=fontsize,
            )


def label_bars(
    ax: plt.Axes,  # type: ignore[type-arg]
    bars: plt.BarContainer,  # type: ignore[type-arg]
    values: list[float],
    *,
    fmt: str = ".2f",
    fontsize: int = 9,
    va: str = "bottom",
) -> None:
    """Add value labels above bars in a bar chart."""
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:{fmt}}",
            ha="center",
            va=va,
            fontsize=fontsize,
        )
