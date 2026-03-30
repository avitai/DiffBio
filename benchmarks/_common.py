"""Shared utilities for DiffBio benchmarks.

Provides reusable infrastructure extracted from existing benchmark scripts:
synthetic data generators, gradient checking, throughput measurement,
result serialization, and a base result dataclass.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

# ---------------------------------------------------------------------------
# Base result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class BaseBenchmarkResult:
    """Common fields shared by all benchmark results.

    Every domain-specific result dataclass should include these fields
    (either by inheriting or by embedding them).
    """

    timestamp: str
    domain: str
    benchmark_name: str
    gradient_nonzero: bool
    gradient_norm: float
    throughput: float
    throughput_unit: str
    wall_time_seconds: float


# ---------------------------------------------------------------------------
# Platform info
# ---------------------------------------------------------------------------


def collect_platform_info() -> dict[str, str]:
    """Collect runtime platform information for reproducibility.

    Returns:
        Dictionary with jax_version, python_version, platform, device,
        and diffbio_version keys.
    """
    devices = jax.devices()
    device_str = str(devices[0]) if devices else "unknown"

    try:
        import diffbio  # noqa: PLC0415

        diffbio_version = getattr(diffbio, "__version__", "unknown")
    except ImportError:
        diffbio_version = "unknown"

    return {
        "jax_version": jax.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "device": device_str,
        "diffbio_version": diffbio_version,
    }


# ---------------------------------------------------------------------------
# Gradient checking
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class GradientFlowResult:
    """Result of a gradient flow check."""

    gradient_norm: float
    gradient_nonzero: bool


def check_gradient_flow(
    loss_fn: Any,
    model: nnx.Module,
    *args: Any,
) -> GradientFlowResult:
    """Verify gradient flow through a model.

    Computes gradients of ``loss_fn(model, *args)`` with respect to
    the model's ``nnx.Param`` leaves and reports the total norm.

    Args:
        loss_fn: Callable ``(model, *args) -> scalar``.
        model: Flax NNX module to differentiate.
        *args: Additional positional arguments forwarded to *loss_fn*.

    Returns:
        :class:`GradientFlowResult` with ``gradient_norm`` and
        ``gradient_nonzero``.
    """
    grad_fn = nnx.grad(loss_fn)
    grads = grad_fn(model, *args)

    total_norm = 0.0
    for _, param in nnx.iter_graph(grads):
        if hasattr(param, "value") and isinstance(param.value, jnp.ndarray):
            total_norm += float(jnp.sum(param.value ** 2))

    total_norm = total_norm ** 0.5
    return GradientFlowResult(
        gradient_norm=total_norm,
        gradient_nonzero=total_norm > 1e-8,
    )


# ---------------------------------------------------------------------------
# Throughput measurement
# ---------------------------------------------------------------------------


def measure_throughput(
    fn: Any,
    args: tuple,
    n_iterations: int = 100,
    warmup: int = 5,
) -> dict[str, float]:
    """Measure function throughput with JIT warmup.

    Calls ``fn(*args)`` for *warmup* iterations (discarded), then
    *n_iterations* timed iterations. Uses ``block_until_ready()`` on
    JAX arrays to ensure accurate timing.

    Args:
        fn: Callable to benchmark.
        args: Tuple of arguments to pass.
        n_iterations: Number of timed iterations.
        warmup: Number of warmup iterations (not timed).

    Returns:
        Dict with ``total_time_s``, ``per_item_ms``, ``items_per_sec``.
    """
    # Warmup
    for _ in range(warmup):
        result = fn(*args)
        if isinstance(result, jnp.ndarray):
            result.block_until_ready()
        elif isinstance(result, tuple) and len(result) > 0:
            first = result[0]
            if isinstance(first, jnp.ndarray):
                first.block_until_ready()
            elif isinstance(first, dict):
                for v in first.values():
                    if isinstance(v, jnp.ndarray):
                        v.block_until_ready()
                        break

    # Timed iterations
    start = time.perf_counter()
    for _ in range(n_iterations):
        result = fn(*args)
        if isinstance(result, jnp.ndarray):
            result.block_until_ready()
        elif isinstance(result, tuple) and len(result) > 0:
            first = result[0]
            if isinstance(first, jnp.ndarray):
                first.block_until_ready()
            elif isinstance(first, dict):
                for v in first.values():
                    if isinstance(v, jnp.ndarray):
                        v.block_until_ready()
                        break
    total = time.perf_counter() - start

    return {
        "total_time_s": total,
        "per_item_ms": (total / n_iterations) * 1000,
        "items_per_sec": n_iterations / total if total > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------


def save_benchmark_result(
    result: dict[str, Any],
    domain: str,
    benchmark_name: str,
    output_dir: Path = Path("benchmarks/results"),
) -> Path:
    """Save benchmark result as timestamped JSON.

    Creates a domain subdirectory under *output_dir* and writes the
    result dict as JSON with a timestamp in the filename.

    Args:
        result: Dictionary of benchmark results.
        domain: Domain name (used as subdirectory).
        benchmark_name: Benchmark identifier (used in filename).
        output_dir: Root directory for results.

    Returns:
        Path to the saved JSON file.
    """
    domain_dir = output_dir / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = domain_dir / f"{benchmark_name}_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return output_file


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def generate_synthetic_expression(
    n_cells: int = 500,
    n_genes: int = 200,
    n_types: int = 3,
    n_batches: int = 2,
    batch_effect_strength: float = 3.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate synthetic single-cell expression data with known structure.

    Creates count data with per-type expression profiles, batch effects,
    and negative binomial sampling for realistic count distributions.

    Args:
        n_cells: Total number of cells.
        n_genes: Number of genes.
        n_types: Number of cell types.
        n_batches: Number of experimental batches.
        batch_effect_strength: Magnitude of additive batch shift (log scale).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys: counts, library_size, batch_labels,
        cell_type_labels, embeddings, n_cells, n_genes, n_batches, n_types.
    """
    key = jax.random.key(seed)
    keys = jax.random.split(key, 6)

    # Per-type mean expression profiles on log scale
    type_log_means = jax.random.normal(keys[0], (n_types, n_genes)) * 1.5 + 2.0

    # Assign cells to types (roughly equal)
    cells_per_type = n_cells // n_types
    type_labels_list: list[int] = []
    for t in range(n_types):
        count = cells_per_type if t < n_types - 1 else n_cells - len(type_labels_list)
        type_labels_list.extend([t] * count)
    cell_type_labels = jnp.array(type_labels_list)

    # Assign cells to batches (roughly equal)
    cells_per_batch = n_cells // n_batches
    batch_labels_list: list[int] = []
    for b in range(n_batches):
        count = cells_per_batch if b < n_batches - 1 else n_cells - len(batch_labels_list)
        batch_labels_list.extend([b] * count)
    batch_labels = jnp.array(batch_labels_list)

    # Per-batch shift and per-cell noise
    batch_shifts = jax.random.normal(keys[1], (n_batches, n_genes)) * batch_effect_strength
    cell_noise = jax.random.normal(keys[2], (n_cells, n_genes)) * 0.3
    cell_log_means = type_log_means[cell_type_labels] + batch_shifts[batch_labels] + cell_noise

    # Negative binomial sampling (Gamma-Poisson mixture)
    rates = jnp.exp(cell_log_means)
    dispersion = 5.0
    gamma_samples = jax.random.gamma(keys[3], dispersion, (n_cells, n_genes))
    scaled_rates = rates * gamma_samples / dispersion
    counts = jax.random.poisson(keys[4], scaled_rates).astype(jnp.float32)

    library_size = jnp.sum(counts, axis=-1)

    # PCA-like embeddings from the clean type structure (no batch effect)
    clean_embeddings = type_log_means[cell_type_labels] + cell_noise
    # Simple dimensionality reduction via random projection
    proj = jax.random.normal(keys[5], (n_genes, min(50, n_genes)))
    embeddings = clean_embeddings @ proj / jnp.sqrt(n_genes)

    return {
        "counts": counts,
        "library_size": library_size,
        "batch_labels": batch_labels,
        "cell_type_labels": cell_type_labels,
        "embeddings": embeddings,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "n_batches": n_batches,
        "n_types": n_types,
    }


def generate_synthetic_sequences(
    n_seqs: int = 100,
    seq_len: int = 50,
    alphabet_size: int = 4,
    seed: int = 42,
) -> jnp.ndarray:
    """Generate random one-hot encoded sequences.

    Args:
        n_seqs: Number of sequences.
        seq_len: Length of each sequence.
        alphabet_size: Size of the alphabet (4 for DNA/RNA).
        seed: Random seed.

    Returns:
        One-hot encoded array of shape ``(n_seqs, seq_len, alphabet_size)``.
    """
    key = jax.random.key(seed)
    indices = jax.random.randint(key, (n_seqs, seq_len), 0, alphabet_size)
    return jax.nn.one_hot(indices, alphabet_size)


def generate_synthetic_coverage(
    length: int = 10000,
    n_peaks: int = 20,
    peak_width_range: tuple[int, int] = (50, 500),
    background_rate: float = 5.0,
    peak_height_range: tuple[float, float] = (20.0, 100.0),
    seed: int = 42,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate synthetic coverage signal with known peak positions.

    Creates a Poisson-distributed background signal with Gaussian peaks
    at random positions. Returns the signal and a binary truth mask
    indicating peak regions.

    Args:
        length: Signal length in positions.
        n_peaks: Number of peaks to insert.
        peak_width_range: (min_width, max_width) of each peak in positions.
        background_rate: Mean Poisson rate for background.
        peak_height_range: (min_height, max_height) above background.
        seed: Random seed.

    Returns:
        Tuple of (signal, truth_mask) where truth_mask is 1.0 at peak
        positions and 0.0 elsewhere.
    """
    key = jax.random.key(seed)
    keys = jax.random.split(key, 4)

    # Background signal
    background = jax.random.poisson(keys[0], background_rate, (length,)).astype(jnp.float32)

    # Peak parameters
    positions = jax.random.randint(keys[1], (n_peaks,), 0, length)
    min_w, max_w = peak_width_range
    widths = jax.random.randint(keys[2], (n_peaks,), min_w, max(min_w + 1, max_w))
    min_h, max_h = peak_height_range
    heights = jax.random.uniform(keys[3], (n_peaks,), minval=min_h, maxval=max_h)

    # Build signal and truth mask using numpy for dynamic indexing
    import numpy as np  # noqa: PLC0415

    signal_np = np.array(background)
    truth_np = np.zeros(length, dtype=np.float32)

    for i in range(n_peaks):
        center = int(positions[i])
        width = int(widths[i])
        height = float(heights[i])
        half_w = width // 2
        start = max(0, center - half_w)
        end = min(length, center + half_w)

        # Gaussian peak shape
        x = np.arange(start, end) - center
        peak = height * np.exp(-0.5 * (x / (width / 6.0)) ** 2)
        signal_np[start:end] += peak
        truth_np[start:end] = 1.0

    return jnp.array(signal_np), jnp.array(truth_np)
