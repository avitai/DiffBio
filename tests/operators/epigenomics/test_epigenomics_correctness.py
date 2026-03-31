#!/usr/bin/env python3
"""Epigenomics correctness tests for DiffBio.

Validates DiffBio's epigenomics operators for output shape
correctness, probability ranges, and gradient flow:
- DifferentiablePeakCaller (CNN-based peak calling)
- FNOPeakCaller (Fourier Neural Operator peak calling)
- ChromatinStateAnnotator (HMM-based chromatin state annotation)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from diffbio.operators.epigenomics.chromatin_state import (
    ChromatinStateAnnotator,
    ChromatinStateConfig,
)
from diffbio.operators.epigenomics.fno_peak_calling import (
    FNOPeakCaller,
    FNOPeakCallerConfig,
)
from diffbio.operators.epigenomics.peak_calling import (
    DifferentiablePeakCaller,
    PeakCallerConfig,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _generate_histone_marks(
    length: int,
    n_marks: int = 6,
    n_states: int = 4,
    seed: int = 99,
) -> jnp.ndarray:
    """Generate synthetic histone mark data with known state structure.

    Creates a signal where positions are assigned to one of
    ``n_states`` latent states, and each state has a characteristic
    pattern of histone marks.

    Args:
        length: Number of genomic positions.
        n_marks: Number of histone marks (channels).
        n_states: Number of latent chromatin states.
        seed: Random seed for reproducibility.

    Returns:
        Histone mark array of shape ``(length, n_marks)``.
    """
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key, 2)

    # State-specific mark profiles (logit scale)
    state_profiles = jax.random.normal(k1, (n_states, n_marks)) * 2.0

    # Assign positions to states in blocks
    block_size = length // n_states
    state_indices = jnp.repeat(
        jnp.arange(n_states),
        block_size,
    )
    # Pad to exact length if needed
    if state_indices.shape[0] < length:
        pad = jnp.full(
            (length - state_indices.shape[0],),
            n_states - 1,
        )
        state_indices = jnp.concatenate([state_indices, pad])
    state_indices = state_indices[:length]

    # Build signal from profiles plus noise
    marks = state_profiles[state_indices]
    noise = jax.random.normal(k2, (length, n_marks)) * 0.5
    return marks + noise


# ------------------------------------------------------------------
# Operator tests
# ------------------------------------------------------------------


def _test_peak_caller(
    coverage: jnp.ndarray,
) -> tuple[dict, DifferentiablePeakCaller]:
    """Test DifferentiablePeakCaller on synthetic coverage.

    Args:
        coverage: Coverage signal of shape ``(length,)``.

    Returns:
        Tuple of (metrics dict, operator instance).
    """
    config = PeakCallerConfig(
        window_size=200,
        num_filters=16,
        kernel_sizes=(5, 11, 21),
        threshold=0.5,
        temperature=1.0,
    )
    rngs = nnx.Rngs(42)
    peak_caller = DifferentiablePeakCaller(config, rngs=rngs)

    data = {"coverage": coverage}
    result, _, _ = peak_caller.apply(data, {}, None)

    length = coverage.shape[0]
    shape_ok = (
        result["peak_scores"].shape == (length,)
        and result["peak_probabilities"].shape == (length,)
        and result["peak_summits"].shape == (length,)
        and result["peak_starts"].shape == (length,)
        and result["peak_ends"].shape == (length,)
    )
    probs = result["peak_probabilities"]
    probs_in_range = bool(jnp.all(probs >= 0.0) and jnp.all(probs <= 1.0))

    return {
        "shape_ok": shape_ok,
        "probs_in_range": probs_in_range,
        "peak_probabilities": probs,
    }, peak_caller


def _test_fno_peak_caller(
    coverage: jnp.ndarray,
) -> tuple[dict, FNOPeakCaller]:
    """Test FNOPeakCaller on synthetic coverage.

    Args:
        coverage: Coverage signal of shape ``(length,)``.

    Returns:
        Tuple of (metrics dict, operator instance).
    """
    config = FNOPeakCallerConfig(
        hidden_channels=16,
        modes=8,
        num_layers=2,
        threshold=0.5,
        temperature=1.0,
    )
    rngs = nnx.Rngs(42)
    fno_caller = FNOPeakCaller(config, rngs=rngs, name="fno_peak")

    data = {"coverage": coverage}
    result, _, _ = fno_caller.apply(data, {}, None)

    length = coverage.shape[0]
    shape_ok = result["peak_scores"].shape == (length,) and result["peak_probabilities"].shape == (
        length,
    )
    probs = result["peak_probabilities"]
    probs_in_range = bool(jnp.all(probs >= 0.0) and jnp.all(probs <= 1.0))

    return {
        "shape_ok": shape_ok,
        "probs_in_range": probs_in_range,
        "peak_probabilities": probs,
    }, fno_caller


def _test_chromatin_state(
    histone_marks: jnp.ndarray,
    n_marks: int,
    n_states: int,
) -> tuple[dict, ChromatinStateAnnotator]:
    """Test ChromatinStateAnnotator on synthetic histone data.

    Args:
        histone_marks: Histone mark array of shape
            ``(length, n_marks)``.
        n_marks: Number of histone mark channels.
        n_states: Number of chromatin states.

    Returns:
        Tuple of (metrics dict, operator instance).
    """
    config = ChromatinStateConfig(
        num_states=n_states,
        num_marks=n_marks,
        temperature=1.0,
    )
    rngs = nnx.Rngs(42)
    annotator = ChromatinStateAnnotator(config, rngs=rngs)

    data = {"histone_marks": histone_marks}
    result, _, _ = annotator.apply(data, {}, None)

    # Posteriors should sum to 1 along the state axis
    posteriors = result["state_posteriors"]
    sums = jnp.sum(posteriors, axis=-1)
    posteriors_sum_ok = bool(jnp.allclose(sums, 1.0, atol=1e-4))

    return {
        "posteriors_sum_ok": posteriors_sum_ok,
    }, annotator


def _compute_pearson_correlation(
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> float:
    """Compute Pearson correlation between two 1-D arrays.

    Args:
        x: First array.
        y: Second array.

    Returns:
        Pearson correlation coefficient.
    """
    x_centered = x - jnp.mean(x)
    y_centered = y - jnp.mean(y)
    numerator = jnp.sum(x_centered * y_centered)
    denominator = jnp.sqrt(jnp.sum(x_centered**2) * jnp.sum(y_centered**2))
    # Guard against zero variance
    safe_denom = jnp.where(denominator > 1e-12, denominator, 1.0)
    return float(numerator / safe_denom)
