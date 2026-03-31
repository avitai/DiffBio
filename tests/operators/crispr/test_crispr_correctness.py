#!/usr/bin/env python3
"""CRISPR guide scoring correctness tests for DiffBio.

Validates DiffBio's DifferentiableCRISPRScorer operator for output
correctness and differentiability on synthetic one-hot encoded guide
RNA sequences.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# -- Constants ---------------------------------------------------------------

GUIDE_LENGTH = 23
ALPHABET_SIZE = 4


# -- Synthetic data ----------------------------------------------------------


def generate_synthetic_guides(
    n_guides: int,
    guide_length: int = GUIDE_LENGTH,
    seed: int = 42,
) -> jnp.ndarray:
    """Generate random one-hot encoded guide RNA sequences.

    Args:
        n_guides: Number of guides to generate.
        guide_length: Nucleotide length per guide (default 23).
        seed: Random seed for reproducibility.

    Returns:
        One-hot array of shape (n_guides, guide_length, 4).
    """
    key = jax.random.key(seed)
    indices = jax.random.randint(key, (n_guides, guide_length), 0, ALPHABET_SIZE)
    return jax.nn.one_hot(indices, ALPHABET_SIZE)


# -- Validation helpers ------------------------------------------------------


def _validate_scores(
    scores: jnp.ndarray,
    n_guides: int,
) -> dict[str, bool]:
    """Validate efficiency score shape and value constraints.

    Args:
        scores: Predicted efficiency scores.
        n_guides: Expected number of guides.

    Returns:
        Dictionary with shape, finiteness, and range flags.
    """
    shape_valid = scores.shape == (n_guides,)
    finite = bool(jnp.all(jnp.isfinite(scores)))
    in_range = bool(jnp.all(scores >= -1e-6) and jnp.all(scores <= 1.0 + 1e-6))
    return {
        "score_shape_valid": shape_valid,
        "scores_finite": finite,
        "scores_in_range": in_range,
    }


def _validate_features(features: jnp.ndarray) -> dict[str, bool]:
    """Validate extracted feature vectors.

    Args:
        features: Feature vectors from the CNN encoder.

    Returns:
        Dictionary with finiteness flag.
    """
    return {"features_finite": bool(jnp.all(jnp.isfinite(features)))}
