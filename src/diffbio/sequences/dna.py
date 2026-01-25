"""DNA sequence data types and encoding utilities for DiffBio.

This module provides functions and utilities for working with DNA sequences
in a JAX-compatible, differentiable manner.
"""

from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


# DNA nucleotide alphabet
DNA_ALPHABET = "ACGT"
DNA_ALPHABET_SIZE = 4

# Mapping from nucleotide to index
_NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": -1}
_IDX_TO_NUC = {0: "A", 1: "C", 2: "G", 3: "T"}


def encode_dna_string(sequence: str, handle_n: Literal["uniform", "zero"] = "uniform") -> Array:
    """Encode a DNA string as a one-hot JAX array.

    Args:
        sequence: DNA string containing only A, C, G, T, N characters.
        handle_n: How to handle N (unknown) nucleotides:
            - "uniform": Encode as uniform distribution [0.25, 0.25, 0.25, 0.25]
            - "zero": Encode as zeros [0, 0, 0, 0]

    Returns:
        One-hot encoded array of shape (len(sequence), 4).
        Columns represent A, C, G, T in that order.

    Example:
        ```python
        encode_dna_string("ACGT")
        ```
        Array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]], dtype=float32)
    """
    sequence = sequence.upper()

    # Convert to indices
    indices = []
    for nuc in sequence:
        if nuc in _NUC_TO_IDX:
            indices.append(_NUC_TO_IDX[nuc])
        else:
            raise ValueError(f"Invalid nucleotide: {nuc}. Expected A, C, G, T, or N.")

    indices_array = jnp.array(indices, dtype=jnp.int32)

    # One-hot encode valid nucleotides (indices 0-3)
    # N nucleotides have index -1 and will be handled separately
    valid_mask = indices_array >= 0
    safe_indices = jnp.where(valid_mask, indices_array, 0)  # Use 0 for safe indexing
    one_hot = jax.nn.one_hot(safe_indices, DNA_ALPHABET_SIZE)

    # Handle N nucleotides
    if handle_n == "uniform":
        n_encoding = jnp.ones(DNA_ALPHABET_SIZE) / DNA_ALPHABET_SIZE
    else:  # "zero"
        n_encoding = jnp.zeros(DNA_ALPHABET_SIZE)

    # Apply N encoding where needed
    n_mask = ~valid_mask
    one_hot = jnp.where(n_mask[:, None], n_encoding, one_hot)

    return one_hot


def decode_dna_onehot(encoded: Array, threshold: float = 0.5) -> str:
    """Decode a one-hot encoded DNA array back to a string.

    Args:
        encoded: One-hot encoded array of shape (length, 4).
        threshold: Minimum confidence to assign a nucleotide.
            Below threshold, returns 'N'.

    Returns:
        DNA string.
    """
    # Get argmax for each position
    indices = jnp.argmax(encoded, axis=-1)
    max_vals = jnp.max(encoded, axis=-1)

    # Build string
    result = []
    for i in range(len(indices)):
        idx = int(indices[i])
        max_val = float(max_vals[i])
        if max_val >= threshold and idx in _IDX_TO_NUC:
            result.append(_IDX_TO_NUC[idx])
        else:
            result.append("N")

    return "".join(result)


def phred_to_probability(phred_scores: Array) -> Array:
    """Convert Phred quality scores to error probabilities.

    Args:
        phred_scores: Array of Phred scores (typically 0-40).

    Returns:
        Array of error probabilities in range [0, 1].

    Note:
        Error probability = 10^(-Q/10) where Q is Phred score.
    """
    return jnp.power(10.0, -phred_scores / 10.0)


def probability_to_phred(error_prob: Array, max_phred: float = 60.0) -> Array:
    """Convert error probabilities to Phred quality scores.

    Args:
        error_prob: Array of error probabilities in range (0, 1].
        max_phred: Maximum Phred score to return (for numerical stability).

    Returns:
        Array of Phred scores.
    """
    # Clip to avoid log(0)
    safe_prob = jnp.clip(error_prob, 1e-10, 1.0)
    phred = -10.0 * jnp.log10(safe_prob)
    return jnp.clip(phred, 0.0, max_phred)


def soft_encode_dna(
    sequence: Array,
    quality_scores: Array,
    temperature: float = 1.0,
) -> Array:
    """Create soft one-hot encoding weighted by quality scores.

    This creates a differentiable encoding where positions with low quality
    have more uniform distributions (higher entropy).

    Args:
        sequence: One-hot encoded sequence of shape (length, 4).
        quality_scores: Phred quality scores of shape (length,).
        temperature: Temperature parameter controlling softness.
            Lower = sharper (more like hard one-hot)
            Higher = softer (more uniform)

    Returns:
        Soft-encoded array of shape (length, 4) where each row
        sums to 1 but may not be exactly one-hot.
    """
    # Convert quality to confidence (higher quality = higher confidence)
    confidence = 1.0 - phred_to_probability(quality_scores)

    # Scale one-hot by confidence and add uniform noise for uncertainty
    uniform = jnp.ones(DNA_ALPHABET_SIZE) / DNA_ALPHABET_SIZE

    # Weighted combination: high confidence -> one-hot, low confidence -> uniform
    soft_encoded = confidence[:, None] * sequence + (1 - confidence[:, None]) * uniform

    # Apply temperature scaling and softmax for normalization
    logits = jnp.log(soft_encoded + 1e-10) / temperature
    return jax.nn.softmax(logits, axis=-1)


def complement_dna(encoded: Array) -> Array:
    """Get the complement of a DNA sequence.

    Complement mapping: A<->T, C<->G

    Args:
        encoded: One-hot encoded DNA of shape (..., 4).

    Returns:
        Complemented sequence of same shape.
    """
    # Complement swaps A<->T (indices 0<->3) and C<->G (indices 1<->2)
    # Permutation: [0,1,2,3] -> [3,2,1,0]
    return encoded[..., ::-1]


def reverse_complement_dna(encoded: Array) -> Array:
    """Get the reverse complement of a DNA sequence.

    Args:
        encoded: One-hot encoded DNA of shape (length, 4).

    Returns:
        Reverse complemented sequence of same shape.
    """
    return complement_dna(encoded[::-1])


def gc_content(encoded: Array) -> Float[Array, ""]:
    """Calculate GC content of a DNA sequence.

    Args:
        encoded: One-hot encoded DNA of shape (length, 4).

    Returns:
        Scalar GC content as fraction in [0, 1].
    """
    # G is index 2, C is index 1
    gc_sum = jnp.sum(encoded[:, 1] + encoded[:, 2])
    total = jnp.sum(encoded)
    return gc_sum / (total + 1e-10)


def create_dna_element_data(
    sequence: str | Array,
    quality_scores: Array | None = None,
) -> dict:
    """Create data dictionary for a DNA sequence Element.

    This creates a data structure compatible with Datarax's Element.

    Args:
        sequence: DNA string or pre-encoded one-hot array.
        quality_scores: Optional Phred quality scores.

    Returns:
        Dictionary suitable for Element(data=...).
    """
    # Encode sequence if string
    if isinstance(sequence, str):
        encoded_seq = encode_dna_string(sequence)
    else:
        encoded_seq = sequence

    data = {"sequence": encoded_seq}

    if quality_scores is not None:
        data["quality_scores"] = quality_scores

    return data
