"""Alignment loss functions for differentiable sequence alignment.

This module provides loss functions for training differentiable alignment
models, including alignment score losses, soft edit distance, and
alignment consistency losses for multi-sequence alignment.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float


class AlignmentScoreLoss(nnx.Module):
    """Loss function based on alignment quality score.

    Computes a loss that measures how well the alignment captures
    the similarity between two sequences. Lower loss indicates
    better alignment of similar positions.

    The loss computes the weighted sum of position-wise mismatches,
    where weights come from the soft alignment matrix.

    Args:
        rngs: Flax NNX random number generators.
    """

    def __init__(self, *, rngs: nnx.Rngs | None = None):
        """Initialize alignment score loss.

        Args:
            rngs: Random number generators (optional).
        """
        super().__init__()

    def __call__(
        self,
        seq1: Float[Array, "len1 alphabet"],
        seq2: Float[Array, "len2 alphabet"],
        alignment: Float[Array, "len1 len2"],
    ) -> Float[Array, ""]:
        """Compute alignment score loss.

        Args:
            seq1: First sequence, soft one-hot encoded (len1, alphabet).
            seq2: Second sequence, soft one-hot encoded (len2, alphabet).
            alignment: Soft alignment matrix where alignment[i,j] indicates
                      probability of aligning position i to position j.

        Returns:
            Scalar loss value. Lower is better alignment.
        """
        # Compute position-wise similarity: seq1[i] dot seq2[j]
        # Higher similarity when same nucleotide
        similarity = jnp.einsum("ia,ja->ij", seq1, seq2)

        # Weight by alignment probabilities
        # High alignment probability * high similarity = good
        weighted_similarity = jnp.sum(alignment * similarity)

        # Convert to loss (negate similarity, normalize)
        max_possible = jnp.sum(alignment)  # If all positions perfectly matched
        loss = 1.0 - (weighted_similarity / jnp.maximum(max_possible, 1e-8))

        return loss


class SoftEditDistanceLoss(nnx.Module):
    """Differentiable approximation of edit distance.

    Computes a soft version of edit distance between two sequences
    that allows gradient flow. Uses the relationship between
    edit distance and alignment scores.

    The edit distance is approximated as the complement of the
    optimal alignment score, scaled appropriately.

    Args:
        normalize: Whether to normalize by sequence length.
        temperature: Temperature for soft minimum operations.
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        normalize: bool = False,
        temperature: float = 0.1,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize soft edit distance loss.

        Args:
            normalize: Whether to normalize by sequence length.
            temperature: Temperature for softmax operations. Lower values
                give sharper approximation of true edit distance.
                Default 0.1 works well for one-hot encoded sequences.
            rngs: Random number generators (optional).
        """
        super().__init__()
        self.normalize = normalize
        self.temperature = nnx.Param(jnp.array(temperature))

    def __call__(
        self,
        seq1: Float[Array, "len1 alphabet"],
        seq2: Float[Array, "len2 alphabet"],
    ) -> Float[Array, ""]:
        """Compute soft edit distance between sequences.

        Args:
            seq1: First sequence, soft one-hot encoded (len1, alphabet).
            seq2: Second sequence, soft one-hot encoded (len2, alphabet).

        Returns:
            Scalar soft edit distance. 0 for identical sequences.
        """
        len1, len2 = seq1.shape[0], seq2.shape[0]

        # Compute position-wise similarity matrix
        # similarity[i,j] = probability that seq1[i] matches seq2[j]
        similarity = jnp.einsum("ia,ja->ij", seq1, seq2)

        temp = self.temperature[...]

        # Use soft assignment to find best match per position
        # Softmax over similarities gives assignment weights
        soft_assign_row = jax.nn.softmax(similarity / temp, axis=1)
        soft_assign_col = jax.nn.softmax(similarity / temp, axis=0)

        # Compute expected similarity under soft assignment
        # For identical sequences: assignment concentrates on diagonal (sim=1)
        # For different sequences: assignment spread out (sim=0 everywhere)
        expected_sim_row = jnp.sum(soft_assign_row * similarity, axis=1)
        expected_sim_col = jnp.sum(soft_assign_col * similarity, axis=0)

        # Total match score = sum of best similarities per position
        match_score_row = jnp.sum(expected_sim_row)
        match_score_col = jnp.sum(expected_sim_col)

        # Distance = unmatched positions
        # Average of (len - match_score) from both perspectives
        dist_row = len1 - match_score_row
        dist_col = len2 - match_score_col
        total_distance = (dist_row + dist_col) / 2.0

        # Ensure non-negative (numerical precision)
        total_distance = jnp.maximum(total_distance, 0.0)

        # Add length difference penalty
        length_penalty = jnp.abs(len1 - len2).astype(jnp.float32)
        total_distance = total_distance + length_penalty

        if self.normalize:
            # Normalize by total length
            total_distance = total_distance / (len1 + len2)

        return total_distance


class AlignmentConsistencyLoss(nnx.Module):
    """Loss for enforcing transitivity in multi-sequence alignments.

    For three sequences A, B, C with pairwise alignments:
    - A->B (align_ab)
    - B->C (align_bc)
    - A->C (align_ac)

    The alignments are consistent if: align_ac ≈ align_ab @ align_bc

    This loss penalizes violations of this transitivity property,
    which is important for producing coherent multiple sequence alignments.

    Args:
        rngs: Flax NNX random number generators.
    """

    def __init__(self, *, rngs: nnx.Rngs | None = None):
        """Initialize alignment consistency loss.

        Args:
            rngs: Random number generators (optional).
        """
        super().__init__()

    def __call__(
        self,
        align_ab: Float[Array, "len_a len_b"],
        align_bc: Float[Array, "len_b len_c"],
        align_ac: Float[Array, "len_a len_c"],
    ) -> Float[Array, ""]:
        """Compute alignment consistency loss.

        Args:
            align_ab: Soft alignment from sequence A to B.
            align_bc: Soft alignment from sequence B to C.
            align_ac: Soft alignment from sequence A to C.

        Returns:
            Scalar loss measuring transitivity violation.
        """
        # Compute expected A->C alignment through B
        # align_ac_expected[i,k] = sum_j align_ab[i,j] * align_bc[j,k]
        align_ac_expected = jnp.matmul(align_ab, align_bc)

        # Normalize to make it a proper probability distribution
        align_ac_expected = align_ac_expected / jnp.maximum(
            jnp.sum(align_ac_expected, axis=1, keepdims=True), 1e-8
        )

        # Compute KL divergence between expected and actual A->C alignment
        # KL(actual || expected) = sum(actual * log(actual / expected))
        eps = 1e-8
        kl_div = jnp.sum(align_ac * jnp.log((align_ac + eps) / (align_ac_expected + eps)))

        # Also compute reverse KL for symmetry
        kl_div_reverse = jnp.sum(
            align_ac_expected * jnp.log((align_ac_expected + eps) / (align_ac + eps))
        )

        # Return symmetric KL (Jensen-Shannon style)
        return (kl_div + kl_div_reverse) / 2.0
