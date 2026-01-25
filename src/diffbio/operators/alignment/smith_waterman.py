"""Smooth Smith-Waterman alignment operator.

This module provides a differentiable implementation of the Smith-Waterman
local alignment algorithm using the logsumexp relaxation (SMURF-style).

Key technique: Replace max with logsumexp and argmax with softmax
to enable gradient flow through the alignment computation.

Reference:
    Petti et al. "End-to-end learning of multiple sequence alignments with
    differentiable Smith-Waterman." Bioinformatics 39(1):btac724, 2023.
"""

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, PyTree

from diffbio.configs import TemperatureConfig
from diffbio.constants import DEFAULT_GAP_EXTEND, DEFAULT_GAP_OPEN
from diffbio.core.base_operators import TemperatureOperator
from diffbio.utils.nn_utils import init_learnable_param


@dataclass
class SmithWatermanConfig(TemperatureConfig):
    """Configuration for SmoothSmithWaterman.

    Attributes:
        temperature: Temperature for logsumexp smoothing.
            Lower = sharper (closer to hard max), Higher = smoother.
        gap_open: Penalty for opening a gap.
        gap_extend: Penalty for extending a gap.
    """

    gap_open: float = DEFAULT_GAP_OPEN
    gap_extend: float = DEFAULT_GAP_EXTEND


class AlignmentResult(NamedTuple):
    """Result of a smooth alignment.

    Attributes:
        score: The soft alignment score.
        alignment_matrix: The DP matrix H[i,j] of shape (len1+1, len2+1).
        soft_alignment: Soft alignment matrix showing position correspondences.
    """

    score: Float[Array, ""]
    alignment_matrix: Float[Array, "len1_plus1 len2_plus1"]
    soft_alignment: Float[Array, "len1 len2"]


class SmoothSmithWaterman(TemperatureOperator):
    """Differentiable Smith-Waterman local alignment.

    This operator implements a smooth version of the Smith-Waterman algorithm
    where max operations are replaced with logsumexp, enabling gradient flow
    through the alignment computation.

    The smoothness is controlled by the temperature parameter:
    - temperature -> 0: Approaches hard max (standard Smith-Waterman)
    - temperature -> inf: Uniform averaging

    Inherits from TemperatureOperator to get:

    - Learnable temperature parameter management
    - soft_max() method using logsumexp relaxation

    Args:
        config: SmithWatermanConfig with alignment parameters.
        scoring_matrix: Scoring matrix for matches/mismatches.
        rngs: Flax NNX random number generators (optional).
        name: Optional operator name.

    Example:
        ```python
        config = SmithWatermanConfig(temperature=1.0)
        scoring = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)
        aligner = SmoothSmithWaterman(config, scoring_matrix=scoring)
        result = aligner.align(seq1, seq2)
        print(result.score)
        ```
    """

    def __init__(
        self,
        config: SmithWatermanConfig,
        scoring_matrix: Array,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the smooth Smith-Waterman aligner.

        Args:
            config: Alignment configuration.
            scoring_matrix: Scoring matrix (alphabet_size, alphabet_size).
            rngs: Random number generators (optional).
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        # Domain-specific learnable parameters (temperature managed by base class)
        self.scoring_matrix = nnx.Param(scoring_matrix)
        self.gap_open = init_learnable_param(config.gap_open)
        self.gap_extend = init_learnable_param(config.gap_extend)

    def _compute_score_matrix(
        self,
        seq1: Float[Array, "len1 alphabet"],
        seq2: Float[Array, "len2 alphabet"],
    ) -> Float[Array, "len1 len2"]:
        """Compute pairwise scoring matrix between sequences.

        Args:
            seq1: First sequence, one-hot encoded (len1, alphabet_size).
            seq2: Second sequence, one-hot encoded (len2, alphabet_size).

        Returns:
            Score matrix of shape (len1, len2).
        """
        # S[i,j] = seq1[i] @ scoring_matrix @ seq2[j].T
        # Using einsum for clarity
        scoring = self.scoring_matrix[...]
        return jnp.einsum("ia,ab,jb->ij", seq1, scoring, seq2)

    def align(
        self,
        seq1: Float[Array, "len1 alphabet"],
        seq2: Float[Array, "len2 alphabet"],
    ) -> AlignmentResult:
        """Perform smooth Smith-Waterman local alignment.

        Args:
            seq1: First sequence, one-hot encoded (len1, alphabet_size).
            seq2: Second sequence, one-hot encoded (len2, alphabet_size).

        Returns:
            AlignmentResult with score, alignment matrix, and soft alignment.
        """
        len1, len2 = seq1.shape[0], seq2.shape[0]

        # Compute pairwise scores
        score_matrix = self._compute_score_matrix(seq1, seq2)

        # Gap penalties
        gap_open = self.gap_open[...]
        gap_extend = self.gap_extend[...]
        # Simplified: use linear gap penalty (gap_open + gap_extend per position)
        gap_penalty = gap_open + gap_extend

        # Initialize DP matrices
        # H[i,j] = alignment score ending at seq1[i-1], seq2[j-1]
        # Using scan for efficient JAX computation

        # Initialize H matrix with zeros
        H = jnp.zeros((len1 + 1, len2 + 1))

        # Fill the DP matrix row by row
        # Note: fori_loop body signature is (i, carry) -> carry
        def fill_row(i, H):
            """Fill row i+1 of the DP matrix."""

            def cell_update(H_prev_col, j_idx):
                """Update single cell H[i+1, j+1]."""
                j = j_idx.astype(jnp.int32)
                s = score_matrix[i, j]
                diag = H[i, j] + s
                up = H[i, j + 1] + gap_penalty
                left = H_prev_col + gap_penalty

                # Use inherited soft_max from TemperatureOperator
                candidates = jnp.stack([jnp.array(0.0), diag, up, left], axis=-1)
                h_new = self.soft_max(candidates, axis=-1)
                return h_new, h_new

            # Scan across columns
            _, new_row = jax.lax.scan(
                cell_update, jnp.array(0.0), jnp.arange(len2, dtype=jnp.int32)
            )

            # Update H matrix
            H = H.at[i + 1, 1:].set(new_row)
            return H

        # Fill all rows using fori_loop for efficiency
        H = jax.lax.fori_loop(0, len1, fill_row, H)

        # Compute final score as smooth max over all positions
        # (local alignment can end anywhere)
        final_candidates = jnp.stack([jnp.array(0.0), jnp.max(H)], axis=-1)
        final_score = self.soft_max(final_candidates, axis=-1)

        # Compute soft alignment (position correspondence probabilities)
        # This is the softmax of the DP matrix (excluding borders)
        H_inner = H[1:, 1:]  # (len1, len2)
        temp = self._temperature  # Use property from TemperatureOperator
        soft_alignment = jax.nn.softmax(H_inner.flatten() / temp).reshape(len1, len2)

        return AlignmentResult(
            score=final_score,
            alignment_matrix=H,
            soft_alignment=soft_alignment,
        )

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply alignment to sequence pair data.

        This method implements the OperatorModule interface for batch processing.
        It expects data containing two sequences and returns alignment results.

        Note: Output preserves input keys for Datarax vmap compatibility,
        while adding alignment result keys.

        Args:
            data: Dictionary containing:
                - "seq1": First sequence, one-hot encoded (len1, alphabet_size)
                - "seq2": Second sequence, one-hot encoded (len2, alphabet_size)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used (deterministic operator)
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains input sequences plus alignment results
                  (score, alignment_matrix, soft_alignment)
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        seq1 = data["seq1"]
        seq2 = data["seq2"]

        # Perform alignment
        result = self.align(seq1, seq2)

        # Build output data - preserve input keys for Datarax vmap compatibility
        transformed_data = {
            "seq1": seq1,
            "seq2": seq2,
            "score": result.score,
            "alignment_matrix": result.alignment_matrix,
            "soft_alignment": result.soft_alignment,
        }

        return transformed_data, state, metadata
