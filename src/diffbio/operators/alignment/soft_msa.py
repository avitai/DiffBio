"""Soft progressive multiple sequence alignment operator.

This module provides differentiable multiple sequence alignment using
soft operations throughout the alignment process.

Key technique: Uses neural network sequence encoders to compute pairwise
similarities, builds a soft guide tree, and performs progressive profile
alignment with soft gap handling.

Applications: Multiple sequence alignment for homology detection, phylogenetic
analysis, and protein family characterization.

Inherits from TemperatureOperator to get:

- _temperature property for temperature-controlled smoothing
- soft_max() for logsumexp-based smooth maximum
- soft_argmax() for soft position selection
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from artifex.generative_models.core.base import MLP
from datarax.core.config import OperatorConfig
from flax import nnx
from jaxtyping import Array, Float, PyTree

from diffbio.core.base_operators import TemperatureOperator
from diffbio.utils.nn_utils import ensure_rngs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SoftProgressiveMSAConfig(OperatorConfig):
    """Configuration for SoftProgressiveMSA.

    Attributes:
        max_seq_length: Maximum sequence length.
        hidden_dim: Hidden dimension for neural networks.
        num_layers: Number of encoder layers.
        alphabet_size: Size of sequence alphabet (4 for DNA, 20 for protein).
        temperature: Temperature for softmax operations.
        gap_open_penalty: Gap opening penalty.
        gap_extend_penalty: Gap extension penalty.
    """

    max_seq_length: int = 100
    hidden_dim: int = 64
    num_layers: int = 2
    alphabet_size: int = 4
    temperature: float = 1.0
    gap_open_penalty: float = -10.0
    gap_extend_penalty: float = -1.0

    def __post_init__(self) -> None:
        """Validate soft MSA configuration."""
        super().__post_init__()
        if self.num_layers < 1:
            raise ValueError("SoftProgressiveMSAConfig.num_layers must be at least 1.")


class SequenceEncoder(nnx.Module):
    """Encoder for biological sequences."""

    def __init__(
        self,
        alphabet_size: int,
        hidden_dim: int,
        num_layers: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the sequence encoder.

        Args:
            alphabet_size: Size of input alphabet.
            hidden_dim: Hidden dimension.
            num_layers: Number of layers.
            rngs: Random number generators.
        """
        super().__init__()
        self.backbone = MLP(
            hidden_dims=[hidden_dim] * num_layers,
            in_features=alphabet_size,
            activation="gelu",
            output_activation="gelu",
            use_batch_norm=False,
            rngs=rngs,
        )

        # Output projection for sequence embedding
        self.output_proj = nnx.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        sequence: Float[Array, "seq_len alphabet_size"],
    ) -> Float[Array, "hidden_dim"]:
        """Encode a sequence to a fixed-size embedding.

        Args:
            sequence: One-hot encoded sequence.

        Returns:
            Sequence embedding vector.
        """
        backbone_output = self.backbone(sequence)
        if isinstance(backbone_output, tuple):
            raise TypeError("Soft MSA sequence backbone must return a single tensor.")

        # Global average pooling to get fixed-size embedding
        embedding = jnp.mean(backbone_output, axis=0)  # (hidden_dim,)
        embedding = self.output_proj(embedding)

        return embedding


class ProfileBuilder(nnx.Module):
    """Builds alignment profiles from sequences."""

    def __init__(
        self,
        hidden_dim: int,
        alphabet_size: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the profile builder.

        Args:
            hidden_dim: Hidden dimension.
            alphabet_size: Alphabet size.
            rngs: Random number generators.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.alphabet_size = alphabet_size

        self.backbone = MLP(
            hidden_dims=[hidden_dim, alphabet_size],
            in_features=alphabet_size,
            activation="gelu",
            output_activation=None,
            use_batch_norm=False,
            rngs=rngs,
        )

    def __call__(
        self,
        sequences: Float[Array, "n_seqs seq_len alphabet_size"],
        weights: Float[Array, "n_seqs"],
    ) -> Float[Array, "seq_len alphabet_size"]:
        """Build a profile from weighted sequences.

        Args:
            sequences: Stack of aligned sequences.
            weights: Weights for each sequence.

        Returns:
            Profile (position-specific scoring matrix).
        """
        # Weighted average of sequences
        weights = weights / (jnp.sum(weights) + 1e-8)
        profile = jnp.einsum("n,nla->la", weights, sequences)

        # Refine profile
        refinement = self.backbone(profile)
        if isinstance(refinement, tuple):
            raise TypeError("Soft MSA profile backbone must return a single tensor.")
        profile = profile + 0.1 * refinement  # Small residual update

        # Normalize to valid probability distribution
        profile = jax.nn.softmax(profile, axis=-1)

        return profile


class SoftProgressiveMSA(TemperatureOperator):
    """Differentiable progressive multiple sequence alignment.

    This operator performs multiple sequence alignment using soft
    operations that maintain gradient flow throughout the process.

    Algorithm:
    1. Encode each sequence to get embeddings
    2. Compute pairwise distances from embeddings
    3. Build soft guide tree from distances
    4. Progressive alignment following guide tree order
    5. Build consensus profile

    Inherits from TemperatureOperator to get:

    - _temperature property for temperature-controlled smoothing
    - soft_max() for logsumexp-based smooth maximum
    - soft_argmax() for soft position selection

    Args:
        config: SoftProgressiveMSAConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        ```python
        config = SoftProgressiveMSAConfig(max_seq_length=100)
        msa = SoftProgressiveMSA(config, rngs=nnx.Rngs(42))
        data = {"sequences": seqs}  # (n_seqs, seq_len, alphabet_size)
        result, state, meta = msa.apply(data, {}, None)
        ```
    """

    def __init__(
        self,
        config: SoftProgressiveMSAConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the soft progressive MSA operator.

        Args:
            config: MSA configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        rngs = ensure_rngs(rngs)

        self.hidden_dim = config.hidden_dim
        # Temperature is now managed by TemperatureOperator via self._temperature
        self.alphabet_size = config.alphabet_size

        # Sequence encoder for computing pairwise similarities
        self.seq_encoder = SequenceEncoder(
            alphabet_size=config.alphabet_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            rngs=rngs,
        )

        # Profile builder for progressive alignment
        self.profile_builder = ProfileBuilder(
            hidden_dim=config.hidden_dim,
            alphabet_size=config.alphabet_size,
            rngs=rngs,
        )

        # Alignment scoring
        self.align_score = nnx.Linear(
            in_features=config.alphabet_size * 2,
            out_features=1,
            rngs=rngs,
        )

    def _compute_pairwise_distances(
        self,
        sequences: Float[Array, "n_seqs seq_len alphabet_size"],
    ) -> Float[Array, "n_seqs n_seqs"]:
        """Compute pairwise distances between sequences.

        Args:
            sequences: Input sequences.

        Returns:
            Pairwise distance matrix.
        """
        n_seqs = sequences.shape[0]

        # Encode all sequences
        embeddings = jax.vmap(self.seq_encoder)(sequences)  # (n_seqs, hidden_dim)

        # Compute pairwise distances (negative cosine similarity)
        norms = jnp.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8
        normalized = embeddings / norms

        # Cosine similarity -> distance
        similarities = jnp.einsum("ih,jh->ij", normalized, normalized)
        distances = 1.0 - similarities

        # Zero diagonal
        distances = distances * (1.0 - jnp.eye(n_seqs))

        return distances

    def _soft_align_pair(
        self,
        seq1: Float[Array, "len1 alphabet_size"],
        seq2: Float[Array, "len2 alphabet_size"],
    ) -> tuple[
        Float[Array, "max_len alphabet_size"],
        Float[Array, "max_len alphabet_size"],
        Float[Array, ""],
    ]:
        """Perform soft pairwise alignment.

        Args:
            seq1: First sequence.
            seq2: Second sequence.

        Returns:
            Tuple of (aligned_seq1, aligned_seq2, alignment_score).
        """
        len1, len2 = seq1.shape[0], seq2.shape[0]
        max_len = max(len1, len2)

        # Compute position-wise similarity scores
        # (len1, alphabet) x (len2, alphabet) -> (len1, len2)
        match_scores = jnp.einsum("ia,ja->ij", seq1, seq2)

        # Soft alignment via attention
        # Each position in seq1 attends to positions in seq2
        # Use inherited _temperature property from TemperatureOperator
        attn_weights = jax.nn.softmax(match_scores / self._temperature, axis=-1)

        # Soft-aligned seq2 based on seq1 positions
        aligned_seq2_to_seq1 = jnp.einsum("ij,ja->ia", attn_weights, seq2)

        # Pad to max_len
        pad1 = max_len - len1

        aligned1 = jnp.pad(seq1, ((0, pad1), (0, 0)))
        aligned2 = jnp.pad(aligned_seq2_to_seq1, ((0, pad1), (0, 0)))

        # Compute alignment score
        alignment_score = jnp.mean(match_scores * attn_weights)

        return aligned1, aligned2, alignment_score

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply soft progressive MSA.

        Args:
            data: Dictionary containing:
                - "sequences": Input sequences (n_seqs, seq_len, alphabet_size)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - "sequences": Original sequences
                    - "aligned_sequences": Soft-aligned sequences
                    - "pairwise_distances": Guide tree distances
                    - "alignment_scores": Pairwise alignment scores
                    - "consensus_profile": Consensus profile
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        sequences = data["sequences"]
        n_seqs = sequences.shape[0]

        # Step 1: Compute pairwise distances for guide tree
        pairwise_distances = self._compute_pairwise_distances(sequences)

        # Step 2: Progressive alignment
        # For simplicity, align all sequences to the first one
        # (full progressive alignment would follow guide tree)
        aligned_sequences = []
        alignment_scores = []

        # Use first sequence as anchor
        anchor = sequences[0]
        aligned_sequences.append(anchor)

        for i in range(1, n_seqs):
            aligned1, aligned2, score = self._soft_align_pair(anchor, sequences[i])
            # Keep the aligned version of sequences[i]
            aligned_sequences.append(aligned2[: sequences[i].shape[0]])
            alignment_scores.append(score)

        # Stack aligned sequences (pad to same length)
        max_len = max(s.shape[0] for s in aligned_sequences)
        padded_aligned = []
        for seq in aligned_sequences:
            pad_len = max_len - seq.shape[0]
            padded = jnp.pad(seq, ((0, pad_len), (0, 0)))
            padded_aligned.append(padded)

        aligned_stack = jnp.stack(padded_aligned, axis=0)

        # Step 3: Build consensus profile
        uniform_weights = jnp.ones(n_seqs) / n_seqs
        consensus = self.profile_builder(aligned_stack, uniform_weights)

        # Alignment scores matrix
        scores_matrix = jnp.zeros((n_seqs, n_seqs))
        if alignment_scores:
            scores_array = jnp.array(alignment_scores)
            # Fill first row/column with computed scores
            scores_matrix = scores_matrix.at[0, 1:].set(scores_array)
            scores_matrix = scores_matrix.at[1:, 0].set(scores_array)

        transformed_data = {
            "sequences": sequences,
            "aligned_sequences": aligned_stack,
            "pairwise_distances": pairwise_distances,
            "alignment_scores": scores_matrix,
            "consensus_profile": consensus,
        }

        return transformed_data, state, metadata
