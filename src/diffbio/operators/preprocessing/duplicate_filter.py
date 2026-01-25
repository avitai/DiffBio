"""Differentiable duplicate weighting operator.

This module provides a probabilistic duplicate weighting operator that assigns
soft weights to reads based on their uniqueness, instead of hard duplicate
removal.

Key technique: Soft clustering of reads by sequence similarity with weights
inversely proportional to cluster size.

Inherits from TemperatureOperator to get:

- _temperature property for temperature-controlled smoothing
- soft_max() for logsumexp-based smooth maximum
- soft_argmax() for soft position selection
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from flax import nnx
from jaxtyping import Array, Float, PyTree

from diffbio.core.base_operators import TemperatureOperator


@dataclass
class DuplicateWeightingConfig(OperatorConfig):
    """Configuration for DifferentiableDuplicateWeighting.

    Attributes:
        temperature: Temperature for soft similarity computation.
            Lower = sharper clustering, Higher = smoother.
        similarity_threshold: Minimum similarity to consider as duplicate.
        embedding_dim: Dimension of learned sequence embedding.
    """

    temperature: float = 1.0
    learnable_temperature: bool = True
    similarity_threshold: float = 0.9
    embedding_dim: int = 32


class DifferentiableDuplicateWeighting(TemperatureOperator):
    """Differentiable duplicate weighting for sequencing reads.

    This operator assigns probabilistic weights to reads based on their
    uniqueness within a batch. Instead of hard duplicate removal, it
    down-weights reads that are similar to others, maintaining gradient flow.

    The algorithm:
    1. Embed sequences using learned convolutional features
    2. Compute pairwise soft similarity matrix
    3. Compute soft cluster sizes from similarity matrix
    4. Assign weights inversely proportional to cluster size

    Note: This operator works on batched data where reads can be compared.
    For single-read processing, it returns weight=1.0.

    Args:
        config: DuplicateWeightingConfig with weighting parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        ```python
        config = DuplicateWeightingConfig(similarity_threshold=0.9)
        weighter = DifferentiableDuplicateWeighting(config, rngs=nnx.Rngs(42))
        data = {"sequence": encoded_seq, "quality_scores": quality}
        result, state, meta = weighter.apply(data, {}, None)
        ```
    """

    def __init__(
        self,
        config: DuplicateWeightingConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the duplicate weighting operator.

        Args:
            config: Duplicate weighting configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        # Learnable parameters
        # Temperature is managed by TemperatureOperator via self._temperature
        self.similarity_threshold = nnx.Param(jnp.array(config.similarity_threshold))

        # Simple embedding: learned projection from one-hot to embedding
        # This will be applied via convolution for position-invariant features
        embedding_dim = config.embedding_dim
        if rngs is not None:
            key = rngs.params()
        else:
            key = jax.random.key(0)

        # Convolution kernel for sequence embedding (kernel_size=7)
        kernel_shape = (7, 4, embedding_dim)  # (kernel_size, in_channels, out_channels)
        self.conv_kernel = nnx.Param(jax.random.normal(key, kernel_shape) * 0.1)

    def _embed_sequence(
        self,
        sequence: Float[Array, "length alphabet"],
    ) -> Float[Array, "embedding_dim"]:
        """Embed a sequence into a fixed-size vector.

        Uses 1D convolution followed by global average pooling to create
        a fixed-size embedding regardless of sequence length.

        Args:
            sequence: One-hot encoded sequence (length, 4).

        Returns:
            Embedding vector of shape (embedding_dim,).
        """
        kernel = self.conv_kernel[...]

        # 1D convolution: (length, alphabet) -> (length - kernel_size + 1, embedding_dim)
        # Using jax.lax.conv for 1D convolution
        # Need to reshape for conv: add batch and channel dims
        seq_reshaped = sequence[None, :, :]  # (1, length, 4)

        # jax.lax.conv expects (batch, in_channels, spatial)
        seq_transposed = jnp.transpose(seq_reshaped, (0, 2, 1))  # (1, 4, length)
        kernel_transposed = jnp.transpose(kernel, (2, 1, 0))  # (out, in, kernel_size)

        # Perform convolution
        conv_out = jax.lax.conv_general_dilated(
            seq_transposed,
            kernel_transposed,
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NCH", "OIH", "NCH"),
        )  # (1, embedding_dim, new_length)

        # Apply ReLU activation
        conv_out = jax.nn.relu(conv_out)

        # Global average pooling
        embedding = jnp.mean(conv_out, axis=-1).squeeze(0)  # (embedding_dim,)

        # L2 normalize for cosine similarity
        embedding = embedding / (jnp.linalg.norm(embedding) + 1e-8)

        return embedding

    def _compute_similarity_matrix(
        self,
        embeddings: Float[Array, "batch embedding_dim"],
    ) -> Float[Array, "batch batch"]:
        """Compute pairwise cosine similarity matrix.

        Args:
            embeddings: Batch of embeddings (batch_size, embedding_dim).

        Returns:
            Similarity matrix of shape (batch_size, batch_size).
        """
        # Cosine similarity: embeddings are already L2 normalized
        similarity = jnp.einsum("ie,je->ij", embeddings, embeddings)
        return similarity

    def _compute_soft_cluster_sizes(
        self,
        similarity: Float[Array, "batch batch"],
    ) -> Float[Array, "batch"]:
        """Compute soft cluster size for each sequence.

        Uses thresholded similarity to compute how many sequences are
        "similar" to each sequence (soft duplicate count).

        Args:
            similarity: Pairwise similarity matrix.

        Returns:
            Soft cluster size for each sequence.
        """
        temp = self._temperature
        threshold = self.similarity_threshold[...]

        # Soft thresholding: sigmoid((similarity - threshold) / temperature)
        soft_membership = jax.nn.sigmoid((similarity - threshold) / temp)

        # Sum memberships for each sequence (including self)
        cluster_sizes = jnp.sum(soft_membership, axis=1)

        return cluster_sizes

    def _compute_uniqueness_weights(
        self,
        cluster_sizes: Float[Array, "batch"],
    ) -> Float[Array, "batch"]:
        """Compute uniqueness weights from cluster sizes.

        Weight = 1 / cluster_size (normalized to sum to batch_size).

        Args:
            cluster_sizes: Soft cluster size for each sequence.

        Returns:
            Uniqueness weight for each sequence.
        """
        # Weight inversely proportional to cluster size
        raw_weights = 1.0 / jnp.maximum(cluster_sizes, 1.0)

        # Normalize weights to have mean 1.0
        weights = raw_weights / jnp.mean(raw_weights)

        return weights

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply duplicate weighting to sequence data.

        For single sequences, returns weight=1.0.
        For batched sequences, computes uniqueness-based weights.

        Args:
            data: Dictionary containing:
                - "sequence": One-hot encoded sequence (length, alphabet_size)
                              or batch (batch, length, alphabet_size)
                - "quality_scores": Quality scores (length,) or (batch, length)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used (deterministic operator)
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - "sequence": Original sequence (unchanged)
                    - "quality_scores": Original quality scores (unchanged)
                    - "uniqueness_weight": Weight based on uniqueness
                    - "embedding": Sequence embedding (for downstream use)
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        sequence = data["sequence"]
        quality_scores = data["quality_scores"]

        # Check if input is batched
        if sequence.ndim == 2:
            # Single sequence: weight = 1.0
            embedding = self._embed_sequence(sequence)
            uniqueness_weight = jnp.array(1.0)
        else:
            # Batched sequences: compute pairwise weights
            batch_size = sequence.shape[0]

            # Embed all sequences
            embeddings = jax.vmap(self._embed_sequence)(sequence)

            # Compute similarity and weights
            similarity = self._compute_similarity_matrix(embeddings)
            cluster_sizes = self._compute_soft_cluster_sizes(similarity)
            weights = self._compute_uniqueness_weights(cluster_sizes)

            # For single sequence output, take first embedding and weight
            embedding = embeddings[0] if batch_size > 0 else jnp.zeros(32)
            uniqueness_weight = weights[0] if batch_size > 0 else jnp.array(1.0)

        # Build output data
        transformed_data = {
            "sequence": sequence,
            "quality_scores": quality_scores,
            "uniqueness_weight": uniqueness_weight,
            "embedding": embedding,
        }

        return transformed_data, state, metadata

    def apply_batch(
        self,
        sequences: Float[Array, "batch length alphabet"],
        quality_scores: Float[Array, "batch length"],
    ) -> tuple[Float[Array, "batch"], Float[Array, "batch embedding_dim"]]:
        """Apply duplicate weighting to a batch of sequences.

        This is a convenience method for processing multiple sequences
        and computing their pairwise uniqueness weights.

        Args:
            sequences: Batch of one-hot encoded sequences.
            quality_scores: Batch of quality scores.

        Returns:
            Tuple of (weights, embeddings):
                - weights: Uniqueness weight for each sequence
                - embeddings: Sequence embeddings
        """
        # Embed all sequences
        embeddings = jax.vmap(self._embed_sequence)(sequences)

        # Compute similarity and weights
        similarity = self._compute_similarity_matrix(embeddings)
        cluster_sizes = self._compute_soft_cluster_sizes(similarity)
        weights = self._compute_uniqueness_weights(cluster_sizes)

        return weights, embeddings
