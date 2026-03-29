"""Type definitions and protocols for DiffBio.

This module provides type aliases, TypedDicts, and protocols that define
the expected interfaces and data structures across the DiffBio codebase.
"""

from typing import Any, Protocol, TypedDict, runtime_checkable

from jaxtyping import Array, Float

# =============================================================================
# Type Aliases for Scalar Values
# =============================================================================

Temperature = float
"""Temperature parameter for soft operations. Must be > 0."""

Probability = float
"""Probability value in range [0, 1]."""

LogProbability = float
"""Log probability value in range (-inf, 0]."""

# =============================================================================
# Type Aliases for Arrays
# =============================================================================

SequenceArray = Float[Array, "length alphabet"]
"""One-hot encoded sequence of shape (length, alphabet_size)."""

BatchArray = Float[Array, "batch ..."]
"""Batched array with batch dimension first."""

ProbabilityArray = Float[Array, "..."]
"""Array of probability values, each in [0, 1]."""

ScoreMatrix = Float[Array, "alphabet alphabet"]
"""Scoring matrix for sequence alignment."""

AlignmentMatrix = Float[Array, "len1_plus1 len2_plus1"]
"""Dynamic programming matrix for alignment."""

PositionWeightMatrix = Float[Array, "length alphabet"]
"""Position weight matrix for motif representation."""


# =============================================================================
# TypedDicts for Data Structures
# =============================================================================


class SequenceData(TypedDict, total=False):
    """Data dictionary for sequence data.

    Required:
        sequence: One-hot encoded sequence.

    Optional:
        quality_scores: Phred quality scores.
        mask: Boolean mask for valid positions.
    """

    sequence: Array
    quality_scores: Array
    mask: Array


class AlignmentResultData(TypedDict, total=False):
    """Data dictionary for alignment results.

    Required:
        score: Alignment score.
        alignment_matrix: DP matrix.

    Optional:
        soft_alignment: Soft position correspondences.
        traceback: Hard alignment path.
    """

    score: Array
    alignment_matrix: Array
    soft_alignment: Array
    traceback: Array


class VariantData(TypedDict, total=False):
    """Data dictionary for variant calling results.

    Required:
        logits: Classification logits.

    Optional:
        probabilities: Softmax probabilities.
        pileup: Pileup representation.
        coverage: Coverage at each position.
    """

    logits: Array
    probabilities: Array
    pileup: Array
    coverage: Array


class LatentData(TypedDict, total=False):
    """Data dictionary for VAE latent representations.

    Required:
        z: Sampled latent representation.

    Optional:
        mean: Mean of latent distribution.
        log_var: Log variance of latent distribution.
    """

    z: Array
    mean: Array
    log_var: Array


class GraphData(TypedDict, total=False):
    """Data dictionary for graph-structured data.

    Required:
        node_features: Node feature matrix.
        edge_index: Edge indices (2, num_edges).

    Optional:
        edge_features: Edge feature matrix.
        batch: Batch assignment for nodes.
    """

    node_features: Array
    edge_index: Array
    edge_features: Array
    batch: Array


# =============================================================================
# Type Aliases for Operator I/O
# =============================================================================

StateDict = dict[str, Any]
"""State dictionary passed between operator calls."""

MetadataDict = dict[str, Any] | None
"""Optional metadata dictionary."""

OperatorOutput = tuple[dict[str, Any], StateDict, MetadataDict]
"""Standard operator output: (data, state, metadata)."""


# =============================================================================
# Protocols for Interfaces
# =============================================================================


@runtime_checkable
class DifferentiableOperator(Protocol):
    """Protocol for differentiable operators.

    All DiffBio operators should implement this interface.
    """

    def apply(
        self,
        data: dict[str, Any],
        state: StateDict,
        metadata: MetadataDict,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> OperatorOutput:
        """Apply the operator to input data.

        Args:
            data: Input data dictionary.
            state: Element state.
            metadata: Element metadata.
            random_params: Random parameters for stochastic operations.
            stats: Statistics dictionary.

        Returns:
            Tuple of (transformed_data, state, metadata).
        """
        ...


@runtime_checkable
class SequenceEncoder(Protocol):
    """Protocol for sequence encoding/decoding.

    Implementations should handle conversion between string
    representations and JAX arrays.
    """

    def encode(self, sequence: str) -> Array:
        """Encode a sequence string to a JAX array.

        Args:
            sequence: String representation of sequence.

        Returns:
            Encoded array representation.
        """
        ...

    def decode(self, encoded: Array) -> str:
        """Decode a JAX array back to a sequence string.

        Args:
            encoded: Array representation of sequence.

        Returns:
            String representation.
        """
        ...


@runtime_checkable
class LossFunction(Protocol):
    """Protocol for loss functions.

    Loss functions compute scalar losses from predictions and targets.
    """

    def __call__(
        self,
        predictions: Array,
        targets: Array,
        **kwargs: Any,
    ) -> Float[Array, ""]:
        """Compute the loss.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional arguments.

        Returns:
            Scalar loss value.
        """
        ...


@runtime_checkable
class Regularizer(Protocol):
    """Protocol for regularization functions.

    Regularizers add penalty terms to loss functions.
    """

    def __call__(self, params: Any) -> Float[Array, ""]:
        """Compute the regularization penalty.

        Args:
            params: Parameters to regularize.

        Returns:
            Scalar regularization term.
        """
        ...
