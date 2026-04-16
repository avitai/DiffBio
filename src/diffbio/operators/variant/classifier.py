"""Variant classifier for differentiable variant calling.

This module provides neural network classifiers for identifying variants
from pileup representations, including a cell-type-aware classifier that
weights variant calls by soft cell-type assignments.
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from artifex.generative_models.core.base import MLP
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, PyTree

from diffbio.configs import ClassifierConfig
from diffbio.constants import DEFAULT_PILEUP_WINDOW_SIZE, DNA_ALPHABET_SIZE
from diffbio.utils.nn_utils import ARTIFEX_RELU_MLP_KWARGS, ensure_rngs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariantClassifierConfig(ClassifierConfig):
    """Configuration for variant classifier.

    Attributes:
        num_classes: Number of variant classes (default: 3 for REF/SNV/INDEL).
        hidden_dim: Hidden layer dimension.
        num_layers: Number of hidden layers.
        dropout_rate: Dropout rate for regularization.
        input_window: Default input window size for pileup.
    """

    input_window: int = DEFAULT_PILEUP_WINDOW_SIZE


class VariantClassifier(OperatorModule):
    """Neural network classifier for variant calling.

    Takes a window of pileup data around a position and classifies it
    as reference, SNV, or indel. Uses a simple MLP architecture that
    is fully differentiable.

    Args:
        config: Classifier configuration.
        rngs: Flax NNX random number generators.
        name: Optional operator name.
    """

    def __init__(
        self,
        config: VariantClassifierConfig,
        *,
        rngs: nnx.Rngs,
        name: str | None = None,
    ) -> None:
        """Initialize variant classifier.

        Args:
            config: Classifier configuration.
            rngs: Random number generators.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        # Input dimension: window_size * alphabet_size (nucleotides)
        input_dim = config.input_window * DNA_ALPHABET_SIZE

        if config.num_layers < 1:
            raise ValueError("VariantClassifierConfig.num_layers must be at least 1.")

        self.backbone = MLP(
            hidden_dims=[config.hidden_dim] * config.num_layers,
            in_features=input_dim,
            dropout_rate=config.dropout_rate,
            rngs=rngs,
            **ARTIFEX_RELU_MLP_KWARGS,
        )

        # Output layer
        self.output_layer = nnx.Linear(config.hidden_dim, config.num_classes, rngs=rngs)

    def classify(
        self,
        pileup_window: Float[Array, "window_size 4"],
    ) -> Float[Array, "num_classes"]:
        """Classify variant from pileup window.

        Args:
            pileup_window: Pileup data for window around position.
                Shape: (window_size, 4) with nucleotide distributions.

        Returns:
            Logits for each variant class. Shape: (num_classes,).
        """
        # Flatten pileup window
        x = pileup_window.reshape(-1)
        backbone_output = self.backbone(x)
        if isinstance(backbone_output, tuple):
            raise TypeError("VariantClassifier backbone must return a single tensor output.")
        x = backbone_output

        # Output
        logits = self.output_layer(x)

        return logits

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply variant classification to pileup data.

        This method implements the OperatorModule interface for batch processing.
        It expects data containing a pileup window and returns classification logits.

        Note: Output preserves input keys for Datarax vmap compatibility,
        while adding classification result keys.

        Args:
            data: Dictionary containing:
                - "pileup_window": Pileup data around position (window_size, 4)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used (dropout handled by eval/train mode)
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains input pileup_window plus logits and
                  probabilities
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        import jax.nn

        pileup_window = data["pileup_window"]

        # Classify
        logits = self.classify(pileup_window)

        # Build output data - preserve input keys for Datarax vmap compatibility
        transformed_data = {
            "pileup_window": pileup_window,
            "logits": logits,
            "probabilities": jax.nn.softmax(logits),
        }

        return transformed_data, state, metadata


@dataclass(frozen=True)
class CellTypeAwareVariantClassifierConfig(OperatorConfig):
    """Configuration for cell-type-aware variant classifier.

    This classifier uses separate classification heads per cell type,
    weighted by soft cell-type assignments to produce cell-type-specific
    variant calling thresholds.

    Attributes:
        n_classes: Number of variant types (e.g., SNP, indel, ref).
        hidden_dim: Hidden layer dimension for the shared feature encoder.
        n_cell_types: Number of cell types for per-type heads.
        pileup_channels: Number of channels in pileup input.
        pileup_width: Width of pileup input.
    """

    n_classes: int = 3
    hidden_dim: int = 64
    n_cell_types: int = 5
    pileup_channels: int = 6
    pileup_width: int = 100


class CellTypeAwareVariantClassifier(OperatorModule):
    """Cell-type-aware variant classifier with per-type classification heads.

    Uses separate classification heads for each cell type, weighted by soft
    cell-type assignment probabilities. This allows different variant calling
    thresholds per cell type, enabling more accurate variant detection in
    heterogeneous cell populations (e.g., single-cell sequencing).

    Architecture:
        1. Shared feature encoder: pileup -> flatten -> Linear -> ReLU -> hidden features
        2. Per-type classification heads: n_cell_types separate Linear(hidden, n_classes)
        3. Each head produces type-specific variant logits -> softmax probabilities
        4. Final aggregation: sum_t(cell_type_weights[:, t] * head_t_probs)

    Args:
        config: CellTypeAwareVariantClassifierConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        ```python
        config = CellTypeAwareVariantClassifierConfig(n_classes=3, n_cell_types=5)
        classifier = CellTypeAwareVariantClassifier(config, rngs=nnx.Rngs(42))
        data = {
            "pileup": pileup_batch,              # (n, channels, width)
            "cell_type_assignments": assignments,  # (n, n_cell_types)
        }
        result, state, meta = classifier.apply(data, {}, None)
        # result["variant_probabilities"]   -> (n, n_classes)
        # result["per_type_probabilities"]  -> (n, n_cell_types, n_classes)
        ```
    """

    def __init__(
        self,
        config: CellTypeAwareVariantClassifierConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize cell-type-aware variant classifier.

        Args:
            config: Classifier configuration.
            rngs: Random number generators for parameter initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        rngs = ensure_rngs(rngs)

        input_dim = config.pileup_channels * config.pileup_width

        # Shared feature encoder: pileup -> hidden features
        self.encoder = nnx.Linear(input_dim, config.hidden_dim, rngs=rngs)

        # Per-cell-type classification heads
        heads = []
        for _ in range(config.n_cell_types):
            heads.append(nnx.Linear(config.hidden_dim, config.n_classes, rngs=rngs))
        self.classification_heads = nnx.List(heads)

    def _encode(
        self,
        pileup: Float[Array, "n channels width"],
    ) -> Float[Array, "n hidden_dim"]:
        """Encode pileup into hidden features via the shared encoder.

        Args:
            pileup: Batch of pileup data, shape (n, channels, width).

        Returns:
            Hidden feature vectors, shape (n, hidden_dim).
        """
        x = pileup.reshape(pileup.shape[0], -1)  # (n, channels * width)
        x = self.encoder(x)
        return nnx.relu(x)

    def _classify_per_type(
        self,
        features: Float[Array, "n hidden_dim"],
    ) -> Float[Array, "n n_cell_types n_classes"]:
        """Run each cell-type classification head on shared features.

        Args:
            features: Shared hidden features, shape (n, hidden_dim).

        Returns:
            Per-type softmax probabilities, shape (n, n_cell_types, n_classes).
        """
        head_outputs = []
        for head in self.classification_heads:
            logits = head(features)  # (n, n_classes)
            probs = jax.nn.softmax(logits, axis=-1)
            head_outputs.append(probs)
        # Stack: (n_cell_types, n, n_classes) -> transpose to (n, n_cell_types, n_classes)
        return jnp.stack(head_outputs, axis=1)

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply cell-type-aware variant classification.

        Computes per-type variant probabilities and aggregates them using
        cell-type assignment weights.

        Args:
            data: Dictionary containing:
                - "pileup": Pileup data, shape (n, channels, width).
                - "cell_type_assignments": Soft cell-type weights, shape (n, n_cell_types).
            state: Element state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: Not used.
            stats: Not used.

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains all input keys plus:
                    - "variant_probabilities": Aggregated probabilities (n, n_classes)
                    - "per_type_probabilities": Per-type probabilities
                      (n, n_cell_types, n_classes)
                - state passed through unchanged
                - metadata passed through unchanged
        """
        pileup = data["pileup"]
        cell_type_assignments = data["cell_type_assignments"]

        # 1. Shared feature encoding
        features = self._encode(pileup)

        # 2. Per-type classification
        per_type_probs = self._classify_per_type(features)  # (n, n_cell_types, n_classes)

        # 3. Weighted aggregation: sum_t( assignments[:, t] * per_type_probs[:, t, :] )
        # assignments: (n, n_cell_types) -> (n, n_cell_types, 1)
        weights = cell_type_assignments[:, :, None]
        variant_probs = jnp.sum(weights * per_type_probs, axis=1)  # (n, n_classes)

        transformed_data = {
            **data,
            "variant_probabilities": variant_probs,
            "per_type_probabilities": per_type_probs,
        }

        return transformed_data, state, metadata
