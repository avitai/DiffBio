"""Variant classifier for differentiable variant calling.

This module provides a neural network classifier for identifying variants
from pileup representations.
"""

from dataclasses import dataclass
from typing import Any

from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, PyTree

from diffbio.configs import ClassifierConfig
from diffbio.constants import DEFAULT_PILEUP_WINDOW_SIZE, DNA_ALPHABET_SIZE


@dataclass
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
    ):
        """Initialize variant classifier.

        Args:
            config: Classifier configuration.
            rngs: Random number generators.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        # Input dimension: window_size * alphabet_size (nucleotides)
        input_dim = config.input_window * DNA_ALPHABET_SIZE

        # Input layer
        self.input_layer = nnx.Linear(input_dim, config.hidden_dim, rngs=rngs)

        # Hidden layers using nnx.List for proper pytree handling
        layers = []
        dropout_layers = []
        for _ in range(config.num_layers - 1):
            layers.append(nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs))
            dropout_layers.append(nnx.Dropout(rate=config.dropout_rate, rngs=rngs))

        self.layers = nnx.List(layers)
        self.dropout_layers = nnx.List(dropout_layers)

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

        # Input projection
        x = self.input_layer(x)
        x = nnx.relu(x)

        # Hidden layers
        for layer, dropout in zip(self.layers, self.dropout_layers, strict=False):
            x = layer(x)
            x = nnx.relu(x)
            x = dropout(x)

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
