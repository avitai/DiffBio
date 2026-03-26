"""Differentiable error correction operator.

This module provides a neural network-based error correction operator that
refines base calls using local sequence context and quality scores.

Key technique: Use a small MLP to predict corrected base probabilities
from a sliding window of sequence and quality data.

Inspired by DeepConsensus approach for consensus calling.

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
from datarax.core.config import OperatorConfig
from flax import nnx
from jaxtyping import Array, Float, PyTree

from diffbio.core.base_operators import TemperatureOperator
from diffbio.utils.nn_utils import build_mlp_layers, ensure_rngs, init_learnable_param

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ErrorCorrectionConfig(OperatorConfig):
    """Configuration for SoftErrorCorrection.

    Attributes:
        window_size: Size of context window around each position.
            Must be odd. Default is 11 (5 bases on each side).
        hidden_dim: Hidden layer dimension in the MLP.
        num_layers: Number of hidden layers in the MLP.
        use_quality: Whether to include quality scores as input.
        temperature: Temperature for output softmax.
    """

    window_size: int = 11
    hidden_dim: int = 64
    num_layers: int = 2
    use_quality: bool = True
    temperature: float = 1.0
    learnable_temperature: bool = True


class SoftErrorCorrection(TemperatureOperator):
    """Differentiable error correction for sequencing reads.

    This operator uses a neural network to refine base calls based on
    local sequence context and quality scores. It outputs soft base
    probabilities that maintain gradient flow.

    The algorithm:
    1. For each position, extract a window of sequence and quality data
    2. Pass through MLP to predict corrected base probabilities
    3. Output soft one-hot representation blending original and corrected

    Args:
        config: ErrorCorrectionConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        ```python
        config = ErrorCorrectionConfig(window_size=11, hidden_dim=64)
        corrector = SoftErrorCorrection(config, rngs=nnx.Rngs(42))
        data = {"sequence": encoded_seq, "quality_scores": quality}
        result, state, meta = corrector.apply(data, {}, None)
        ```
    """

    def __init__(
        self,
        config: ErrorCorrectionConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the error correction operator.

        Args:
            config: Error correction configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        rngs = ensure_rngs(rngs)

        self.window_size = config.window_size
        self.use_quality = config.use_quality

        # Input dimension: window_size * (4 alphabet + 1 quality if used)
        alphabet_size = 4
        features_per_position = alphabet_size + (1 if config.use_quality else 0)
        input_dim = config.window_size * features_per_position

        # Build MLP layers using utility
        self.layers, _, out_dim = build_mlp_layers(
            in_features=input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            rngs=rngs,
        )

        # Output layer (predicts 4 base probabilities)
        self.output_layer = nnx.Linear(in_features=out_dim, out_features=alphabet_size, rngs=rngs)

        # Temperature is managed by TemperatureOperator via self._temperature

        # Learnable blending weight (how much to trust correction vs original)
        self.correction_weight = init_learnable_param(0.5)

    def _extract_window(
        self,
        sequence: Float[Array, "length alphabet"],
        quality_scores: Float[Array, "length"],
        position: Array | int,
    ) -> Float[Array, "window_features"]:
        """Extract feature window around a position.

        Args:
            sequence: One-hot encoded sequence.
            quality_scores: Quality scores for each position.
            position: Center position for the window.

        Returns:
            Flattened feature vector for the window.
        """
        seq_len = sequence.shape[0]
        half_window = self.window_size // 2

        # Build window features with padding for edge positions
        def get_position_features(offset: Array | int) -> Float[Array, "features"]:
            pos = position + offset - half_window
            # Handle boundary conditions with zeros
            in_bounds = (pos >= 0) & (pos < seq_len)

            # Get sequence features
            clipped_pos = jnp.clip(pos, 0, seq_len - 1)
            slice_val = jax.lax.dynamic_slice(sequence, (clipped_pos, 0), (1, 4))
            seq_features = jnp.where(in_bounds, slice_val.squeeze(0), jnp.zeros(4))

            if self.use_quality:
                # Get quality feature (normalized to [0, 1])
                qual_feature = jnp.where(
                    in_bounds,
                    quality_scores[jnp.clip(pos, 0, seq_len - 1)] / 40.0,  # Normalize by Q40
                    0.0,
                )
                return jnp.concatenate([seq_features, jnp.array([qual_feature])])
            return seq_features

        # Extract all window positions
        features = jax.vmap(get_position_features)(jnp.arange(self.window_size))
        return features.flatten()

    def _predict_correction(
        self,
        window_features: Float[Array, "window_features"],
    ) -> Float[Array, "alphabet"]:
        """Predict corrected base probabilities from window features.

        Args:
            window_features: Flattened window feature vector.

        Returns:
            Soft base probabilities (4,).
        """
        x = window_features

        # Pass through MLP layers with ReLU activation
        for layer in self.layers:
            x = layer(x)
            x = nnx.relu(x)

        # Output layer
        logits = self.output_layer(x)

        # Apply temperature-scaled softmax
        temp = self._temperature
        probs = jax.nn.softmax(logits / temp)

        return probs

    def _correct_position(
        self,
        sequence: Float[Array, "length alphabet"],
        quality_scores: Float[Array, "length"],
        position: Array | int,
    ) -> Float[Array, "alphabet"]:
        """Correct a single position using context.

        Args:
            sequence: One-hot encoded sequence.
            quality_scores: Quality scores.
            position: Position to correct.

        Returns:
            Corrected soft one-hot for this position.
        """
        # Extract window features
        window = self._extract_window(sequence, quality_scores, position)

        # Predict correction
        correction = self._predict_correction(window)

        # Blend with original based on correction weight
        original = sequence[position]
        weight = jax.nn.sigmoid(self.correction_weight[...])
        corrected = weight * correction + (1 - weight) * original

        # Renormalize to ensure valid probability distribution
        corrected = corrected / (jnp.sum(corrected) + 1e-8)

        return corrected

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply error correction to sequence data.

        This method corrects each position in the sequence using the
        neural network model, producing soft corrected base probabilities.

        Args:
            data: Dictionary containing:
                - "sequence": One-hot encoded sequence (length, alphabet_size)
                - "quality_scores": Phred quality scores (length,)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used (deterministic operator)
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - "sequence": Corrected soft one-hot sequence
                    - "quality_scores": Original quality scores
                    - "correction_confidence": Average correction weight
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        sequence = data["sequence"]
        quality_scores = data["quality_scores"]
        seq_len = sequence.shape[0]

        # Correct each position
        def correct_fn(position: Array | int) -> Float[Array, "alphabet"]:
            return self._correct_position(sequence, quality_scores, position)

        corrected_sequence = jax.vmap(correct_fn)(jnp.arange(seq_len))

        # Compute confidence metric (based on correction weight)
        confidence = jax.nn.sigmoid(self.correction_weight[...])

        # Build output data
        transformed_data = {
            "sequence": corrected_sequence,
            "quality_scores": quality_scores,
            "correction_confidence": confidence,
        }

        return transformed_data, state, metadata
