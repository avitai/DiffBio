"""CNN Variant Classifier for DeepVariant-style pileup classification.

This module provides a convolutional neural network classifier for
variant calling from pileup images, inspired by DeepVariant.

Key technique: 2D convolutions on pileup images enable learning
spatial patterns in read alignments for accurate variant detection.

Applications: Germline/somatic variant calling, variant quality scoring.
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from artifex.generative_models.core.base import CNN
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, PyTree

from diffbio.constants import DEFAULT_DROPOUT_RATE, DEFAULT_NUM_CLASSES
from diffbio.utils.nn_utils import ensure_rngs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CNNVariantClassifierConfig(OperatorConfig):
    """Configuration for CNNVariantClassifier.

    Attributes:
        num_classes: Number of variant classes (default: 3 for REF/SNV/INDEL).
        input_height: Height of pileup image (coverage depth).
        input_width: Width of pileup image (context window).
        num_channels: Number of input channels (A, C, G, T, quality, strand).
        hidden_channels: Number of channels in each conv layer.
        fc_dims: Dimensions of fully connected layers.
        dropout_rate: Dropout rate for regularization.
    """

    num_classes: int = DEFAULT_NUM_CLASSES
    input_height: int = 100  # coverage depth
    input_width: int = 221  # context window
    num_channels: int = 6  # A, C, G, T, quality, strand
    hidden_channels: tuple[int, ...] = (64, 128, 256)
    fc_dims: tuple[int, ...] = (256, 128)
    dropout_rate: float = DEFAULT_DROPOUT_RATE

    def __post_init__(self) -> None:
        """Set stochastic config based on dropout usage."""
        if self.dropout_rate > 0:
            object.__setattr__(self, "stochastic", True)
            if self.stream_name is None:
                object.__setattr__(self, "stream_name", "dropout")
        super().__post_init__()


class CNNVariantClassifier(OperatorModule):
    """CNN classifier for DeepVariant-style variant calling.

    This operator implements a convolutional neural network that processes
    pileup images to classify genomic positions as reference, SNV, or indel.

    Architecture:
    - Strided Conv2D feature extractor (artifex ``CNN``) with ReLU
    - Global average pooling before FC layers
    - Fully connected layers with dropout
    - Softmax output for class probabilities

    Args:
        config: CNNVariantClassifierConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        ```python
        config = CNNVariantClassifierConfig(num_classes=3)
        classifier = CNNVariantClassifier(config, rngs=nnx.Rngs(42))
        data = {"pileup_image": image_batch}  # (B, H, W, C)
        result, state, meta = classifier.apply(data, {}, None)
        ```
    """

    def __init__(
        self,
        config: CNNVariantClassifierConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the CNN variant classifier.

        Args:
            config: Classifier configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        rngs = ensure_rngs(rngs)

        self.num_classes = config.num_classes
        self.dropout_rate = config.dropout_rate

        # Convolutional feature extractor (shared artifex CNN). Strided
        # convolutions provide the spatial downsampling the previous
        # max-pooling did; a global average pool follows in ``_features``.
        self.cnn = (
            CNN(
                list(config.hidden_channels),
                in_features=config.num_channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                activation="relu",
                use_batch_norm=False,
                dropout_rate=0.0,
                rngs=rngs,
            )
            if config.hidden_channels
            else None
        )

        # Fully connected layers
        fc_layers = []
        # After global average pooling, input dim is last conv channel count
        fc_in_dim = config.hidden_channels[-1] if config.hidden_channels else config.num_channels
        for fc_dim in config.fc_dims:
            fc_layers.append(nnx.Linear(fc_in_dim, fc_dim, rngs=rngs))
            fc_in_dim = fc_dim
        self.fc_layers = nnx.List(fc_layers)

        # Dropout layer
        if config.dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        else:
            self.dropout = None

        # Output layer
        self.output_layer = nnx.Linear(fc_in_dim, config.num_classes, rngs=rngs)

    def _features(
        self,
        pileup_image: Float[Array, "batch height width channels"],
    ) -> Float[Array, "batch num_classes"]:
        """Run the CNN feature extractor and classification head.

        Args:
            pileup_image: Batch of pileup images (B, H, W, C).

        Returns:
            Logits for each variant class per image (B, num_classes).
        """
        x = pileup_image
        if self.cnn is not None:
            x = self.cnn(x)

        # Global average pooling -> (batch, channels)
        x = jnp.mean(x, axis=(1, 2))

        # Fully connected head
        for fc in self.fc_layers:
            x = fc(x)
            x = nnx.relu(x)
            if self.dropout is not None:
                x = self.dropout(x)

        return self.output_layer(x)

    def _classify_single(
        self,
        pileup_image: Float[Array, "height width channels"],
    ) -> Float[Array, "num_classes"]:
        """Classify a single pileup image.

        Args:
            pileup_image: Single pileup image (H, W, C).

        Returns:
            Logits for each variant class.
        """
        return self._features(pileup_image[None, ...])[0]

    def classify(
        self,
        pileup_image: Float[Array, "batch height width channels"],
    ) -> Float[Array, "batch num_classes"]:
        """Classify batch of pileup images.

        Args:
            pileup_image: Batch of pileup images (B, H, W, C).

        Returns:
            Logits for each variant class per image.
        """
        return self._features(pileup_image)

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply CNN classification to pileup images.

        Args:
            data: Dictionary containing:
                - "pileup_image": Pileup images (batch, height, width, channels)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - "pileup_image": Original input
                    - "logits": Raw classification scores
                    - "class_probs": Softmax probabilities
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        pileup_image = data["pileup_image"]

        # Classify
        logits = self.classify(pileup_image)

        # Build output data
        transformed_data = {
            "pileup_image": pileup_image,
            "logits": logits,
            "class_probs": jax.nn.softmax(logits, axis=-1),
        }

        return transformed_data, state, metadata
