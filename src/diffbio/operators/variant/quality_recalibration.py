"""Soft Variant Quality Filter for VQSR-style recalibration.

This module provides a differentiable variant quality filter using
a Gaussian Mixture Model for scoring and sigmoid-based soft filtering.

Key technique: Differentiable GMM enables end-to-end learning of
quality distributions, with sigmoid thresholds maintaining gradients.

Applications: VQSR-style variant filtering, quality score recalibration.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, PyTree


@dataclass
class VariantQualityFilterConfig(OperatorConfig):
    """Configuration for SoftVariantQualityFilter.

    Attributes:
        n_components: Number of GMM components.
        n_features: Number of variant features.
        threshold: Quality score threshold for filtering.
        temperature: Temperature for softmax/sigmoid operations.
    """

    n_components: int = 3
    n_features: int = 4  # depth, qual, strand_bias, mapq
    threshold: float = 0.5
    temperature: float = 1.0


class SoftVariantQualityFilter(OperatorModule):
    """Differentiable variant quality filter using GMM.

    This operator implements VQSR-style variant quality recalibration
    using a learnable Gaussian Mixture Model. Variants are scored
    by their likelihood under the GMM, and soft filtering is applied
    via sigmoid thresholds.

    Algorithm:
    1. Compute GMM component responsibilities (E-step style)
    2. Score variants by weighted log-likelihood
    3. Apply sigmoid threshold for soft filtering

    Args:
        config: VariantQualityFilterConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = VariantQualityFilterConfig(n_components=3)
        >>> filter_op = SoftVariantQualityFilter(config, rngs=nnx.Rngs(42))
        >>> data = {"variant_features": features}  # (n_variants, n_features)
        >>> result, state, meta = filter_op.apply(data, {}, None)
    """

    def __init__(
        self,
        config: VariantQualityFilterConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the quality filter.

        Args:
            config: Filter configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.n_components = config.n_components
        self.n_features = config.n_features
        self.threshold = config.threshold
        self.temperature = config.temperature

        # Initialize GMM parameters
        # Component means: (n_components, n_features)
        key = rngs.params()
        init_means = jax.random.normal(key, (config.n_components, config.n_features)) * 0.5
        self.means = nnx.Param(init_means)

        # Component log variances (diagonal covariance): (n_components, n_features)
        key = rngs.params()
        init_log_var = jax.random.normal(key, (config.n_components, config.n_features)) * 0.1
        self.log_variances = nnx.Param(init_log_var)

        # Component mixing weights (unnormalized): (n_components,)
        key = rngs.params()
        noise = jax.random.normal(key, (config.n_components,)) * 0.1
        init_weights = jnp.ones(config.n_components) + noise
        self.log_mixing_weights = nnx.Param(init_weights)

        # Learned quality score projection
        self.quality_proj = nnx.Linear(config.n_features, 1, rngs=rngs)

    def get_mixing_weights(self) -> Float[Array, "n_components"]:
        """Get normalized mixing weights.

        Returns:
            Mixing weights summing to 1.
        """
        return jax.nn.softmax(self.log_mixing_weights[...] / self.temperature)

    def get_variances(self) -> Float[Array, "n_components n_features"]:
        """Get positive variances from log parameters.

        Returns:
            Variance values.
        """
        return jnp.exp(self.log_variances[...])

    def compute_component_log_probs(
        self,
        features: Float[Array, "n_variants n_features"],
    ) -> Float[Array, "n_variants n_components"]:
        """Compute log probability under each GMM component.

        Args:
            features: Variant feature vectors.

        Returns:
            Log probability for each variant under each component.
        """
        means = self.means[...]  # (n_components, n_features)
        variances = self.get_variances()  # (n_components, n_features)

        # Expand for broadcasting
        # features: (n_variants, 1, n_features)
        # means: (1, n_components, n_features)
        features_exp = features[:, None, :]
        means_exp = means[None, :, :]
        variances_exp = variances[None, :, :]

        # Gaussian log probability (diagonal covariance)
        # log p(x | mu, sigma^2) = -0.5 * sum((x - mu)^2 / sigma^2 + log(2*pi*sigma^2))
        diff_sq = (features_exp - means_exp) ** 2
        log_prob = -0.5 * jnp.sum(
            diff_sq / variances_exp + jnp.log(2 * jnp.pi * variances_exp), axis=-1
        )  # (n_variants, n_components)

        return log_prob

    def compute_responsibilities(
        self,
        features: Float[Array, "n_variants n_features"],
    ) -> Float[Array, "n_variants n_components"]:
        """Compute component responsibilities (soft assignments).

        Args:
            features: Variant feature vectors.

        Returns:
            Responsibility matrix (probability each variant belongs to each component).
        """
        log_probs = self.compute_component_log_probs(features)  # (n_variants, n_components)
        mixing_weights = self.get_mixing_weights()  # (n_components,)

        # log P(x, z) = log P(x | z) + log P(z)
        log_joint = log_probs + jnp.log(mixing_weights + 1e-10)

        # Responsibilities via softmax (with temperature)
        responsibilities = jax.nn.softmax(log_joint / self.temperature, axis=-1)

        return responsibilities

    def compute_quality_scores(
        self,
        features: Float[Array, "n_variants n_features"],
    ) -> Float[Array, "n_variants"]:
        """Compute quality scores for variants.

        Combines GMM likelihood with learned projection.

        Args:
            features: Variant feature vectors.

        Returns:
            Quality scores in [0, 1] range.
        """
        # GMM-based score
        log_probs = self.compute_component_log_probs(features)  # (n_variants, n_components)
        mixing_weights = self.get_mixing_weights()  # (n_components,)

        # Total log likelihood under mixture
        log_likelihood = jax.scipy.special.logsumexp(
            log_probs + jnp.log(mixing_weights + 1e-10), axis=-1
        )  # (n_variants,)

        # Learned quality projection
        learned_quality = self.quality_proj(features).squeeze(-1)  # (n_variants,)

        # Combine and normalize to [0, 1]
        combined_score = log_likelihood + learned_quality
        quality_scores = jax.nn.sigmoid(combined_score)

        return quality_scores

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply quality filtering to variants.

        Args:
            data: Dictionary containing:
                - "variant_features": Feature vectors (n_variants, n_features)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:
                    - "variant_features": Original features
                    - "quality_scores": Computed quality scores [0, 1]
                    - "filter_weights": Soft filter weights [0, 1]
                    - "component_probs": GMM component responsibilities
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        features = data["variant_features"]

        # Compute quality scores
        quality_scores = self.compute_quality_scores(features)

        # Soft filter weights using sigmoid threshold
        filter_weights = jax.nn.sigmoid((quality_scores - self.threshold) / self.temperature)

        # Component responsibilities
        component_probs = self.compute_responsibilities(features)

        # Build output data
        transformed_data = {
            "variant_features": features,
            "quality_scores": quality_scores,
            "filter_weights": filter_weights,
            "component_probs": component_probs,
        }

        return transformed_data, state, metadata
