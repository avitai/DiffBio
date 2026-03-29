"""Differentiable spatial gene detection operators.

This module provides Gaussian process-based approaches to spatial gene detection
inspired by SpatialDE for identifying spatially variable genes in spatial
transcriptomics data.

SpatialDE decomposes expression variability into spatial and non-spatial components
using GP regression with RBF kernels. The Fraction of Spatial Variance (FSV)
quantifies how much variance is explained by spatial structure.

References:
    Svensson et al. (2018) "SpatialDE: identification of spatially variable genes"
    https://www.nature.com/articles/nmeth.4636
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from flax import nnx
from jaxtyping import Array, Float

from diffbio.core.base_operators import TemperatureOperator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpatialGeneDetectorConfig(OperatorConfig):
    # pylint: disable=too-many-instance-attributes
    """Configuration for spatial gene detection.

    Attributes:
        n_genes: Number of genes to analyze.
        lengthscale: Initial lengthscale for RBF kernel (characteristic length).
        variance: Signal variance (sigma^2_s in SpatialDE).
        noise_variance: Observation noise variance (sigma^2_e in SpatialDE).
        n_inducing_points: Number of inducing points for sparse GP.
        hidden_dims: Hidden dimensions for smoothing network.
        temperature: Temperature for soft thresholding.
        pvalue_threshold: Threshold for spatial gene classification.
        learnable_kernel: Whether kernel parameters are learnable.
        compute_field_ops: Whether to compute spatial expression gradients
            and Laplacians using opifex field operations.
    """

    n_genes: int = 2000
    lengthscale: float = 1.0
    variance: float = 1.0
    noise_variance: float = 0.1
    n_inducing_points: int = 100
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    temperature: float = 1.0
    pvalue_threshold: float = 0.05
    learnable_kernel: bool = True
    compute_field_ops: bool = False


class DifferentiableSpatialGeneDetector(TemperatureOperator):
    """SpatialDE-style differentiable spatial gene detection.

    This operator identifies spatially variable genes using a differentiable
    Gaussian process approach. It computes a spatial variance score for each
    gene and provides soft assignments for spatial vs non-spatial genes.

    The model decomposes gene expression as:
        y = f(x) + epsilon
    where f(x) ~ GP(0, K) is the spatial component and epsilon ~ N(0, sigma^2)
    is the non-spatial noise.

    The Fraction of Spatial Variance (FSV) is:
        FSV = sigma^2_s / (sigma^2_s + sigma^2_e)

    Input data structure:
        - spatial_coords: Float[Array, "n_spots 2"] - Spatial coordinates
        - expression: Float[Array, "n_spots n_genes"] - Gene expression
        - total_counts: Float[Array, "n_spots"] - Total counts per spot

    Output data structure (adds):
        - spatial_variance: Float[Array, "n_genes"] - Spatial variance per gene
        - spatial_pvalues: Float[Array, "n_genes"] - P-values for spatial patterns
        - is_spatial: Float[Array, "n_genes"] - Soft spatial gene indicator
        - smoothed_expression: Float[Array, "n_spots n_genes"] - GP smoothed expression
        - fsv: Float[Array, "n_genes"] - Fraction of Spatial Variance

    Example:
        ```python
        config = SpatialGeneDetectorConfig(n_genes=2000)
        detector = DifferentiableSpatialGeneDetector(config, rngs=nnx.Rngs(42))
        result, state, meta = detector.apply(data, {}, None)
        spatial_genes = result["is_spatial"] > 0.5
        ```
    """

    def __init__(
        self,
        config: SpatialGeneDetectorConfig,
        *,
        rngs: nnx.Rngs,
        name: str | None = None,
    ):
        """Initialize the spatial gene detector.

        Args:
            config: Detector configuration.
            rngs: Random number generators.
            name: Optional name for the operator.
        """
        super().__init__(config, rngs=rngs, name=name)

        self.config: SpatialGeneDetectorConfig = config

        # Kernel parameters (learnable in log-space for positivity)
        if config.learnable_kernel:
            self.log_lengthscale = nnx.Param(jnp.log(jnp.array(config.lengthscale)))
            self.log_variance = nnx.Param(jnp.log(jnp.array(config.variance)))
            self.log_noise_variance = nnx.Param(jnp.log(jnp.array(config.noise_variance)))
        else:
            self._lengthscale = config.lengthscale
            self._variance = config.variance
            self._noise_variance = config.noise_variance

        # Smoothing network for expression (neural approximation to GP mean)
        smoothing_layers = []
        prev_dim = 2  # Spatial coordinates
        for hidden_dim in config.hidden_dims:
            smoothing_layers.append(nnx.Linear(prev_dim, hidden_dim, rngs=rngs))
            prev_dim = hidden_dim
        self.smoothing_layers = nnx.List(smoothing_layers)
        self.smoothing_out = nnx.Linear(prev_dim, config.n_genes, rngs=rngs)

    @property
    def lengthscale(self) -> Float[Array, ""] | float:
        """Get the characteristic length for RBF kernel."""
        if hasattr(self, "log_lengthscale"):
            return jnp.exp(self.log_lengthscale[...])
        return self._lengthscale

    @property
    def variance(self) -> Float[Array, ""] | float:
        """Get the signal variance parameter."""
        if hasattr(self, "log_variance"):
            return jnp.exp(self.log_variance[...])
        return self._variance

    @property
    def noise_variance(self) -> Float[Array, ""] | float:
        """Get current noise variance (sigma^2_e)."""
        if hasattr(self, "log_noise_variance"):
            return jnp.exp(self.log_noise_variance[...])
        return self._noise_variance

    def compute_kernel(
        self,
        X1: Float[Array, "n1 2"],
        X2: Float[Array, "n2 2"],
    ) -> Float[Array, "n1 n2"]:
        """Compute squared exponential (RBF) kernel matrix.

        K(x1, x2) = variance * exp(-||x1 - x2||^2 / (2 * lengthscale^2))

        This is the standard kernel used in SpatialDE for modeling
        spatial covariance.

        Args:
            X1: First set of spatial coordinates.
            X2: Second set of spatial coordinates.

        Returns:
            Kernel matrix.
        """
        # Compute squared Euclidean distances
        sq_dist = jnp.sum(
            (X1[:, None, :] - X2[None, :, :]) ** 2,
            axis=-1,
        )

        # Squared exponential (RBF) kernel
        lengthscale = self.lengthscale
        variance = self.variance

        K = variance * jnp.exp(-sq_dist / (2 * lengthscale**2))
        return K

    def compute_spatial_variance(
        self,
        coords: Float[Array, "n_spots 2"],
        expression: Float[Array, "n_spots n_genes"],
    ) -> tuple[Float[Array, "n_genes"], Float[Array, "n_genes"]]:
        """Compute spatial variance and FSV for each gene.

        Uses neural network approximation to GP posterior mean for efficiency.
        Computes variance decomposition: total = spatial + residual.

        Args:
            coords: Spatial coordinates.
            expression: Normalized gene expression.

        Returns:
            Tuple of (spatial_variance, fsv) per gene.
        """
        # Compute smoothed expression (approximate GP mean)
        smoothed = self._smooth_expression(coords, expression)

        # Center expression
        expression_centered = expression - jnp.mean(expression, axis=0, keepdims=True)

        # Total variance per gene
        total_var = jnp.var(expression_centered, axis=0)

        # Residual after spatial smoothing
        residual = expression - smoothed

        # Residual variance (non-spatial component)
        residual_var = jnp.var(residual, axis=0)

        # Spatial variance = total - residual (variance explained by space)
        spatial_var = jnp.maximum(total_var - residual_var, 0.0)

        # Fraction of Spatial Variance (FSV)
        fsv = spatial_var / (total_var + 1e-8)

        return spatial_var, fsv

    def _smooth_expression(
        self,
        coords: Float[Array, "n_spots 2"],
        expression: Float[Array, "n_spots n_genes"],
    ) -> Float[Array, "n_spots n_genes"]:
        """Compute smoothed expression using neural network.

        This provides a differentiable approximation to the GP posterior mean.

        Args:
            coords: Spatial coordinates.
            expression: Gene expression.

        Returns:
            Smoothed expression.
        """
        # Normalize coordinates for stable training
        coords_norm = (coords - jnp.mean(coords, axis=0)) / (jnp.std(coords, axis=0) + 1e-6)

        # Apply smoothing network
        h = coords_norm
        for layer in self.smoothing_layers:
            h = nnx.relu(layer(h))

        # Output layer predicts deviation from mean
        deviation = self.smoothing_out(h)

        # Smoothed = mean + learned spatial deviation
        smoothed = jnp.mean(expression, axis=0, keepdims=True) + deviation

        return smoothed

    def compute_pvalues(
        self,
        fsv: Float[Array, "n_genes"],
        n_spots: int,
    ) -> Float[Array, "n_genes"]:
        """Compute differentiable pseudo-p-values for spatial patterns.

        Uses a soft approximation to the likelihood ratio test.
        In SpatialDE, p-values come from comparing the spatial model
        to a null model without spatial structure.

        Args:
            fsv: Fraction of Spatial Variance per gene.
            n_spots: Number of spatial locations.

        Returns:
            Soft p-values (lower = more spatially variable).
        """
        # Approximate likelihood ratio statistic
        # Higher FSV -> larger LR statistic -> smaller p-value
        # Scale by n_spots to approximate degrees of freedom effect
        lr_stat = fsv * n_spots

        # Transform to pseudo-pvalue using sigmoid
        # This gives a differentiable approximation to the chi-squared CDF
        pvalues = nnx.sigmoid(-lr_stat + 2.0)

        return pvalues

    def apply(
        self,
        data: dict[str, Array],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,  # noqa: ARG002
        stats: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[dict[str, Array], dict[str, Any], dict[str, Any] | None]:
        """Apply spatial gene detection.

        Args:
            data: Input data containing:
                - spatial_coords: Float[Array, "n_spots 2"]
                - expression: Float[Array, "n_spots n_genes"]
                - total_counts: Float[Array, "n_spots"] (optional)
            state: Element state (passed through).
            metadata: Element metadata (passed through).

        Returns:
            Tuple of (output_data, state, metadata).
        """
        coords = data["spatial_coords"]
        expression = data["expression"]
        n_spots = coords.shape[0]

        # Normalize expression if total counts provided
        if "total_counts" in data:
            total_counts = data["total_counts"]
            expression_norm = expression / (total_counts[:, None] + 1e-6)
            expression_norm = expression_norm * jnp.median(total_counts)
        else:
            expression_norm = expression

        # Compute smoothed expression
        smoothed = self._smooth_expression(coords, expression_norm)

        # Compute spatial variance and FSV
        spatial_variance, fsv = self.compute_spatial_variance(coords, expression_norm)

        # Compute p-values
        pvalues = self.compute_pvalues(fsv, n_spots)

        # Soft spatial classification using temperature-controlled sigmoid
        threshold = self.config.pvalue_threshold
        temp = self._temperature
        is_spatial = nnx.sigmoid((threshold - pvalues) / temp)

        # Build output
        output_data = {
            **data,
            "spatial_variance": spatial_variance,
            "fsv": fsv,
            "spatial_pvalues": pvalues,
            "is_spatial": is_spatial,
            "smoothed_expression": smoothed,
        }

        # Optionally compute spatial field operations (gradient, laplacian)
        if self.config.compute_field_ops:
            field_ops = _compute_spatial_field_ops(coords, smoothed)
            output_data.update(field_ops)

        return output_data, state, metadata


def _compute_spatial_field_ops(
    coords: Float[Array, "n_spots 2"],
    smoothed: Float[Array, "n_spots n_genes"],
) -> dict[str, Array]:
    """Compute spatial gradient and Laplacian of smoothed expression.

    Uses opifex's autodiff-based field operations to compute per-gene
    spatial gradients and Laplacians at each spot location.

    Args:
        coords: Spatial coordinates (n_spots, 2).
        smoothed: Smoothed expression (n_spots, n_genes).

    Returns:
        Dict with:
            - ``expression_gradient``: Per-gene spatial gradient magnitude (n_genes,).
            - ``expression_laplacian``: Per-gene mean Laplacian (n_genes,).
    """
    from opifex.core.physics import compute_gradient, compute_laplacian  # noqa: PLC0415

    n_genes = smoothed.shape[1]
    gradient_magnitudes = []
    laplacian_means = []

    for gene_idx in range(n_genes):
        gene_expr = smoothed[:, gene_idx]

        # Build a scalar function f(x) -> expression at nearest point
        # Use RBF interpolation as a differentiable field
        def _gene_field(x: Array, _expr: Array = gene_expr, _coords: Array = coords) -> Array:
            dists = jnp.sum((x - _coords) ** 2, axis=-1)
            weights = jnp.exp(-dists / 2.0)
            weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-8)
            return jnp.sum(weights * _expr, axis=-1)

        # Compute gradient magnitude at each spot
        grads = compute_gradient(_gene_field, coords)  # (n_spots, 2)
        grad_mag = jnp.mean(jnp.sqrt(jnp.sum(grads**2, axis=-1) + 1e-8))
        gradient_magnitudes.append(grad_mag)

        # Compute Laplacian at each spot
        lap = compute_laplacian(_gene_field, coords)  # (n_spots,)
        laplacian_means.append(jnp.mean(jnp.abs(lap)))

    return {
        "expression_gradient": jnp.stack(gradient_magnitudes),
        "expression_laplacian": jnp.stack(laplacian_means),
    }


def create_spatial_gene_detector(
    n_genes: int = 2000,
    n_inducing_points: int = 100,
    lengthscale: float = 1.0,
    variance: float = 1.0,
    seed: int = 42,
) -> DifferentiableSpatialGeneDetector:
    """Factory function to create a spatial gene detector.

    Args:
        n_genes: Number of genes to analyze.
        n_inducing_points: Number of inducing points for sparse GP.
        lengthscale: Initial kernel lengthscale.
        variance: Initial signal variance.
        seed: Random seed.

    Returns:
        Configured DifferentiableSpatialGeneDetector instance.
    """
    config = SpatialGeneDetectorConfig(
        n_genes=n_genes,
        n_inducing_points=n_inducing_points,
        lengthscale=lengthscale,
        variance=variance,
    )
    rngs = nnx.Rngs(seed)
    return DifferentiableSpatialGeneDetector(config, rngs=rngs)
