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
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from flax import nnx
from jaxtyping import Array, Float

from diffbio.core import soft_ops
from diffbio.core.base_operators import TemperatureOperator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SpatialKernelConfig:
    """Gaussian-process kernel configuration."""

    lengthscale: float = 1.0
    variance: float = 1.0
    noise_variance: float = 0.1
    n_inducing_points: int = 100
    learnable_kernel: bool = True


@dataclass(frozen=True)
class _SpatialDetectionConfig:
    """Spatial gene classification configuration."""

    n_genes: int = 2000
    hidden_dims: tuple[int, ...] | list[int] = (64, 32)
    temperature: float = 1.0
    pvalue_threshold: float = 0.05
    compute_field_ops: bool = False


@dataclass(frozen=True)
class SpatialGeneDetectorConfig(
    _SpatialKernelConfig,
    _SpatialDetectionConfig,
    OperatorConfig,
):
    """Configuration for spatial gene detection."""

    def __post_init__(self) -> None:
        """Validate the spatial gene detector configuration."""
        super().__post_init__()

        hidden_dims = tuple(self.hidden_dims)
        object.__setattr__(self, "hidden_dims", hidden_dims)

        if self.n_genes <= 0:
            raise ValueError("n_genes must be positive.")
        if self.lengthscale <= 0.0:
            raise ValueError("lengthscale must be positive.")
        if self.variance <= 0.0:
            raise ValueError("variance must be positive.")
        if self.noise_variance <= 0.0:
            raise ValueError("noise_variance must be positive.")
        if self.n_inducing_points <= 0:
            raise ValueError("n_inducing_points must be positive.")
        if not hidden_dims or any(hidden_dim <= 0 for hidden_dim in hidden_dims):
            raise ValueError("hidden_dims must contain only positive integers.")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if not 0.0 <= self.pvalue_threshold <= 1.0:
            raise ValueError("pvalue_threshold must be between 0.0 and 1.0.")


@dataclass(frozen=True, slots=True)
class _FixedKernelState:
    """Static kernel parameters for non-learnable mode."""

    log_lengthscale: float
    log_variance: float
    log_noise_variance: float


class _LearnableKernelState(nnx.Module):
    """Learnable kernel parameters stored in log-space."""

    def __init__(self, config: SpatialGeneDetectorConfig) -> None:
        self.log_lengthscale = nnx.Param(jnp.log(jnp.array(config.lengthscale)))
        self.log_variance = nnx.Param(jnp.log(jnp.array(config.variance)))
        self.log_noise_variance = nnx.Param(jnp.log(jnp.array(config.noise_variance)))


class _SpatialSmoothingNetwork(nnx.Module):
    """Neural approximation to the GP posterior mean."""

    def __init__(
        self,
        *,
        hidden_dims: tuple[int, ...],
        n_genes: int,
        rngs: nnx.Rngs,
    ) -> None:
        smoothing_layers = []
        prev_dim = 2
        for hidden_dim in hidden_dims:
            smoothing_layers.append(nnx.Linear(prev_dim, hidden_dim, rngs=rngs))
            prev_dim = hidden_dim
        self.layers = nnx.List(smoothing_layers)
        self.output = nnx.Linear(prev_dim, n_genes, rngs=rngs)

    def __call__(self, coords: Float[Array, "n_spots 2"]) -> Float[Array, "n_spots n_genes"]:
        """Predict a spatial deviation field from normalized coordinates."""
        hidden = coords
        for layer in self.layers:
            hidden = nnx.relu(layer(hidden))
        return self.output(hidden)


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

        # Kernel parameters (learnable in log-space for positivity)
        if config.learnable_kernel:
            self.kernel_state = _LearnableKernelState(config)
        else:
            self.kernel_state = nnx.static(
                _FixedKernelState(
                    log_lengthscale=float(jnp.log(jnp.array(config.lengthscale))),
                    log_variance=float(jnp.log(jnp.array(config.variance))),
                    log_noise_variance=float(jnp.log(jnp.array(config.noise_variance))),
                )
            )

        # Smoothing network for expression (neural approximation to GP mean)
        self.smoothing_network = _SpatialSmoothingNetwork(
            hidden_dims=tuple(config.hidden_dims),
            n_genes=config.n_genes,
            rngs=rngs,
        )

    @property
    def lengthscale(self) -> Float[Array, ""] | float:
        """Get the characteristic length for RBF kernel."""
        kernel_state = self.kernel_state
        if isinstance(kernel_state, _LearnableKernelState):
            return jnp.exp(kernel_state.log_lengthscale[...])
        return jnp.exp(jnp.asarray(kernel_state.log_lengthscale))

    @property
    def variance(self) -> Float[Array, ""] | float:
        """Get the signal variance parameter."""
        kernel_state = self.kernel_state
        if isinstance(kernel_state, _LearnableKernelState):
            return jnp.exp(kernel_state.log_variance[...])
        return jnp.exp(jnp.asarray(kernel_state.log_variance))

    @property
    def noise_variance(self) -> Float[Array, ""] | float:
        """Get current noise variance (sigma^2_e)."""
        kernel_state = self.kernel_state
        if isinstance(kernel_state, _LearnableKernelState):
            return jnp.exp(kernel_state.log_noise_variance[...])
        return jnp.exp(jnp.asarray(kernel_state.log_noise_variance))

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
        deviation = self.smoothing_network(coords_norm)

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
            expression_norm = expression_norm * soft_ops.median(total_counts, softness=0.1)
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
        is_spatial = soft_ops.less(pvalues, threshold, softness=temp)

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
    spatial gradients and Laplacians at each spot location. Vectorized
    over genes via ``jax.vmap`` — no Python for-loops.

    Args:
        coords: Spatial coordinates (n_spots, 2).
        smoothed: Smoothed expression (n_spots, n_genes).

    Returns:
        Dict with:
            - ``expression_gradient``: Per-gene spatial gradient magnitude (n_genes,).
            - ``expression_laplacian``: Per-gene mean Laplacian (n_genes,).
    """
    from opifex.core.physics import compute_gradient, compute_laplacian  # noqa: PLC0415

    def _gene_field(x: Array, gene_expr: Array) -> Array:
        """RBF-interpolated scalar field for a single gene."""
        dists = jnp.sum((x - coords) ** 2, axis=-1)
        weights = jnp.exp(-dists / 2.0)
        weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-8)
        return jnp.sum(weights * gene_expr, axis=-1)

    def _per_gene_metrics(gene_expr: Array) -> tuple[Array, Array]:
        """Compute gradient magnitude and Laplacian for one gene."""
        field_fn = lambda x: _gene_field(x, gene_expr)  # noqa: E731
        grads = compute_gradient(field_fn, coords)
        grad_mag = jnp.mean(jnp.sqrt(jnp.sum(grads**2, axis=-1) + 1e-8))
        lap = compute_laplacian(field_fn, coords)
        lap_mean = jnp.mean(jnp.abs(lap))
        return grad_mag, lap_mean

    # Vectorize over genes (columns of smoothed)
    gradient_mags, laplacian_means = jax.vmap(_per_gene_metrics)(
        smoothed.T
    )  # smoothed.T is (n_genes, n_spots)

    return {
        "expression_gradient": gradient_mags,
        "expression_laplacian": laplacian_means,
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
