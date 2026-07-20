"""Fully differentiable, matrix-free PCA via a randomized subspace-iteration solver.

Where :mod:`diffbio.operators.normalization.differentiable_pca` forms the ``d x d``
covariance and differentiates its dense eigendecomposition (with a Lorentzian-regularized
eigengap VJP), this operator never forms the covariance and never calls ``eigh`` on it. It
computes the top principal directions with a *matrix-free* randomized subspace iteration
(``opifex.uncertainty.linalg.randomized_svd``; Halko-Martinsson-Tropp, SIAM Review 2011)
applied to the centered data: a Gaussian sketch is pushed through repeated ``Xc`` /
``Xc^T`` products (block power iterations) and re-orthonormalized by a QR, so the leading
singular directions -- the PCA loadings -- emerge without the ghost-eigenvalue artifacts of
single-vector Lanczos.

Subspace iteration is the block generalization of power iteration the paper's future work
calls for: the whole solve is a sequence of differentiable matmuls and a QR, so gradients
flow through the eigensolver to whatever produced the features -- making the PCA basis
itself trainable end to end, without a fixed anchor -- while never materializing the
covariance, so it scales to many features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jax.typing import ArrayLike
from opifex.uncertainty.linalg import randomized_svd

_PRECISION = jax.lax.Precision.HIGHEST
"""Force exact float32 accumulation so the sketch products stay orthonormal-safe."""


def matfree_pca(
    features: ArrayLike,
    n_components: int,
    *,
    num_iterations: int = 2,
    oversampling: int = 10,
    init_seed: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Differentiable matrix-free PCA of ``features`` via randomized subspace iteration.

    Zero-centers the data, runs a randomized SVD of the centered matrix through matrix-free
    ``Xc`` / ``Xc^T`` products, and applies the ``svd_flip`` sign convention (largest-absolute
    loading positive) for a deterministic basis. Explained variance is ``sigma^2 / (n - 1)``.

    Args:
        features: A ``(n_samples, n_features)`` matrix.
        n_components: Number of principal components to keep.
        num_iterations: Subspace (power) iterations; 1--2 is the robust standard. Large
            values can overflow on large-magnitude spectra, as the sketch is not
            renormalized between iterations.
        oversampling: Extra sketch columns for robust spectral capture (Halko+ §4.4);
            ``n_components + oversampling`` must not exceed ``min(n_samples, n_features)``.
        init_seed: Seed for the deterministic Gaussian sketch.

    Returns:
        Scores ``(n_samples, n_components)``, components ``(n_components, n_features)``,
        and explained variance ``(n_components,)``.
    """
    features = jnp.asarray(features)
    n_samples, n_features = features.shape

    mean = jnp.mean(features, axis=0, keepdims=True)
    centered = features - mean

    def matvec(vector: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(centered, vector, precision=_PRECISION)

    def matvec_transpose(vector: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(centered.T, vector, precision=_PRECISION)

    _, singular_values, right_vectors = randomized_svd(
        matvec=matvec,
        matvec_transpose=matvec_transpose,
        dim_rows=n_samples,
        dim_cols=n_features,
        rank=n_components,
        oversampling=oversampling,
        num_iterations=num_iterations,
        key=jax.random.key(init_seed),
    )

    top_vectors = right_vectors
    max_abs_index = jnp.argmax(jnp.abs(top_vectors), axis=0)
    signs = jnp.sign(top_vectors[max_abs_index, jnp.arange(n_components)])
    signs = jnp.where(signs == 0.0, 1.0, signs)
    top_vectors = top_vectors * signs[None, :]

    scores = jnp.matmul(centered, top_vectors, precision=_PRECISION)
    components = top_vectors.T
    explained_variance = singular_values**2 / jnp.maximum(n_samples - 1, 1)
    return scores, components, explained_variance


@dataclass(frozen=True)
class MatrixFreePCAConfig(OperatorConfig):
    """Configuration for :class:`MatrixFreePCA`.

    Attributes:
        n_components: Number of principal components to keep.
        num_iterations: Subspace (power) iterations; 1--2 is the robust standard.
        oversampling: Extra sketch columns for robust spectral capture;
            ``n_components + oversampling`` must not exceed ``min(n_samples, n_features)``
            at ``apply`` time.
        init_seed: Seed for the deterministic Gaussian sketch.
    """

    n_components: int = 50
    num_iterations: int = 2
    oversampling: int = 10
    init_seed: int = 0

    def __post_init__(self) -> None:
        """Validate the sizes, failing fast on inconsistent values.

        Raises:
            ValueError: If ``n_components`` is non-positive, ``num_iterations`` is
                negative, or ``oversampling`` is negative.
        """
        super().__post_init__()
        if self.n_components <= 0:
            raise ValueError(f"n_components must be strictly positive, got {self.n_components}")
        if self.num_iterations < 0:
            raise ValueError(f"num_iterations must be non-negative, got {self.num_iterations}")
        if self.oversampling < 0:
            raise ValueError(f"oversampling must be non-negative, got {self.oversampling}")


class MatrixFreePCA(OperatorModule):
    """Differentiable matrix-free PCA whose basis is recomputed from the data each call.

    A whole-matrix (cross-sample) operator, like :class:`DifferentiablePCA`: one element is
    the full ``(n_samples, n_features)`` matrix and ``apply`` reduces over the sample axis
    through the covariance ``matvec``. It carries no trainable parameters -- the basis is a
    differentiable function of the input, so gradients reach upstream (learnable)
    preprocessing. Only ``apply`` is implemented, per the ``OperatorModule`` contract; it is
    vmap-safe so the framework batches it over a stack of matrices.
    """

    def __init__(
        self,
        config: MatrixFreePCAConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the operator.

        Args:
            config: Matrix-free PCA configuration.
            rngs: Optional RNG state (unused; kept for interface compatibility).
            name: Optional module name.
        """
        super().__init__(config, rngs=rngs, name=name)

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict | None,
        random_params: dict | None = None,
        stats: dict | None = None,
    ) -> tuple[dict, dict, dict | None]:
        """Project ``data["features"]`` onto its leading principal components.

        Args:
            data: Dictionary containing ``"features"`` ``(n_samples, n_features)``.
            state: Operator state dictionary.
            metadata: Optional metadata dictionary.
            random_params: Optional random parameters (unused).
            stats: Optional statistics dictionary (unused).

        Returns:
            Tuple of ``(output_data, state, metadata)`` where ``output_data`` adds ``"pca"``
            scores, ``"pca_components"`` loadings, and ``"explained_variance"``.
        """
        del random_params, stats
        config: MatrixFreePCAConfig = self.config
        scores, components, explained_variance = matfree_pca(
            data["features"],
            config.n_components,
            num_iterations=config.num_iterations,
            oversampling=config.oversampling,
            init_seed=config.init_seed,
        )
        output_data = {
            **data,
            "pca": scores,
            "pca_components": components,
            "explained_variance": explained_variance,
        }
        return output_data, state, metadata
