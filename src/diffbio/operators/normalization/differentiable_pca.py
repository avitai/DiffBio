"""Robust differentiable PCA operator.

Provides a scanpy-parity PCA (zero-center then eigendecomposition of the
covariance, with the sklearn ``svd_flip`` sign convention) whose gradients stay
finite and bounded under degenerate or near-tied eigenvalues, so a downstream
loss can tune the projection during joint optimization.

The eigendecomposition gradient replaces the fragile ``1 / (w_j - w_i)`` divided
differences with a bounded, Tikhonov (Lorentzian) form ``d / (d^2 + gap_eps^2)``
that caps the gradient magnitude at ``1 / (2 * gap_eps)`` instead of diverging at
a tie, and jitters the covariance to break exact multiplicities. This is in the
spirit of the robust-differentiable-SVD literature (Wang et al. 2021); the
specific regularizer here is the Lorentzian form. Away from ties the reciprocal
equals ``1 / (w_j - w_i)``, so the gradient matches the exact eigendecomposition
VJP; only near a tie is it regularized. Eigenvalue gradients are always exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jax.typing import ArrayLike

_DEFAULT_GAP_EPS = 1.0e-6
_DEFAULT_JITTER = 1.0e-8
# Match JAX's own eigh gradient: default matmul precision on GPU/TPU uses
# reduced-precision (tf32-style) accumulation, which destroys the orthonormality
# of the eigenvectors and the accuracy of the covariance. HIGHEST forces exact
# float32 accumulation.
_PRECISION = jax.lax.Precision.HIGHEST


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def safe_eigh(
    matrix: jnp.ndarray,
    gap_eps: float = _DEFAULT_GAP_EPS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Symmetric eigendecomposition with gradients robust to degeneracies.

    Returns ``(eigenvalues, eigenvectors)`` with eigenvalues in ascending order,
    identical to ``jnp.linalg.eigh`` in the forward pass. The backward pass uses a
    bounded divided-difference so gradients stay finite when eigenvalues coincide.

    Args:
        matrix: A symmetric ``(dim, dim)`` matrix.
        gap_eps: Regularization scale for the eigenvalue-gap reciprocal.

    Returns:
        Ascending eigenvalues ``(dim,)`` and their eigenvectors ``(dim, dim)``.
    """
    del gap_eps  # Only consumed by the backward pass.
    return jnp.linalg.eigh(matrix)


def _safe_eigh_fwd(
    matrix: jnp.ndarray,
    gap_eps: float,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    """Forward pass: decompose and stash the factors for the backward pass."""
    del gap_eps  # Only consumed by the backward pass.
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    return (eigenvalues, eigenvectors), (eigenvalues, eigenvectors)


def _safe_eigh_bwd(
    gap_eps: float,
    residuals: tuple[jnp.ndarray, jnp.ndarray],
    cotangents: tuple[jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray]:
    """Backward pass with a bounded eigenvalue-gap reciprocal."""
    eigenvalues, eigenvectors = residuals
    grad_eigenvalues, grad_eigenvectors = cotangents

    # F[i, j] = 1 / (w_j - w_i), matching jnp.linalg.eigh's own VJP convention.
    gap = eigenvalues[None, :] - eigenvalues[:, None]
    # Bounded reciprocal of the gap: approaches 1/gap away from ties, is zero at a
    # tie, and never exceeds 1 / (2 * gap_eps) in magnitude. This is the core that
    # keeps gradients finite under degenerate/near-tied eigenvalues.
    reciprocal = gap / (gap**2 + gap_eps**2)
    dim = eigenvalues.shape[0]
    reciprocal = reciprocal * (1.0 - jnp.eye(dim, dtype=reciprocal.dtype))

    vt_grad_vectors = jnp.matmul(eigenvectors.T, grad_eigenvectors, precision=_PRECISION)
    inner = jnp.diag(grad_eigenvalues) + reciprocal * vt_grad_vectors
    grad_matrix = jnp.matmul(
        eigenvectors,
        jnp.matmul(inner, eigenvectors.T, precision=_PRECISION),
        precision=_PRECISION,
    )
    grad_matrix = 0.5 * (grad_matrix + grad_matrix.T)
    return (grad_matrix,)


safe_eigh.defvjp(_safe_eigh_fwd, _safe_eigh_bwd)


def robust_pca(
    features: ArrayLike,
    n_components: int,
    *,
    gap_eps: float = _DEFAULT_GAP_EPS,
    jitter: float = _DEFAULT_JITTER,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Differentiable PCA via the eigendecomposition of the covariance.

    Mirrors ``scanpy``/``sklearn`` PCA: zero-center the data, decompose the
    covariance, keep the top components in descending eigenvalue order, and apply
    the ``svd_flip`` sign convention (largest-absolute loading positive). The
    gradient path is robust to degenerate eigenvalues via :func:`safe_eigh`.

    Args:
        features: A ``(n_cells, n_genes)`` matrix.
        n_components: Number of principal components to keep. Clamped to
            ``min(n_components, n_genes, n_cells)``.
        gap_eps: Robustness scale for the eigenvector gradient.
        jitter: Diagonal jitter added to the covariance to break exact ties.

    Returns:
        Scores ``(n_cells, k)``, components ``(k, n_genes)``, and explained
        variance ``(k,)``.
    """
    features = jnp.asarray(features)
    n_samples, n_features = features.shape
    n_output = min(n_components, n_features, n_samples)

    mean = jnp.mean(features, axis=0, keepdims=True)
    centered = features - mean
    covariance = jnp.matmul(centered.T, centered, precision=_PRECISION) / jnp.maximum(
        n_samples - 1, 1
    )
    covariance = covariance + jitter * jnp.eye(n_features, dtype=covariance.dtype)

    eigenvalues, eigenvectors = safe_eigh(covariance, gap_eps)
    top_values = eigenvalues[::-1][:n_output]
    top_vectors = eigenvectors[:, ::-1][:, :n_output]

    max_abs_index = jnp.argmax(jnp.abs(top_vectors), axis=0)
    signs = jnp.sign(top_vectors[max_abs_index, jnp.arange(n_output)])
    signs = jnp.where(signs == 0.0, 1.0, signs)
    top_vectors = top_vectors * signs[None, :]

    scores = jnp.matmul(centered, top_vectors, precision=_PRECISION)
    components = top_vectors.T
    return scores, components, top_values


@dataclass(frozen=True)
class DifferentiablePCAConfig(OperatorConfig):
    """Configuration for :class:`DifferentiablePCA`.

    Attributes:
        n_components: Number of principal components to keep.
        gap_eps: Robustness scale for the eigenvector gradient reciprocal.
        jitter: Diagonal jitter added to the covariance to break exact ties.
    """

    n_components: int = 50
    gap_eps: float = _DEFAULT_GAP_EPS
    jitter: float = _DEFAULT_JITTER


class DifferentiablePCA(OperatorModule):
    """Scanpy-parity PCA with degeneracy-robust, differentiable gradients.

    Like ``DifferentiablePHATE`` and ``DifferentiableUMAP``, this is a whole-matrix
    (cross-cell) operator: one element is the full ``(n_cells, n_genes)`` feature
    matrix, and ``apply`` reduces over the cell axis to form the covariance. Only
    ``apply`` is implemented, per the ``OperatorModule`` contract; the framework's
    ``apply_batch`` maps it over a batch of such matrices (``apply`` is vmap-safe).
    """

    def __init__(
        self,
        config: DifferentiablePCAConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the operator.

        Args:
            config: PCA configuration.
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
            data: Dictionary containing ``"features"`` ``(n_cells, n_genes)``.
            state: Operator state dictionary.
            metadata: Optional metadata dictionary.
            random_params: Optional random parameters (unused).
            stats: Optional statistics dictionary (unused).

        Returns:
            Tuple of ``(output_data, state, metadata)`` where ``output_data`` adds
            ``"pca"`` scores, ``"pca_components"`` loadings, and
            ``"explained_variance"``.
        """
        del random_params, stats
        config: DifferentiablePCAConfig = self.config
        scores, components, explained_variance = robust_pca(
            data["features"],
            config.n_components,
            gap_eps=config.gap_eps,
            jitter=config.jitter,
        )
        output_data = {
            **data,
            "pca": scores,
            "pca_components": components,
            "explained_variance": explained_variance,
        }
        return output_data, state, metadata
