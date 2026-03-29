"""SINDy-based gene regulatory network inference.

Implements Sparse Identification of Nonlinear Dynamics (SINDy) for
discovering gene regulatory equations from expression time-series or
pseudotime-ordered single-cell data.

Unlike the GATv2-based ``DifferentiableGRN`` which learns regulatory
strengths via attention weights, SINDy discovers explicit governing
equations by fitting sparse coefficient vectors to a polynomial library
of candidate terms.

Algorithm:
    1. Build a polynomial feature library from expression data.
    2. Compute numerical derivatives (finite differences along time axis).
    3. Solve for sparse coefficients via differentiable soft thresholding
       (proximal gradient descent).
    4. The coefficient matrix encodes which genes regulate which.

Reference:
    Brunton, Proctor & Kutz (2016). Discovering governing equations from
    data by sparse identification of nonlinear dynamical systems. PNAS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, PyTree

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SINDyGRNConfig(OperatorConfig):
    """Configuration for SINDy GRN inference.

    Attributes:
        n_genes: Number of genes in the expression matrix.
        polynomial_degree: Maximum polynomial degree for the feature library.
        sparsity_threshold: Soft thresholding level for sparse regression.
            Higher values produce sparser coefficient matrices.
        n_iterations: Number of iterative thresholding steps (STRidge).
        ridge_alpha: Ridge regression regularisation parameter.
    """

    n_genes: int = 100
    polynomial_degree: int = 2
    sparsity_threshold: float = 0.1
    n_iterations: int = 10
    ridge_alpha: float = 0.01


def build_polynomial_library(
    x: Float[Array, "n_samples n_features"],
    degree: int = 2,
) -> Float[Array, "n_samples n_library"]:
    """Build a polynomial feature library from input data.

    For degree=1, returns x unchanged (linear terms only).
    For degree=2, appends all pairwise products (x_i * x_j for i <= j).

    Args:
        x: Input data of shape (n_samples, n_features).
        degree: Maximum polynomial degree (1 or 2).

    Returns:
        Library matrix with columns for each polynomial term.
    """
    terms = [x]

    if degree >= 2:
        n_features = x.shape[1]
        quadratic_terms = []
        for i in range(n_features):
            for j in range(i, n_features):
                quadratic_terms.append(x[:, i] * x[:, j])
        if quadratic_terms:
            terms.append(jnp.stack(quadratic_terms, axis=1))

    return jnp.concatenate(terms, axis=1)


def _soft_threshold(
    x: Array,
    threshold: float,
) -> Array:
    """Differentiable soft thresholding (proximal operator for L1).

    Args:
        x: Input array.
        threshold: Threshold level.

    Returns:
        Soft-thresholded array: sign(x) * max(|x| - threshold, 0).
    """
    return jnp.sign(x) * jax.nn.relu(jnp.abs(x) - threshold)


def _stridge_solve(
    theta: Float[Array, "n_samples n_library"],
    dx: Float[Array, "n_samples n_genes"],
    *,
    sparsity_threshold: float,
    n_iterations: int,
    ridge_alpha: float,
) -> Float[Array, "n_library n_genes"]:
    """Sequentially Thresholded Ridge Regression (STRidge).

    Iteratively solves ridge regression then soft-thresholds small
    coefficients to promote sparsity.

    Args:
        theta: Feature library matrix (n_samples, n_library).
        dx: Numerical derivatives (n_samples, n_genes).
        sparsity_threshold: Soft thresholding level.
        n_iterations: Number of threshold iterations.
        ridge_alpha: Ridge regularisation parameter.

    Returns:
        Sparse coefficient matrix (n_library, n_genes).
    """
    n_lib = theta.shape[1]
    regulariser = ridge_alpha * jnp.eye(n_lib)

    # Initial ridge regression: xi = (Theta^T Theta + alpha*I)^{-1} Theta^T dx
    gram = theta.T @ theta + regulariser
    xi = jnp.linalg.solve(gram, theta.T @ dx)

    def _threshold_step(
        xi: Float[Array, "n_library n_genes"],
        _: None,
    ) -> tuple[Float[Array, "n_library n_genes"], None]:
        return _soft_threshold(xi, sparsity_threshold), None

    xi, _ = jax.lax.scan(_threshold_step, xi, None, length=n_iterations)
    return xi


class SINDyGRNOperator(OperatorModule):
    """SINDy-based differentiable gene regulatory network inference.

    Discovers sparse governing equations for gene expression dynamics
    by fitting a polynomial library to numerical derivatives of expression
    data. The resulting coefficient matrix encodes which genes (and their
    interactions) regulate which target genes.

    Complements ``DifferentiableGRN`` (GATv2-based) by providing an
    equation-discovery approach rather than an attention-based one.

    Input data:
        - ``"counts"``: Expression matrix ``(n_timepoints, n_genes)``
          ordered by time or pseudotime.

    Output adds:
        - ``"grn_coefficients"``: Sparse coefficient matrix
          ``(n_library, n_genes)`` encoding regulatory relationships.
        - ``"grn_equations"``: Same as coefficients (alias for clarity).

    Args:
        config: SINDyGRNConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = SINDyGRNConfig(n_genes=5, polynomial_degree=1)
        >>> op = SINDyGRNOperator(config, rngs=nnx.Rngs(0))
        >>> data = {"counts": time_ordered_expression}
        >>> result, _, _ = op.apply(data, {}, None)
        >>> result["grn_coefficients"].shape
        (5, 5)
    """

    def __init__(
        self,
        config: SINDyGRNConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize SINDy GRN operator.

        Args:
            config: SINDy configuration.
            rngs: Random number generators.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)
        self.config: SINDyGRNConfig = config

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,  # noqa: ARG002
        stats: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply SINDy GRN inference.

        Args:
            data: Dictionary containing:
                - ``"counts"``: Expression matrix ``(n_timepoints, n_genes)``
            state: Element state (passed through).
            metadata: Element metadata (passed through).
            random_params: Unused.
            stats: Unused.

        Returns:
            Tuple of (output_data, state, metadata).
        """
        counts = data["counts"]  # (n_timepoints, n_genes)
        cfg = self.config

        # Compute numerical derivatives (forward differences)
        dx = counts[1:] - counts[:-1]  # (n_timepoints - 1, n_genes)
        x_mid = counts[:-1]  # Use start-of-interval values

        # Build polynomial feature library
        theta = build_polynomial_library(x_mid, degree=cfg.polynomial_degree)

        # Solve sparse regression
        coefficients = _stridge_solve(
            theta,
            dx,
            sparsity_threshold=cfg.sparsity_threshold,
            n_iterations=cfg.n_iterations,
            ridge_alpha=cfg.ridge_alpha,
        )

        return (
            {
                **data,
                "grn_coefficients": coefficients,
                "grn_equations": coefficients,
            },
            state,
            metadata,
        )
