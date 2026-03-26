"""Optimal-transport trajectory inference between two single-cell timepoints.

This module implements Waddington-OT-style trajectory inference using
entropy-regularised optimal transport. Given gene-expression matrices at
two timepoints, the operator:

1. Computes a squared-Euclidean expression cost matrix between all cell
   pairs across the two timepoints.
2. Solves for the transport plan via the shared ``SinkhornLayer`` from
   ``diffbio.core.optimal_transport`` (DRY).
3. Estimates per-cell growth rates from the transport-plan row sums.
4. Interpolates an intermediate cell distribution at a configurable time
   between the two observations.

All operations are fully differentiable through JAX, enabling gradient-based
optimisation of upstream embeddings or transport parameters.

References:
    Schiebinger et al., "Optimal-Transport Analysis of Single-Cell Gene
    Expression Identifies Developmental Trajectories in Reprogramming",
    Cell 2019.
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, PyTree

from diffbio.constants import EPSILON
from diffbio.core.optimal_transport import SinkhornLayer
from diffbio.utils.nn_utils import ensure_rngs

logger = logging.getLogger(__name__)

__all__ = [
    "OTTrajectoryConfig",
    "DifferentiableOTTrajectory",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OTTrajectoryConfig(OperatorConfig):
    """Configuration for OT-based trajectory inference.

    Attributes:
        n_genes: Number of input genes per cell.
        sinkhorn_epsilon: Entropy regularisation strength for the Sinkhorn
            solver. Larger values produce smoother transport plans.
        sinkhorn_iters: Number of Sinkhorn iterations.
        growth_rate_regularization: Scaling factor applied to raw row-sums
            before normalisation. Higher values amplify growth-rate variation.
        interpolation_time: Fraction in (0, 1) at which to compute the
            interpolated cell distribution between t1 and t2.
    """

    n_genes: int = 200
    sinkhorn_epsilon: float = 0.1
    sinkhorn_iters: int = 100
    growth_rate_regularization: float = 1.0
    interpolation_time: float = 0.5


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------


class DifferentiableOTTrajectory(OperatorModule):
    """Waddington-OT-style differentiable trajectory inference.

    Computes an optimal-transport plan between cell populations at two
    timepoints, estimates per-cell growth (proliferation) rates, and
    interpolates an intermediate cell distribution.

    Algorithm:
        1. Build the squared-Euclidean cost matrix ``C[i,j] = ||x_i - y_j||^2``
           between cells at t1 and t2.
        2. Compute the entropy-regularised transport plan via ``SinkhornLayer``.
        3. Derive growth rates from the transport plan: cells that transport to
           more targets in t2 have higher proliferation. Normalise so that
           ``mean(growth_rates) == 1``.
        4. Interpolate an intermediate distribution at time *s*:
           ``x_interp = (1-s) * x_t1 + s * (T @ x_t2) / T.sum(axis=1)``

    Args:
        config: OTTrajectoryConfig with operator parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = OTTrajectoryConfig(n_genes=100, sinkhorn_iters=50)
        >>> op = DifferentiableOTTrajectory(config)
        >>> data = {
        ...     "counts_t1": jnp.ones((20, 100)),
        ...     "counts_t2": jnp.ones((25, 100)),
        ... }
        >>> result, state, meta = op.apply(data, {}, None)
        >>> result["transport_plan"].shape
        (20, 25)
    """

    def __init__(
        self,
        config: OTTrajectoryConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the OT trajectory operator.

        Args:
            config: OT trajectory configuration.
            rngs: Random number generators (for API consistency).
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        rngs = ensure_rngs(rngs)

        self.sinkhorn = SinkhornLayer(
            epsilon=config.sinkhorn_epsilon,
            num_iters=config.sinkhorn_iters,
            rngs=rngs,
        )

    # -- Internal helpers ---------------------------------------------------

    def _compute_expression_cost(
        self,
        counts_t1: Float[Array, "n1 g"],
        counts_t2: Float[Array, "n2 g"],
    ) -> Float[Array, "n1 n2"]:
        """Compute the squared-Euclidean expression cost matrix.

        Uses the expansion ``||a - b||^2 = ||a||^2 + ||b||^2 - 2 a . b``
        for efficiency.

        Args:
            counts_t1: Expression matrix at timepoint 1.
            counts_t2: Expression matrix at timepoint 2.

        Returns:
            Cost matrix of shape ``(n1, n2)``.
        """
        sq1 = jnp.sum(counts_t1**2, axis=-1, keepdims=True)  # (n1, 1)
        sq2 = jnp.sum(counts_t2**2, axis=-1)  # (n2,)
        dot = jnp.dot(counts_t1, counts_t2.T)  # (n1, n2)
        cost = sq1 + sq2 - 2.0 * dot
        return jnp.maximum(cost, 0.0)

    def _estimate_growth_rates(
        self,
        transport_plan: Float[Array, "n1 n2"],
    ) -> Float[Array, "n1"]:
        """Estimate per-cell growth rates from the transport plan.

        Cells whose row in the transport plan sums to a larger value are
        inferred to be proliferating (they contribute mass to more cells
        in the next timepoint). The rates are normalised so that
        ``mean(growth_rates) == 1``.

        Args:
            transport_plan: Optimal transport plan ``(n1, n2)``.

        Returns:
            Normalised growth rates ``(n1,)``.
        """
        raw_rates = jnp.sum(transport_plan, axis=1)
        mean_rate = jnp.mean(raw_rates) + EPSILON
        return raw_rates / mean_rate

    def _interpolate_trajectory(
        self,
        counts_t1: Float[Array, "n1 g"],
        counts_t2: Float[Array, "n2 g"],
        transport_plan: Float[Array, "n1 n2"],
        interpolation_time: float,
    ) -> Float[Array, "n1 g"]:
        """Interpolate cell states at an intermediate timepoint.

        For each cell *i* in t1, the transported expression is the weighted
        average of t2 cells according to the transport plan. The interpolated
        state is a convex combination of the original t1 expression and the
        transported expression.

        Args:
            counts_t1: Expression at timepoint 1.
            counts_t2: Expression at timepoint 2.
            transport_plan: Optimal transport plan.
            interpolation_time: Fraction *s* in (0, 1).

        Returns:
            Interpolated expression matrix ``(n1, g)``.
        """
        # Row-normalise the transport plan so each row sums to 1
        row_sums = jnp.sum(transport_plan, axis=1, keepdims=True) + EPSILON
        plan_normalised = transport_plan / row_sums

        # Transported expression: weighted average of t2 cells
        transported = plan_normalised @ counts_t2  # (n1, g)

        # Convex interpolation
        s = interpolation_time
        return (1.0 - s) * counts_t1 + s * transported

    # -- Public apply -------------------------------------------------------

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply OT-based trajectory inference to two-timepoint expression data.

        Args:
            data: Dictionary containing:
                - ``"counts_t1"``: Expression matrix at timepoint 1 ``(n1, g)``
                - ``"counts_t2"``: Expression matrix at timepoint 2 ``(n2, g)``
            state: Element state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: Not used (non-stochastic operator).
            stats: Not used.

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains all original keys plus:

                    - ``"transport_plan"``: OT plan ``(n1, n2)``
                    - ``"growth_rates"``: Per-cell growth rates ``(n1,)``
                    - ``"interpolated_counts"``: Interpolated expression
                      at the configured midpoint ``(n1, g)``
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        counts_t1: Float[Array, "n1 g"] = data["counts_t1"]
        counts_t2: Float[Array, "n2 g"] = data["counts_t2"]

        n1 = counts_t1.shape[0]
        n2 = counts_t2.shape[0]

        # Step 1: Expression cost matrix
        cost = self._compute_expression_cost(counts_t1, counts_t2)

        # Step 2: Solve OT via Sinkhorn with uniform marginals
        a = jnp.ones(n1) / n1
        b = jnp.ones(n2) / n2
        transport_plan = self.sinkhorn(cost, a, b)

        # Step 3: Growth rates from row sums
        growth_rates = self._estimate_growth_rates(transport_plan)

        # Step 4: Interpolated expression at configured time
        interpolated = self._interpolate_trajectory(
            counts_t1, counts_t2, transport_plan, self.config.interpolation_time
        )

        transformed_data = {
            **data,
            "transport_plan": transport_plan,
            "growth_rates": growth_rates,
            "interpolated_counts": interpolated,
        }

        return transformed_data, state, metadata
