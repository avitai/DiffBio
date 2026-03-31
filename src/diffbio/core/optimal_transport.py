"""Optimal transport layers for differentiable assignment and matching.

# TODO: Migrate to artifex

This module provides differentiable optimal transport solvers using the
Sinkhorn algorithm in log-domain for numerical stability.

Components:

- **SinkhornLayer**: Computes the entropy-regularised optimal transport plan
  between two discrete distributions given a cost matrix, using the
  Sinkhorn-Knopp algorithm in log-domain.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

__all__ = [
    "SinkhornLayer",
]


class SinkhornLayer(nnx.Module):
    """Sinkhorn optimal transport layer (log-domain).

    # TODO: Migrate to artifex

    Computes the entropy-regularised optimal transport plan between two
    discrete marginal distributions ``a`` and ``b`` given a cost matrix ``C``,
    by solving::

        min_{P >= 0}  <P, C> - epsilon * H(P)
        s.t.  P @ 1 = a,   P^T @ 1 = b

    The algorithm runs in log-domain for numerical stability::

        f, g = 0, 0
        for _ in range(num_iters):
            f = epsilon * log(a) - epsilon * logsumexp((-C + g) / epsilon, axis=1)
            g = epsilon * log(b) - epsilon * logsumexp((-C + f) / epsilon, axis=0)
        P = exp((f[:, None] + g[None, :] - C) / epsilon)

    Args:
        epsilon: Entropy regularisation strength (larger = smoother plan).
        num_iters: Number of Sinkhorn iterations.
        rngs: Flax NNX random number generators (unused, kept for API consistency).
    """

    def __init__(
        self,
        epsilon: float,
        num_iters: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the Sinkhorn layer.

        Args:
            epsilon: Regularisation strength.
            num_iters: Number of Sinkhorn iterations.
            rngs: Random number generators (for API consistency).
        """
        super().__init__()
        self.epsilon = nnx.static(epsilon)
        self.num_iters = nnx.static(num_iters)

    def __call__(
        self,
        cost: Float[Array, "n m"],
        a: Float[Array, " n"],
        b: Float[Array, " m"],
    ) -> Float[Array, "n m"]:
        """Compute the optimal transport plan.

        Args:
            cost: Cost matrix of shape ``(n, m)``.
            a: Source marginal distribution of shape ``(n,)``, must sum to 1.
            b: Target marginal distribution of shape ``(m,)``, must sum to 1.

        Returns:
            Transport plan of shape ``(n, m)`` satisfying (approximately)
            ``P @ 1 = a`` and ``P^T @ 1 = b``.
        """
        return _sinkhorn_log_domain(cost, a, b, self.epsilon, self.num_iters)


def _sinkhorn_log_domain(
    cost: Float[Array, "n m"],
    a: Float[Array, " n"],
    b: Float[Array, " m"],
    epsilon: float,
    num_iters: int,
) -> Float[Array, "n m"]:
    """Run the Sinkhorn algorithm in log-domain.

    Args:
        cost: Cost matrix ``(n, m)``.
        a: Source marginal ``(n,)``.
        b: Target marginal ``(m,)``.
        epsilon: Regularisation parameter.
        num_iters: Number of iterations.

    Returns:
        Transport plan ``(n, m)``.
    """
    log_a = jnp.log(a + 1e-30)
    log_b = jnp.log(b + 1e-30)

    f = jnp.zeros_like(a)
    g = jnp.zeros_like(b)

    def _step(
        carry: tuple[Float[Array, " n"], Float[Array, " m"]], _: None
    ) -> tuple[tuple[Float[Array, " n"], Float[Array, " m"]], None]:
        """Perform one Sinkhorn iteration updating dual variables f and g."""
        f_prev, g_prev = carry
        # f update: epsilon * log(a) - epsilon * logsumexp((-C + g) / epsilon, axis=1)
        f_new = epsilon * log_a - epsilon * jax.scipy.special.logsumexp(
            (-cost + g_prev[None, :]) / epsilon, axis=1
        )
        # g update: epsilon * log(b) - epsilon * logsumexp((-C + f) / epsilon, axis=0)
        g_new = epsilon * log_b - epsilon * jax.scipy.special.logsumexp(
            (-cost + f_new[:, None]) / epsilon, axis=0
        )
        return (f_new, g_new), None

    (f, g), _ = jax.lax.scan(_step, (f, g), None, length=num_iters)

    # Recover the transport plan
    log_plan = (f[:, None] + g[None, :] - cost) / epsilon
    return jnp.exp(log_plan)
