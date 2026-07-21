"""Stochastic-gate (STG) differentiable feature selection for gene panels.

Implements the Stochastic Gates relaxation of L0 feature selection (Yamada et al.,
*Feature Selection using Stochastic Gates*, MLSys 2020): each gene carries a learnable
gate mean ``mu`` and a hard-sigmoid gate ``z = clip(mu + sigma * eps, 0, 1)`` with
``eps ~ N(0, 1)`` injected during training. At evaluation the noise is dropped and the
gate is the deterministic clamp ``clip(mu, 0, 1)``. A differentiable L0 surrogate --
``sum_g Phi(mu_g / sigma)`` with ``Phi`` the standard normal CDF -- estimates the expected
number of active gates, so adding ``l0_lambda * l0_penalty`` to a task loss drives the panel
toward a small gene set while gradients still flow to every gate through the hard-sigmoid.

Unlike :class:`SoftHVG` (a dispersion-ranked straight-through top-k, which fixes the panel
size exactly), this gate learns the panel *size* through the L0 penalty rather than fixing
it, and can be initialized from a frozen selection mask so it starts at the frozen panel
(``init_gate``) for a clean frozen-vs-joint comparison.
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

from diffbio.utils.nn_utils import get_rng_key

_DEFAULT_SIGMA = 0.5
_DEFAULT_MU_INIT = 0.5
_DEFAULT_L0_LAMBDA = 1.0


def hard_sigmoid_gate(mu: ArrayLike, noise: ArrayLike) -> jnp.ndarray:
    """Return the hard-sigmoid stochastic gate ``clip(mu + noise, 0, 1)``."""
    return jnp.clip(jnp.asarray(mu) + jnp.asarray(noise), 0.0, 1.0)


def l0_penalty(mu: ArrayLike, sigma: float) -> jnp.ndarray:
    """Differentiable L0 surrogate: the expected number of active gates.

    Equals ``sum_g Phi(mu_g / sigma)`` where ``Phi`` is the standard normal CDF -- the
    probability that gate ``g`` exceeds zero under the injected Gaussian noise.

    Args:
        mu: Per-gene gate means ``(n_genes,)``.
        sigma: Gate noise scale (strictly positive).

    Returns:
        A scalar equal to the expected count of open gates.
    """
    return jnp.sum(jax.scipy.special.ndtr(jnp.asarray(mu) / sigma))


@dataclass(frozen=True)
class StochasticGateSelectorConfig(OperatorConfig):
    """Configuration for :class:`StochasticGateSelector`.

    Attributes:
        n_genes: Number of input genes; sizes the learnable gate vector.
        sigma: Gate noise scale used during training and in the L0 surrogate.
        l0_lambda: Weight applied to the exposed L0 penalty.
        mu_init: Initial gate mean for every gene when no ``init_gate`` is given.
    """

    n_genes: int = 1
    sigma: float = _DEFAULT_SIGMA
    l0_lambda: float = _DEFAULT_L0_LAMBDA
    mu_init: float = _DEFAULT_MU_INIT

    def __post_init__(self) -> None:
        """Validate the configuration at construction, failing fast on bad values.

        Raises:
            ValueError: If ``n_genes`` or ``sigma`` is not strictly positive, or
                ``l0_lambda`` is negative.
        """
        super().__post_init__()
        if self.n_genes <= 0:
            raise ValueError(f"n_genes must be strictly positive, got {self.n_genes}")
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be strictly positive, got {self.sigma}")
        if self.l0_lambda < 0.0:
            raise ValueError(f"l0_lambda must be non-negative, got {self.l0_lambda}")


class StochasticGateSelector(OperatorModule):
    """Learnable stochastic-gate feature selection over a gene panel.

    Like :class:`SoftHVG`, this is a whole-matrix (cross-cell) operator: one element is the
    full ``(n_cells, n_genes)`` matrix and ``apply`` gates each gene column. Injects Gaussian
    gate noise when ``config.stochastic`` is set (training) and is deterministic otherwise
    (evaluation). The learnable gate means ``mu`` and the probe receive gradients; adding the
    exposed ``"l0_penalty"`` to the task loss sparsifies the panel.
    """

    def __init__(
        self,
        config: StochasticGateSelectorConfig,
        *,
        init_gate: ArrayLike | None = None,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the operator with per-gene gate means.

        Args:
            config: Gate selection configuration.
            init_gate: Optional ``(n_genes,)`` initial gate values (e.g. a frozen 0/1
                selection mask); the deterministic gate then reproduces it. Defaults to a
                uniform ``config.mu_init``.
            rngs: RNG state supplying the ``"sample"`` stream for gate noise.
            name: Optional module name.
        """
        super().__init__(config, rngs=rngs, name=name)
        if init_gate is None:
            mu = jnp.full((config.n_genes,), config.mu_init, dtype=jnp.float32)
        else:
            mu = jnp.asarray(init_gate, dtype=jnp.float32)
        self.mu = nnx.Param(mu)
        self.rngs = rngs

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict | None,
        random_params: dict | None = None,
        stats: dict | None = None,
    ) -> tuple[dict, dict, dict | None]:
        """Gate ``data["features"]`` by the stochastic gates and expose the L0 penalty.

        Args:
            data: Dictionary containing ``"features"`` ``(n_cells, n_genes)``.
            state: Operator state dictionary.
            metadata: Optional metadata dictionary.
            random_params: Optional random parameters (unused).
            stats: Optional statistics dictionary (unused).

        Returns:
            Tuple of ``(output_data, state, metadata)`` where ``output_data`` gates
            ``"features"`` and adds ``"gate"`` (the per-gene gate) and ``"l0_penalty"``.
        """
        del random_params, stats
        config: StochasticGateSelectorConfig = self.config
        mu = self.mu[...]
        if config.stochastic:
            # stream_name is validated non-None whenever stochastic is set.
            key = get_rng_key(self.rngs, config.stream_name or "gate_noise", fallback_seed=0)
            noise = config.sigma * jax.random.normal(key, mu.shape)
        else:
            noise = jnp.zeros_like(mu)
        gate = hard_sigmoid_gate(mu, noise)
        gated = data["features"] * gate[None, :]
        output_data = {
            **data,
            "features": gated,
            "gate": gate,
            "l0_penalty": config.l0_lambda * l0_penalty(mu, config.sigma),
        }
        return output_data, state, metadata
