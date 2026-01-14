"""RNA velocity estimation via Neural ODEs.

This module provides differentiable RNA velocity estimation using neural
networks to learn splicing kinetics and integrate ODEs.

Key technique: Uses neural networks to learn per-gene kinetics parameters
(transcription, splicing, degradation rates) and integrates the splicing
ODE differentiably using Euler method.

Applications: Inferring cell state transitions and developmental trajectories
from single-cell RNA-seq data with spliced/unspliced counts.
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
class VelocityConfig(OperatorConfig):
    """Configuration for DifferentiableVelocity.

    Attributes:
        n_genes: Number of genes.
        hidden_dim: Hidden dimension for neural networks.
        dt: Time step for ODE integration.
        n_steps: Number of integration steps.
        kinetics_model: Type of kinetics model ("standard" or "dynamical").
        stochastic: Whether the operator uses randomness.
        stream_name: RNG stream name.
    """

    n_genes: int = 2000
    hidden_dim: int = 64
    dt: float = 0.1
    n_steps: int = 10
    kinetics_model: str = "standard"
    stochastic: bool = False
    stream_name: str | None = None


class TimeEncoder(nnx.Module):
    """Encoder for estimating latent time from expression."""

    def __init__(
        self,
        n_genes: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the time encoder.

        Args:
            n_genes: Number of input genes.
            hidden_dim: Hidden dimension.
            rngs: Random number generators.
        """
        super().__init__()

        # Encode spliced + unspliced to latent time
        self.linear1 = nnx.Linear(
            in_features=n_genes * 2,
            out_features=hidden_dim,
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            rngs=rngs,
        )
        self.time_proj = nnx.Linear(
            in_features=hidden_dim,
            out_features=1,
            rngs=rngs,
        )
        self.norm = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)

    def __call__(
        self,
        spliced: Float[Array, "n_cells n_genes"],
        unspliced: Float[Array, "n_cells n_genes"],
    ) -> Float[Array, "n_cells"]:
        """Estimate latent time for each cell.

        Args:
            spliced: Spliced (mature) mRNA counts.
            unspliced: Unspliced (nascent) mRNA counts.

        Returns:
            Latent time estimates per cell (bounded 0-1).
        """
        # Concatenate and log-transform
        x = jnp.concatenate([jnp.log1p(spliced), jnp.log1p(unspliced)], axis=-1)

        # Encode
        x = nnx.gelu(self.linear1(x))
        x = self.norm(nnx.gelu(self.linear2(x)))

        # Project to time (sigmoid for 0-1 range)
        time = jax.nn.sigmoid(self.time_proj(x)).squeeze(-1)

        return time


class KineticsEncoder(nnx.Module):
    """Encoder for learning per-gene kinetics parameters."""

    def __init__(
        self,
        n_genes: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the kinetics encoder.

        Args:
            n_genes: Number of genes.
            rngs: Random number generators.
        """
        super().__init__()

        # Learnable per-gene kinetics (alpha, beta, gamma)
        # Initialize with reasonable defaults
        key = rngs.params()
        k1, k2, k3 = jax.random.split(key, 3)

        # Transcription rate (alpha): typically 0.1-10
        self.log_alpha = nnx.Param(jax.random.normal(k1, (n_genes,)) * 0.5)

        # Splicing rate (beta): typically 0.1-1
        self.log_beta = nnx.Param(jax.random.normal(k2, (n_genes,)) * 0.5 - 1.0)

        # Degradation rate (gamma): typically 0.01-0.5
        self.log_gamma = nnx.Param(jax.random.normal(k3, (n_genes,)) * 0.5 - 2.0)

    def __call__(self) -> tuple[
        Float[Array, "n_genes"],
        Float[Array, "n_genes"],
        Float[Array, "n_genes"],
    ]:
        """Get kinetics parameters.

        Returns:
            Tuple of (alpha, beta, gamma) - all positive via softplus.
        """
        alpha = jax.nn.softplus(self.log_alpha.value)
        beta = jax.nn.softplus(self.log_beta.value)
        gamma = jax.nn.softplus(self.log_gamma.value)

        return alpha, beta, gamma


class DifferentiableVelocity(OperatorModule):
    """Differentiable RNA velocity estimation via Neural ODEs.

    This operator estimates RNA velocity from spliced and unspliced
    counts using learned kinetics parameters and differentiable ODE
    integration.

    Algorithm:
    1. Encode expression to latent time per cell
    2. Learn per-gene kinetics (alpha, beta, gamma)
    3. Compute velocity from splicing ODE:
       ds/dt = beta * u - gamma * s
       du/dt = alpha - beta * u
    4. Integrate ODE using Euler method

    Args:
        config: VelocityConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = VelocityConfig(n_genes=2000)
        >>> velocity = DifferentiableVelocity(config, rngs=nnx.Rngs(42))
        >>> data = {"spliced": spliced, "unspliced": unspliced}
        >>> result, state, meta = velocity.apply(data, {}, None)
    """

    def __init__(
        self,
        config: VelocityConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the velocity operator.

        Args:
            config: Velocity configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.n_genes = config.n_genes
        self.dt = config.dt
        self.n_steps = config.n_steps

        # Time encoder
        self.time_encoder = TimeEncoder(
            n_genes=config.n_genes,
            hidden_dim=config.hidden_dim,
            rngs=rngs,
        )

        # Kinetics encoder
        self.kinetics_encoder = KineticsEncoder(
            n_genes=config.n_genes,
            rngs=rngs,
        )

    def _compute_velocity(
        self,
        spliced: Float[Array, "n_cells n_genes"],
        unspliced: Float[Array, "n_cells n_genes"],
        beta: Float[Array, "n_genes"],
        gamma: Float[Array, "n_genes"],
    ) -> Float[Array, "n_cells n_genes"]:
        """Compute RNA velocity from splicing dynamics.

        The standard RNA velocity model:
        ds/dt = beta * u - gamma * s

        Args:
            spliced: Spliced counts.
            unspliced: Unspliced counts.
            beta: Splicing rate.
            gamma: Degradation rate.

        Returns:
            Velocity (ds/dt) for each cell-gene pair.
        """
        # Velocity is the rate of change of spliced mRNA
        # ds/dt = splicing_in - degradation_out
        # ds/dt = beta * unspliced - gamma * spliced
        velocity = beta[None, :] * unspliced - gamma[None, :] * spliced

        return velocity

    def _euler_step(
        self,
        s: Float[Array, "n_cells n_genes"],
        u: Float[Array, "n_cells n_genes"],
        alpha: Float[Array, "n_genes"],
        beta: Float[Array, "n_genes"],
        gamma: Float[Array, "n_genes"],
        dt: float,
    ) -> tuple[Float[Array, "n_cells n_genes"], Float[Array, "n_cells n_genes"]]:
        """Single Euler integration step.

        Args:
            s: Current spliced counts.
            u: Current unspliced counts.
            alpha: Transcription rate.
            beta: Splicing rate.
            gamma: Degradation rate.
            dt: Time step.

        Returns:
            Tuple of (new_spliced, new_unspliced).
        """
        # ODE system:
        # du/dt = alpha - beta * u
        # ds/dt = beta * u - gamma * s
        du_dt = alpha[None, :] - beta[None, :] * u
        ds_dt = beta[None, :] * u - gamma[None, :] * s

        # Euler update
        u_new = u + dt * du_dt
        s_new = s + dt * ds_dt

        # Ensure non-negative
        u_new = jnp.maximum(u_new, 0.0)
        s_new = jnp.maximum(s_new, 0.0)

        return s_new, u_new

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply RNA velocity estimation.

        Args:
            data: Dictionary containing:
                - "spliced": Spliced mRNA counts (n_cells, n_genes)
                - "unspliced": Unspliced mRNA counts (n_cells, n_genes)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:
                    - "spliced": Original spliced counts
                    - "unspliced": Original unspliced counts
                    - "velocity": RNA velocity estimates
                    - "latent_time": Estimated latent time per cell
                    - "alpha": Transcription rate per gene
                    - "beta": Splicing rate per gene
                    - "gamma": Degradation rate per gene
                    - "projected_spliced": Projected future spliced
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        spliced = data["spliced"]
        unspliced = data["unspliced"]

        # Estimate latent time per cell
        latent_time = self.time_encoder(spliced, unspliced)

        # Get kinetics parameters
        alpha, beta, gamma = self.kinetics_encoder()

        # Compute velocity
        velocity = self._compute_velocity(spliced, unspliced, beta, gamma)

        # Project forward using learned dynamics
        s_proj, u_proj = spliced, unspliced
        for _ in range(self.n_steps):
            s_proj, u_proj = self._euler_step(s_proj, u_proj, alpha, beta, gamma, self.dt)

        transformed_data = {
            "spliced": spliced,
            "unspliced": unspliced,
            "velocity": velocity,
            "latent_time": latent_time,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "projected_spliced": s_proj,
        }

        return transformed_data, state, metadata
