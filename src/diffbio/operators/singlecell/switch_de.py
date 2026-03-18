"""Sigmoidal switch differential expression operator.

This module provides a differentiable model of gene expression as a
sigmoidal function of pseudotime. Each gene has a learnable switch time,
amplitude, and baseline, enabling gradient-based identification of
dynamically regulated genes along a trajectory.

Key technique: Models expression as ``a * sigmoid((t - t_switch) / T) + b``
where the sigmoid temperature controls smoothness of the transition.

Applications: Identifying switch-like gene regulation events in
single-cell pseudotime trajectories.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from flax import nnx
from jaxtyping import Array, Float, PyTree

from diffbio.core.base_operators import TemperatureOperator


@dataclass
class SwitchDEConfig(OperatorConfig):
    """Configuration for sigmoidal switch differential expression.

    Attributes:
        n_genes: Number of genes to model.
        temperature: Temperature controlling sigmoid smoothness.
            Lower values produce sharper switch transitions.
        learnable_temperature: Whether temperature is a learnable parameter.
    """

    n_genes: int = 2000
    temperature: float = 1.0
    learnable_temperature: bool = False


class DifferentiableSwitchDE(TemperatureOperator):
    """Differentiable sigmoidal switch model for differential expression.

    Models gene expression as a sigmoidal function of pseudotime:
    ``g(t) = amplitude * sigmoid((t - t_switch) / temperature) + baseline``

    Each gene has learnable parameters for switch time, amplitude, and
    baseline expression level. The switch score quantifies how strongly
    a gene switches, computed as the maximum sigmoid derivative scaled
    by amplitude.

    Inherits from TemperatureOperator to get:

    - _temperature property for temperature-controlled smoothing
    - soft_max() for logsumexp-based smooth maximum
    - soft_argmax() for soft position selection

    Args:
        config: SwitchDEConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        ```python
        config = SwitchDEConfig(n_genes=2000, temperature=1.0)
        op = DifferentiableSwitchDE(config, rngs=nnx.Rngs(42))
        data = {"counts": counts, "pseudotime": pseudotime}
        result, state, meta = op.apply(data, {}, None)
        ```
    """

    def __init__(
        self,
        config: SwitchDEConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the sigmoidal switch DE operator.

        Args:
            config: Switch DE configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        n_genes = config.n_genes

        # Switch time per gene (init to 0.5 = midpoint of pseudotime)
        self.t_switch = nnx.Param(jnp.full((n_genes,), 0.5))

        # Sigmoid amplitude per gene (init to 1.0)
        self.amplitude = nnx.Param(jnp.ones((n_genes,)))

        # Baseline expression per gene (init to 0.0)
        self.baseline = nnx.Param(jnp.zeros((n_genes,)))

    def _compute_predicted_expression(
        self,
        pseudotime: Float[Array, "n_cells"],
    ) -> Float[Array, "n_cells n_genes"]:
        """Compute predicted expression from the sigmoidal model.

        Args:
            pseudotime: Pseudotime values per cell.

        Returns:
            Predicted expression matrix (n_cells, n_genes).
        """
        temp = self._temperature
        t_switch = self.t_switch[...]
        amplitude = self.amplitude[...]
        baseline = self.baseline[...]

        # Sigmoid argument: (t - t_switch) / temperature
        # pseudotime: (n_cells,) -> (n_cells, 1), t_switch: (n_genes,) -> (1, n_genes)
        sigmoid_arg = (pseudotime[:, None] - t_switch[None, :]) / temp
        predicted = amplitude[None, :] * jax.nn.sigmoid(sigmoid_arg) + baseline[None, :]

        return predicted

    def _compute_switch_scores(self) -> Float[Array, "n_genes"]:
        """Compute switch score per gene.

        The switch score is the maximum sigmoid derivative scaled by
        amplitude: ``amplitude * (1 / (4 * temperature))``.

        Returns:
            Switch scores per gene.
        """
        temp = self._temperature
        amplitude = self.amplitude[...]
        return amplitude * (1.0 / (4.0 * temp))

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply sigmoidal switch DE model to single-cell data.

        Args:
            data: Dictionary containing:
                - "counts": Gene expression counts (n_cells, n_genes)
                - "pseudotime": Pseudotime values per cell (n_cells,)
            state: Element state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: Not used.
            stats: Not used.

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - "counts": Original expression counts
                    - "pseudotime": Original pseudotime
                    - "switch_times": Learned switch time per gene
                    - "switch_scores": Switch score per gene
                    - "predicted_expression": Predicted expression from model
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        pseudotime = data["pseudotime"]

        predicted = self._compute_predicted_expression(pseudotime)
        scores = self._compute_switch_scores()

        transformed_data = {
            **data,
            "switch_times": self.t_switch[...],
            "switch_scores": scores,
            "predicted_expression": predicted,
        }

        return transformed_data, state, metadata
