"""Differentiable splicing PSI (Percent Spliced In) calculation.

This module implements a differentiable PSI calculation operator for
alternative splicing analysis with end-to-end gradient flow.

Inherits from TemperatureOperator to get:

- _temperature property for temperature-controlled smoothing
- soft_max() for logsumexp-based smooth maximum
- soft_argmax() for soft position selection
"""

from dataclasses import dataclass
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig

from diffbio.core.base_operators import TemperatureOperator


@dataclass
class SplicingPSIConfig(OperatorConfig):
    """Configuration for differentiable PSI calculation.

    Attributes:
        pseudocount: Pseudocount added for numerical stability and regularization.
        temperature: Temperature for confidence calculation.
        min_total_reads: Minimum total reads for reliable PSI estimation.
        stream_name: Name of the data stream to process.
    """

    pseudocount: float = 1.0
    temperature: float = 1.0
    learnable_temperature: bool = True
    min_total_reads: int = 10


class SplicingPSI(TemperatureOperator):
    """Differentiable PSI calculation for alternative splicing analysis.

    PSI (Percent Spliced In) quantifies alternative splicing by computing
    the fraction of transcripts that include a specific exon or splice site.

    The standard PSI formula is:
        PSI = inclusion_reads / (inclusion_reads + exclusion_reads)

    This operator adds:
    - Learnable pseudocount for regularization
    - Confidence estimation based on read coverage
    - Full differentiability for end-to-end training

    Example:
        ```python
        config = SplicingPSIConfig(
            pseudocount=1.0,
            min_total_reads=10,
        )
        psi_op = SplicingPSI(config, rngs=rngs)

        data = {
            "inclusion_counts": inclusion_reads,  # Junction reads supporting inclusion
            "exclusion_counts": exclusion_reads,  # Junction reads supporting exclusion
        }
        result, state, metadata = psi_op.apply(data, {}, None)
        psi_values = result["psi"]
        confidence = result["psi_confidence"]
        ```
    """

    def __init__(self, config: SplicingPSIConfig, *, rngs: nnx.Rngs | None = None):
        """Initialize the PSI operator.

        Args:
            config: Configuration for the operator.
            rngs: Random number generators for initialization.
        """
        super().__init__(config, rngs=rngs)
        self.config = config

        # Learnable pseudocount (must be positive)
        self.pseudocount = nnx.Param(jnp.array(config.pseudocount))

        # Temperature is managed by TemperatureOperator via self._temperature

    def _compute_psi(
        self, inclusion: jax.Array, exclusion: jax.Array, pseudocount: jax.Array | float
    ) -> jax.Array:
        """Compute PSI with pseudocount for numerical stability.

        Args:
            inclusion: Inclusion junction read counts.
            exclusion: Exclusion junction read counts.
            pseudocount: Pseudocount for regularization.

        Returns:
            PSI values in [0, 1].
        """
        # Add pseudocount to both numerator and denominator terms
        inc_adj = inclusion + pseudocount
        exc_adj = exclusion + pseudocount

        # PSI = inclusion / (inclusion + exclusion)
        psi = inc_adj / (inc_adj + exc_adj)

        return psi

    def _compute_confidence(
        self, inclusion: jax.Array, exclusion: jax.Array, temperature: jax.Array | float
    ) -> jax.Array:
        """Compute confidence in PSI estimate based on read coverage.

        Higher total reads = higher confidence in the PSI estimate.

        Args:
            inclusion: Inclusion junction read counts.
            exclusion: Exclusion junction read counts.
            temperature: Temperature for sigmoid scaling.

        Returns:
            Confidence values in [0, 1].
        """
        total_reads = inclusion + exclusion
        min_reads = self.config.min_total_reads

        # Sigmoid-based confidence: approaches 1 as reads increase
        # Centered around min_total_reads
        confidence = jax.nn.sigmoid((total_reads - min_reads) / (temperature * min_reads + 1e-6))

        return confidence

    def _compute_delta_psi_variance(
        self, inclusion: jax.Array, exclusion: jax.Array, psi: jax.Array
    ) -> jax.Array:
        """Compute variance of PSI estimate using beta-binomial model.

        This approximates the variance of PSI under a binomial model,
        which can be used for significance testing.

        Args:
            inclusion: Inclusion junction read counts.
            exclusion: Exclusion junction read counts.
            psi: Computed PSI values.

        Returns:
            Variance estimates for each PSI value.
        """
        total = inclusion + exclusion + 2 * jnp.abs(self.pseudocount[...])

        # Variance of beta distribution: psi * (1 - psi) / (n + 1)
        variance = (psi * (1 - psi)) / (total + 1)

        return variance

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict | None,
        random_params: dict | None = None,
        stats: dict | None = None,
    ) -> tuple[dict, dict, dict | None]:
        """Apply PSI calculation to junction read counts.

        Args:
            data: Dictionary containing:
                - 'inclusion_counts': Reads supporting exon inclusion
                - 'exclusion_counts': Reads supporting exon exclusion
            state: Operator state dictionary.
            metadata: Optional metadata dictionary.
            random_params: Optional random parameters (unused).
            stats: Optional statistics dictionary (unused).

        Returns:
            Tuple of (output_data, state, metadata) where output_data contains:

                - 'inclusion_counts': Original inclusion counts
                - 'exclusion_counts': Original exclusion counts
                - 'psi': Computed PSI values
                - 'psi_confidence': Confidence in PSI estimates
                - 'psi_variance': Variance of PSI estimates
        """
        del random_params, stats  # Unused

        inclusion = data["inclusion_counts"]
        exclusion = data["exclusion_counts"]

        # Get learnable parameters (ensure positive)
        pseudocount = jnp.abs(self.pseudocount[...]) + 1e-6
        temperature = jnp.abs(self._temperature) + 1e-6

        # Compute PSI
        psi = self._compute_psi(inclusion, exclusion, pseudocount)

        # Compute confidence
        confidence = self._compute_confidence(inclusion, exclusion, temperature)

        # Compute variance for significance testing
        variance = self._compute_delta_psi_variance(inclusion, exclusion, psi)

        output_data = {
            **data,
            "psi": psi,
            "psi_confidence": confidence,
            "psi_variance": variance,
        }

        return output_data, state, metadata
