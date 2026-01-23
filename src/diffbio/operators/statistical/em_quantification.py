"""Differentiable EM-based transcript quantification operator.

This module provides an unrolled EM algorithm for transcript
quantification, inspired by Salmon and Kallisto.

Key technique: Fixed number of EM iterations enables gradient flow
through all steps of the algorithm.

Applications: RNA-seq transcript quantification, isoform abundance estimation.

Inherits from TemperatureOperator to get:
- _temperature property for temperature-controlled smoothing
- soft_max() for logsumexp-based smooth maximum
- soft_argmax() for soft position selection
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
class EMQuantifierConfig(OperatorConfig):
    """Configuration for DifferentiableEMQuantifier.

    Attributes:
        n_transcripts: Number of transcripts to quantify.
        n_iterations: Fixed number of EM iterations (for unrolling).
        temperature: Temperature for softmax in E-step.
    """

    n_transcripts: int = 1000
    n_iterations: int = 10
    temperature: float = 1.0


class DifferentiableEMQuantifier(TemperatureOperator):
    """Differentiable EM for transcript quantification.

    This operator implements the EM algorithm for estimating transcript
    abundances from read-to-transcript compatibility data. The fixed
    number of iterations enables gradient flow through the entire
    quantification process.

    Algorithm:
    1. Initialize abundances (learnable prior)
    2. E-step: Probabilistic assignment of reads to transcripts
       weights = softmax(compatibility * abundances / temperature)
    3. M-step: Update abundances from weighted counts
       abundances = sum(weights) / effective_lengths
       abundances = abundances / sum(abundances)
    4. Repeat for n_iterations

    Inherits from TemperatureOperator to get:
    - _temperature property for temperature-controlled smoothing
    - soft_max() for logsumexp-based smooth maximum
    - soft_argmax() for soft position selection

    Args:
        config: EMQuantifierConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = EMQuantifierConfig(n_transcripts=1000, n_iterations=10)
        >>> quantifier = DifferentiableEMQuantifier(config, rngs=nnx.Rngs(42))
        >>> data = {"compatibility": compat_matrix, "effective_lengths": eff_lens}
        >>> result, state, meta = quantifier.apply(data, {}, None)
    """

    def __init__(
        self,
        config: EMQuantifierConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the EM quantifier operator.

        Args:
            config: EM quantifier configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.n_transcripts = config.n_transcripts
        self.n_iterations = config.n_iterations
        # Temperature is now managed by TemperatureOperator via self._temperature

        # Initialize log abundances (will be normalized via softmax)
        # Shape: (n_transcripts,)
        key = rngs.params()
        init_log_abundances = jax.random.normal(key, (config.n_transcripts,)) * 0.01
        self.log_initial_abundances = nnx.Param(init_log_abundances)

    def get_initial_abundances(self) -> Float[Array, "n_transcripts"]:
        """Get normalized initial abundances.

        Returns:
            Initial abundance distribution (n_transcripts,), sums to 1.
        """
        return jax.nn.softmax(self.log_initial_abundances[...])

    def em_step(
        self,
        abundances: Float[Array, "n_transcripts"],
        compatibility: Float[Array, "n_reads n_transcripts"],
        effective_lengths: Float[Array, "n_transcripts"],
    ) -> Float[Array, "n_transcripts"]:
        """Perform one EM iteration.

        Args:
            abundances: Current abundance estimates.
            compatibility: Read-transcript compatibility matrix.
            effective_lengths: Effective transcript lengths.

        Returns:
            Updated abundance estimates.
        """
        # E-step: Compute read assignment probabilities
        # P(transcript | read) propto compatibility * abundance / length
        rate = abundances / (effective_lengths + 1e-8)  # (n_transcripts,)

        # Score for each read-transcript pair
        scores = compatibility * rate  # (n_reads, n_transcripts)

        # Normalize per read (softmax with temperature)
        # Use inherited _temperature property from TemperatureOperator
        weights = jax.nn.softmax(
            jnp.log(scores + 1e-10) / self._temperature, axis=1
        )  # (n_reads, n_transcripts)

        # M-step: Update abundances
        # Expected count for each transcript
        expected_counts = jnp.sum(weights, axis=0)  # (n_transcripts,)

        # Normalize by effective length
        new_abundances = expected_counts / (effective_lengths + 1e-8)

        # Normalize to sum to 1
        new_abundances = new_abundances / (jnp.sum(new_abundances) + 1e-8)

        return new_abundances

    def quantify(
        self,
        compatibility: Float[Array, "n_reads n_transcripts"],
        effective_lengths: Float[Array, "n_transcripts"],
    ) -> Float[Array, "n_transcripts"]:
        """Run EM algorithm for quantification.

        Args:
            compatibility: Read-transcript compatibility matrix.
            effective_lengths: Effective transcript lengths.

        Returns:
            Final transcript abundance estimates.
        """
        # Initialize abundances
        abundances = self.get_initial_abundances()

        # Run fixed number of EM iterations
        def em_iteration(abundances, _):
            new_abundances = self.em_step(abundances, compatibility, effective_lengths)
            return new_abundances, None

        final_abundances, _ = jax.lax.scan(em_iteration, abundances, None, length=self.n_iterations)

        return final_abundances

    def compute_tpm(
        self,
        abundances: Float[Array, "n_transcripts"],
        effective_lengths: Float[Array, "n_transcripts"],
    ) -> Float[Array, "n_transcripts"]:
        """Convert abundances to TPM (Transcripts Per Million).

        Args:
            abundances: Normalized abundance estimates.
            effective_lengths: Effective transcript lengths.

        Returns:
            TPM values (sum to 1 million).
        """
        # TPM = (abundance / length) / sum(abundance / length) * 1e6
        rate = abundances / (effective_lengths + 1e-8)
        tpm = rate / (jnp.sum(rate) + 1e-8) * 1e6
        return tpm

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply EM quantification to read assignment data.

        This method runs the EM algorithm to estimate transcript
        abundances from read-transcript compatibility data.

        Args:
            data: Dictionary containing:
                - "compatibility": Read-transcript compatibility matrix
                  (n_reads, n_transcripts)
                - "effective_lengths": Effective transcript lengths
                  (n_transcripts,)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used (deterministic operator)
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:
                    - "compatibility": Original compatibility matrix
                    - "effective_lengths": Original effective lengths
                    - "abundances": Estimated transcript abundances
                    - "tpm": TPM (Transcripts Per Million) values
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        compatibility = data["compatibility"]
        effective_lengths = data["effective_lengths"]

        # Run EM quantification
        abundances = self.quantify(compatibility, effective_lengths)

        # Compute TPM
        tpm = self.compute_tpm(abundances, effective_lengths)

        # Build output data
        transformed_data = {
            "compatibility": compatibility,
            "effective_lengths": effective_lengths,
            "abundances": abundances,
            "tpm": tpm,
        }

        return transformed_data, state, metadata
