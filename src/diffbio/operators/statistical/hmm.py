"""Differentiable Hidden Markov Model operator.

This module provides a differentiable implementation of the HMM
forward algorithm using logsumexp for numerical stability.

Key technique: Use logsumexp instead of direct probability multiplication
to maintain numerical stability and enable gradient flow.

Applications: Gene finding, chromatin state annotation, profile search.

Inherits from HMMOperator to get:

- forward_pass() for likelihood computation
- forward_backward_posteriors() for posterior computation
- get_log_transition_matrix(), get_log_emission_matrix(),
  get_log_initial_distribution() for parameter access
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from flax import nnx
from jaxtyping import Array, Float, Int, PyTree

from diffbio.core.base_operators import HMMOperator


@dataclass
class HMMConfig(OperatorConfig):
    """Configuration for DifferentiableHMM.

    Attributes:
        num_states: Number of hidden states.
        num_emissions: Number of possible emissions (e.g., 4 for DNA).
        temperature: Temperature for softmax operations.
        learnable_transitions: Whether transition probabilities are learnable.
        learnable_emissions: Whether emission probabilities are learnable.
    """

    num_states: int = 3
    num_emissions: int = 4
    temperature: float = 1.0
    learnable_transitions: bool = True
    learnable_emissions: bool = True


class DifferentiableHMM(HMMOperator):
    """Differentiable Hidden Markov Model.

    This operator implements the HMM forward algorithm with differentiable
    operations, enabling gradient-based learning of transition and emission
    parameters.

    The forward algorithm computes P(observations | model) using dynamic
    programming with logsumexp for numerical stability:

    alpha[t, j] = sum_i(alpha[t-1, i] * A[i,j]) * B[j, o_t]

    In log space:
    log_alpha[t, j] = logsumexp_i(log_alpha[t-1, i] + log_A[i,j]) + log_B[j, o_t]

    Inherits from HMMOperator to get:

    - forward_pass() for likelihood computation
    - forward_backward_posteriors() for posterior computation
    - get_log_transition_matrix(), get_log_emission_matrix(),
      get_log_initial_distribution() for parameter access

    Args:
        config: HMMConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        ```python
        config = HMMConfig(num_states=3, num_emissions=4)
        hmm = DifferentiableHMM(config, rngs=nnx.Rngs(42))
        data = {"observations": jnp.array([0, 1, 2, 3])}
        result, state, meta = hmm.apply(data, {}, None)
        ```
    """

    def __init__(
        self,
        config: HMMConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the HMM operator.

        Args:
            config: HMM configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        # HMMOperator handles parameter initialization
        super().__init__(config, rngs=rngs, name=name)

    # get_log_transition_matrix() is inherited from HMMOperator
    # get_log_emission_matrix() is inherited from HMMOperator
    # get_log_initial_distribution() is inherited from HMMOperator

    def forward(
        self,
        observations: Int[Array, "seq_len"],
    ) -> Float[Array, ""]:
        """Compute log probability of observations using forward algorithm.

        Delegates to inherited forward_pass() from HMMOperator.

        Args:
            observations: Integer-encoded observations (seq_len,).

        Returns:
            Log probability of the observation sequence.
        """
        return self.forward_pass(observations)

    def forward_soft(
        self,
        observations: Float[Array, "seq_len n_emissions"],
    ) -> Float[Array, ""]:
        """Compute log probability with soft (probabilistic) observations.

        This variant accepts soft observations (probability distributions
        over emissions) for fully differentiable operation.

        Args:
            observations: Soft observations (seq_len, n_emissions).

        Returns:
            Log probability of the observation sequence.
        """
        log_trans = self.get_log_transition_matrix()
        log_emit = self.get_log_emission_matrix()
        log_init = self.get_log_initial_distribution()

        # Soft emission: sum over emissions weighted by observation probs
        # log P(o_t | state) = logsumexp(log_emit + log(o_t))
        def soft_emission(obs: Float[Array, "n_emissions"]) -> Float[Array, "n_states"]:
            # obs is (n_emissions,), log_emit is (n_states, n_emissions)
            log_obs = jnp.log(obs + 1e-10)
            return jax.scipy.special.logsumexp(log_emit + log_obs, axis=1)

        # Initialize
        log_alpha = log_init + soft_emission(observations[0])

        # Forward pass
        def forward_step(
            log_alpha: Float[Array, "n_states"], obs: Float[Array, "n_emissions"]
        ) -> tuple[Float[Array, "n_states"], None]:
            log_alpha_expanded = log_alpha[:, None]
            log_alpha_new = jax.scipy.special.logsumexp(log_alpha_expanded + log_trans, axis=0)
            log_alpha_new = log_alpha_new + soft_emission(obs)
            return log_alpha_new, None

        log_alpha, _ = jax.lax.scan(forward_step, log_alpha, observations[1:])

        log_prob = jax.scipy.special.logsumexp(log_alpha)

        return log_prob

    def forward_backward(
        self,
        observations: Int[Array, "seq_len"],
    ) -> Float[Array, "seq_len num_states"]:
        """Compute state posteriors using forward-backward algorithm.

        Delegates to inherited forward_backward_posteriors() from HMMOperator.

        Args:
            observations: Integer-encoded observations.

        Returns:
            State posteriors P(state_t | observations) for each position.
        """
        return self.forward_backward_posteriors(observations)

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply HMM to observation sequence.

        This method computes the log-likelihood and state posteriors
        for a given observation sequence.

        Args:
            data: Dictionary containing:
                - "observations": Integer-encoded observations (seq_len,)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used (deterministic operator)
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - "observations": Original observations
                    - "log_likelihood": Log probability of sequence
                    - "state_posteriors": P(state | observations) at each position
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        observations = data["observations"]

        # Compute log likelihood
        log_likelihood = self.forward(observations)

        # Compute state posteriors
        state_posteriors = self.forward_backward(observations)

        # Build output data
        transformed_data = {
            "observations": observations,
            "log_likelihood": log_likelihood,
            "state_posteriors": state_posteriors,
        }

        return transformed_data, state, metadata
