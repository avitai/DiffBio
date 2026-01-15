"""Differentiable Hidden Markov Model operator.

This module provides a differentiable implementation of the HMM
forward algorithm using logsumexp for numerical stability.

Key technique: Use logsumexp instead of direct probability multiplication
to maintain numerical stability and enable gradient flow.

Applications: Gene finding, chromatin state annotation, profile search.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, Int, PyTree


@dataclass
class HMMConfig(OperatorConfig):
    """Configuration for DifferentiableHMM.

    Attributes:
        n_states: Number of hidden states.
        n_emissions: Number of possible emissions (e.g., 4 for DNA).
        temperature: Temperature for softmax operations.
        learnable_transitions: Whether transition probabilities are learnable.
        learnable_emissions: Whether emission probabilities are learnable.
    """

    n_states: int = 3
    n_emissions: int = 4
    temperature: float = 1.0
    learnable_transitions: bool = True
    learnable_emissions: bool = True


class DifferentiableHMM(OperatorModule):
    """Differentiable Hidden Markov Model.

    This operator implements the HMM forward algorithm with differentiable
    operations, enabling gradient-based learning of transition and emission
    parameters.

    The forward algorithm computes P(observations | model) using dynamic
    programming with logsumexp for numerical stability:

    alpha[t, j] = sum_i(alpha[t-1, i] * A[i,j]) * B[j, o_t]

    In log space:
    log_alpha[t, j] = logsumexp_i(log_alpha[t-1, i] + log_A[i,j]) + log_B[j, o_t]

    Args:
        config: HMMConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = HMMConfig(n_states=3, n_emissions=4)
        >>> hmm = DifferentiableHMM(config, rngs=nnx.Rngs(42))
        >>> data = {"observations": jnp.array([0, 1, 2, 3])}
        >>> result, state, meta = hmm.apply(data, {}, None)
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
        super().__init__(config, rngs=rngs, name=name)

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.n_states = config.n_states
        self.n_emissions = config.n_emissions
        self.temperature = config.temperature

        # Initialize transition parameters (will be normalized via softmax)
        # Shape: (n_states, n_states)
        key = rngs.params()
        init_trans = jax.random.normal(key, (config.n_states, config.n_states)) * 0.1
        self.log_transition_params = nnx.Param(init_trans)

        # Initialize emission parameters (will be normalized via softmax)
        # Shape: (n_states, n_emissions)
        key = rngs.params()
        init_emit = jax.random.normal(key, (config.n_states, config.n_emissions)) * 0.1
        self.log_emission_params = nnx.Param(init_emit)

        # Initialize initial state distribution
        # Shape: (n_states,)
        key = rngs.params()
        init_initial = jax.random.normal(key, (config.n_states,)) * 0.1
        self.log_initial_params = nnx.Param(init_initial)

    def get_log_transition_matrix(self) -> Float[Array, "n_states n_states"]:
        """Get normalized log transition matrix.

        Returns:
            Log transition probabilities (n_states, n_states).
        """
        # Apply softmax along rows to get valid probabilities
        log_trans = jax.nn.log_softmax(self.log_transition_params[...] / self.temperature, axis=1)
        return log_trans

    def get_log_emission_matrix(self) -> Float[Array, "n_states n_emissions"]:
        """Get normalized log emission matrix.

        Returns:
            Log emission probabilities (n_states, n_emissions).
        """
        # Apply softmax along rows to get valid probabilities
        log_emit = jax.nn.log_softmax(self.log_emission_params[...] / self.temperature, axis=1)
        return log_emit

    def get_log_initial_distribution(self) -> Float[Array, "n_states"]:
        """Get normalized log initial state distribution.

        Returns:
            Log initial probabilities (n_states,).
        """
        log_init = jax.nn.log_softmax(self.log_initial_params[...] / self.temperature)
        return log_init

    def forward(
        self,
        observations: Int[Array, "seq_len"],
    ) -> Float[Array, ""]:
        """Compute log probability of observations using forward algorithm.

        Args:
            observations: Integer-encoded observations (seq_len,).

        Returns:
            Log probability of the observation sequence.
        """
        log_trans = self.get_log_transition_matrix()
        log_emit = self.get_log_emission_matrix()
        log_init = self.get_log_initial_distribution()

        # Initialize: alpha[0, j] = pi[j] * B[j, o_0]
        log_alpha = log_init + log_emit[:, observations[0]]

        # Forward pass
        def forward_step(
            log_alpha: Float[Array, "n_states"],
            obs: int,
        ) -> tuple[Float[Array, "n_states"], None]:
            # log_alpha_new[j] = logsumexp_i(log_alpha[i] + log_A[i,j]) + log_B[j, obs]
            # Expand dimensions for broadcasting
            log_alpha_expanded = log_alpha[:, None]  # (n_states, 1)
            # log_alpha_expanded + log_trans has shape (n_states, n_states)
            # logsumexp over axis 0 gives (n_states,)
            log_alpha_new = jax.scipy.special.logsumexp(log_alpha_expanded + log_trans, axis=0)
            # Add emission probability
            log_alpha_new = log_alpha_new + log_emit[:, obs]
            return log_alpha_new, None

        # Scan over observations (skip first one, already handled in init)
        log_alpha, _ = jax.lax.scan(forward_step, log_alpha, observations[1:])

        # Total probability: sum over final states
        log_prob = jax.scipy.special.logsumexp(log_alpha)

        return log_prob

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
    ) -> Float[Array, "seq_len n_states"]:
        """Compute state posteriors using forward-backward algorithm.

        Args:
            observations: Integer-encoded observations.

        Returns:
            State posteriors P(state_t | observations) for each position.
        """
        log_trans = self.get_log_transition_matrix()
        log_emit = self.get_log_emission_matrix()
        log_init = self.get_log_initial_distribution()

        # Forward pass - store all alpha values
        def forward_step(
            log_alpha: Float[Array, "n_states"], obs: int
        ) -> tuple[Float[Array, "n_states"], Float[Array, "n_states"]]:
            log_alpha_expanded = log_alpha[:, None]
            log_alpha_new = jax.scipy.special.logsumexp(log_alpha_expanded + log_trans, axis=0)
            log_alpha_new = log_alpha_new + log_emit[:, obs]
            return log_alpha_new, log_alpha_new

        log_alpha_init = log_init + log_emit[:, observations[0]]
        _, log_alphas = jax.lax.scan(forward_step, log_alpha_init, observations[1:])
        # Prepend initial alpha
        log_alphas = jnp.concatenate([log_alpha_init[None, :], log_alphas], axis=0)

        # Backward pass
        def backward_step(
            log_beta: Float[Array, "n_states"], obs: int
        ) -> tuple[Float[Array, "n_states"], Float[Array, "n_states"]]:
            # log_beta_new[i] = logsumexp_j(log_A[i,j] + log_B[j, obs] + log_beta[j])
            log_beta_expanded = log_beta + log_emit[:, obs]  # (n_states,)
            log_beta_new = jax.scipy.special.logsumexp(log_trans + log_beta_expanded, axis=1)
            return log_beta_new, log_beta_new

        log_beta_init = jnp.zeros(self.n_states)  # log(1) = 0
        # Backward pass goes in reverse
        _, log_betas_rev = jax.lax.scan(backward_step, log_beta_init, observations[1:][::-1])
        # Reverse and append final beta
        log_betas = jnp.concatenate([log_betas_rev[::-1], log_beta_init[None, :]], axis=0)

        # Compute posteriors: P(state_t | obs) = alpha_t * beta_t / P(obs)
        log_posteriors = log_alphas + log_betas
        # Normalize at each position
        log_posteriors = log_posteriors - jax.scipy.special.logsumexp(
            log_posteriors, axis=1, keepdims=True
        )

        return jnp.exp(log_posteriors)

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
