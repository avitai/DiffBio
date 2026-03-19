"""Differentiable chromatin state annotation (ChromHMM-style).

This module implements a differentiable HMM-based chromatin state annotator
that can be used for learning chromatin states from histone modification data.

Optionally supports cell-type-conditioned emission probabilities, where each
chromatin state has per-cell-type Gaussian emission parameters. GMM-style soft
state assignment (gamma/responsibility) is computed via softmax over
log-likelihoods, inspired by SCALE.

Inherits from TemperatureOperator to get:

- _temperature property for temperature-controlled smoothing
- soft_max() for logsumexp-based smooth maximum
- soft_argmax() for soft Viterbi decoding

Note: This uses a Bernoulli emission model (for histone marks) rather than
categorical emissions, so it doesn't inherit from HMMOperator which assumes
categorical emissions. The conditioned mode uses Gaussian emissions.
"""

import math
from dataclasses import dataclass
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig

from diffbio.core.base_operators import TemperatureOperator


@dataclass
class ChromatinStateConfig(OperatorConfig):
    """Configuration for differentiable chromatin state annotator.

    Attributes:
        num_states: Number of chromatin states to learn.
        num_marks: Number of histone marks in input.
        temperature: Temperature for soft operations.
        use_cell_type_conditioning: Whether to condition emission
            probabilities on cell type. When enabled, each state has
            per-cell-type Gaussian emission parameters.
        num_cell_types: Number of cell types for conditioning.
        stream_name: Name of the data stream to process.
    """

    num_states: int = 15
    num_marks: int = 6
    temperature: float = 1.0
    use_cell_type_conditioning: bool = False
    num_cell_types: int = 1


class ChromatinStateAnnotator(TemperatureOperator):
    """Differentiable chromatin state annotator using HMM.

    This operator implements a differentiable Hidden Markov Model for
    annotating chromatin states from histone modification data. It uses
    the forward algorithm in log-space for numeric stability and provides
    soft Viterbi decoding for end-to-end differentiability.

    The HMM has:
    - Learnable transition probabilities between states
    - Learnable emission probabilities for each histone mark per state
    - Learnable initial state distribution

    When cell-type conditioning is enabled:
    - Each state has per-cell-type Gaussian emission parameters (mean, variance)
    - The cell type vector is used to blend emission parameters
    - GMM-style soft assignment (gamma) is computed via softmax over
      log-likelihoods, per SCALE

    Inherits from TemperatureOperator to get:

    - _temperature property for temperature-controlled smoothing
    - soft_max() for logsumexp-based smooth maximum
    - soft_argmax() for soft Viterbi decoding

    Example:
        ```python
        config = ChromatinStateConfig(
            num_states=15,
            num_marks=6,
            use_cell_type_conditioning=True,
            num_cell_types=5,
        )
        annotator = ChromatinStateAnnotator(config, rngs=rngs)

        data = {
            "histone_marks": marks,  # (length, num_marks)
            "cell_type": cell_type,  # (num_cell_types,) soft vector
        }
        result, state, metadata = annotator.apply(data, {}, None)
        state_probs = result["state_probabilities"]
        gamma = result["gamma"]  # soft state assignment
        ```
    """

    def __init__(self, config: ChromatinStateConfig, *, rngs: nnx.Rngs | None = None):
        """Initialize the chromatin state annotator.

        Args:
            config: Configuration for the annotator.
            rngs: Random number generators for initialization.
        """
        super().__init__(config, rngs=rngs)
        self.config = config

        if rngs is None:
            rngs = nnx.Rngs(0)

        num_states = config.num_states
        num_marks = config.num_marks

        # Initialize transition matrix (log-space)
        # Start with slight preference for self-transitions
        key = rngs.params() if hasattr(rngs, "params") else jax.random.key(0)
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        transition_init = jax.random.normal(k1, (num_states, num_states)) * 0.1
        transition_init = transition_init + jnp.eye(num_states) * 2.0  # Self-loop bias
        self.transition_logits = nnx.Param(transition_init)

        # Initialize emission parameters (Bernoulli model for default mode)
        emission_init = jax.random.normal(k2, (num_states, num_marks)) * 0.5
        self.emission_logits = nnx.Param(emission_init)

        # Initial state distribution
        initial_init = jax.random.normal(k3, (num_states,)) * 0.1
        self.initial_logits = nnx.Param(initial_init)
        # Temperature is now managed by TemperatureOperator via self._temperature

        # Cell-type conditioned emission parameters
        self._use_conditioning = config.use_cell_type_conditioning
        if self._use_conditioning:
            num_cell_types = config.num_cell_types

            # Per-type emission means: (num_cell_types, num_states, num_marks)
            means_init = jax.random.normal(k4, (num_cell_types, num_states, num_marks)) * 0.5
            self.emission_means = nnx.Param(means_init)

            # Per-type emission log-variances: (num_cell_types, num_states, num_marks)
            logvar_init = jax.random.normal(k5, (num_cell_types, num_states, num_marks)) * 0.1
            self.emission_logvars = nnx.Param(logvar_init)

    def _log_transition_matrix(self) -> jax.Array:
        """Get log transition probabilities (row-normalized)."""
        return jax.nn.log_softmax(self.transition_logits[...], axis=-1)

    def _log_initial_distribution(self) -> jax.Array:
        """Get log initial state probabilities."""
        return jax.nn.log_softmax(self.initial_logits[...])

    def _log_emission_probs(self, observations: jax.Array) -> jax.Array:
        """Compute log emission probabilities using Bernoulli model.

        Args:
            observations: Histone mark signals of shape (..., num_marks).

        Returns:
            Log emission probabilities of shape (..., num_states).
        """
        # Emission model: product of Bernoulli for each mark
        # P(obs | state) = prod_m P(mark_m | state)
        # For continuous input, use sigmoid to get probabilities

        # emission_logits: (num_states, num_marks)
        # observations: (..., num_marks)

        # Compute P(mark=1 | state) using sigmoid
        mark_probs = jax.nn.sigmoid(self.emission_logits[...])  # (num_states, num_marks)

        # Normalize observations to [0, 1] range using sigmoid
        obs_probs = jax.nn.sigmoid(observations)  # (..., num_marks)

        # Log probability of observing the marks given each state
        # Using soft Bernoulli: obs * log(p) + (1-obs) * log(1-p)
        log_mark_probs = jnp.log(mark_probs + 1e-8)  # (num_states, num_marks)
        log_not_mark_probs = jnp.log(1 - mark_probs + 1e-8)  # (num_states, num_marks)

        # Broadcast and compute log likelihood
        # obs_probs: (..., num_marks), mark_probs: (num_states, num_marks)
        log_emission = (
            obs_probs[..., None, :] * log_mark_probs
            + (1 - obs_probs[..., None, :]) * log_not_mark_probs
        )  # (..., num_states, num_marks)

        # Sum over marks
        return log_emission.sum(axis=-1)  # (..., num_states)

    def _log_emission_probs_conditioned(
        self,
        observations: jax.Array,
        cell_type: jax.Array,
    ) -> jax.Array:
        """Compute log emission probabilities conditioned on cell type.

        Uses Gaussian emission model where parameters are a weighted blend
        of per-cell-type parameters.

        Args:
            observations: Histone mark signals of shape (..., num_marks).
            cell_type: Cell type vector of shape (num_cell_types,), soft assignment.

        Returns:
            Log emission probabilities of shape (..., num_states).
        """
        # Blend per-type emission parameters using cell_type weights
        # emission_means: (num_cell_types, num_states, num_marks)
        # cell_type: (num_cell_types,)
        blended_means = jnp.einsum(
            "c,csm->sm", cell_type, self.emission_means[...]
        )  # (num_states, num_marks)

        blended_logvars = jnp.einsum(
            "c,csm->sm", cell_type, self.emission_logvars[...]
        )  # (num_states, num_marks)

        # Gaussian log-likelihood: -0.5 * (log(2*pi*var) + (x-mu)^2 / var)
        variance = jnp.exp(blended_logvars) + 1e-8  # (num_states, num_marks)

        # observations: (..., num_marks) -> (..., 1, num_marks) for broadcasting
        obs_expanded = observations[..., None, :]  # (..., 1, num_marks)

        log_emission = -0.5 * (
            jnp.log(2 * math.pi * variance) + (obs_expanded - blended_means) ** 2 / variance
        )  # (..., num_states, num_marks)

        # Sum over marks
        return log_emission.sum(axis=-1)  # (..., num_states)

    def _compute_gamma(self, log_emissions: jax.Array) -> jax.Array:
        """Compute GMM-style soft state assignment (responsibility).

        Gamma is the soft assignment probability of each position to each
        state, computed via softmax over log-likelihoods. This follows the
        SCALE approach where gamma = p(c|z) = p(c)*p(z|c) / p(z).

        Args:
            log_emissions: Log emission probabilities, shape (length, num_states).

        Returns:
            Gamma (responsibility) of shape (length, num_states).
        """
        # Use log initial distribution as prior p(c)
        log_prior = self._log_initial_distribution()  # (num_states,)

        # p(c,z) = p(c) * p(z|c) in log space
        log_joint = log_prior + log_emissions  # (length, num_states)

        # Softmax to normalize: gamma = p(c|z) = softmax(log_joint)
        gamma = jax.nn.softmax(log_joint, axis=-1)

        return gamma

    def _forward_algorithm(self, log_emissions: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Run forward algorithm in log-space.

        Args:
            log_emissions: Log emission probabilities of shape (length, num_states).

        Returns:
            Tuple of (alpha, log_likelihood) where alpha has shape (length, num_states).
        """
        log_trans = self._log_transition_matrix()
        log_init = self._log_initial_distribution()

        def forward_step(
            log_alpha_prev: jax.Array,
            log_emit_t: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            # log_alpha_prev: (num_states,)
            # log_emit_t: (num_states,)
            # log_alpha_t[j] = log_emit_t[j] + logsumexp_i(log_alpha_prev[i] + log_trans[i,j])

            # Expand for broadcasting: (num_states, 1) + (num_states, num_states)
            log_alpha_trans = log_alpha_prev[:, None] + log_trans
            log_alpha_t = log_emit_t + jax.scipy.special.logsumexp(log_alpha_trans, axis=0)
            return log_alpha_t, log_alpha_t

        # Initialize with initial distribution
        log_alpha_0 = log_init + log_emissions[0]

        # Run forward pass
        _, log_alphas = jax.lax.scan(forward_step, log_alpha_0, log_emissions[1:])

        # Prepend initial alpha
        log_alphas = jnp.concatenate([log_alpha_0[None, :], log_alphas], axis=0)

        # Compute log likelihood
        log_likelihood = jax.scipy.special.logsumexp(log_alphas[-1])

        return log_alphas, log_likelihood

    def _backward_algorithm(self, log_emissions: jax.Array) -> jax.Array:
        """Run backward algorithm in log-space.

        Args:
            log_emissions: Log emission probabilities of shape (length, num_states).

        Returns:
            Beta values of shape (length, num_states).
        """
        log_trans = self._log_transition_matrix()
        num_states = self.config.num_states

        def backward_step(
            log_beta_next: jax.Array,
            log_emit_next: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            # log_beta_next: (num_states,)
            # log_emit_next: (num_states,)
            # log_beta_t[i] = logsumexp_j(log_trans[i,j] + log_emit_next[j] + log_beta_next[j])

            log_beta_t = jax.scipy.special.logsumexp(
                log_trans + log_emit_next[None, :] + log_beta_next[None, :], axis=1
            )
            return log_beta_t, log_beta_t

        # Initialize with zeros (log(1) = 0)
        log_beta_T = jnp.zeros(num_states)

        # Run backward pass (reversed)
        _, log_betas_rev = jax.lax.scan(backward_step, log_beta_T, log_emissions[1:][::-1])

        # Reverse and prepend final beta
        log_betas = jnp.concatenate([log_betas_rev[::-1], log_beta_T[None, :]], axis=0)

        return log_betas

    def _compute_posteriors(self, log_alphas: jax.Array, log_betas: jax.Array) -> jax.Array:
        """Compute posterior state probabilities.

        Args:
            log_alphas: Forward probabilities of shape (length, num_states).
            log_betas: Backward probabilities of shape (length, num_states).

        Returns:
            Posterior probabilities of shape (length, num_states).
        """
        # P(state_t | observations) = alpha_t * beta_t / P(observations)
        log_posteriors = log_alphas + log_betas
        posteriors = jax.nn.softmax(log_posteriors, axis=-1)
        return posteriors

    def _soft_viterbi(self, log_emissions: jax.Array) -> jax.Array:
        """Compute soft Viterbi decoding using temperature-scaled max.

        Args:
            log_emissions: Log emission probabilities of shape (length, num_states).

        Returns:
            Most likely state at each position (as soft argmax).
        """
        log_trans = self._log_transition_matrix()
        log_init = self._log_initial_distribution()
        # Use inherited _temperature property from TemperatureOperator
        temperature = jnp.abs(self._temperature) + 1e-6

        def viterbi_step(
            log_delta_prev: jax.Array,
            log_emit_t: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            # Soft max over previous states
            log_delta_trans = log_delta_prev[:, None] + log_trans
            # Use logsumexp with temperature scaling for soft max
            log_delta_t = log_emit_t + temperature * jax.scipy.special.logsumexp(
                log_delta_trans / temperature, axis=0
            )
            return log_delta_t, log_delta_t

        # Initialize
        log_delta_0 = log_init + log_emissions[0]

        # Forward pass
        _, log_deltas = jax.lax.scan(viterbi_step, log_delta_0, log_emissions[1:])
        log_deltas = jnp.concatenate([log_delta_0[None, :], log_deltas], axis=0)

        # Soft argmax for most likely state
        state_probs = jax.nn.softmax(log_deltas / temperature, axis=-1)
        most_likely = jnp.sum(state_probs * jnp.arange(self.config.num_states)[None, :], axis=-1)

        return most_likely

    def _apply_single(
        self,
        marks: jax.Array,
        cell_type: jax.Array | None = None,
    ) -> dict:
        """Apply HMM to a single sequence.

        Args:
            marks: Histone mark signals of shape (length, num_marks).
            cell_type: Optional cell type vector of shape (num_cell_types,).

        Returns:
            Dictionary of outputs.
        """
        # Compute log emissions based on mode
        if self._use_conditioning and cell_type is not None:
            log_emissions = self._log_emission_probs_conditioned(
                marks, cell_type
            )  # (length, num_states)
        else:
            log_emissions = self._log_emission_probs(marks)  # (length, num_states)

        # Forward algorithm
        log_alphas, log_likelihood = self._forward_algorithm(log_emissions)

        # Backward algorithm
        log_betas = self._backward_algorithm(log_emissions)

        # Posterior probabilities
        posteriors = self._compute_posteriors(log_alphas, log_betas)

        # State probabilities (normalized alphas for online prediction)
        state_probs = jax.nn.softmax(log_alphas, axis=-1)

        # Soft Viterbi path
        viterbi_path = self._soft_viterbi(log_emissions)

        result = {
            "state_probabilities": state_probs,
            "state_posteriors": posteriors,
            "viterbi_path": viterbi_path,
            "log_likelihood": log_likelihood,
        }

        # Add gamma (GMM-style soft assignment) for conditioned mode
        if self._use_conditioning and cell_type is not None:
            gamma = self._compute_gamma(log_emissions)
            result["gamma"] = gamma

        return result

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict | None,
        random_params: dict | None = None,
        stats: dict | None = None,
    ) -> tuple[dict, dict, dict | None]:
        """Apply chromatin state annotation to histone mark data.

        Args:
            data: Dictionary containing:
                - 'histone_marks': Signals of shape (length, num_marks) or
                  (batch, length, num_marks)
                - 'cell_type': Optional cell type vector of shape
                  (num_cell_types,) when conditioning is enabled
            state: Operator state dictionary.
            metadata: Optional metadata dictionary.
            random_params: Optional random parameters (unused).
            stats: Optional statistics dictionary (unused).

        Returns:
            Tuple of (output_data, state, metadata) where output_data contains:

                - 'histone_marks': Original histone mark signals
                - 'state_probabilities': State probabilities at each position
                - 'state_posteriors': Posterior state probabilities
                - 'viterbi_path': Soft Viterbi decoding result
                - 'log_likelihood': Log likelihood of the sequence
                - 'gamma': Soft state assignment (only when conditioning enabled)
        """
        del random_params, stats  # Unused

        marks = data["histone_marks"]
        cell_type = data.get("cell_type") if self._use_conditioning else None

        # Handle single vs batched input
        single_input = marks.ndim == 2
        if single_input:
            result = self._apply_single(marks, cell_type)
        else:
            # Batched input - vmap over batch dimension
            # For conditioned mode, cell_type is shared across batch
            if cell_type is not None:
                result = jax.vmap(lambda m: self._apply_single(m, cell_type))(marks)
            else:
                result = jax.vmap(self._apply_single)(marks)

        output_data = {**data, **result}

        return output_data, state, metadata
