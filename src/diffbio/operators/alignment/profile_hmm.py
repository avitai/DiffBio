"""Profile HMM search operator for HMMER-style sequence alignment.

This module provides a differentiable implementation of profile HMM
search, enabling gradient-based learning of profile parameters.

Key technique: Use the forward algorithm with logsumexp for
differentiable profile-sequence alignment scoring.

Applications: Protein domain detection, remote homology search.
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
class ProfileHMMConfig(OperatorConfig):
    """Configuration for ProfileHMMSearch.

    Attributes:
        profile_length: Length of the profile (number of match states).
        alphabet_size: Size of sequence alphabet (20 for protein, 4 for DNA).
        temperature: Temperature for softmax operations.
        learnable_profile: Whether profile parameters are learnable.
        stochastic: Whether the operator uses randomness.
        stream_name: RNG stream name (not used).
    """

    profile_length: int = 100
    alphabet_size: int = 20  # Amino acids by default
    temperature: float = 1.0
    learnable_profile: bool = True
    stochastic: bool = False
    stream_name: str | None = None


class ProfileHMMSearch(OperatorModule):
    """Profile HMM search with differentiable scoring.

    This operator implements a simplified profile HMM with match, insert,
    and delete states. The forward algorithm computes the alignment score
    differentiably using logsumexp.

    Profile HMM structure (per position):
    - Match state: emits according to position-specific distribution
    - Insert state: emits according to background distribution
    - Delete state: silent (no emission)

    Transitions:
    - M->M, M->I, M->D (from match)
    - I->M, I->I (from insert)
    - D->M, D->D (from delete)

    Args:
        config: ProfileHMMConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = ProfileHMMConfig(profile_length=100, alphabet_size=20)
        >>> profiler = ProfileHMMSearch(config, rngs=nnx.Rngs(42))
        >>> data = {"sequence": one_hot_sequence}
        >>> result, state, meta = profiler.apply(data, {}, None)
    """

    def __init__(
        self,
        config: ProfileHMMConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the profile HMM operator.

        Args:
            config: Profile HMM configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.profile_length = config.profile_length
        self.alphabet_size = config.alphabet_size
        self.temperature = config.temperature

        # Initialize match emissions (profile_length, alphabet_size)
        key = rngs.params()
        init_match = jax.random.normal(
            key, (config.profile_length, config.alphabet_size)
        ) * 0.1
        self.log_match_emissions = nnx.Param(init_match)

        # Initialize insert emissions (profile_length, alphabet_size)
        # Insert states use near-uniform distribution
        key = rngs.params()
        init_insert = jax.random.normal(
            key, (config.profile_length, config.alphabet_size)
        ) * 0.01
        self.log_insert_emissions = nnx.Param(init_insert)

        # Initialize transition parameters
        # For each position: [M->M, M->I, M->D, I->M, I->I, D->M, D->D]
        key = rngs.params()
        # Default: prefer M->M transitions
        init_trans = jnp.zeros((config.profile_length, 7))
        init_trans = init_trans.at[:, 0].set(2.0)  # M->M bias
        init_trans = init_trans.at[:, 3].set(1.0)  # I->M bias
        init_trans = init_trans.at[:, 5].set(1.0)  # D->M bias
        init_trans = init_trans + jax.random.normal(key, init_trans.shape) * 0.1
        self.log_transitions = nnx.Param(init_trans)

    def get_match_emissions(self) -> Float[Array, "profile_length alphabet_size"]:
        """Get normalized match emission probabilities.

        Returns:
            Log match emission probabilities.
        """
        return jax.nn.log_softmax(
            self.log_match_emissions[...] / self.temperature, axis=1
        )

    def get_insert_emissions(self) -> Float[Array, "profile_length alphabet_size"]:
        """Get normalized insert emission probabilities.

        Returns:
            Log insert emission probabilities.
        """
        return jax.nn.log_softmax(
            self.log_insert_emissions[...] / self.temperature, axis=1
        )

    def get_transitions(self) -> dict[str, Float[Array, "profile_length"]]:
        """Get normalized transition probabilities.

        Returns:
            Dictionary with log transition probabilities for each type.
        """
        trans = self.log_transitions[...]

        # Normalize M->* transitions (positions 0, 1, 2)
        log_m_trans = jax.nn.log_softmax(trans[:, :3] / self.temperature, axis=1)

        # Normalize I->* transitions (positions 3, 4)
        log_i_trans = jax.nn.log_softmax(trans[:, 3:5] / self.temperature, axis=1)

        # Normalize D->* transitions (positions 5, 6)
        log_d_trans = jax.nn.log_softmax(trans[:, 5:7] / self.temperature, axis=1)

        return {
            "m_to_m": log_m_trans[:, 0],
            "m_to_i": log_m_trans[:, 1],
            "m_to_d": log_m_trans[:, 2],
            "i_to_m": log_i_trans[:, 0],
            "i_to_i": log_i_trans[:, 1],
            "d_to_m": log_d_trans[:, 0],
            "d_to_d": log_d_trans[:, 1],
        }

    def score_sequence(
        self,
        sequence: Float[Array, "seq_len alphabet_size"],
    ) -> Float[Array, ""]:
        """Score a sequence against the profile using forward algorithm.

        Computes log P(sequence | profile) using dynamic programming.

        Args:
            sequence: One-hot encoded sequence.

        Returns:
            Log probability (alignment score).
        """
        seq_len = sequence.shape[0]
        log_match = self.get_match_emissions()
        log_insert = self.get_insert_emissions()
        trans = self.get_transitions()

        # DP matrices: (seq_pos, profile_pos)
        # We use a simplified forward algorithm

        # Emission scores for each sequence position at each profile position
        # match_scores[i, j] = log P(seq[i] | Match[j])
        match_scores = jnp.einsum("sa,pa->sp", sequence, jnp.exp(log_match))
        match_scores = jnp.log(match_scores + 1e-10)

        # Insert scores per position (use position-specific insert emissions)
        insert_scores = jnp.einsum("sa,pa->sp", sequence, jnp.exp(log_insert))
        insert_scores = jnp.log(insert_scores + 1e-10)

        # Initialize DP
        # log_M[j] = log probability of being in Match state j
        # log_I[j] = log probability of being in Insert state j
        # log_D[j] = log probability of being in Delete state j

        neg_inf = -1e10

        # Initial state: can start at any match state with decreasing probability
        # or go through leading deletes
        log_M = jnp.full(self.profile_length, neg_inf)
        log_I = jnp.full(self.profile_length, neg_inf)
        log_D = jnp.full(self.profile_length, neg_inf)

        # Start: emit first sequence position at first match state
        log_M = log_M.at[0].set(match_scores[0, 0])
        log_I = log_I.at[0].set(insert_scores[0, 0])

        # Forward pass
        def forward_step(carry, seq_idx):
            log_M, log_I, log_D = carry
            obs_match = match_scores[seq_idx]
            obs_insert = insert_scores[seq_idx]

            # New M states: can come from M, I, or D at previous profile position
            # M[j] <- M[j-1] + M->M + emit[j]
            #      <- I[j-1] + I->M + emit[j]
            #      <- D[j-1] + D->M + emit[j]
            new_log_M = jnp.full(self.profile_length, neg_inf)

            # From previous M (M[j-1] -> M[j] for j=1..L-1)
            from_M = log_M[:-1] + trans["m_to_m"][:-1]
            # From previous I (I[j-1] -> M[j] for j=1..L-1)
            from_I = log_I[:-1] + trans["i_to_m"][:-1]
            # From previous D (D[j-1] -> M[j] for j=1..L-1)
            from_D = log_D[:-1] + trans["d_to_m"][:-1]

            # Combine and add emission
            combined = jax.scipy.special.logsumexp(
                jnp.stack([from_M, from_I, from_D]), axis=0
            )
            new_log_M = new_log_M.at[1:].set(combined + obs_match[1:])

            # Can also start fresh at position 0
            new_log_M = new_log_M.at[0].set(
                jax.scipy.special.logsumexp(
                    jnp.array([new_log_M[0], obs_match[0]])
                )
            )

            # New I states: can come from M or I at same profile position
            from_M_to_I = log_M + trans["m_to_i"]
            from_I_to_I = log_I + trans["i_to_i"]
            new_log_I = jax.scipy.special.logsumexp(
                jnp.stack([from_M_to_I, from_I_to_I]), axis=0
            ) + obs_insert

            # D states don't emit, handled separately
            # For simplicity, we skip explicit D state tracking in emissions
            # and just allow gaps through the transition structure
            new_log_D = log_D  # Simplified: D states updated via M transitions

            return (new_log_M, new_log_I, new_log_D), None

        # Scan over sequence positions (skip first, handled in init)
        (final_M, final_I, final_D), _ = jax.lax.scan(
            forward_step,
            (log_M, log_I, log_D),
            jnp.arange(1, seq_len)
        )

        # Final score: sum over all final states
        score = jax.scipy.special.logsumexp(
            jnp.concatenate([final_M, final_I])
        )

        return score

    def compute_posteriors(
        self,
        sequence: Float[Array, "seq_len alphabet_size"],
    ) -> Float[Array, "seq_len profile_length 3"]:
        """Compute state posteriors (simplified).

        Returns soft alignment between sequence and profile positions.

        Args:
            sequence: One-hot encoded sequence.

        Returns:
            Posterior probabilities for M/I/D states at each position.
        """
        seq_len = sequence.shape[0]
        log_match = self.get_match_emissions()

        # Simplified: just compute match scores as posteriors
        match_scores = jnp.einsum("sa,pa->sp", sequence, jnp.exp(log_match))

        # Normalize across profile positions for each sequence position
        posteriors = jax.nn.softmax(match_scores / self.temperature, axis=1)

        # Expand to include I/D placeholder dimensions
        # Shape: (seq_len, profile_length, 3) where dim 2 is [M, I, D]
        full_posteriors = jnp.zeros((seq_len, self.profile_length, 3))
        full_posteriors = full_posteriors.at[:, :, 0].set(posteriors)

        return full_posteriors

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply profile HMM search to sequence.

        Args:
            data: Dictionary containing:
                - "sequence": One-hot encoded sequence (seq_len, alphabet_size)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used (deterministic operator)
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:
                    - "sequence": Original sequence
                    - "score": Profile alignment score
                    - "state_posteriors": Soft state assignments
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        sequence = data["sequence"]

        # Compute alignment score
        score = self.score_sequence(sequence)

        # Compute state posteriors
        state_posteriors = self.compute_posteriors(sequence)

        # Build output data
        transformed_data = {
            "sequence": sequence,
            "score": score,
            "state_posteriors": state_posteriors,
        }

        return transformed_data, state, metadata
