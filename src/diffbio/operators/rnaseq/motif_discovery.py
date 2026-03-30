"""Differentiable motif discovery (MEME-style).

This module implements a differentiable version of motif discovery with
PWM (Position Weight Matrix) learning for end-to-end gradient flow.

Inherits from TemperatureOperator to get:

- _temperature property for temperature-controlled smoothing
- soft_max() for logsumexp-based smooth maximum
- soft_argmax() for soft position selection
"""

import logging
from dataclasses import dataclass
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig

from diffbio.core import soft_ops
from diffbio.core.base_operators import TemperatureOperator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MotifDiscoveryConfig(OperatorConfig):
    """Configuration for differentiable motif discovery.

    Attributes:
        motif_width: Width of the motif (number of positions).
        num_motifs: Number of motifs to discover.
        alphabet_size: Size of the sequence alphabet (4 for DNA).
        temperature: Temperature for soft operations.
        background_prior: Prior probability for background model.
        stream_name: Name of the data stream to process.
    """

    motif_width: int = 12
    num_motifs: int = 1
    alphabet_size: int = 4
    temperature: float = 1.0
    learnable_temperature: bool = True
    background_prior: float = 0.25  # Uniform for DNA


class DifferentiableMotifDiscovery(TemperatureOperator):
    """Differentiable motif discovery with PWM learning.

    This operator implements a simplified differentiable version of MEME-style
    motif discovery. It learns Position Weight Matrices (PWMs) that represent
    sequence motifs and scans sequences to find motif occurrences.

    The motif score at position i is computed as:
        score(i) = sum_j PWM[j, seq[i+j]]

    For one-hot encoded sequences, this is equivalent to:
        score(i) = sum_j sum_k seq[i+j, k] * log(PWM[j, k])

    Example:
        ```python
        config = MotifDiscoveryConfig(
            motif_width=12,
            num_motifs=3,
        )
        motif_op = DifferentiableMotifDiscovery(config, rngs=rngs)

        data = {"sequence": one_hot_sequence}  # (length, alphabet_size)
        result, state, metadata = motif_op.apply(data, {}, None)
        motif_scores = result["motif_scores"]  # (num_positions, num_motifs)
        pwm = result["pwm"]  # (num_motifs, motif_width, alphabet_size)
        ```
    """

    def __init__(self, config: MotifDiscoveryConfig, *, rngs: nnx.Rngs | None = None):
        """Initialize the motif discovery operator.

        Args:
            config: Configuration for the operator.
            rngs: Random number generators for initialization.
        """
        super().__init__(config, rngs=rngs)
        self.config = config

        if rngs is None:
            rngs = nnx.Rngs(0)

        key = rngs.params() if hasattr(rngs, "params") else jax.random.key(0)

        # Initialize PWM logits (before softmax normalization)
        # Shape: (num_motifs, motif_width, alphabet_size)
        # Initialize near uniform with small random noise
        pwm_init = (
            jax.random.normal(
                key,
                (config.num_motifs, config.motif_width, config.alphabet_size),
            )
            * 0.1
        )

        self.pwm_logits = nnx.Param(pwm_init)

        # Temperature is managed by TemperatureOperator via self._temperature

    def _get_pwm(self) -> jax.Array:
        """Get normalized PWM from logits.

        Returns:
            PWM of shape (num_motifs, motif_width, alphabet_size) with
            probabilities summing to 1 over the alphabet dimension.
        """
        temperature = jnp.abs(self._temperature) + 1e-6
        return jax.nn.softmax(self.pwm_logits[...] / temperature, axis=-1)

    def _scan_single_motif(self, sequence: jax.Array, pwm: jax.Array) -> jax.Array:
        """Scan a sequence with a single PWM using convolution.

        Args:
            sequence: One-hot encoded sequence of shape (length, alphabet_size).
            pwm: PWM of shape (motif_width, alphabet_size).

        Returns:
            Motif scores at each valid position, shape (num_positions,).
        """
        motif_width = pwm.shape[0]
        seq_length = sequence.shape[0]
        num_positions = seq_length - motif_width + 1

        # Use log-odds scoring
        # log_pwm = log(PWM) - log(background)
        background = self.config.background_prior
        log_pwm = jnp.log(pwm + 1e-8) - jnp.log(background)

        # Compute score at each position using sliding window
        def score_at_position(start_idx):
            window = jax.lax.dynamic_slice(
                sequence, (start_idx, 0), (motif_width, self.config.alphabet_size)
            )
            # Score = sum of log-odds weighted by sequence
            score = jnp.sum(window * log_pwm)
            return score

        positions = jnp.arange(num_positions)
        scores = jax.vmap(score_at_position)(positions)

        return scores

    def _scan_sequence(self, sequence: jax.Array) -> jax.Array:
        """Scan a sequence with all motifs.

        Args:
            sequence: One-hot encoded sequence of shape (length, alphabet_size).

        Returns:
            Motif scores of shape (num_positions, num_motifs).
        """
        pwm = self._get_pwm()

        # Scan with each motif
        def scan_with_motif(pwm_single):
            return self._scan_single_motif(sequence, pwm_single)

        # Shape: (num_motifs, num_positions)
        all_scores = jax.vmap(scan_with_motif)(pwm)

        # Transpose to (num_positions, num_motifs)
        return all_scores.T

    def _find_motif_positions(self, scores: jax.Array, threshold: float = 0.0) -> jax.Array:
        """Find soft motif positions based on scores.

        Args:
            scores: Motif scores of shape (num_positions, num_motifs).
            threshold: Score threshold for calling a motif hit.

        Returns:
            Soft position indicators of shape (num_positions, num_motifs).
        """
        temperature = jnp.abs(self._temperature) + 1e-6
        return soft_ops.greater(scores, threshold, softness=temperature)

    def _apply_single(self, sequence: jax.Array) -> dict:
        """Apply motif discovery to a single sequence.

        Args:
            sequence: One-hot encoded sequence of shape (length, alphabet_size).

        Returns:
            Dictionary with motif scores, positions, and PWM.
        """
        # Get current PWM
        pwm = self._get_pwm()

        # Scan sequence
        motif_scores = self._scan_sequence(sequence)

        # Find soft motif positions
        motif_positions = self._find_motif_positions(motif_scores)

        return {
            "motif_scores": motif_scores,
            "motif_positions": motif_positions,
            "pwm": pwm,
        }

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict | None,
        random_params: dict | None = None,
        stats: dict | None = None,
    ) -> tuple[dict, dict, dict | None]:
        """Apply motif discovery to sequence data.

        Args:
            data: Dictionary containing:
                - 'sequence': One-hot encoded sequence(s) of shape
                  (length, alphabet_size) or (batch, length, alphabet_size)
            state: Operator state dictionary.
            metadata: Optional metadata dictionary.
            random_params: Optional random parameters (unused).
            stats: Optional statistics dictionary (unused).

        Returns:
            Tuple of (output_data, state, metadata) where output_data contains:

                - 'sequence': Original sequence data
                - 'motif_scores': Log-odds scores at each position
                - 'motif_positions': Soft motif occurrence indicators
                - 'pwm': Current Position Weight Matrix
        """
        del random_params, stats  # Unused

        sequence = data["sequence"]

        # Handle single vs batched input
        single_input = sequence.ndim == 2
        if single_input:
            result = self._apply_single(sequence)
        else:
            # Batched input - vmap over batch dimension
            result = jax.vmap(self._apply_single)(sequence)
            # PWM is shared, take from first (they're all the same)
            result["pwm"] = self._get_pwm()

        output_data = {**data, **result}

        return output_data, state, metadata
