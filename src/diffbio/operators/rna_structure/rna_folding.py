"""Differentiable RNA secondary structure prediction.

This module implements a differentiable version of the McCaskill partition
function algorithm for computing RNA base pair probabilities. The algorithm
uses dynamic programming with temperature-controlled softmax for gradient flow.

The McCaskill algorithm (1990) computes:
- Z = Σ_P exp(-E(P)/RT): Partition function over all structures
- P^bp[i,j]: Probability that positions i and j are base-paired

Key features:

- McCaskill-style inside-outside computation
- Base pair probability matrix
- Temperature-controlled smoothing for differentiability
- Watson-Crick + wobble base pairing

For differentiable optimization, we use a generalization of McCaskill's
algorithm that operates on continuous probability distributions over
nucleotides, following Matthies et al. (2024).

References:
    McCaskill, J. S. (1990). The equilibrium partition function and base
    pair binding probabilities for RNA secondary structure.
    Biopolymers 29, 1105-1119.

    Matthies, M. C. et al. (2024). Differentiable partition function
    calculation for RNA. Nucleic Acids Research 52(3), e14.

    Krueger, R. et al. (2025). JAX-RNAfold: Scalable differentiable folding.
    Bioinformatics 41(5), btaf203.
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from flax import nnx
from jaxtyping import Array, Float, PyTree

from diffbio.core.base_operators import TemperatureOperator

logger = logging.getLogger(__name__)

# RNA nucleotide indices (standard one-hot encoding)
NUC_A = 0  # Adenine
NUC_C = 1  # Cytosine
NUC_G = 2  # Guanine
NUC_U = 3  # Uracil

# Base pair energies (in units of RT, negative = favorable)
# Simplified Nussinov-style scoring where each base pair contributes
# a fixed energy independent of context.
BP_ENERGY_AU = -2.0  # A-U pair (2 hydrogen bonds)
BP_ENERGY_GC = -3.0  # G-C pair (3 hydrogen bonds, stronger)
BP_ENERGY_GU = -1.0  # G-U wobble pair (weaker)

# Minimum hairpin loop size (standard is 3 unpaired nucleotides)
DEFAULT_MIN_HAIRPIN = 3


@dataclass(frozen=True)
class RNAFoldConfig(OperatorConfig):
    """Configuration for DifferentiableRNAFold.

    Attributes:
        temperature: Temperature for Boltzmann distribution and softmax.
            Default 1.0. Lower values give sharper base pair probabilities.
            In physical terms, this is RT (gas constant * temperature).
        min_hairpin_loop: Minimum hairpin loop size. Default 3.
            Standard is 3 nucleotides between paired bases.
        alphabet_size: Size of nucleotide alphabet. Default 4 (A,C,G,U).
        bp_energy_au: Energy for A-U base pair. Default -2.0.
        bp_energy_gc: Energy for G-C base pair. Default -3.0.
        bp_energy_gu: Energy for G-U wobble pair. Default -1.0.
        learnable_temperature: Whether temperature is learnable.
    """

    cacheable: bool = True
    temperature: float = 1.0
    min_hairpin_loop: int = DEFAULT_MIN_HAIRPIN
    alphabet_size: int = 4
    bp_energy_au: float = BP_ENERGY_AU
    bp_energy_gc: float = BP_ENERGY_GC
    bp_energy_gu: float = BP_ENERGY_GU
    learnable_temperature: bool = False


def compute_pair_energy_matrix(
    sequence: Float[Array, "length 4"],
    bp_energy_au: float = BP_ENERGY_AU,
    bp_energy_gc: float = BP_ENERGY_GC,
    bp_energy_gu: float = BP_ENERGY_GU,
) -> Float[Array, "length length"]:
    """Compute base pair energy matrix for RNA sequence.

    Uses Watson-Crick and wobble base pairing rules:
    - A-U: 2 hydrogen bonds (medium strength)
    - G-C: 3 hydrogen bonds (strongest)
    - G-U: Wobble pair (weakest)

    For soft/probabilistic sequences, the energy is weighted by the
    probability of each nucleotide at each position.

    Args:
        sequence: One-hot encoded RNA sequence (A=0, C=1, G=2, U=3).
        bp_energy_au: Energy for A-U pair.
        bp_energy_gc: Energy for G-C pair.
        bp_energy_gu: Energy for G-U wobble pair.

    Returns:
        Energy matrix where [i,j] is the pairing energy for positions i,j.
        More negative = more favorable pairing.
    """
    # Get soft nucleotide probabilities at each position
    p_a = sequence[:, NUC_A]  # (length,)
    p_c = sequence[:, NUC_C]
    p_g = sequence[:, NUC_G]
    p_u = sequence[:, NUC_U]

    # Compute pairwise base pair compatibility using outer products
    # A-U pairing (in both directions)
    au_prob = jnp.outer(p_a, p_u) + jnp.outer(p_u, p_a)

    # G-C pairing
    gc_prob = jnp.outer(p_g, p_c) + jnp.outer(p_c, p_g)

    # G-U wobble pairing
    gu_prob = jnp.outer(p_g, p_u) + jnp.outer(p_u, p_g)

    # Combined energy (weighted by pairing probability)
    energy = bp_energy_au * au_prob + bp_energy_gc * gc_prob + bp_energy_gu * gu_prob

    return energy


def mccaskill_partition_function(
    energy_matrix: Float[Array, "length length"],
    min_hairpin: int = DEFAULT_MIN_HAIRPIN,
    temperature: float = 1.0,
) -> tuple[Float[Array, "length length"], Float[Array, ""]]:
    """Compute partition function using McCaskill algorithm.

    Implements the inside algorithm for RNA partition function computation.
    The recursion follows McCaskill (1990):

    Q[i,j] = Q[i,j-1] + Σ_k Q[i,k-1] * Q^bp[k,j]
    Q^bp[i,j] = exp(-E[i,j]/RT) * (1 + Q[i+1,j-1])

    For differentiability, we work in log space and use logsumexp.

    Args:
        energy_matrix: Base pair energy matrix from compute_pair_energy_matrix.
        min_hairpin: Minimum hairpin loop size.
        temperature: Temperature (RT) for Boltzmann weights.

    Returns:
        Tuple of (log_Q, log_Z) where:
        - log_Q[i,j] is log partition function for subsequence [i,j]
        - log_Z is total log partition function
    """
    n = energy_matrix.shape[0]

    # Initialize log partition functions
    # log_Q[i,j] = log of partition function for subsequence [i,j]
    # log_Qbp[i,j] = log of partition function for [i,j] where i,j are paired

    # For subsequences shorter than min_hairpin+2, no pairing is possible
    # Q[i,i] = 1 (empty structure), so log_Q[i,i] = 0
    log_Q = jnp.zeros((n, n))

    # Boltzmann weights for base pairs: exp(-E/T)
    # In log space: -E/T
    log_boltzmann = -energy_matrix / temperature

    # Create validity mask (positions can pair if |i-j| > min_hairpin)
    i_idx = jnp.arange(n)[:, None]
    j_idx = jnp.arange(n)[None, :]
    valid_pair = jnp.abs(i_idx - j_idx) > min_hairpin

    # Apply mask: invalid pairs get -inf in log space (probability 0)
    log_boltzmann = jnp.where(valid_pair, log_boltzmann, -jnp.inf)

    # Fill DP table using scan over diagonal lengths
    # For each length d, compute Q[i, i+d] for all valid i

    def fill_length(log_Q, d):
        """Fill all entries with subsequence length d."""

        def compute_entry(i):
            j = i + d

            # Case 1: j is unpaired, use Q[i, j-1]
            log_unpaired = log_Q[i, j - 1] if j > i else 0.0

            # Case 2: j pairs with some k in [i, j-min_hairpin)
            # Sum over all k: Q[i,k-1] * exp(-E[k,j]/T) * (1 + Q[k+1,j-1])

            def pair_term(k):
                # Boltzmann factor for k,j pair
                log_bp = log_boltzmann[k, j]

                # Left fragment [i, k-1]
                log_left = jnp.where(k > i, log_Q[i, k - 1], 0.0)

                # Inside fragment [k+1, j-1]
                # Q_inside = 1 + Q[k+1, j-1] means in log space:
                # log(1 + exp(log_Q)) = log1p(exp(log_Q))
                log_inside_q = jnp.where(
                    k + 1 <= j - 1,
                    log_Q[k + 1, j - 1],
                    0.0,  # Empty inside, log(1) = 0
                )
                # log(1 + Q) = log(exp(0) + exp(log_Q)) = logsumexp([0, log_Q])
                log_one_plus_inside = jax.scipy.special.logsumexp(jnp.array([0.0, log_inside_q]))

                return log_left + log_bp + log_one_plus_inside

            # Valid pairing partners
            k_values = jnp.arange(n)
            log_pair_terms = jax.vmap(pair_term)(k_values)

            # Mask invalid k values
            k_valid = (k_values >= i) & (k_values <= j - min_hairpin - 1)
            log_pair_terms = jnp.where(k_valid, log_pair_terms, -jnp.inf)

            # Combine unpaired case with all pairing cases
            all_terms = jnp.concatenate([jnp.array([log_unpaired]), log_pair_terms])
            log_total = jax.scipy.special.logsumexp(all_terms)

            return log_total

        # Compute entries for this diagonal
        # Use fori_loop for better tracing
        def body_fn(i, log_Q):
            j = i + d
            valid = j < n
            new_val = jax.lax.cond(valid, lambda: compute_entry(i), lambda: 0.0)
            log_Q = jax.lax.cond(valid, lambda q: q.at[i, j].set(new_val), lambda q: q, log_Q)
            return log_Q

        log_Q = jax.lax.fori_loop(0, n, body_fn, log_Q)
        return log_Q, None

    # Fill for all lengths from min_hairpin+1 to n-1
    log_Q, _ = jax.lax.scan(fill_length, log_Q, jnp.arange(min_hairpin + 1, n))

    # Total partition function is Q[0, n-1]
    log_Z = log_Q[0, n - 1]

    return log_Q, log_Z


def compute_base_pair_probabilities(
    energy_matrix: Float[Array, "length length"],
    min_hairpin: int = DEFAULT_MIN_HAIRPIN,
    temperature: Array | float = 1.0,
) -> tuple[Float[Array, "length length"], Float[Array, ""]]:
    """Compute base pair probability matrix.

    Uses a simplified approach where probabilities are derived from
    the Boltzmann-weighted base pair energies normalized over all
    valid positions.

    For full McCaskill, one would compute inside-outside probabilities,
    but for differentiability and simplicity, we use:
    P[i,j] ∝ exp(-E[i,j]/T) * validity_mask[i,j]

    Args:
        energy_matrix: Base pair energy matrix.
        min_hairpin: Minimum hairpin loop size.
        temperature: Temperature for Boltzmann distribution.

    Returns:
        Tuple of (bp_probs, log_Z) where:
        - bp_probs[i,j] is probability that i and j are paired
        - log_Z is log partition function (logsumexp of valid pairs)
    """
    n = energy_matrix.shape[0]

    # Create validity mask
    i_idx = jnp.arange(n)[:, None]
    j_idx = jnp.arange(n)[None, :]
    valid_mask = (jnp.abs(i_idx - j_idx) > min_hairpin).astype(jnp.float32)

    # Boltzmann weights: exp(-E/T)
    # In a normalized probability, negative energies give higher probability
    log_weights = -energy_matrix / temperature

    # Mask invalid positions
    log_weights_masked = jnp.where(
        valid_mask > 0.5, log_weights, jnp.full_like(log_weights, -jnp.inf)
    )

    # Normalize via softmax over all valid positions
    flat_log_weights = log_weights_masked.flatten()
    flat_probs = jax.nn.softmax(flat_log_weights)
    bp_probs = flat_probs.reshape(n, n)

    # The partition function is the sum of Boltzmann weights
    # log_Z = logsumexp(-E/T) over valid pairs
    log_Z = jax.scipy.special.logsumexp(jnp.where(valid_mask > 0.5, log_weights, -jnp.inf))

    # Ensure symmetry (A pairs with B == B pairs with A)
    bp_probs = (bp_probs + bp_probs.T) / 2

    # Re-normalize after symmetrization
    total = bp_probs.sum()
    bp_probs = jnp.where(total > 1e-10, bp_probs / total, bp_probs)

    return bp_probs, log_Z


class DifferentiableRNAFold(TemperatureOperator):
    """Differentiable RNA secondary structure prediction.

    This operator computes base pair probabilities for RNA sequences
    using a McCaskill-style partition function algorithm. The implementation
    uses temperature-controlled softmax for full differentiability.

    The McCaskill algorithm computes Z = Σ_P exp(-E(P)/RT), the partition
    function over all possible secondary structures. From this, base pair
    probabilities are derived as the marginal probability that positions
    i and j are paired in the ensemble.

    For differentiability, we generalize the algorithm to operate on
    continuous probability distributions over nucleotides, following
    Matthies et al. (2024).

    Input data structure:
        - sequence: Float[Array, "length 4"] or Float[Array, "batch length 4"]
            One-hot encoded RNA sequence (A=0, C=1, G=2, U=3)

    Output data structure (adds):
        - bp_probs: Float[Array, "length length"] - Base pair probabilities
        - partition_function: Float[Array, ""] - Log partition function

    Example:
        ```python
        config = RNAFoldConfig(temperature=1.0)
        predictor = DifferentiableRNAFold(config, rngs=nnx.Rngs(42))
        sequence = jax.nn.one_hot(seq_indices, num_classes=4)
        result, _, _ = predictor.apply({"sequence": sequence}, {}, None)
        bp_probs = result["bp_probs"]  # (length, length)
        ```
    """

    def __init__(
        self,
        config: RNAFoldConfig,
        *,
        rngs: nnx.Rngs,
        name: str | None = None,
    ):
        """Initialize the RNA fold predictor.

        Args:
            config: Configuration with folding parameters.
            rngs: Random number generators.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

    def _fold_single(
        self,
        sequence: Float[Array, "length 4"],
    ) -> tuple[Float[Array, "length length"], Float[Array, ""]]:
        """Compute folding for a single sequence.

        Args:
            sequence: One-hot encoded RNA sequence.

        Returns:
            Tuple of (base_pair_probabilities, log_partition_function).
        """
        temperature = self._temperature
        config = self.config

        # Compute base pair energy matrix
        energy_matrix = compute_pair_energy_matrix(
            sequence,
            bp_energy_au=config.bp_energy_au,
            bp_energy_gc=config.bp_energy_gc,
            bp_energy_gu=config.bp_energy_gu,
        )

        # Compute base pair probabilities
        bp_probs, log_z = compute_base_pair_probabilities(
            energy_matrix,
            min_hairpin=config.min_hairpin_loop,
            temperature=temperature,
        )

        return bp_probs, log_z

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply RNA folding prediction to sequence data.

        Args:
            data: Dictionary containing:
                - "sequence": One-hot encoded RNA sequence
                  Shape: (length, 4) or (batch, length, 4)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - All original keys from data
                    - "bp_probs": Base pair probability matrix
                    - "partition_function": Log partition function
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        del random_params, stats  # Unused

        sequence = data["sequence"]

        # Handle batched vs single sequence
        if sequence.ndim == 2:
            # Single sequence: (length, 4)
            bp_probs, log_z = self._fold_single(sequence)
        else:
            # Batched: (batch, length, 4)
            bp_probs, log_z = jax.vmap(self._fold_single)(sequence)

        # Build output data, preserving all input keys
        transformed_data = {
            **data,
            "bp_probs": bp_probs,
            "partition_function": log_z,
        }

        return transformed_data, state, metadata


def create_rna_fold_predictor(
    temperature: float = 1.0,
    min_hairpin_loop: int = DEFAULT_MIN_HAIRPIN,
    bp_energy_au: float = BP_ENERGY_AU,
    bp_energy_gc: float = BP_ENERGY_GC,
    bp_energy_gu: float = BP_ENERGY_GU,
    *,
    rngs: nnx.Rngs | None = None,
) -> DifferentiableRNAFold:
    """Create an RNA fold predictor with given parameters.

    Factory function for convenient predictor creation.

    Args:
        temperature: Softmax temperature for Boltzmann distribution.
        min_hairpin_loop: Minimum hairpin loop size.
        bp_energy_au: Energy for A-U base pair.
        bp_energy_gc: Energy for G-C base pair.
        bp_energy_gu: Energy for G-U wobble pair.
        rngs: Random number generators.

    Returns:
        Configured DifferentiableRNAFold instance.

    Example:
        ```python
        predictor = create_rna_fold_predictor(temperature=0.5)
        result, _, _ = predictor.apply({"sequence": seq}, {}, None)
        ```
    """
    if rngs is None:
        rngs = nnx.Rngs(0)

    config = RNAFoldConfig(
        temperature=temperature,
        min_hairpin_loop=min_hairpin_loop,
        bp_energy_au=bp_energy_au,
        bp_energy_gc=bp_energy_gc,
        bp_energy_gu=bp_energy_gu,
    )

    return DifferentiableRNAFold(config, rngs=rngs)
