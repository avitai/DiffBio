"""Differentiable secondary structure prediction (PyDSSP-style).

This module implements a JAX/Flax NNX version of the DSSP algorithm for
assigning secondary structure to protein backbone atoms. The key innovation
is a continuous hydrogen bond matrix that enables gradient-based optimization.

The algorithm computes hydrogen bond energies using the Kabsch-Sander formula,
then applies a smooth transformation to create a differentiable H-bond matrix.
Secondary structure is assigned based on characteristic H-bond patterns.

Reference:
    Kabsch & Sander (1983). Dictionary of protein secondary structure:
    pattern recognition of hydrogen-bonded and geometrical features.
    Biopolymers 22, 2577-2637.

    Minami (2023). PyDSSP: A simplified implementation of DSSP algorithm
    for PyTorch and NumPy. https://github.com/ShintaroMinami/PyDSSP
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float

logger = logging.getLogger(__name__)

# DSSP constants from Kabsch & Sander (1983)
CONST_Q1Q2 = 0.084  # Partial charges (e units)
CONST_F = 332.0  # Conversion factor to kcal/mol
DEFAULT_CUTOFF = -0.5  # H-bond energy threshold (kcal/mol)
DEFAULT_MARGIN = 1.0  # Smoothing margin for continuous H-bond map

# Atom indices in the coordinate array
ATOM_N = 0
ATOM_CA = 1
ATOM_C = 2
ATOM_O = 3

# Secondary structure class indices
SS_LOOP = 0  # '-' (coil/loop)
SS_HELIX = 1  # 'H' (alpha-helix)
SS_STRAND = 2  # 'E' (beta-strand)


@dataclass(frozen=True)
class SecondaryStructureConfig(OperatorConfig):
    """Configuration for DifferentiableSecondaryStructure.

    Attributes:
        margin: Smoothing margin for continuous H-bond matrix. Default 1.0.
            Controls the sharpness of the sigmoid-like transformation.
            Smaller values = sharper transitions.
        cutoff: Hydrogen bond energy threshold in kcal/mol. Default -0.5.
            Bonds with energy below this are considered hydrogen bonds.
        min_helix_length: Minimum consecutive residues for helix assignment.
            Default 4 (standard for alpha-helix i→i+4 pattern).
        temperature: Temperature for soft secondary structure assignment.
            Default 1.0. Lower = sharper assignments.
    """

    margin: float = DEFAULT_MARGIN
    cutoff: float = DEFAULT_CUTOFF
    min_helix_length: int = 4
    temperature: float = 1.0


def compute_hydrogen_position(
    n_pos: Float[Array, "... 3"],
    ca_pos: Float[Array, "... 3"],
    c_prev_pos: Float[Array, "... 3"],
) -> Float[Array, "... 3"]:
    """Compute hydrogen atom position from backbone atoms.

    The amide hydrogen is placed along the N-H bond direction, which is
    approximately opposite to the bisector of CA-N and C_prev-N vectors.

    Args:
        n_pos: Nitrogen atom positions.
        ca_pos: Alpha carbon positions.
        c_prev_pos: Carbonyl carbon from previous residue.

    Returns:
        Estimated hydrogen atom positions.
    """
    # Vectors from N to neighboring atoms
    vec_n_ca = ca_pos - n_pos
    vec_n_c = c_prev_pos - n_pos

    # Normalize
    vec_n_ca = vec_n_ca / (jnp.linalg.norm(vec_n_ca, axis=-1, keepdims=True) + 1e-8)
    vec_n_c = vec_n_c / (jnp.linalg.norm(vec_n_c, axis=-1, keepdims=True) + 1e-8)

    # H is opposite to the average direction (bisector)
    h_direction = -(vec_n_ca + vec_n_c)
    h_direction = h_direction / (jnp.linalg.norm(h_direction, axis=-1, keepdims=True) + 1e-8)

    # Standard N-H bond length is ~1.0 Angstrom
    h_pos = n_pos + h_direction * 1.0

    return h_pos


class DifferentiableSecondaryStructure(OperatorModule):
    """Differentiable secondary structure prediction using DSSP algorithm.

    This operator computes secondary structure assignments for protein
    backbone atoms using a differentiable version of the DSSP algorithm.
    The key innovation is a continuous hydrogen bond matrix that enables
    gradient flow through the secondary structure prediction.

    The algorithm:
    1. Compute hydrogen bond energies using Kabsch-Sander electrostatic formula
    2. Apply smooth transformation to create continuous H-bond matrix in [0,1]
    3. Detect helix patterns (i→i+4 H-bonds) and strand patterns
    4. Output soft secondary structure assignments

    Input data structure:
        - coordinates: Float[Array, "batch length 4 3"] - Backbone atoms (N, CA, C, O)

    Output data structure (adds):
        - ss_onehot: Float[Array, "batch length 3"] - Soft SS probabilities
        - hbond_map: Float[Array, "batch length length"] - Continuous H-bond matrix
        - ss_indices: Int[Array, "batch length"] - Hard SS assignments (0=loop, 1=helix, 2=strand)

    Example:
        ```python
        config = SecondaryStructureConfig(margin=1.0, cutoff=-0.5)
        predictor = DifferentiableSecondaryStructure(config, rngs=nnx.Rngs(42))
        coords = jax.random.uniform(key, (1, 50, 4, 3)) * 10  # 50 residues
        result, _, _ = predictor.apply({"coordinates": coords}, {}, None)
        ss_probs = result["ss_onehot"]  # (1, 50, 3)
        ```
    """

    def __init__(
        self,
        config: SecondaryStructureConfig,
        *,
        rngs: nnx.Rngs,
        name: str | None = None,
    ):
        """Initialize the secondary structure predictor.

        Args:
            config: Configuration with DSSP parameters.
            rngs: Random number generators.
            name: Optional name for the operator.
        """
        super().__init__(config, rngs=rngs, name=name)
        self.config: SecondaryStructureConfig = config

    def compute_hbond_energy(
        self,
        coords: Float[Array, "batch length 4 3"],
    ) -> Float[Array, "batch length length"]:
        """Compute hydrogen bond energy matrix.

        Uses the Kabsch-Sander electrostatic energy formula:
            E = q1*q2 * f * (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN)

        where r_XY is the distance between atoms X and Y.

        Donor: N-H from residue i
        Acceptor: C=O from residue j

        Args:
            coords: Backbone coordinates (batch, length, 4, 3).
                    Atom order: N, CA, C, O

        Returns:
            Energy matrix (batch, length, length) in kcal/mol.
            E[b, i, j] = energy of H-bond from donor i to acceptor j.
        """
        batch, length, _, _ = coords.shape

        # Extract atom positions
        n_pos = coords[:, :, ATOM_N, :]  # (batch, length, 3)
        ca_pos = coords[:, :, ATOM_CA, :]
        c_pos = coords[:, :, ATOM_C, :]
        o_pos = coords[:, :, ATOM_O, :]

        # Compute hydrogen positions (shifted by 1 for C from previous residue)
        # For first residue, use self C as approximation
        c_prev = jnp.concatenate([c_pos[:, :1, :], c_pos[:, :-1, :]], axis=1)
        h_pos = compute_hydrogen_position(n_pos, ca_pos, c_prev)

        # Expand for pairwise computation
        # Donors: N, H from residue i
        # Acceptors: C, O from residue j
        n_i = n_pos[:, :, None, :]  # (batch, length, 1, 3)
        h_i = h_pos[:, :, None, :]
        c_j = c_pos[:, None, :, :]  # (batch, 1, length, 3)
        o_j = o_pos[:, None, :, :]

        # Compute distances with numerical stability
        def safe_distance(pos1, pos2, min_dist=0.1):
            diff = pos1 - pos2
            dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-10)
            return jnp.maximum(dist, min_dist)

        d_on = safe_distance(o_j, n_i)  # O(acceptor) to N(donor)
        d_ch = safe_distance(c_j, h_i)  # C(acceptor) to H(donor)
        d_oh = safe_distance(o_j, h_i)  # O(acceptor) to H(donor)
        d_cn = safe_distance(c_j, n_i)  # C(acceptor) to N(donor)

        # Kabsch-Sander energy formula
        energy = CONST_Q1Q2 * CONST_F * (1.0 / d_on + 1.0 / d_ch - 1.0 / d_oh - 1.0 / d_cn)

        return energy

    def compute_hbond_map(
        self,
        coords: Float[Array, "batch length 4 3"],
    ) -> Float[Array, "batch length length"]:
        """Compute continuous hydrogen bond matrix.

        Transforms the energy matrix into a continuous [0,1] matrix using
        a smooth sigmoid-like function based on sine:
            HbondMat(i,j) = (1 + sin((cutoff - E - margin) / margin * pi/2)) / 2

        This allows gradients to flow through the H-bond detection.

        Args:
            coords: Backbone coordinates (batch, length, 4, 3).

        Returns:
            Continuous H-bond matrix (batch, length, length) in [0, 1].
        """
        energy = self.compute_hbond_energy(coords)

        margin = self.config.margin
        cutoff = self.config.cutoff

        # Smooth transformation: maps energy to [0, 1]
        # More negative energy (stronger H-bond) -> higher value
        x = (cutoff - energy - margin) / margin

        # Clamp to [-1, 1] for sin input to stay in valid range
        x = jnp.clip(x, -1.0, 1.0)

        # Sine-based sigmoid: smooth transition in [0, 1]
        hbond_map = (1.0 + jnp.sin(x * jnp.pi / 2)) / 2.0

        return hbond_map

    def detect_helix_pattern(
        self,
        hbond_map: Float[Array, "batch length length"],
    ) -> Float[Array, "batch length"]:
        """Detect alpha-helix pattern (i→i+4 hydrogen bonds).

        Alpha-helices are characterized by H-bonds from residue i (donor)
        to residue i-4 (acceptor), creating i→i+4 backbone H-bonds.

        Args:
            hbond_map: Continuous H-bond matrix.

        Returns:
            Soft helix assignment for each residue.
        """
        batch, length, _ = hbond_map.shape

        # Extract i→i+4 diagonal (donor i to acceptor i-4)
        # This means hbond_map[i, i-4] should be high
        helix_score = jnp.zeros((batch, length))

        # For residues 4 and beyond, check if they donate to i-4
        # and residue i+4 donates to them
        for offset in [3, 4]:  # Check both i→i+3 (3-10 helix) and i→i+4 (alpha helix)
            # Create shifted versions to extract diagonal
            if offset < length:
                # Score for being in a helix: both i→i-offset and i+offset→i should be H-bonded
                # Simplified: just check i→i-offset pattern
                indices = jnp.arange(length)
                donor_indices = indices
                acceptor_indices = indices - offset

                # Mask for valid indices
                valid_mask = acceptor_indices >= 0

                # Get H-bond scores for valid pairs
                scores = jnp.where(
                    valid_mask,
                    hbond_map[:, donor_indices, jnp.maximum(acceptor_indices, 0)],
                    0.0,
                )

                helix_score = helix_score + scores

        # Normalize and apply temperature
        helix_score = helix_score / 2.0  # Average of two patterns
        helix_score = jnp.clip(helix_score, 0.0, 1.0)

        return helix_score

    def detect_strand_pattern(
        self,
        hbond_map: Float[Array, "batch length length"],
    ) -> Float[Array, "batch length"]:
        """Detect beta-strand pattern (parallel/antiparallel H-bonds).

        Beta-strands are characterized by H-bonds between distant residues
        forming ladder-like patterns (parallel or antiparallel).

        Args:
            hbond_map: Continuous H-bond matrix.

        Returns:
            Soft strand assignment for each residue.
        """
        batch, length, _ = hbond_map.shape

        # For strands, we look for H-bonds to non-local residues (|i-j| > 4)
        # Create distance mask
        i_idx = jnp.arange(length)[:, None]
        j_idx = jnp.arange(length)[None, :]
        distance_mask = jnp.abs(i_idx - j_idx) > 4

        # Mask H-bond map to only consider non-local bonds
        masked_hbond = hbond_map * distance_mask[None, :, :]

        # Score for each residue: max of incoming and outgoing non-local H-bonds
        incoming = jnp.max(masked_hbond, axis=1)  # Max over donors
        outgoing = jnp.max(masked_hbond, axis=2)  # Max over acceptors

        strand_score = jnp.maximum(incoming, outgoing)

        return strand_score

    def assign_secondary_structure(
        self,
        hbond_map: Float[Array, "batch length length"],
    ) -> Float[Array, "batch length 3"]:
        """Assign secondary structure based on H-bond patterns.

        Combines helix and strand detection into soft assignments.

        Args:
            hbond_map: Continuous H-bond matrix.

        Returns:
            One-hot encoded SS assignments (batch, length, 3).
            Classes: 0=loop, 1=helix, 2=strand
        """
        helix_score = self.detect_helix_pattern(hbond_map)
        strand_score = self.detect_strand_pattern(hbond_map)

        # Stack scores: [loop, helix, strand]
        # Loop score is complement of max(helix, strand)
        loop_score = 1.0 - jnp.maximum(helix_score, strand_score)

        scores = jnp.stack([loop_score, helix_score, strand_score], axis=-1)

        # Apply softmax with temperature for soft assignments
        temp = self.config.temperature
        ss_probs = nnx.softmax(scores / temp, axis=-1)

        return ss_probs

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,  # noqa: ARG002
        stats: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        """Apply secondary structure prediction.

        Args:
            data: Input data containing:
                - coordinates: Float[Array, "batch length 4 3"]
            state: Element state (passed through).
            metadata: Element metadata (passed through).
            random_params: Random parameters (unused).
            stats: Optional statistics (unused).

        Returns:
            Tuple of (output_data, state, metadata).
        """
        coords = data["coordinates"]

        # Compute H-bond matrix
        hbond_map = self.compute_hbond_map(coords)

        # Assign secondary structure
        ss_onehot = self.assign_secondary_structure(hbond_map)

        # Hard assignments for convenience
        ss_indices = jnp.argmax(ss_onehot, axis=-1)

        # Build output
        output_data = {
            **data,
            "hbond_map": hbond_map,
            "ss_onehot": ss_onehot,
            "ss_indices": ss_indices,
        }

        return output_data, state, metadata


def create_secondary_structure_predictor(
    margin: float = DEFAULT_MARGIN,
    cutoff: float = DEFAULT_CUTOFF,
    min_helix_length: int = 4,
    temperature: float = 1.0,
    seed: int = 42,
) -> DifferentiableSecondaryStructure:
    """Factory function to create a secondary structure predictor.

    Args:
        margin: Smoothing margin for H-bond matrix. Default 1.0.
        cutoff: H-bond energy threshold in kcal/mol. Default -0.5.
        min_helix_length: Minimum residues for helix. Default 4.
        temperature: Softmax temperature. Default 1.0.
        seed: Random seed. Default 42.

    Returns:
        Configured DifferentiableSecondaryStructure instance.
    """
    config = SecondaryStructureConfig(
        margin=margin,
        cutoff=cutoff,
        min_helix_length=min_helix_length,
        temperature=temperature,
    )
    rngs = nnx.Rngs(seed)
    return DifferentiableSecondaryStructure(config, rngs=rngs)
