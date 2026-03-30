"""Differentiable molecular fingerprint operators.

This module implements neural graph fingerprints that provide
differentiable alternatives to traditional molecular fingerprints.

Operators:
    DifferentiableMolecularFingerprint: General neural graph fingerprint
    CircularFingerprintOperator: Differentiable ECFP/Morgan fingerprints
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx

from diffbio.core import soft_ops
from diffbio.operators.drug_discovery._graph_utils import (
    attach_fingerprint,
    build_encoder,
    ensure_rngs,
    graph_sum_readout,
    initialize_graph_encoder,
    stabilize_operator_id,
    unpack_graph_inputs,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MolecularFingerprintConfig(OperatorConfig):
    """Configuration for molecular fingerprint operator.

    Attributes:
        fingerprint_dim: Dimension of output fingerprint vector.
        hidden_dim: Hidden dimension for graph convolutions.
        num_layers: Number of graph convolution layers.
        in_features: Number of input node features (default: DEFAULT_ATOM_FEATURES=34).
        normalize: Whether to L2-normalize the fingerprint.
    """

    fingerprint_dim: int = 256
    hidden_dim: int = 128
    num_layers: int = 3
    in_features: int = 4  # Default for tests; use DEFAULT_ATOM_FEATURES for real molecules
    normalize: bool = False


class DifferentiableMolecularFingerprint(OperatorModule):
    """Neural graph fingerprint operator.

    Computes learned molecular fingerprints using graph neural networks.
    Unlike traditional fingerprints (e.g., ECFP/Morgan), these are fully
    differentiable and can be optimized for specific tasks.

    The fingerprint is computed by:
    1. Message passing to compute atom representations
    2. Sum pooling to get graph-level representation
    3. Linear projection to fingerprint dimension
    4. Optional L2 normalization

    Example:
        ```python
        config = MolecularFingerprintConfig(fingerprint_dim=128)
        fp_op = DifferentiableMolecularFingerprint(config, rngs=nnx.Rngs(42))
        data = {"node_features": nodes, "adjacency": adj, "node_mask": mask}
        result, _, _ = fp_op.apply(data, {}, None)
        fingerprint = result["fingerprint"]  # shape: (128,)
        ```
    """

    def __init__(
        self,
        config: MolecularFingerprintConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize fingerprint operator.

        Args:
            config: Fingerprint configuration.
            rngs: Flax NNX random number generators.
            name: Optional name for the operator.
        """
        super().__init__(config, rngs=rngs, name=name)

        rngs = initialize_graph_encoder(
            self,
            rngs=rngs,
            in_features=config.in_features,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
        )

        # Projection to fingerprint dimension
        self.projection = nnx.Linear(
            in_features=config.hidden_dim,
            out_features=config.fingerprint_dim,
            rngs=rngs,
        )

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        """Compute molecular fingerprint.

        Args:
            data: Input data containing:
                - node_features: (num_nodes, num_features) atom features
                - adjacency: (num_nodes, num_nodes) adjacency matrix
                - node_mask: (num_nodes,) mask for valid nodes
            state: Per-element state (passed through).
            metadata: Optional metadata.
            random_params: Unused random parameters.
            stats: Optional statistics dictionary.

        Returns:
            Tuple of:
                - data with added "fingerprint" key
                - unchanged state
                - unchanged metadata
        """
        graph_repr = graph_sum_readout(data, self.encoder)

        # Project to fingerprint dimension
        fingerprint = self.projection(graph_repr)

        # Optional normalization
        if self.config.normalize:
            fingerprint = fingerprint / (jnp.linalg.norm(fingerprint) + 1e-8)

        return attach_fingerprint(data, fingerprint), state, metadata


def create_fingerprint_operator(
    fingerprint_dim: int = 256,
    num_layers: int = 3,
    normalize: bool = False,
    seed: int = 42,
) -> DifferentiableMolecularFingerprint:
    """Create a molecular fingerprint operator.

    Args:
        fingerprint_dim: Output fingerprint dimension.
        num_layers: Number of message passing layers.
        normalize: Whether to L2-normalize output.
        seed: Random seed.

    Returns:
        Configured DifferentiableMolecularFingerprint.
    """
    config = MolecularFingerprintConfig(
        fingerprint_dim=fingerprint_dim,
        num_layers=num_layers,
        normalize=normalize,
    )
    return DifferentiableMolecularFingerprint(config, rngs=nnx.Rngs(seed))


# =============================================================================
# Circular Fingerprint (ECFP/Morgan) Operator
# =============================================================================


@dataclass(frozen=True)
class CircularFingerprintConfig(OperatorConfig):
    # pylint: disable=too-many-instance-attributes
    """Configuration for circular fingerprint operator (ECFP/Morgan).

    Attributes:
        radius: Fingerprint radius. ECFP4 = radius 2, ECFP6 = radius 3.
        n_bits: Number of bits in fingerprint (default: 2048).
        use_chirality: Include chirality in fingerprint (default: False).
        use_bond_types: Include bond type information (default: True).
        use_features: Use pharmacophoric features (FCFP variant, default: False).
        differentiable: Use learned hash functions for gradients (default: True).
        hash_hidden_dim: Hidden dimension for hash network (default: 128).
        temperature: Temperature for soft bit assignment (default: 1.0).
        in_features: Number of input node features (default: 4).
    """

    radius: int = 2  # ECFP4 = radius 2, ECFP6 = radius 3
    n_bits: int = 2048
    use_chirality: bool = False
    use_bond_types: bool = True
    use_features: bool = False  # FCFP variant if True
    differentiable: bool = True  # Use learned hash functions
    hash_hidden_dim: int = 128
    temperature: float = 1.0  # For soft bit assignment
    in_features: int = 4  # Number of input node features


class CircularFingerprintOperator(OperatorModule):
    """Differentiable circular fingerprints (ECFP/Morgan).

    For differentiable=True:
        Uses message passing to aggregate substructure information,
        then learned "soft hash" functions for bit assignment.
        Gradients flow through the entire computation.

    For differentiable=False:
        Wraps RDKit implementation for exact ECFP.
        No gradient flow (useful for inference/comparison).

    The differentiable version approximates ECFP behavior while
    enabling end-to-end optimization of the fingerprint representation.

    Example:
        ```python
        config = CircularFingerprintConfig(radius=2, n_bits=1024)
        fp_op = CircularFingerprintOperator(config, rngs=nnx.Rngs(0))
        data = {"node_features": node_feats, "adjacency": adj}
        result, state, meta = fp_op.apply(data, {}, None)
        fingerprint = result["fingerprint"]  # Shape: (n_bits,)
        ```

    References:
        Rogers, David, and Mathew Hahn. "Extended-connectivity fingerprints."
        Journal of chemical information and modeling 50.5 (2010): 742-754.
    """

    def __init__(
        self,
        config: CircularFingerprintConfig,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize CircularFingerprintOperator.

        Args:
            config: Circular fingerprint configuration.
            rngs: Flax NNX random number generators.
        """
        super().__init__(config, rngs=rngs)

        stabilize_operator_id(self)
        rngs = ensure_rngs(rngs)

        if config.differentiable:
            # Message passing layers for substructure aggregation
            # Each layer corresponds to one radius step
            self.message_passing = build_encoder(
                hidden_dim=config.hash_hidden_dim,
                num_layers=config.radius,
                in_features=config.in_features,
                rngs=rngs,
            )

            # Learned hash function: maps substructure embedding to bit indices
            self.hash_network = nnx.Sequential(
                nnx.Linear(config.hash_hidden_dim, config.hash_hidden_dim, rngs=rngs),
                nnx.relu,
                nnx.Linear(config.hash_hidden_dim, config.n_bits, rngs=rngs),
            )
        else:
            # RDKit mode - no learnable parameters needed
            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem

                self._Chem = Chem
                self._AllChem = AllChem
            except ImportError as e:
                raise ImportError(
                    "CircularFingerprintOperator with differentiable=False "
                    "requires RDKit: pip install rdkit"
                ) from e

    @nnx.jit
    def _compute_differentiable_fp(
        self,
        node_features: jnp.ndarray,
        adjacency: jnp.ndarray,
        edge_features: jnp.ndarray | None = None,
        node_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute differentiable circular fingerprint (JIT compiled).

        Uses message passing to aggregate local substructure information,
        then applies learned hash functions for soft bit assignment.

        Args:
            node_features: (num_nodes, num_features) atom features
            adjacency: (num_nodes, num_nodes) adjacency matrix
            edge_features: Optional edge features
            node_mask: Optional mask for valid nodes

        Returns:
            Fingerprint vector of shape (n_bits,)
        """
        # Message passing to compute atom representations with substructure info
        # After 'radius' layers, each atom embedding contains info about
        # atoms within 'radius' bonds
        node_hidden = self.message_passing(node_features, adjacency, edge_features)

        # Apply node mask if provided
        if node_mask is not None:
            node_hidden = node_hidden * node_mask[:, None]

        # Compute soft hash for each atom's environment
        # hash_logits: (num_nodes, n_bits)
        hash_logits = self.hash_network(node_hidden)

        # Apply temperature-scaled softmax for soft bit assignment
        # Higher temperature = softer bits, lower = sharper (more binary-like)
        soft_bits = soft_ops.greater(hash_logits, 0.0, softness=self.config.temperature)

        # Aggregate across atoms using max (OR-like) operation
        # This mimics how ECFP sets bits based on any substructure match
        fingerprint = soft_ops.max(soft_bits, axis=0, softness=self.config.temperature)

        return fingerprint

    def _compute_rdkit_fp(self, smiles: str) -> jnp.ndarray:
        """Compute exact ECFP using RDKit.

        Args:
            smiles: SMILES string

        Returns:
            Binary fingerprint vector of shape (n_bits,)
        """
        mol = self._Chem.MolFromSmiles(smiles)
        if mol is None:
            # Return zero fingerprint for invalid SMILES
            return jnp.zeros(self.config.n_bits, dtype=jnp.float32)

        # Compute Morgan/ECFP fingerprint
        if self.config.use_features:
            # FCFP variant - uses pharmacophoric features
            fp = self._AllChem.GetMorganFingerprintAsBitVect(
                mol,
                self.config.radius,
                nBits=self.config.n_bits,
                useChirality=self.config.use_chirality,
                useBondTypes=self.config.use_bond_types,
                useFeatures=True,
            )
        else:
            # Standard ECFP
            fp = self._AllChem.GetMorganFingerprintAsBitVect(
                mol,
                self.config.radius,
                nBits=self.config.n_bits,
                useChirality=self.config.use_chirality,
                useBondTypes=self.config.use_bond_types,
            )

        # Convert to numpy first (faster), then to JAX
        # RDKit provides efficient conversion to numpy
        arr = np.zeros(self.config.n_bits, dtype=np.float32)
        self._AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        return jnp.asarray(arr)

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        """Compute circular fingerprint.

        Args:
            data: Input data containing either:
                For differentiable=True:
                    - node_features: (num_nodes, num_features) atom features
                    - adjacency: (num_nodes, num_nodes) adjacency matrix
                    - node_mask: (num_nodes,) optional mask for valid nodes
                For differentiable=False:
                    - smiles: SMILES string
            state: Per-element state (passed through).
            metadata: Optional metadata.
            random_params: Unused random parameters.
            stats: Optional statistics dictionary.

        Returns:
            Tuple of:
                - data with added "fingerprint" key
                - unchanged state
                - unchanged metadata
        """
        if self.config.differentiable:
            node_features, adjacency, edge_features, node_mask = unpack_graph_inputs(data)

            fp = self._compute_differentiable_fp(
                node_features,
                adjacency,
                edge_features,  # pyright: ignore[reportArgumentType]
                node_mask,  # pyright: ignore[reportArgumentType]
            )
        else:
            smiles = data["smiles"]
            fp = self._compute_rdkit_fp(smiles)

        return attach_fingerprint(data, fp), state, metadata


# =============================================================================
# Factory Functions for Common ECFP Configurations
# =============================================================================


def create_ecfp4_operator(
    n_bits: int = 2048,
    differentiable: bool = True,
    rngs: nnx.Rngs | None = None,
) -> CircularFingerprintOperator:
    """Create ECFP4 (radius=2) fingerprint operator.

    ECFP4 captures substructures within 4 bonds (radius 2).

    Args:
        n_bits: Number of fingerprint bits (default: 2048).
        differentiable: Use learned hash functions (default: True).
        rngs: Random number generators.

    Returns:
        Configured CircularFingerprintOperator.
    """
    config = CircularFingerprintConfig(
        radius=2,
        n_bits=n_bits,
        differentiable=differentiable,
    )
    return CircularFingerprintOperator(config, rngs=rngs or nnx.Rngs(0))


def create_ecfp6_operator(
    n_bits: int = 2048,
    differentiable: bool = True,
    rngs: nnx.Rngs | None = None,
) -> CircularFingerprintOperator:
    """Create ECFP6 (radius=3) fingerprint operator.

    ECFP6 captures substructures within 6 bonds (radius 3).

    Args:
        n_bits: Number of fingerprint bits (default: 2048).
        differentiable: Use learned hash functions (default: True).
        rngs: Random number generators.

    Returns:
        Configured CircularFingerprintOperator.
    """
    config = CircularFingerprintConfig(
        radius=3,
        n_bits=n_bits,
        differentiable=differentiable,
    )
    return CircularFingerprintOperator(config, rngs=rngs or nnx.Rngs(0))


def create_fcfp4_operator(
    n_bits: int = 2048,
    differentiable: bool = True,
    rngs: nnx.Rngs | None = None,
) -> CircularFingerprintOperator:
    """Create FCFP4 (feature-based, radius=2) fingerprint operator.

    FCFP4 uses pharmacophoric atom features instead of atomic properties.
    Better for finding molecules with similar biological activity.

    Args:
        n_bits: Number of fingerprint bits (default: 2048).
        differentiable: Use learned hash functions (default: True).
        rngs: Random number generators.

    Returns:
        Configured CircularFingerprintOperator.
    """
    config = CircularFingerprintConfig(
        radius=2,
        n_bits=n_bits,
        use_features=True,
        differentiable=differentiable,
    )
    return CircularFingerprintOperator(config, rngs=rngs or nnx.Rngs(0))
