"""Primitive functions for molecular graph processing.

This module provides utility functions for converting SMILES strings to
molecular graphs suitable for differentiable neural network processing.
RDKit parses each molecule on the host, where its graph is assembled with NumPy; the
functions return float32 JAX arrays, one transfer per array.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _rdkit_chem() -> Any:
    """Return `rdkit.Chem`, importing it on first use.

    RDKit is an optional dependency, installed with `pip install "diffbio[chem]"`.
    Importing it at module scope would pull the whole toolkit into every `import
    diffbio`, for a capability most pipelines never call. The other RDKit call sites
    in this package are already lazy in the same way; this keeps them consistent.
    """
    try:
        from rdkit import Chem
    except ImportError as e:  # pragma: no cover - exercised only without RDKit
        raise ImportError(
            'Molecular graph primitives require RDKit: pip install "diffbio[chem]"'
        ) from e
    return Chem


@dataclass(frozen=True)
class AtomFeatureConfig:
    """Configuration for atom feature extraction.

    The default configuration produces 34 features:

    - Atom type: 12 dimensions (C, N, O, S, F, Cl, Br, I, P, Si, B, Other)
    - Degree: 7 dimensions (0-6)
    - Formal charge: 5 dimensions (-2 to +2)
    - Hybridization: 4 dimensions (SP, SP2, SP3, SP3D)
    - Aromaticity: 1 dimension (binary)
    - Num hydrogens: 5 dimensions (0-4)
    """

    num_atom_types: int = 12
    max_degree: int = 6  # Creates max_degree + 1 dimensions
    charge_range: tuple[int, int] = (-2, 2)  # Creates 5 dimensions
    num_hybridization_types: int = 4
    max_num_hydrogens: int = 4  # Creates max_num_hydrogens + 1 dimensions

    @property
    def total_features(self) -> int:
        """Calculate total number of atom features."""
        return (
            self.num_atom_types
            + (self.max_degree + 1)
            + (self.charge_range[1] - self.charge_range[0] + 1)
            + self.num_hybridization_types
            + 1  # aromaticity
            + (self.max_num_hydrogens + 1)
        )


# Default configuration
DEFAULT_ATOM_CONFIG = AtomFeatureConfig()
DEFAULT_ATOM_FEATURES = DEFAULT_ATOM_CONFIG.total_features  # 34 features

# Atom type vocabulary for one-hot encoding
ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si", "B", "Other"]

# Bond-type one-hot width: single, double, triple, aromatic.
DEFAULT_BOND_FEATURES = 4


def _atom_feature_values(atom: Any, config: AtomFeatureConfig) -> list[float]:
    """Return an RDKit atom's feature values as a host list (see `get_atom_features`)."""
    features: list[float] = []

    # Atom type
    symbol = atom.GetSymbol()
    atom_type_idx = ATOM_TYPES.index(symbol) if symbol in ATOM_TYPES else len(ATOM_TYPES) - 1
    atom_type_onehot = [0.0] * config.num_atom_types
    atom_type_onehot[min(atom_type_idx, config.num_atom_types - 1)] = 1.0
    features.extend(atom_type_onehot)

    # Degree
    degree = min(atom.GetDegree(), config.max_degree)
    degree_onehot = [0.0] * (config.max_degree + 1)
    degree_onehot[degree] = 1.0
    features.extend(degree_onehot)

    # Formal charge
    charge = atom.GetFormalCharge()
    charge_min, charge_max = config.charge_range
    charge_idx = max(charge_min, min(charge_max, charge)) - charge_min
    charge_dim = charge_max - charge_min + 1
    charge_onehot = [0.0] * charge_dim
    charge_onehot[charge_idx] = 1.0
    features.extend(charge_onehot)

    # Hybridization
    hybridization = atom.GetHybridization()
    chem = _rdkit_chem()
    hyb_types = [
        chem.rdchem.HybridizationType.SP,
        chem.rdchem.HybridizationType.SP2,
        chem.rdchem.HybridizationType.SP3,
        chem.rdchem.HybridizationType.SP3D,
    ]
    hyb_onehot = [0.0] * config.num_hybridization_types
    for i, h in enumerate(hyb_types[: config.num_hybridization_types]):
        if hybridization == h:
            hyb_onehot[i] = 1.0
            break
    features.extend(hyb_onehot)

    # Aromaticity
    features.append(1.0 if atom.GetIsAromatic() else 0.0)

    # Number of hydrogens
    num_h = min(atom.GetTotalNumHs(), config.max_num_hydrogens)
    h_onehot = [0.0] * (config.max_num_hydrogens + 1)
    h_onehot[num_h] = 1.0
    features.extend(h_onehot)

    return features


def get_atom_features(atom: Any, config: AtomFeatureConfig | None = None) -> jnp.ndarray:
    """Extract features from an RDKit atom object.

    Features include:
    - Atom type (one-hot, config.num_atom_types)
    - Degree (one-hot, 0 to config.max_degree)
    - Formal charge (one-hot, config.charge_range)
    - Hybridization (one-hot, config.num_hybridization_types)
    - Aromaticity (binary)
    - Number of hydrogens (one-hot, 0 to config.max_num_hydrogens)

    Args:
        atom: RDKit atom object.
        config: Feature extraction configuration. Defaults to DEFAULT_ATOM_CONFIG.

    Returns:
        Feature vector of shape (config.total_features,).
    """
    resolved = DEFAULT_ATOM_CONFIG if config is None else config
    return jnp.asarray(_atom_feature_values(atom, resolved), dtype=jnp.float32)


def get_bond_features(bond: Any) -> jnp.ndarray:
    """Extract features from an RDKit bond object.

    Features include:
    - Bond type (one-hot: single, double, triple, aromatic)

    Args:
        bond: RDKit bond object.

    Returns:
        Feature vector of shape (4,).
    """
    return jnp.asarray(_bond_feature_values(bond), dtype=jnp.float32)


def _bond_feature_values(bond: Any) -> list[float]:
    """Return an RDKit bond's one-hot bond type as a host list (see `get_bond_features`)."""
    bond_type = bond.GetBondType()
    bond_types = _rdkit_chem().rdchem.BondType
    kinds = (bond_types.SINGLE, bond_types.DOUBLE, bond_types.TRIPLE, bond_types.AROMATIC)
    return [1.0 if bond_type == kind else 0.0 for kind in kinds]


def _host_graph(smiles: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse `smiles` and return its node features, adjacency and edge features as NumPy.

    Raises:
        ValueError: If the SMILES string is invalid or describes no atoms.
    """
    mol = _rdkit_chem().MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        raise ValueError(f"SMILES string has no atoms: {smiles!r}")

    node_features = np.asarray(
        [_atom_feature_values(atom, DEFAULT_ATOM_CONFIG) for atom in mol.GetAtoms()],
        dtype=np.float32,
    )
    adjacency = np.zeros((num_atoms, num_atoms), dtype=np.float32)
    edge_features = np.zeros((num_atoms, num_atoms, DEFAULT_BOND_FEATURES), dtype=np.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Symmetric (undirected graph)
        adjacency[i, j] = adjacency[j, i] = 1.0
        edge_features[i, j] = edge_features[j, i] = _bond_feature_values(bond)
    return node_features, adjacency, edge_features


def smiles_to_graph(smiles: str) -> dict[str, Any]:
    """Convert a SMILES string to a molecular graph.

    Args:
        smiles: SMILES string representing a molecule.

    Returns:
        Dictionary containing:
            - node_features: (num_atoms, num_features) atom feature matrix
            - adjacency: (num_atoms, num_atoms) adjacency matrix
            - edge_features: (num_atoms, num_atoms, num_edge_features) bond features
            - num_nodes: number of atoms

    Raises:
        ValueError: If the SMILES string is invalid or describes no atoms.
    """
    node_features, adjacency, edge_features = _host_graph(smiles)
    return {
        "node_features": jnp.asarray(node_features),
        "adjacency": jnp.asarray(adjacency),
        "edge_features": jnp.asarray(edge_features),
        "num_nodes": node_features.shape[0],
    }


def batch_smiles_to_graphs(smiles_list: list[str]) -> dict[str, Any]:
    """Convert a batch of SMILES strings to padded graph tensors.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        Dictionary containing:
            - node_features: (batch_size, max_nodes, num_features)
            - adjacency: (batch_size, max_nodes, max_nodes)
            - edge_features: (batch_size, max_nodes, max_nodes, num_edge_features)
            - node_mask: (batch_size, max_nodes) mask for valid nodes

    Raises:
        ValueError: If the list is empty, or a SMILES string is invalid or describes no atoms.
    """
    if not smiles_list:
        raise ValueError("batch_smiles_to_graphs needs at least one SMILES string")
    graphs = [_host_graph(smiles) for smiles in smiles_list]
    max_nodes = max(nodes.shape[0] for nodes, _, _ in graphs)
    batch_size = len(graphs)
    num_features = graphs[0][0].shape[1]

    node_features = np.zeros((batch_size, max_nodes, num_features), dtype=np.float32)
    adjacency = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.float32)
    edge_features = np.zeros(
        (batch_size, max_nodes, max_nodes, DEFAULT_BOND_FEATURES), dtype=np.float32
    )
    node_mask = np.zeros((batch_size, max_nodes), dtype=np.float32)
    for i, (nodes, adj, edges) in enumerate(graphs):
        n = nodes.shape[0]
        node_features[i, :n] = nodes
        adjacency[i, :n, :n] = adj
        edge_features[i, :n, :n] = edges
        node_mask[i, :n] = 1.0

    return {
        "node_features": jnp.asarray(node_features),
        "adjacency": jnp.asarray(adjacency),
        "edge_features": jnp.asarray(edge_features),
        "node_mask": jnp.asarray(node_mask),
    }
