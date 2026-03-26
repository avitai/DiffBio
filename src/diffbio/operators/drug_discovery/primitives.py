"""Primitive functions for molecular graph processing.

This module provides utility functions for converting SMILES strings to
molecular graphs suitable for differentiable neural network processing.
RDKit is used for parsing only; all graph operations use JAX arrays.
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from rdkit import Chem

logger = logging.getLogger(__name__)


@dataclass
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
    if config is None:
        config = DEFAULT_ATOM_CONFIG

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
    hyb_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
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

    return jnp.array(features, dtype=jnp.float32)


def get_bond_features(bond: Any) -> jnp.ndarray:
    """Extract features from an RDKit bond object.

    Features include:
    - Bond type (one-hot: single, double, triple, aromatic)

    Args:
        bond: RDKit bond object.

    Returns:
        Feature vector of shape (4,).
    """
    bond_type = bond.GetBondType()
    features = [
        1 if bond_type == Chem.rdchem.BondType.SINGLE else 0,
        1 if bond_type == Chem.rdchem.BondType.DOUBLE else 0,
        1 if bond_type == Chem.rdchem.BondType.TRIPLE else 0,
        1 if bond_type == Chem.rdchem.BondType.AROMATIC else 0,
    ]
    return jnp.array(features, dtype=jnp.float32)


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
        ValueError: If SMILES string is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    num_atoms = mol.GetNumAtoms()

    # Extract node features
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(get_atom_features(atom))
    node_features = jnp.stack(node_features)

    # Build adjacency matrix and edge features
    adjacency = jnp.zeros((num_atoms, num_atoms), dtype=jnp.float32)
    edge_features = jnp.zeros((num_atoms, num_atoms, 4), dtype=jnp.float32)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feat = get_bond_features(bond)

        # Symmetric (undirected graph)
        adjacency = adjacency.at[i, j].set(1.0)
        adjacency = adjacency.at[j, i].set(1.0)
        edge_features = edge_features.at[i, j].set(bond_feat)
        edge_features = edge_features.at[j, i].set(bond_feat)

    return {
        "node_features": node_features,
        "adjacency": adjacency,
        "edge_features": edge_features,
        "num_nodes": num_atoms,
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
    """
    graphs = [smiles_to_graph(s) for s in smiles_list]

    max_nodes = max(g["num_nodes"] for g in graphs)
    batch_size = len(graphs)
    num_features = graphs[0]["node_features"].shape[1]
    num_edge_features = graphs[0]["edge_features"].shape[2]

    # Initialize padded tensors
    node_features = jnp.zeros((batch_size, max_nodes, num_features))
    adjacency = jnp.zeros((batch_size, max_nodes, max_nodes))
    edge_features = jnp.zeros((batch_size, max_nodes, max_nodes, num_edge_features))
    node_mask = jnp.zeros((batch_size, max_nodes))

    for i, g in enumerate(graphs):
        n = g["num_nodes"]
        node_features = node_features.at[i, :n, :].set(g["node_features"])
        adjacency = adjacency.at[i, :n, :n].set(g["adjacency"])
        edge_features = edge_features.at[i, :n, :n, :].set(g["edge_features"])
        node_mask = node_mask.at[i, :n].set(1.0)

    return {
        "node_features": node_features,
        "adjacency": adjacency,
        "edge_features": edge_features,
        "node_mask": node_mask,
    }
