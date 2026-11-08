"""Primitive functions for molecular graph processing.

This module provides utility functions for converting SMILES strings to
molecular graphs suitable for differentiable neural network processing.
RDKit parses each molecule on the host, where its graph is assembled with NumPy; the
functions return float32 JAX arrays, one transfer per array.

The atom and bond features are the AttentiveFP featurization (Xiong et al. 2019, "Pushing the
Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism",
J. Med. Chem. 63, 8749), as DGL-LifeSci's `AttentiveFPAtomFeaturizer` and
`AttentiveFPBondFeaturizer` define it (Apache-2.0): 39 atom and 10 bond features, including
chirality, bond stereo, conjugation and ring membership, so stereoisomers get distinct graphs.
"""

import logging
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


# The AttentiveFP featurization's value sets, one per one-hot block. RDKit enums are matched by
# name, so these widths are known without importing RDKit. A block with an unknown slot sets it
# for a value outside its set; the degree, hydrogen-count, bond-type and stereo blocks have none
# and stay all zero, as in the reference.
ATOM_ELEMENTS = ("B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "Te", "I", "At")
ATOM_DEGREES = (0, 1, 2, 3, 4, 5)
ATOM_HYBRIDIZATIONS = ("SP", "SP2", "SP3", "SP3D", "SP3D2")
ATOM_HYDROGEN_COUNTS = (0, 1, 2, 3, 4)
ATOM_CIP_CODES = ("R", "S")
BOND_TYPES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
BOND_STEREOS = ("STEREONONE", "STEREOANY", "STEREOZ", "STEREOE")

# Element (+ unknown), degree, formal charge, radical electrons, hybridization (+ unknown),
# aromaticity, total hydrogens, chiral centre, CIP label: 39.
DEFAULT_ATOM_FEATURES = (
    len(ATOM_ELEMENTS)
    + 1
    + len(ATOM_DEGREES)
    + 2
    + len(ATOM_HYBRIDIZATIONS)
    + 1
    + 1
    + len(ATOM_HYDROGEN_COUNTS)
    + 1
    + len(ATOM_CIP_CODES)
)
# Bond type, conjugated, in ring, stereo configuration: 10.
DEFAULT_BOND_FEATURES = len(BOND_TYPES) + 2 + len(BOND_STEREOS)


def _one_hot(value: Any, allowed: tuple[Any, ...], *, unknown: bool = False) -> list[float]:
    """Return the one-hot encoding of `value` over `allowed`, with an optional unknown slot."""
    encoding = [1.0 if value == item else 0.0 for item in allowed]
    if unknown:
        encoding.append(0.0 if value in allowed else 1.0)
    return encoding


def _atom_feature_values(atom: Any) -> list[float]:
    """Return an RDKit atom's feature values as a host list (see `get_atom_features`)."""
    cip_code = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
    return [
        *_one_hot(atom.GetSymbol(), ATOM_ELEMENTS, unknown=True),
        *_one_hot(atom.GetDegree(), ATOM_DEGREES),
        float(atom.GetFormalCharge()),
        float(atom.GetNumRadicalElectrons()),
        *_one_hot(str(atom.GetHybridization()), ATOM_HYBRIDIZATIONS, unknown=True),
        float(atom.GetIsAromatic()),
        *_one_hot(atom.GetTotalNumHs(), ATOM_HYDROGEN_COUNTS),
        float(atom.HasProp("_ChiralityPossible")),
        *_one_hot(cip_code, ATOM_CIP_CODES),
    ]


def get_atom_features(atom: Any) -> jnp.ndarray:
    """Extract an RDKit atom's AttentiveFP features.

    Features, in order:

    - Element: one-hot over `ATOM_ELEMENTS`, plus an unknown slot
    - Degree: one-hot over 0-5 (all zero above 5)
    - Formal charge and number of radical electrons, as counts
    - Hybridization: one-hot over `ATOM_HYBRIDIZATIONS`, plus an unknown slot
    - Aromaticity (binary)
    - Total hydrogens: one-hot over 0-4 (all zero above 4)
    - Chiral centre (binary, RDKit's `_ChiralityPossible`)
    - CIP label: one-hot over R and S (all zero when RDKit assigns none)

    Args:
        atom: RDKit atom object.

    Returns:
        Feature vector of shape (`DEFAULT_ATOM_FEATURES`,).
    """
    return jnp.asarray(_atom_feature_values(atom), dtype=jnp.float32)


def get_bond_features(bond: Any) -> jnp.ndarray:
    """Extract an RDKit bond's AttentiveFP features.

    Features, in order:

    - Bond type: one-hot over single, double, triple and aromatic
    - Conjugated and in a ring (binary each)
    - Stereo configuration: one-hot over none, any, Z and E

    Args:
        bond: RDKit bond object.

    Returns:
        Feature vector of shape (`DEFAULT_BOND_FEATURES`,).
    """
    return jnp.asarray(_bond_feature_values(bond), dtype=jnp.float32)


def _bond_feature_values(bond: Any) -> list[float]:
    """Return an RDKit bond's feature values as a host list (see `get_bond_features`)."""
    return [
        *_one_hot(str(bond.GetBondType()), BOND_TYPES),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        *_one_hot(str(bond.GetStereo()), BOND_STEREOS),
    ]


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
        [_atom_feature_values(atom) for atom in mol.GetAtoms()],
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
