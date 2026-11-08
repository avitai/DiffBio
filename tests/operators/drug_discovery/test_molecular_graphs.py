"""Molecular graphs built from SMILES: exact contents, batch padding and dtypes.

`smiles_to_graph` and `batch_smiles_to_graphs` read RDKit on the host; their arrays are float32
JAX arrays whatever JAX's default float width is, and a batch row is its molecule's graph padded
with zeros.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffbio.operators.drug_discovery import (
    DEFAULT_ATOM_FEATURES,
    DEFAULT_BOND_FEATURES,
    batch_smiles_to_graphs,
    smiles_to_graph,
)

# Bond vectors in the AttentiveFP layout (type, conjugated, in ring, stereo): a single bond with no
# stereo, and an aromatic bond, which is conjugated and in a ring.
SINGLE = (0, 6)
AROMATIC = (3, 4, 5, 6)
ARRAY_KEYS = ("node_features", "adjacency", "edge_features")


def _expected_bonds(num_atoms: int, bonds: list[tuple[int, int, tuple[int, ...]]]):
    adjacency = np.zeros((num_atoms, num_atoms), np.float32)
    edges = np.zeros((num_atoms, num_atoms, DEFAULT_BOND_FEATURES), np.float32)
    for i, j, positions in bonds:
        adjacency[i, j] = adjacency[j, i] = 1.0
        edges[i, j, list(positions)] = edges[j, i, list(positions)] = 1.0
    return adjacency, edges


@pytest.mark.parametrize(
    ("smiles", "num_atoms", "bonds"),
    [
        ("CCO", 3, [(0, 1, SINGLE), (1, 2, SINGLE)]),
        ("c1ccccc1", 6, [(k, (k + 1) % 6, AROMATIC) for k in range(6)]),
    ],
    ids=["ethanol", "benzene"],
)
def test_a_graph_holds_the_molecules_atoms_and_bonds(
    smiles: str,
    num_atoms: int,
    bonds: list[tuple[int, int, tuple[int, ...]]],
) -> None:
    graph = smiles_to_graph(smiles)
    adjacency, edges = _expected_bonds(num_atoms, bonds)

    assert graph["num_nodes"] == num_atoms
    assert graph["node_features"].shape == (num_atoms, DEFAULT_ATOM_FEATURES)
    np.testing.assert_array_equal(np.asarray(graph["adjacency"]), adjacency)
    np.testing.assert_array_equal(np.asarray(graph["edge_features"]), edges)


def test_a_batch_row_is_its_molecules_graph_padded_with_zeros() -> None:
    smiles = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    batch = batch_smiles_to_graphs(smiles)
    max_nodes = max(smiles_to_graph(s)["num_nodes"] for s in smiles)

    assert batch["node_features"].shape == (3, max_nodes, DEFAULT_ATOM_FEATURES)
    for row, molecule in enumerate(smiles):
        graph = smiles_to_graph(molecule)
        n = graph["num_nodes"]
        np.testing.assert_array_equal(np.asarray(batch["node_mask"][row]), np.arange(max_nodes) < n)
        for key in ARRAY_KEYS:
            padded = np.asarray(batch[key][row])
            corner = padded[:n] if key == "node_features" else padded[:n, :n]
            np.testing.assert_array_equal(corner, np.asarray(graph[key]))
            assert padded.sum() == corner.sum()


@pytest.mark.parametrize("enable_x64", [False, True], ids=["float32_default", "x64"])
def test_graph_arrays_are_float32_jax_arrays(enable_x64: bool) -> None:
    with jax.enable_x64(enable_x64):
        graph = smiles_to_graph("CCO")
        batch = batch_smiles_to_graphs(["CCO", "c1ccccc1"])

    for arrays in (graph, batch):
        for key, value in arrays.items():
            if key == "num_nodes":
                continue
            assert isinstance(value, jax.Array), key
            assert value.dtype == jnp.float32, key


def test_an_invalid_smiles_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="not_a_smiles"):
        smiles_to_graph("not_a_smiles")


def test_a_smiles_without_atoms_is_refused() -> None:
    with pytest.raises(ValueError, match="no atoms"):
        smiles_to_graph("")


def test_an_empty_batch_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one SMILES"):
        batch_smiles_to_graphs([])
