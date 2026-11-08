"""Molecular graphs carry the AttentiveFP featurization (Xiong et al. 2019).

The reference is DGL-LifeSci's `AttentiveFPAtomFeaturizer` and `AttentiveFPBondFeaturizer`
(Apache-2.0); the ethanol vectors below are the output its documentation shows for 'CCO'. Atom
layout: element (15 named + unknown), degree 0-5, formal charge, radical electrons, hybridization
(SP to SP3D2 + unknown), aromatic, total hydrogens 0-4, chiral centre, CIP R/S. Bond layout: type
(single, double, triple, aromatic), conjugated, in ring, stereo (none, any, Z, E).
"""

from __future__ import annotations

import numpy as np
import pytest

from diffbio.operators.drug_discovery import (
    DEFAULT_ATOM_FEATURES,
    DEFAULT_BOND_FEATURES,
    smiles_to_graph,
)

# Atom feature positions in the reference layout.
UNKNOWN_ELEMENT, DEGREES, CHARGE, RADICALS = 15, slice(16, 22), 22, 23
UNKNOWN_HYBRIDIZATION, CHIRAL_CENTRE, CIP_R, CIP_S = 29, 36, 37, 38
# Bond feature positions in the reference layout.
STEREO_Z, STEREO_E = 8, 9


def _hot(width: int, positions: list[int]) -> np.ndarray:
    vector = np.zeros(width, np.float32)
    vector[positions] = 1.0
    return vector


def _atoms(smiles: str) -> np.ndarray:
    return np.asarray(smiles_to_graph(smiles)["node_features"])


def _graph_bytes(smiles: str) -> bytes:
    graph = smiles_to_graph(smiles)
    keys = ("node_features", "adjacency", "edge_features")
    return b"".join(np.asarray(graph[key]).tobytes() for key in keys)


def test_the_widths_are_the_reference_featurizers() -> None:
    assert DEFAULT_ATOM_FEATURES == 39
    assert DEFAULT_BOND_FEATURES == 10


def test_ethanol_atoms_match_the_reference_output() -> None:
    expected = np.stack(
        [
            _hot(39, [1, 17, 26, 34]),  # C, degree 1, SP3, 3 H
            _hot(39, [1, 18, 26, 33]),  # C, degree 2, SP3, 2 H
            _hot(39, [3, 17, 26, 32]),  # O, degree 1, SP3, 1 H
        ]
    )

    np.testing.assert_array_equal(_atoms("CCO"), expected)


def test_ethanol_bonds_match_the_reference_output() -> None:
    edges = np.asarray(smiles_to_graph("CCO")["edge_features"])
    single = _hot(10, [0, 6])  # single, not conjugated, not in a ring, no stereo

    expected = np.zeros((3, 3, 10), np.float32)
    for i, j in ((0, 1), (1, 2)):
        expected[i, j] = expected[j, i] = single
    np.testing.assert_array_equal(edges, expected)


def test_aromatic_bonds_are_conjugated_and_in_a_ring() -> None:
    edges = np.asarray(smiles_to_graph("c1ccccc1")["edge_features"])

    np.testing.assert_array_equal(edges[0, 1], _hot(10, [3, 4, 5, 6]))


@pytest.mark.parametrize(
    ("smiles", "cip"), [("N[C@@H](C)C(=O)O", CIP_S), ("N[C@H](C)C(=O)O", CIP_R)]
)
def test_a_stereocentre_carries_its_cip_label(smiles: str, cip: int) -> None:
    alpha_carbon = _atoms(smiles)[1]

    assert alpha_carbon[CHIRAL_CENTRE] == 1.0
    assert alpha_carbon[cip] == 1.0
    assert alpha_carbon[CIP_R] + alpha_carbon[CIP_S] == 1.0


@pytest.mark.parametrize(("smiles", "stereo"), [("C/C=C/C", STEREO_E), ("C/C=C\\C", STEREO_Z)])
def test_a_stereo_double_bond_carries_its_configuration(smiles: str, stereo: int) -> None:
    double_bond = np.asarray(smiles_to_graph(smiles)["edge_features"])[1, 2]

    assert double_bond[stereo] == 1.0
    assert double_bond[6:].sum() == 1.0


# BBBP's six groups of distinct molecules that had one identical 34/4 graph each; in two of them
# the molecules carry opposite labels.
BBBP_STEREOISOMERS = [
    ("CN1C2CCC1CC(OC(=O)[C@H](CO)c1ccccc1)C2", "CN1C2CCC1CC(OC(=O)C(CO)c1ccccc1)C2"),
    ("C[C@H](N)Cc1ccccc1", "CC(N)Cc1ccccc1"),
    (
        "C[C@H]1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23",
        "CC1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23",
    ),
    (
        "C=C[C@H]1CN2CC[C@H]1C[C@@H]2[C@@H](O)c1ccnc2ccc(OC)cc12",
        "C=C[C@H]1CN2CC[C@@H]1C[C@H]2[C@H](O)c1ccnc2ccc(OC)cc12",
    ),
    (
        "CN(C)[C@@H]1C(=O)/C(=C(\\N)O)C(=O)[C@@]2(O)C(=O)C3=C(O)c4c(O)ccc(Cl)c4[C@@](C)(O)"
        "[C@H]3C[C@@H]12.[Cl-].[H+]",
        "CN(C)[C@@H]1C(=O)/C(=C(/N)O)C(=O)[C@@]2(O)C(=O)C3=C(O)c4c(O)ccc(Cl)c4[C@@](C)(O)"
        "[C@H]3C[C@@H]12.[Cl-].[H+]",
    ),
    ("CNC[C@H](O)c1ccc(O)c(O)c1", "CNCC(O)c1ccc(O)c(O)c1"),
]


@pytest.mark.parametrize(("first", "second"), BBBP_STEREOISOMERS)
def test_stereoisomers_get_distinct_graphs(first: str, second: str) -> None:
    assert _graph_bytes(first) != _graph_bytes(second)


def test_an_unlisted_element_and_an_ion_fall_back_as_the_reference_does() -> None:
    sodium, chloride = _atoms("[Na+].[Cl-]")

    assert sodium[UNKNOWN_ELEMENT] == 1.0
    assert sodium[CHARGE] == 1.0
    assert chloride[CHARGE] == -1.0
    assert sodium[UNKNOWN_HYBRIDIZATION] == 1.0


def test_radical_electrons_are_counted() -> None:
    assert _atoms("[CH3]")[0][RADICALS] == 1.0


def test_a_degree_above_five_leaves_the_degree_block_empty() -> None:
    sulfur = _atoms("FS(F)(F)(F)(F)F")[1]

    assert sulfur[DEGREES].sum() == 0.0
