"""A drug-discovery operator built with its defaults accepts the graph builder's output.

Every default input width is the featurizer's width, written as its symbol (chemprop derives
`DEFAULT_ATOM_FDIM` from its featurizer the same way), so the two cannot disagree.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.drug_discovery import (
    DEFAULT_ATOM_FEATURES,
    DEFAULT_BOND_FEATURES,
    ADMETConfig,
    AttentiveFPConfig,
    CircularFingerprintConfig,
    CircularFingerprintOperator,
    MACCSKeysConfig,
    MACCSKeysOperator,
    MolecularFingerprintConfig,
    MolecularPropertyConfig,
    smiles_to_graph,
)
from diffbio.operators.drug_discovery.admet_predictor import ADMETPredictor
from diffbio.operators.drug_discovery.attentive_fp import AttentiveFP
from diffbio.operators.drug_discovery.fingerprint import DifferentiableMolecularFingerprint
from diffbio.operators.drug_discovery.message_passing import (
    MessagePassingLayer,
    StackedMessagePassing,
)
from diffbio.operators.drug_discovery.property_predictor import MolecularPropertyPredictor

OPERATORS: list[tuple[Callable[..., Any], Callable[[], Any]]] = [
    (CircularFingerprintOperator, CircularFingerprintConfig),
    (MACCSKeysOperator, MACCSKeysConfig),
    (DifferentiableMolecularFingerprint, MolecularFingerprintConfig),
    (AttentiveFP, AttentiveFPConfig),
    (MolecularPropertyPredictor, MolecularPropertyConfig),
    (ADMETPredictor, ADMETConfig),
]


def test_the_featurizer_widths_are_the_graph_builders() -> None:
    graph = smiles_to_graph("CCO")

    assert graph["node_features"].shape[-1] == DEFAULT_ATOM_FEATURES
    assert graph["edge_features"].shape[-1] == DEFAULT_BOND_FEATURES


@pytest.mark.parametrize(
    "config",
    [
        CircularFingerprintConfig,
        MACCSKeysConfig,
        MolecularFingerprintConfig,
        AttentiveFPConfig,
        MolecularPropertyConfig,
        ADMETConfig,
    ],
)
def test_config_widths_default_to_the_featurizers(config: Callable[[], Any]) -> None:
    defaults = config()

    assert defaults.in_features == DEFAULT_ATOM_FEATURES
    for edge_field in ("num_edge_features", "edge_dim"):
        if hasattr(defaults, edge_field):
            assert getattr(defaults, edge_field) == DEFAULT_BOND_FEATURES


def test_message_passing_with_default_widths_accepts_a_real_molecule() -> None:
    graph = smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O")
    layer = MessagePassingLayer(hidden_dim=8, rngs=nnx.Rngs(0))
    stack = StackedMessagePassing(hidden_dim=8, num_layers=2, rngs=nnx.Rngs(0))

    for module in (layer, stack):
        hidden = module(graph["node_features"], graph["adjacency"], graph["edge_features"])
        assert hidden.shape == (graph["num_nodes"], 8)
        assert bool(jnp.all(jnp.isfinite(hidden)))


@pytest.mark.parametrize(
    ("operator", "config"), OPERATORS, ids=lambda x: getattr(x, "__name__", "")
)
def test_an_operator_with_default_widths_accepts_a_real_molecule(
    operator: Callable[..., Any], config: Callable[[], Any]
) -> None:
    module = operator(config(), rngs=nnx.Rngs(0))
    graph = smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O")
    data = {key: graph[key] for key in ("node_features", "adjacency", "edge_features")}

    result, _, _ = module.apply(data, {}, None)

    outputs = [value for key, value in result.items() if key not in data]
    assert outputs
    for value in jax.tree.leaves(outputs):
        assert bool(jnp.all(jnp.isfinite(value)))
