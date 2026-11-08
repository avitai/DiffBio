"""A padded molecule's fingerprint equals its unpadded fingerprint.

The fingerprint operators read out with a soft max over atoms; padding atoms must take no part
in it. The comparison is made exact by construction rather than held to a tolerance: every
parameter is drawn from {-1/4, 0, 1/4}, the atom features are one-hot and the adjacency is 0/1,
so each linear layer and ReLU in the forward pass is exact in float32; the only inexact steps
are elementwise, and they see identical inputs in the padded and unpadded runs; and methanol
has two heavy atoms, so every sum over atoms has at most two nonzero terms, which round the
same in either order. Any difference is therefore padding taking part in the computation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from diffbio.operators.drug_discovery import (
    DEFAULT_ATOM_FEATURES,
    CircularFingerprintConfig,
    CircularFingerprintOperator,
    MACCSKeysConfig,
    MACCSKeysOperator,
    batch_smiles_to_graphs,
    smiles_to_graph,
)

GRAPH_KEYS = ("node_features", "adjacency", "edge_features")


def _with_exact_parameters(operator: nnx.Module) -> nnx.Module:
    """Replace every parameter with values from {-1/4, 0, 1/4}, deterministically."""
    params = nnx.state(operator, nnx.Param)
    leaves, treedef = jax.tree.flatten(params)
    rng = np.random.default_rng(0)
    exact = [
        jnp.asarray(rng.choice([-0.25, 0.0, 0.25], size=leaf.shape), dtype=leaf.dtype)
        for leaf in leaves
    ]
    nnx.update(operator, jax.tree.unflatten(treedef, exact))
    return operator


def _circular() -> nnx.Module:
    config = CircularFingerprintConfig(
        radius=1, n_bits=16, hash_hidden_dim=2, in_features=DEFAULT_ATOM_FEATURES
    )
    return _with_exact_parameters(CircularFingerprintOperator(config, rngs=nnx.Rngs(0)))


def _maccs() -> nnx.Module:
    config = MACCSKeysConfig(
        n_bits=16, hidden_dim=2, num_layers=1, in_features=DEFAULT_ATOM_FEATURES
    )
    return _with_exact_parameters(MACCSKeysOperator(config, rngs=nnx.Rngs(0)))


def _fingerprint(operator: nnx.Module, graph: dict[str, Any]) -> np.ndarray:
    result, _, _ = operator.apply(graph, {}, None)
    return np.asarray(result["fingerprint"])


@pytest.mark.parametrize("build", [_circular, _maccs], ids=["circular", "maccs"])
def test_a_padded_molecule_has_its_unpadded_fingerprint(build: Callable[[], nnx.Module]) -> None:
    operator = build()
    methanol = smiles_to_graph("CO")
    batch = batch_smiles_to_graphs(["CO", "CC(=O)Oc1ccccc1C(=O)O"])  # padded to 13 atoms
    padded = {key: batch[key][0] for key in (*GRAPH_KEYS, "node_mask")}

    unpadded = _fingerprint(operator, {key: methanol[key] for key in GRAPH_KEYS})

    np.testing.assert_array_equal(_fingerprint(operator, padded), unpadded)
