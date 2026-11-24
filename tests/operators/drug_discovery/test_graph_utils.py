"""Tests for shared graph utility helpers used by drug-discovery operators."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from diffbio.operators.drug_discovery._graph_utils import (
    build_optional_dropout,
    graph_readout,
    initialize_graph_encoder_from_config,
)
from diffbio.operators.drug_discovery.message_passing import StackedMessagePassing


class _DummyModule:
    """Minimal object compatible with graph utility initialization helpers."""


@dataclass(frozen=True)
class _MessagePassingConfig:
    hidden_dim: int = 8
    num_message_passing_steps: int = 2
    in_features: int = 4
    num_edge_features: int = 3


@dataclass(frozen=True)
class _FingerprintLikeConfig:
    hidden_dim: int = 10
    num_layers: int = 3
    in_features: int = 4


@dataclass(frozen=True)
class _IncompleteConfig:
    hidden_dim: int = 8
    in_features: int = 4


def test_initialize_graph_encoder_from_config_supports_message_passing_configs():
    """Initialize a standard encoder from configs using num_message_passing_steps."""
    module = _DummyModule()
    config = _MessagePassingConfig()

    initialize_graph_encoder_from_config(module, config, rngs=nnx.Rngs(0))

    assert isinstance(module.encoder, StackedMessagePassing)

    encoded = module.encoder(
        jnp.ones((5, config.in_features)),
        jnp.eye(5),
        jnp.ones((5, 5, config.num_edge_features)),
    )

    assert encoded.shape == (5, config.hidden_dim)


def test_initialize_graph_encoder_from_config_supports_custom_layer_fields():
    """Initialize an encoder from configs that store layer count under num_layers."""
    module = _DummyModule()
    config = _FingerprintLikeConfig()

    initialize_graph_encoder_from_config(
        module,
        config,
        rngs=nnx.Rngs(1),
        num_layers_attr="num_layers",
        attr="message_passing",
    )

    assert isinstance(module.message_passing, StackedMessagePassing)

    encoded = module.message_passing(
        jnp.ones((4, config.in_features)),
        jnp.eye(4),
    )

    assert encoded.shape == (4, config.hidden_dim)


def test_initialize_graph_encoder_from_config_fails_fast_for_missing_fields():
    """Reject configs that do not declare the required layer-count field."""
    module = _DummyModule()

    with pytest.raises(AttributeError, match="num_message_passing_steps"):
        initialize_graph_encoder_from_config(module, _IncompleteConfig(), rngs=nnx.Rngs(2))


def test_build_optional_dropout_omits_zero_rate():
    """Skip dropout modules when the configured rate is non-positive."""
    assert build_optional_dropout(0.0, rngs=nnx.Rngs(3)) is None


def test_build_optional_dropout_builds_positive_rate_dropout():
    """Create a reusable dropout module for positive rates."""
    dropout = build_optional_dropout(0.25, rngs=nnx.Rngs(4))

    assert isinstance(dropout, nnx.Dropout)


# --- graph_readout -----------------------------------------------------------------------------
#
# The readout is the only route by which molecule size reaches a graph-level prediction: nothing
# in `get_atom_features` encodes an atom count, and message passing is local -- messages are
# masked by the adjacency and summed over neighbors (`message_passing.py:126-129`), with
# `num_nodes` used only for broadcasting. So `sum` pooling makes the output extensive in the atom
# count and `mean` pooling makes it intensive, and these tests pin that distinction.
#
# Exactness is by construction wherever it is claimed: atom features are one-hot, the adjacency
# and edge features are 0/1, and padded pairs contribute terms that are exactly zero, so adding
# them changes nothing in float32. The duplication test is the one place a tolerance appears,
# because it compares sums taken in different associations; its tolerance is derived in situ.

_HIDDEN_DIM = 8
_NUM_LAYERS = 2
_IN_FEATURES = 4
_NUM_EDGE_FEATURES = 3


def _readout_encoder(seed: int = 0) -> StackedMessagePassing:
    """Build a small encoder whose node states depend only on each atom's neighborhood."""
    return StackedMessagePassing(
        hidden_dim=_HIDDEN_DIM,
        num_layers=_NUM_LAYERS,
        in_features=_IN_FEATURES,
        num_edge_features=_NUM_EDGE_FEATURES,
        rngs=nnx.Rngs(seed),
    )


def _chain_graph(num_atoms: int, num_rows: int) -> dict[str, jnp.ndarray]:
    """Build one bonded chain of `num_atoms` atoms, padded out to `num_rows` rows."""
    adjacency = jnp.zeros((num_rows, num_rows), dtype=jnp.float32)
    bonded = jnp.arange(num_atoms - 1)
    adjacency = adjacency.at[bonded, bonded + 1].set(1.0).at[bonded + 1, bonded].set(1.0)
    return {
        "node_features": jnp.eye(num_rows, _IN_FEATURES, dtype=jnp.float32),
        "adjacency": adjacency,
        "edge_features": jnp.ones((num_rows, num_rows, _NUM_EDGE_FEATURES), dtype=jnp.float32),
        "node_mask": (jnp.arange(num_rows) < num_atoms).astype(jnp.float32),
    }


def _duplicated_graph(num_atoms: int) -> dict[str, jnp.ndarray]:
    """Build two disconnected copies of one chain: twice the atoms, identical node states."""
    single = _chain_graph(num_atoms, num_atoms)
    adjacency = jnp.zeros((2 * num_atoms, 2 * num_atoms), dtype=jnp.float32)
    adjacency = adjacency.at[:num_atoms, :num_atoms].set(single["adjacency"])
    adjacency = adjacency.at[num_atoms:, num_atoms:].set(single["adjacency"])
    return {
        "node_features": jnp.concatenate([single["node_features"]] * 2, axis=0),
        "adjacency": adjacency,
        "edge_features": jnp.ones(
            (2 * num_atoms, 2 * num_atoms, _NUM_EDGE_FEATURES), dtype=jnp.float32
        ),
        "node_mask": jnp.ones((2 * num_atoms,), dtype=jnp.float32),
    }


def _node_states(graph: dict[str, jnp.ndarray], encoder: StackedMessagePassing) -> jnp.ndarray:
    """Run the encoder exactly as the readout does, for an independent reference."""
    return encoder(graph["node_features"], graph["adjacency"], graph["edge_features"])


def test_graph_readout_sums_only_the_unmasked_atoms():
    """Sum pooling reproduces the masked sum, so padding takes no part in it."""
    encoder = _readout_encoder()
    graph = _chain_graph(num_atoms=3, num_rows=7)

    states = _node_states(graph, encoder)
    expected = jnp.sum(states * graph["node_mask"][:, None], axis=0)

    np.testing.assert_array_equal(graph_readout(graph, encoder, aggregation="sum"), expected)


def test_graph_readout_means_over_the_mask_and_not_the_padded_width():
    """Mean pooling divides by the mask's own sum, never by the padded row count."""
    encoder = _readout_encoder()
    graph = _chain_graph(num_atoms=3, num_rows=7)

    states = _node_states(graph, encoder)
    masked_total = jnp.sum(states * graph["node_mask"][:, None], axis=0)
    over_the_mask = masked_total / jnp.sum(graph["node_mask"])
    over_the_padded_width = masked_total / graph["node_mask"].shape[0]

    actual = graph_readout(graph, encoder, aggregation="mean")

    np.testing.assert_array_equal(actual, over_the_mask)
    assert not bool(jnp.allclose(over_the_mask, over_the_padded_width))


def test_graph_readout_ignores_whatever_sits_in_the_padded_rows():
    """Changing the padded rows changes nothing: this is padding invariance, stated exactly.

    Held at one array shape, so the comparison is exact. Padded rows reach the output by two
    routes and both are closed: the readout masks them out, and they cannot influence a real
    atom's state because every pad-to-real entry of the adjacency is zero.
    """
    encoder = _readout_encoder()
    graph = _chain_graph(num_atoms=3, num_rows=11)
    occupied_padding = {
        **graph,
        "node_features": graph["node_features"].at[3:, :].set(7.0),
    }

    for aggregation in ("sum", "mean"):
        np.testing.assert_array_equal(
            graph_readout(occupied_padding, encoder, aggregation=aggregation),
            graph_readout(graph, encoder, aggregation=aggregation),
        )


def test_mean_readout_is_unchanged_by_how_far_the_graph_is_padded():
    """The same molecule at two padded widths agrees to float32 reassociation, not to the bit.

    The tolerance here is an allowance and not a derived bound, which is why it is labelled as
    one. Widening the array changes the *shape* of every reduction inside the encoder, so XLA
    groups the same nonzero terms differently and the result moves by a few float32 ULP --
    measured at 3.58e-07 absolute and 2.41e-07 relative on states of magnitude ~1.48, which is
    2-3 ULP there. Adding exactly-zero terms is exact arithmetic but it is not an exact
    reduction tree. rtol=1e-05 is roughly 84 ULP at that magnitude: loose enough to absorb the
    association, tight enough that a divisor taken over the padded width instead of the mask
    (11/3, a factor of 3.7) could not hide in it. The exact statements about padding live in
    the three tests above, which assert bit equality and get it.
    """
    encoder = _readout_encoder()

    unpadded = graph_readout(_chain_graph(num_atoms=3, num_rows=3), encoder, aggregation="mean")
    padded = graph_readout(_chain_graph(num_atoms=3, num_rows=11), encoder, aggregation="mean")

    np.testing.assert_allclose(padded, unpadded, rtol=1e-05, atol=1e-06)


def test_sum_readout_is_extensive_and_mean_readout_is_intensive():
    """Doubling the molecule doubles a sum readout and leaves a mean readout alone.

    Two disconnected copies of one chain have identical per-atom states, so this separates the
    two poolings by exactly the factor the choice is about. The tolerance is derived, not
    measured: each readout sums at most 8 float32 terms, whose recursive-summation relative
    error is bounded by n * eps = 8 * 1.19e-07 = 9.5e-07, so rtol=1e-05 sits above that bound
    and five orders of magnitude below the factor of two under test. atol covers the exact
    zeros a ReLU leaves in the states.
    """
    encoder = _readout_encoder()
    single = _chain_graph(num_atoms=4, num_rows=4)
    doubled = _duplicated_graph(num_atoms=4)

    single_sum = graph_readout(single, encoder, aggregation="sum")
    doubled_sum = graph_readout(doubled, encoder, aggregation="sum")
    single_mean = graph_readout(single, encoder, aggregation="mean")
    doubled_mean = graph_readout(doubled, encoder, aggregation="mean")

    np.testing.assert_allclose(doubled_sum, 2.0 * single_sum, rtol=1e-05, atol=1e-06)
    np.testing.assert_allclose(doubled_mean, single_mean, rtol=1e-05, atol=1e-06)


def test_graph_readout_returns_zeros_for_a_graph_with_no_unmasked_atoms():
    """An all-padding graph reads out as zeros rather than as a division by zero.

    `jnp.mean(..., where=)` is the standard-library masked mean and divides by the mask's sum
    (`jax/_src/numpy/reductions.py:889,913,921`), but it has no guard for an empty mask -- unlike
    `_var`, which returns NaN deliberately at `:1162` -- so 0/0 would reach the caller as NaN.
    chemprop clamps the divisor (`sums / counts.clamp(min=1)`) and jraph takes
    `maximum(count, 1)`; this asserts that behavior.
    """
    encoder = _readout_encoder()
    graph = _chain_graph(num_atoms=1, num_rows=5)
    empty = {**graph, "node_mask": jnp.zeros((5,), dtype=jnp.float32)}

    for aggregation in ("sum", "mean"):
        actual = graph_readout(empty, encoder, aggregation=aggregation)
        np.testing.assert_array_equal(actual, jnp.zeros((_HIDDEN_DIM,), dtype=jnp.float32))


def test_mean_readout_has_finite_gradients_through_an_all_padding_graph():
    """The clamped divisor keeps reverse-mode gradients finite where 0/0 would poison them."""
    encoder = _readout_encoder()
    graph = _chain_graph(num_atoms=1, num_rows=5)
    empty = {**graph, "node_mask": jnp.zeros((5,), dtype=jnp.float32)}

    def objective(node_features: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(
            graph_readout({**empty, "node_features": node_features}, encoder, aggregation="mean")
        )

    gradient = jax.grad(objective)(empty["node_features"])

    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_graph_readout_rejects_an_unknown_aggregation():
    """Fail fast on a pooling mode this readout does not implement."""
    encoder = _readout_encoder()
    graph = _chain_graph(num_atoms=3, num_rows=7)

    with pytest.raises(ValueError, match="aggregation"):
        graph_readout(graph, encoder, aggregation="max")  # type: ignore[arg-type]
