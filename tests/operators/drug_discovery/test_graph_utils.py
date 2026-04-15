"""Tests for shared graph utility helpers used by drug-discovery operators."""

from dataclasses import dataclass

import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.drug_discovery._graph_utils import (
    build_optional_dropout,
    initialize_graph_encoder_from_config,
)
from diffbio.operators.drug_discovery.message_passing import StackedMessagePassing


class _DummyModule:
    """Minimal object compatible with graph utility initialization helpers."""

    def __init__(self) -> None:
        self._unique_id = "dummy"


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
