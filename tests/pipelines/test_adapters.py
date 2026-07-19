"""Tests for pipeline composition adapters (ticket 05)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from diffbio.pipelines.adapters import RenameField, RenameFieldConfig


def _adapter(source: str, target: str) -> RenameField:
    return RenameField(RenameFieldConfig(source=source, target=target), rngs=nnx.Rngs(0))


def test_moves_source_value_to_target() -> None:
    value = jnp.arange(6.0)
    adapter = _adapter("normalized", "features")
    output, _, _ = adapter.apply({"normalized": value}, {}, None)
    np.testing.assert_array_equal(np.asarray(output["features"]), np.asarray(value))


def test_removes_source_key() -> None:
    adapter = _adapter("pca", "embeddings")
    output, _, _ = adapter.apply({"pca": jnp.ones(3)}, {}, None)
    assert "pca" not in output
    assert "embeddings" in output


def test_passes_through_other_keys() -> None:
    adapter = _adapter("normalized", "features")
    output, _, _ = adapter.apply(
        {"normalized": jnp.ones(3), "labels": jnp.arange(3), "counts": jnp.zeros(3)}, {}, None
    )
    assert "labels" in output
    assert "counts" in output


def test_missing_source_raises() -> None:
    adapter = _adapter("normalized", "features")
    with pytest.raises(KeyError, match="normalized"):
        adapter.apply({"counts": jnp.ones(3)}, {}, None)


def test_gradient_flows_through_renamed_value() -> None:
    adapter = _adapter("normalized", "features")

    def loss(x: jnp.ndarray) -> jnp.ndarray:
        output, _, _ = adapter.apply({"normalized": x}, {}, None)
        return jnp.sum(output["features"] ** 2)

    grad = jax.grad(loss)(jnp.arange(5.0))
    assert bool(jnp.all(jnp.isfinite(grad)))
    np.testing.assert_allclose(np.asarray(grad), np.asarray(2.0 * jnp.arange(5.0)), atol=1e-5)


def test_composes_under_nnx_jit() -> None:
    adapter = _adapter("normalized", "features")

    @nnx.jit
    def run(module: RenameField, x: jnp.ndarray) -> jnp.ndarray:
        output, _, _ = module.apply({"normalized": x}, {}, None)
        return output["features"]

    result = run(adapter, jnp.ones(4))
    np.testing.assert_array_equal(np.asarray(result), np.ones(4))


def test_config_rejects_empty_source() -> None:
    with pytest.raises(ValueError, match="source"):
        RenameFieldConfig(source="", target="features")


def test_config_rejects_empty_target() -> None:
    with pytest.raises(ValueError, match="target"):
        RenameFieldConfig(source="pca", target="")


def test_has_no_learnable_parameters() -> None:
    adapter = _adapter("pca", "embeddings")
    params = nnx.state(adapter, nnx.Param)
    assert len(jax.tree.leaves(params)) == 0
