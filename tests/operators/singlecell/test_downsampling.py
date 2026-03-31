"""Tests for ReadDownsampler."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from diffbio.operators.singlecell.downsampling import (
    DownsamplingConfig,
    ReadDownsampler,
)


class TestReadDownsampler:
    """Tests for ReadDownsampler."""

    def test_fraction_reduces_counts(self) -> None:
        config = DownsamplingConfig(
            mode="fraction", fraction=0.5, apply_log1p=False, is_log1p_input=False
        )
        op = ReadDownsampler(config, rngs=nnx.Rngs(0))
        data = {"counts": jnp.array([[100.0, 200.0, 50.0]])}
        result, _, _ = op.apply(data, {}, None)
        # Downsampled counts should be lower on average
        assert float(result["counts"].sum()) < float(data["counts"].sum())

    def test_fraction_one_preserves_counts(self) -> None:
        config = DownsamplingConfig(
            mode="fraction", fraction=1.0, apply_log1p=False, is_log1p_input=False
        )
        op = ReadDownsampler(config, rngs=nnx.Rngs(0))
        data = {"counts": jnp.array([[10.0, 20.0, 30.0]])}
        result, _, _ = op.apply(data, {}, None)
        np.testing.assert_allclose(result["counts"], data["counts"], atol=1e-5)

    def test_target_depth_mode(self) -> None:
        config = DownsamplingConfig(
            mode="target_depth",
            target_depth=100,
            apply_log1p=False,
            is_log1p_input=False,
        )
        op = ReadDownsampler(config, rngs=nnx.Rngs(0))
        # Cell with 1000 total reads -> target 100 -> fraction ~0.1
        data = {"counts": jnp.array([[500.0, 300.0, 200.0]])}
        result, _, _ = op.apply(data, {}, None)
        total = float(result["counts"].sum())
        # Should be roughly around 100 (stochastic)
        assert total < 500

    def test_log1p_handling(self) -> None:
        config = DownsamplingConfig(
            mode="fraction",
            fraction=0.5,
            apply_log1p=True,
            is_log1p_input=True,
        )
        op = ReadDownsampler(config, rngs=nnx.Rngs(0))
        raw_counts = jnp.array([[100.0, 200.0, 50.0]])
        log_counts = jnp.log1p(raw_counts)
        data = {"counts": log_counts}
        result, _, _ = op.apply(data, {}, None)
        # Output should be log1p-transformed (non-negative, smaller than input)
        assert jnp.all(result["counts"] >= 0)

    def test_differentiable(self) -> None:
        config = DownsamplingConfig(
            mode="fraction", fraction=0.5, apply_log1p=False, is_log1p_input=False
        )
        op = ReadDownsampler(config, rngs=nnx.Rngs(0))

        def loss_fn(operator: ReadDownsampler, counts: jnp.ndarray) -> jnp.ndarray:
            data = {"counts": counts}
            result, _, _ = operator.apply(data, {}, None)
            return result["counts"].sum()

        counts = jnp.array([[100.0, 200.0, 50.0]])
        # Should not raise
        grads = jax.grad(loss_fn, argnums=1)(op, counts)
        assert grads is not None
        assert grads.shape == counts.shape

    def test_preserves_other_keys(self) -> None:
        config = DownsamplingConfig(
            mode="fraction", fraction=0.5, apply_log1p=False, is_log1p_input=False
        )
        op = ReadDownsampler(config, rngs=nnx.Rngs(0))
        data = {
            "counts": jnp.array([[10.0, 20.0]]),
            "pert_code": 1,
            "other": "preserved",
        }
        result, _, _ = op.apply(data, {}, None)
        assert result["pert_code"] == 1
        assert result["other"] == "preserved"

    def test_jit_compatible(self) -> None:
        """Test JIT compilation works for ReadDownsampler."""
        config = DownsamplingConfig(
            mode="fraction", fraction=0.5, apply_log1p=False, is_log1p_input=False
        )
        op = ReadDownsampler(config, rngs=nnx.Rngs(0))

        @jax.jit
        def compute(operator: ReadDownsampler, counts: jnp.ndarray) -> jnp.ndarray:
            data = {"counts": counts}
            result, _, _ = operator.apply(data, {}, None)
            return result["counts"]

        counts = jnp.array([[100.0, 200.0, 50.0]])
        result = compute(op, counts)
        assert result.shape == counts.shape
        assert jnp.all(jnp.isfinite(result))

    def test_non_negative_output(self) -> None:
        config = DownsamplingConfig(
            mode="fraction", fraction=0.3, apply_log1p=False, is_log1p_input=False
        )
        op = ReadDownsampler(config, rngs=nnx.Rngs(42))
        data = {"counts": jnp.array([[50.0, 100.0, 200.0, 10.0, 5.0]])}
        result, _, _ = op.apply(data, {}, None)
        assert jnp.all(result["counts"] >= 0)
