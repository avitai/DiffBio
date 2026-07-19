"""Tests for the LearnablePileup operator.

Test-driven: these define the contract. The central property is *init-at-frozen* --
at initialization the learnable pileup must reproduce the frozen DeepVariant pileup
exactly, so joint optimization starts at the hand-designed baseline and can only
improve. The learnable encoding parameters (base embedding + per-channel affine maps)
must then receive gradients and be the operator's only trainable state.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from diffbio.operators.variant.deepvariant_pileup import (
    DeepVariantPileupConfig,
    DeepVariantStylePileup,
)
from diffbio.operators.variant.learnable_pileup import LearnablePileup


def _sample_data(num_reads: int = 12, read_length: int = 20, window_size: int = 60):
    """Build a small but realistic set of pileup inputs from fixed RNG seeds."""
    keys = jax.random.split(jax.random.PRNGKey(0), 6)
    reads = jax.nn.one_hot(jax.random.randint(keys[0], (num_reads, read_length), 0, 4), 4).astype(
        jnp.float32
    )
    reference = jax.nn.one_hot(jax.random.randint(keys[1], (window_size,), 0, 4), 4).astype(
        jnp.float32
    )
    qualities = jax.random.uniform(keys[2], (num_reads, read_length), minval=10.0, maxval=40.0)
    mapq = jax.random.uniform(keys[3], (num_reads,), minval=20.0, maxval=60.0)
    strands = jax.random.bernoulli(keys[4], 0.5, (num_reads,)).astype(jnp.float32)
    positions = jax.random.randint(keys[5], (num_reads,), 0, window_size - read_length)
    return {
        "reads": reads,
        "reference": reference,
        "base_qualities": qualities,
        "mapping_qualities": mapq,
        "strands": strands,
        "positions": positions,
    }


def _config(**overrides) -> DeepVariantPileupConfig:
    base = {"window_size": 60, "max_reads": 16}
    base.update(overrides)
    return DeepVariantPileupConfig(**base)


# --- Init-at-frozen: reproduces the DeepVariant pileup exactly -------------------


def test_init_reproduces_frozen_pileup_all_channels() -> None:
    config = _config()
    data = _sample_data(window_size=config.window_size)
    frozen = DeepVariantStylePileup(config, rngs=nnx.Rngs(0))
    learnable = LearnablePileup(config, rngs=nnx.Rngs(0))

    frozen_image = frozen.apply(data, {}, None)[0]["pileup_image"]
    learnable_image = learnable.apply(data, {}, None)[0]["pileup_image"]

    np.testing.assert_allclose(np.asarray(learnable_image), np.asarray(frozen_image), atol=1e-5)


@pytest.mark.parametrize(
    "channels",
    [
        ("base",),
        ("base_quality",),
        ("mapping_quality",),
        ("strand",),
        ("supports_variant",),
        ("differs_from_ref",),
        ("base", "base_quality", "strand"),
    ],
)
def test_init_reproduces_frozen_per_channel_subset(channels) -> None:
    config = _config(channels=channels)
    data = _sample_data(window_size=config.window_size)
    frozen_image = DeepVariantStylePileup(config, rngs=nnx.Rngs(1)).apply(data, {}, None)[0][
        "pileup_image"
    ]
    learnable_image = LearnablePileup(config, rngs=nnx.Rngs(1)).apply(data, {}, None)[0][
        "pileup_image"
    ]
    np.testing.assert_allclose(np.asarray(learnable_image), np.asarray(frozen_image), atol=1e-5)


def test_base_embedding_initialized_to_identity() -> None:
    learnable = LearnablePileup(_config(), rngs=nnx.Rngs(0))
    np.testing.assert_allclose(np.asarray(learnable.base_embedding[...]), np.eye(4), atol=0.0)


def test_scalar_affines_initialized_to_gain_one_bias_zero() -> None:
    learnable = LearnablePileup(_config(), rngs=nnx.Rngs(0))
    for affine in (
        learnable.quality_affine,
        learnable.mapq_affine,
        learnable.strand_affine,
        learnable.mismatch_affine,
    ):
        np.testing.assert_array_equal(np.asarray(affine[...]), np.array([1.0, 0.0]))


# --- Output contract ------------------------------------------------------------


def test_output_shape_and_channels() -> None:
    config = _config()
    learnable = LearnablePileup(config, rngs=nnx.Rngs(0))
    assert learnable.num_channels == 9
    result = learnable.apply(_sample_data(window_size=config.window_size), {}, None)[0]
    assert result["pileup_image"].shape == (config.max_reads, config.window_size, 9)
    assert bool(jnp.all(jnp.isfinite(result["pileup_image"])))


def test_preserves_input_data() -> None:
    config = _config()
    data = _sample_data(window_size=config.window_size)
    result = LearnablePileup(config, rngs=nnx.Rngs(0)).apply(data, {}, None)[0]
    assert "reads" in result and "reference" in result
    assert jnp.array_equal(result["reads"], data["reads"])


# --- Learnability: encoding params are trainable and on the gradient path -------


def test_only_encoding_params_are_trainable() -> None:
    learnable = LearnablePileup(_config(), rngs=nnx.Rngs(0))
    params = nnx.state(learnable, nnx.Param)
    # base_embedding + four scalar affines = five trainable leaves, nothing else.
    assert len(jax.tree.leaves(params)) == 5


def test_gradient_flows_to_all_encoding_params() -> None:
    config = _config()
    data = _sample_data(window_size=config.window_size)
    learnable = LearnablePileup(config, rngs=nnx.Rngs(0))
    graphdef, params, rest = nnx.split(learnable, nnx.Param, ...)

    def loss(param_state: nnx.State) -> jnp.ndarray:
        model = nnx.merge(graphdef, param_state, rest)
        image = model.apply(data, {}, None)[0]["pileup_image"]
        # Square so the mismatch-channel gradient does not vanish at the frozen values.
        return jnp.sum(image**2)

    grads = jax.grad(loss)(params)
    leaves = jax.tree.leaves(grads)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    # every encoding parameter receives a non-zero gradient.
    assert all(float(jnp.linalg.norm(leaf)) > 0.0 for leaf in leaves)


def test_training_step_changes_pileup_away_from_frozen() -> None:
    config = _config()
    data = _sample_data(window_size=config.window_size)
    frozen_image = DeepVariantStylePileup(config, rngs=nnx.Rngs(0)).apply(data, {}, None)[0][
        "pileup_image"
    ]
    learnable = LearnablePileup(config, rngs=nnx.Rngs(0))
    graphdef, params, rest = nnx.split(learnable, nnx.Param, ...)

    def loss(param_state: nnx.State) -> jnp.ndarray:
        model = nnx.merge(graphdef, param_state, rest)
        return jnp.sum(model.apply(data, {}, None)[0]["pileup_image"] ** 2)

    grads = jax.grad(loss)(params)
    updated = jax.tree.map(lambda p, g: p + 0.1 * g, params, grads)
    moved_image = nnx.merge(graphdef, updated, rest).apply(data, {}, None)[0]["pileup_image"]
    # after one gradient step the encoding no longer matches the frozen baseline.
    assert float(jnp.linalg.norm(moved_image - frozen_image)) > 0.0


# --- Transform compatibility ----------------------------------------------------


def test_jit_compatible() -> None:
    config = _config()
    data = _sample_data(window_size=config.window_size)
    learnable = LearnablePileup(config, rngs=nnx.Rngs(0))

    @nnx.jit
    def run(module: LearnablePileup, batch: dict) -> jnp.ndarray:
        return module.apply(batch, {}, None)[0]["pileup_image"]

    image = run(learnable, data)
    assert image.shape == (config.max_reads, config.window_size, 9)
    assert bool(jnp.all(jnp.isfinite(image)))


def test_differentiable_through_reads() -> None:
    config = _config()
    data = _sample_data(window_size=config.window_size)
    learnable = LearnablePileup(config, rngs=nnx.Rngs(0))

    def loss_fn(reads: jnp.ndarray) -> jnp.ndarray:
        result = learnable.apply({**data, "reads": reads}, {}, None)[0]
        return jnp.sum(result["pileup_image"])

    grad = jax.grad(loss_fn)(data["reads"])
    assert grad.shape == data["reads"].shape
    assert bool(jnp.all(jnp.isfinite(grad)))
