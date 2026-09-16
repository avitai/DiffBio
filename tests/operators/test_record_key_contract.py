"""Stochastic operators draw from the record key their caller passes.

datarax 0.1.10 hands ``apply`` each record's PRNG key as its fourth argument, and a
stochastic operator draws everything it applies from that key: the same key gives the same
output, another key gives another, and no key is refused rather than replaced by a fixed
draw. These tests pin that contract for every DiffBio operator that draws randomness.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.foundation_models.foundation_model import (
    DifferentiableFoundationModel,
    FoundationModelConfig,
)
from diffbio.operators.multiomics.multiomics_vae import (
    DifferentiableMultiOmicsVAE,
    MultiOmicsVAEConfig,
)
from diffbio.operators.normalization.vae_normalizer import VAENormalizer, VAENormalizerConfig
from diffbio.operators.singlecell.ambient_removal import (
    AmbientRemovalConfig,
    DifferentiableAmbientRemoval,
)
from diffbio.operators.singlecell.doublet_detection import (
    DifferentiableDoubletScorer,
    DoubletScorerConfig,
)
from diffbio.operators.singlecell.downsampling import DownsamplingConfig, ReadDownsampler
from diffbio.operators.singlecell.imputation import (
    DifferentiableTransformerDenoiser,
    TransformerDenoiserConfig,
)
from diffbio.operators.singlecell.simulation import DifferentiableSimulator, SimulationConfig
from diffbio.operators.singlecell.stochastic_gate_selector import (
    StochasticGateSelector,
    StochasticGateSelectorConfig,
)

Case = tuple[str, Callable[[], Any], Callable[[], dict[str, Any]], str]


def _counts(shape: tuple[int, ...], seed: int = 0) -> jax.Array:
    return jnp.abs(jax.random.normal(jax.random.key(seed), shape)) * 5.0 + 0.1


def _downsampler() -> ReadDownsampler:
    config = DownsamplingConfig(
        mode="fraction", fraction=0.5, apply_log1p=False, is_log1p_input=False
    )
    return ReadDownsampler(config, rngs=nnx.Rngs(0))


def _gate_selector() -> StochasticGateSelector:
    config = StochasticGateSelectorConfig(
        n_genes=40, sigma=0.5, mu_init=0.5, stochastic=True, stream_name="gate_noise"
    )
    return StochasticGateSelector(config, rngs=nnx.Rngs(0, gate_noise=0))


def _vae_normalizer() -> VAENormalizer:
    return VAENormalizer(VAENormalizerConfig(n_genes=30, latent_dim=4), rngs=nnx.Rngs(0))


def _ambient_removal() -> DifferentiableAmbientRemoval:
    config = AmbientRemovalConfig(n_genes=30, latent_dim=8, hidden_dims=[16])
    return DifferentiableAmbientRemoval(config, rngs=nnx.Rngs(0))


def _multiomics_vae() -> DifferentiableMultiOmicsVAE:
    config = MultiOmicsVAEConfig(
        modality_dims=[20, 10], latent_dim=5, hidden_dim=16, modality_weight_mode="equal"
    )
    return DifferentiableMultiOmicsVAE(config, rngs=nnx.Rngs(0))


def _simulator() -> DifferentiableSimulator:
    config = SimulationConfig(n_cells=20, n_genes=15, n_groups=2, n_batches=1)
    return DifferentiableSimulator(config, rngs=nnx.Rngs(0))


def _doublet_scorer() -> DifferentiableDoubletScorer:
    config = DoubletScorerConfig(n_neighbors=5, n_pca_components=5, n_genes=20)
    return DifferentiableDoubletScorer(config, rngs=nnx.Rngs(0))


def _foundation_model() -> DifferentiableFoundationModel:
    config = FoundationModelConfig(
        n_genes=20, hidden_dim=16, num_layers=1, num_heads=2, mask_ratio=0.3, dropout_rate=0.0
    )
    return DifferentiableFoundationModel(config, rngs=nnx.Rngs(params=0, sample=1, dropout=2))


def _transformer_denoiser() -> DifferentiableTransformerDenoiser:
    config = TransformerDenoiserConfig(
        n_genes=20, hidden_dim=16, num_layers=1, num_heads=2, mask_ratio=0.3, dropout_rate=0.0
    )
    return DifferentiableTransformerDenoiser(config, rngs=nnx.Rngs(params=0, sample=1, dropout=2))


def _vae_data() -> dict[str, Any]:
    counts = _counts((30,))
    return {"counts": counts, "library_size": jnp.sum(counts)}


def _gene_data() -> dict[str, Any]:
    return {"counts": _counts((6, 20)), "gene_ids": jnp.arange(20, dtype=jnp.int32)}


CASES: list[Case] = [
    ("ReadDownsampler", _downsampler, lambda: {"counts": _counts((3, 8))}, "counts"),
    ("StochasticGateSelector", _gate_selector, lambda: {"features": _counts((2, 40))}, "gate"),
    ("VAENormalizer", _vae_normalizer, _vae_data, "normalized"),
    (
        "DifferentiableAmbientRemoval",
        _ambient_removal,
        lambda: {
            "counts": _counts((10, 30)),
            "ambient_profile": jax.nn.softmax(jax.random.normal(jax.random.key(1), (30,))),
        },
        "latent",
    ),
    (
        "DifferentiableMultiOmicsVAE",
        _multiomics_vae,
        lambda: {"rna_counts": _counts((6, 20)), "atac_counts": _counts((6, 10), 1)},
        "joint_latent",
    ),
    ("DifferentiableSimulator", _simulator, dict, "counts"),
    (
        "DifferentiableDoubletScorer",
        _doublet_scorer,
        lambda: {"counts": _counts((30, 20))},
        "doublet_scores",
    ),
    ("DifferentiableFoundationModel", _foundation_model, _gene_data, "predicted_expression"),
    ("DifferentiableTransformerDenoiser", _transformer_denoiser, _gene_data, "mask"),
]


def _field(result: dict[str, Any], name: str) -> jax.Array:
    return jnp.asarray(result[name])


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_the_same_key_gives_the_same_output(case: Case) -> None:
    name, build, data, field = case
    operator = build()
    first, _, _ = operator.apply(data(), {}, None, jax.random.key(7))
    again, _, _ = operator.apply(data(), {}, None, jax.random.key(7))
    assert jnp.array_equal(_field(first, field), _field(again, field)), name


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_another_key_gives_another_output(case: Case) -> None:
    name, build, data, field = case
    operator = build()
    first, _, _ = operator.apply(data(), {}, None, jax.random.key(7))
    other, _, _ = operator.apply(data(), {}, None, jax.random.key(8))
    assert not jnp.array_equal(_field(first, field), _field(other, field)), name


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_a_missing_key_is_refused(case: Case) -> None:
    name, build, data, _ = case
    operator = build()
    with pytest.raises(ValueError, match=name):
        operator.apply(data(), {}, None)
