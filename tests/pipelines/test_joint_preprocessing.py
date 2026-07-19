"""Tests for the JointPreprocessingPipeline and its Gate 1 parity (ticket 05)."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from datarax.operators import CompositeOperatorModule, CompositionStrategy
from flax import nnx

from benchmarks.singlecell.frozen_annotation_baseline import frozen_preprocess
from diffbio.pipelines.joint_preprocessing import (
    JointPreprocessingPipeline,
    JointPreprocessingPipelineConfig,
)


def _counts(n_cells: int, n_genes: int, seed: int) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32))


def _structured_counts(
    n_cells: int, n_genes: int, n_classes: int, seed: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Counts whose class identity is encoded in a block of marker genes."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n_cells)
    counts = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    markers = max(1, n_genes // (n_classes * 2))
    for class_index in range(n_classes):
        block = slice(class_index * markers, (class_index + 1) * markers)
        counts[labels == class_index, block] += 20.0
    return jnp.asarray(counts), jnp.asarray(labels)


def _pipeline(
    n_genes: int,
    n_classes: int,
    *,
    n_top_genes: int,
    n_components: int,
    seed: int = 0,
    hvg_hard: bool = True,
) -> JointPreprocessingPipeline:
    config = JointPreprocessingPipelineConfig(
        n_genes=n_genes,
        n_classes=n_classes,
        n_top_genes=n_top_genes,
        n_components=n_components,
        hvg_hard=hvg_hard,
    )
    return JointPreprocessingPipeline(config, rngs=nnx.Rngs(seed))


def _subspace_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Sign- and rotation-invariant distance between the column spaces of a and b.
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    return float(np.linalg.norm(qa @ qa.T - qb @ qb.T))


# --- Gate 1: frozen-mode parity with the scanpy baseline ------------------------


def test_frozen_features_match_baseline_subspace() -> None:
    counts = _counts(200, 60, seed=0)
    pipeline = _pipeline(60, 3, n_top_genes=25, n_components=10)
    output, _, _ = pipeline.apply({"counts": counts}, {}, None)

    baseline = frozen_preprocess(np.asarray(counts), n_top_genes=25, n_components=10)
    # The gated (not subset) PCA scores span the same subspace as the baseline's
    # subset PCA, so a linear probe sees the same separability (Gate 1).
    assert _subspace_distance(np.asarray(output["embeddings"]), baseline) < 1.0e-3


def test_frozen_features_match_baseline_per_component() -> None:
    counts = _counts(200, 60, seed=1)
    pipeline = _pipeline(60, 3, n_top_genes=25, n_components=10)
    output, _, _ = pipeline.apply({"counts": counts}, {}, None)
    ours = np.asarray(output["embeddings"])
    baseline = frozen_preprocess(np.asarray(counts), n_top_genes=25, n_components=10)
    for component in range(10):
        aligned = min(
            np.linalg.norm(ours[:, component] - baseline[:, component]),
            np.linalg.norm(ours[:, component] + baseline[:, component]),
        )
        assert aligned / (np.linalg.norm(baseline[:, component]) + 1e-9) < 1.0e-2


# --- Pipeline outputs -----------------------------------------------------------


def test_pipeline_produces_annotation_outputs() -> None:
    counts = _counts(120, 40, seed=2)
    pipeline = _pipeline(40, 4, n_top_genes=20, n_components=8)
    output, _, _ = pipeline.apply({"counts": counts}, {}, None)
    assert output["embeddings"].shape == (120, 8)
    assert output["logits"].shape == (120, 4)
    assert output["probabilities"].shape == (120, 4)
    assert output["predicted_labels"].shape == (120,)
    assert bool(jnp.all(jnp.isfinite(output["logits"])))


# --- End-to-end differentiability -----------------------------------------------


def _classification_loss(
    pipeline: JointPreprocessingPipeline,
    counts: jnp.ndarray,
    labels: jnp.ndarray,
) -> tuple[nnx.State, Callable[[nnx.State], jnp.ndarray]]:
    graphdef, params, rest = nnx.split(pipeline, nnx.Param, ...)
    n_cells = counts.shape[0]

    def loss(param_state: nnx.State) -> jnp.ndarray:
        model = nnx.merge(graphdef, param_state, rest)
        output, _, _ = model.apply({"counts": counts}, {}, None)
        log_probs = jax.nn.log_softmax(output["logits"])
        return -jnp.mean(log_probs[jnp.arange(n_cells), labels])

    return params, loss


def test_end_to_end_gradient_is_finite_and_reaches_preprocessing() -> None:
    counts = _counts(150, 40, seed=3)
    labels = jnp.asarray(np.random.default_rng(3).integers(0, 3, size=150))
    pipeline = _pipeline(40, 3, n_top_genes=20, n_components=8)
    params, loss = _classification_loss(pipeline, counts, labels)

    grads = jax.grad(loss)(params)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(grads))
    # Gradient must reach the normalization params (pseudocount, depth) through the
    # full HVG -> scale -> PCA -> probe stack, proving the pipeline is differentiable
    # end to end rather than NaN or zero upstream. These live on the first composite
    # child; assert both receive a substantial gradient.
    norm_grads = grads["composite"]["operators"][0]
    assert float(jnp.abs(norm_grads["raw_pseudocount"][...])) > 1e-4
    assert float(jnp.abs(norm_grads["depth_exponent"][...])) > 1e-4
    # The SoftHVG gene weights (third child) sit on a nonzero, finite gradient path
    # too; their magnitude is small because dispersion already ranks the informative
    # genes, and is tuned via the softness/mode in joint optimization.
    gene_weight_grad = grads["composite"]["operators"][2]["gene_weights"][...]
    assert bool(jnp.all(jnp.isfinite(gene_weight_grad)))
    assert float(jnp.linalg.norm(gene_weight_grad)) > 0.0


def test_pipeline_trains_end_to_end_via_gradient_descent() -> None:
    # The strongest differentiability check: gradient descent on every learnable
    # parameter of the full stack (normalization, gene weights, probe) drives the
    # annotation loss down on class-structured data -- the joint-optimization claim.
    counts, labels = _structured_counts(150, 40, 3, seed=4)
    pipeline = _pipeline(40, 3, n_top_genes=20, n_components=8, hvg_hard=False)
    params, loss = _classification_loss(pipeline, counts, labels)

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    initial_loss = float(loss(params))
    learning_rate = 0.1
    for _ in range(40):
        _, grads = value_and_grad(params)
        params = jax.tree.map(lambda p, g: p - learning_rate * g, params, grads)
    final_loss = float(loss(params))
    assert final_loss < 0.5 * initial_loss


def test_output_is_deterministic() -> None:
    counts = _counts(100, 30, seed=5)
    first = _pipeline(30, 3, n_top_genes=15, n_components=6, seed=7)
    second = _pipeline(30, 3, n_top_genes=15, n_components=6, seed=7)
    out_first, _, _ = first.apply({"counts": counts}, {}, None)
    out_second, _, _ = second.apply({"counts": counts}, {}, None)
    np.testing.assert_array_equal(
        np.asarray(out_first["embeddings"]), np.asarray(out_second["embeddings"])
    )


# --- Composition contract -------------------------------------------------------


def test_composition_is_sequential_composite() -> None:
    pipeline = _pipeline(30, 3, n_top_genes=15, n_components=6)
    assert isinstance(pipeline.composite, CompositeOperatorModule)
    assert pipeline.composite.config.strategy is CompositionStrategy.SEQUENTIAL
    operators = pipeline.composite.config.operators
    assert operators is not None
    # Norm, rename, SoftHVG, scaler, PCA, rename, probe.
    assert len(operators) == 7


def test_child_accessor_fails_fast_for_absent_operator() -> None:
    from diffbio.operators.normalization.umap import DifferentiableUMAP

    pipeline = _pipeline(30, 3, n_top_genes=15, n_components=6)
    with pytest.raises(ValueError, match="DifferentiableUMAP"):
        pipeline._child(DifferentiableUMAP)


def test_params_are_discoverable_for_joint_training() -> None:
    pipeline = _pipeline(30, 3, n_top_genes=15, n_components=6)
    params = nnx.state(pipeline, nnx.Param)
    # Normalization (2) + SoftHVG (1) + probe kernel/bias (2); PCA/scaler have none.
    assert len(jax.tree.leaves(params)) == 5


def test_apply_is_nnx_jit_compatible() -> None:
    counts = _counts(80, 30, seed=6)
    pipeline = _pipeline(30, 3, n_top_genes=15, n_components=6)

    @nnx.jit
    def run(module: JointPreprocessingPipeline, matrix: jnp.ndarray) -> jnp.ndarray:
        output, _, _ = module.apply({"counts": matrix}, {}, None)
        return output["logits"]

    logits = run(pipeline, counts)
    assert logits.shape == (80, 3)
    assert bool(jnp.all(jnp.isfinite(logits)))


# --- Config validation ----------------------------------------------------------


def test_config_rejects_non_positive_sizes() -> None:
    for field, value in (
        ("n_genes", 0),
        ("n_classes", 0),
        ("n_top_genes", -1),
        ("n_components", 0),
    ):
        with pytest.raises(ValueError, match=field):
            JointPreprocessingPipelineConfig(**{field: value})


def test_config_rejects_n_components_exceeding_n_genes() -> None:
    with pytest.raises(ValueError, match="n_components"):
        JointPreprocessingPipelineConfig(n_genes=10, n_components=50)
