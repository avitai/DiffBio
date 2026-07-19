"""Tests for fit_jointly joint-optimization mode (ticket 06)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from diffbio.pipelines.joint_preprocessing import (
    JointPreprocessingPipeline,
    JointPreprocessingPipelineConfig,
)
from diffbio.pipelines.joint_training import (
    JointTrainingConfig,
    JointTrainingResult,
    fit_jointly,
)
from diffbio.utils.training import cross_entropy_loss


def _structured(
    n_cells: int, n_genes: int, n_classes: int, seed: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n_cells)
    counts = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    markers = max(1, n_genes // (n_classes * 2))
    for class_index in range(n_classes):
        block = slice(class_index * markers, (class_index + 1) * markers)
        counts[labels == class_index, block] += 15.0
    return jnp.asarray(counts), jnp.asarray(labels)


def _pipeline(seed: int = 0, hvg_hard: bool = False) -> JointPreprocessingPipeline:
    config = JointPreprocessingPipelineConfig(
        n_genes=50, n_classes=3, n_top_genes=25, n_components=10, hvg_hard=hvg_hard
    )
    return JointPreprocessingPipeline(config, rngs=nnx.Rngs(seed))


# --- End-to-end gradient flow to every preprocessing parameter -------------------


def test_gradient_reaches_each_preprocessing_parameter() -> None:
    counts, labels = _structured(200, 50, 3, seed=0)
    pipeline = _pipeline()

    def classification(model: JointPreprocessingPipeline) -> jnp.ndarray:
        output, _, _ = model.apply({"counts": counts}, {}, None)
        return cross_entropy_loss(output["logits"], labels, num_classes=3)

    grads = nnx.grad(classification)(pipeline)
    norm_grads = grads["composite"]["operators"][0]
    gene_weight_grad = grads["composite"]["operators"][2]["gene_weights"][...]

    # Each learnable preprocessing parameter must receive a finite, non-zero
    # gradient. The gene-weight gradient is small (dispersion already ranks the
    # informative genes, so only a small correction is needed) but non-zero -- the
    # differentiable path exists, and the weights move measurably over training.
    assert float(jnp.abs(norm_grads["raw_pseudocount"][...])) > 1e-6
    assert float(jnp.abs(norm_grads["depth_exponent"][...])) > 1e-6
    assert bool(jnp.all(jnp.isfinite(gene_weight_grad)))
    assert float(jnp.linalg.norm(gene_weight_grad)) > 1e-7


# --- Training behavior ----------------------------------------------------------


def test_fit_jointly_converges_to_low_loss_and_high_accuracy() -> None:
    # Trained to completion, joint optimization drives the label loss near zero and
    # the pipeline to (near-)perfect annotation on class-structured data.
    counts, labels = _structured(200, 50, 3, seed=1)
    pipeline = _pipeline()
    result = fit_jointly(
        pipeline, counts, labels, config=JointTrainingConfig(n_steps=200, learning_rate=5e-2)
    )
    assert isinstance(result, JointTrainingResult)
    assert result.loss_history[-1] < 0.1
    logits = pipeline.apply({"counts": counts}, {}, None)[0]["logits"]
    accuracy = float(jnp.mean(jnp.argmax(logits, axis=-1) == labels))
    assert accuracy > 0.95


def test_training_is_numerically_stable() -> None:
    counts, labels = _structured(150, 50, 3, seed=2)
    pipeline = _pipeline()
    result = fit_jointly(pipeline, counts, labels, config=JointTrainingConfig(n_steps=40))
    assert all(np.isfinite(loss) for loss in result.loss_history)


def test_gradnorm_weights_adapt_from_unit() -> None:
    counts, labels = _structured(150, 50, 3, seed=3)
    pipeline = _pipeline()
    result = fit_jointly(pipeline, counts, labels, config=JointTrainingConfig(n_steps=40))
    # The balancer starts at unit weights; GradNorm must move them.
    assert len(result.final_loss_weights) == 2
    assert any(abs(weight - 1.0) > 1e-3 for weight in result.final_loss_weights)


def test_fit_jointly_updates_preprocessing_parameters() -> None:
    counts, labels = _structured(150, 50, 3, seed=4)
    pipeline = _pipeline()

    def snapshot() -> dict[str, float]:
        return {
            "pseudocount": float(pipeline.normalization.raw_pseudocount[...]),
            "depth": float(pipeline.normalization.depth_exponent[...]),
            "gene_weights": float(jnp.linalg.norm(pipeline.soft_hvg.gene_weights[...])),
        }

    before = snapshot()
    fit_jointly(pipeline, counts, labels, config=JointTrainingConfig(n_steps=40))
    after = snapshot()
    # Joint optimization unfreezes and moves every preprocessing parameter.
    assert abs(after["pseudocount"] - before["pseudocount"]) > 1e-4
    assert abs(after["depth"] - before["depth"]) > 1e-4
    assert abs(after["gene_weights"] - before["gene_weights"]) > 1e-4


def test_hard_mode_trains_stably() -> None:
    # The straight-through (frozen-parity) HVG mode also trains end to end.
    counts, labels = _structured(150, 50, 3, seed=5)
    pipeline = _pipeline(hvg_hard=True)
    result = fit_jointly(pipeline, counts, labels, config=JointTrainingConfig(n_steps=30))
    assert all(np.isfinite(loss) for loss in result.loss_history)
    assert result.loss_history[-1] < result.loss_history[0]


def test_fit_jointly_is_deterministic() -> None:
    counts, labels = _structured(120, 50, 3, seed=6)
    config = JointTrainingConfig(n_steps=25, seed=0)
    first = fit_jointly(_pipeline(seed=1), counts, labels, config=config)
    second = fit_jointly(_pipeline(seed=1), counts, labels, config=config)
    np.testing.assert_allclose(
        np.asarray(first.loss_history), np.asarray(second.loss_history), atol=1e-6
    )


# --- Frozen mode remains available and unchanged --------------------------------


def test_frozen_apply_is_unaffected_by_joint_training() -> None:
    counts, labels = _structured(120, 50, 3, seed=7)
    reference = _pipeline(seed=2)
    reference_logits = np.asarray(reference.apply({"counts": counts}, {}, None)[0]["logits"])

    trained = _pipeline(seed=2)
    fit_jointly(trained, counts, labels, config=JointTrainingConfig(n_steps=20))

    # Joint training mutates its own instance in place; the frozen apply path is
    # unchanged, so a freshly built pipeline still reproduces the reference logits
    # exactly (training one instance does not perturb the shared code path).
    fresh = _pipeline(seed=2)
    np.testing.assert_array_equal(
        np.asarray(fresh.apply({"counts": counts}, {}, None)[0]["logits"]), reference_logits
    )


# --- Config validation ----------------------------------------------------------


def test_config_rejects_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="n_steps"):
        JointTrainingConfig(n_steps=0)


def test_config_rejects_non_positive_learning_rate() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        JointTrainingConfig(learning_rate=0.0)


def test_config_rejects_non_positive_grad_clip_norm() -> None:
    with pytest.raises(ValueError, match="grad_clip_norm"):
        JointTrainingConfig(grad_clip_norm=0.0)


def test_pipeline_exposes_trainable_stage_accessors() -> None:
    from diffbio.operators.normalization.learnable_normalization import LearnableNormalization
    from diffbio.operators.singlecell.soft_hvg import SoftHVG

    pipeline = _pipeline()
    assert isinstance(pipeline.normalization, LearnableNormalization)
    assert isinstance(pipeline.soft_hvg, SoftHVG)
