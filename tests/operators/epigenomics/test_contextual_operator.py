"""Tests for the contextual epigenomics operator path."""

from __future__ import annotations

from typing import Any, Literal, cast

import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from diffbio.operators.epigenomics.contextual import (
    ContextualEpigenomicsConfig,
    ContextualEpigenomicsOperator,
    compute_contextual_epigenomics_loss,
)
from diffbio.sources.contextual_epigenomics import (
    build_synthetic_contextual_epigenomics_dataset,
)


def _make_batch(
    *,
    target_semantics: Literal["binary_peak_mask", "chromatin_state_id"] = "binary_peak_mask",
    num_outputs: int = 1,
) -> dict[str, jnp.ndarray]:
    """Build a deterministic contextual epigenomics batch."""
    return build_synthetic_contextual_epigenomics_dataset(
        n_examples=4,
        sequence_length=20,
        num_tf_features=3,
        target_semantics=target_semantics,
        num_output_classes=num_outputs,
    )


class TestContextualEpigenomicsOperator:
    """Tests for one configurable contextual epigenomics operator path."""

    def test_invalid_dropout_rate_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="dropout_rate"):
            ContextualEpigenomicsConfig(
                hidden_dim=16,
                num_layers=1,
                num_heads=2,
                intermediate_dim=32,
                max_length=20,
                num_tf_features=3,
                num_outputs=1,
                dropout_rate=1.5,
            )

    def test_invalid_head_divisibility_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="hidden_dim"):
            ContextualEpigenomicsConfig(
                hidden_dim=15,
                num_layers=1,
                num_heads=2,
                intermediate_dim=32,
                max_length=20,
                num_tf_features=3,
                num_outputs=1,
            )

    def test_sequence_only_mode_ignores_tf_conditioning(self, rngs: nnx.Rngs) -> None:
        data = _make_batch()
        config = ContextualEpigenomicsConfig(
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            intermediate_dim=32,
            max_length=20,
            num_tf_features=3,
            num_outputs=1,
            use_tf_context=False,
            use_chromatin_guidance=False,
        )
        operator = ContextualEpigenomicsOperator(config, rngs=rngs)

        variant_a = dict(data)
        variant_b = dict(data)
        variant_b["tf_context"] = jnp.roll(data["tf_context"], shift=1, axis=1)

        logits_a = operator.apply(variant_a, {}, None)[0]["logits"]
        logits_b = operator.apply(variant_b, {}, None)[0]["logits"]

        assert logits_a.shape == (4, 20)
        assert jnp.allclose(logits_a, logits_b)

    def test_tf_conditioning_changes_outputs_when_enabled(self, rngs: nnx.Rngs) -> None:
        data = _make_batch()
        config = ContextualEpigenomicsConfig(
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            intermediate_dim=32,
            max_length=20,
            num_tf_features=3,
            num_outputs=1,
            use_tf_context=True,
            use_chromatin_guidance=False,
        )
        operator = ContextualEpigenomicsOperator(config, rngs=rngs)

        variant_a = dict(data)
        variant_b = dict(data)
        variant_b["tf_context"] = jnp.roll(data["tf_context"], shift=1, axis=1)

        logits_a = operator.apply(variant_a, {}, None)[0]["logits"]
        logits_b = operator.apply(variant_b, {}, None)[0]["logits"]

        assert logits_a.shape == (4, 20)
        assert not jnp.allclose(logits_a, logits_b)

    def test_masked_positions_are_ignored_by_chromatin_guidance(self, rngs: nnx.Rngs) -> None:
        data = _make_batch()
        sequence_mask = jnp.ones((4, 20), dtype=jnp.float32)
        sequence_mask = sequence_mask.at[:, -4:].set(0.0)
        data["sequence_mask"] = sequence_mask

        config = ContextualEpigenomicsConfig(
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            intermediate_dim=32,
            max_length=20,
            num_tf_features=3,
            num_outputs=1,
            use_tf_context=True,
            use_chromatin_guidance=True,
            chromatin_guidance_weight=0.5,
        )
        operator = ContextualEpigenomicsOperator(config, rngs=rngs)

        variant_a = dict(data)
        variant_b = dict(data)
        variant_b["chromatin_contacts"] = data["chromatin_contacts"].at[:, -4:, :].set(1.0)
        variant_b["chromatin_contacts"] = variant_b["chromatin_contacts"].at[:, :, -4:].set(1.0)

        result_a = operator.apply(variant_a, {}, None)[0]
        result_b = operator.apply(variant_b, {}, None)[0]

        assert result_a["chromatin_guidance_loss"] >= 0.0
        assert jnp.allclose(
            result_a["chromatin_guidance_loss"],
            result_b["chromatin_guidance_loss"],
            atol=1e-6,
        )

    def test_loss_toggles_follow_ablation_mode(self, rngs: nnx.Rngs) -> None:
        data = _make_batch()

        without_guidance = ContextualEpigenomicsOperator(
            ContextualEpigenomicsConfig(
                hidden_dim=16,
                num_layers=1,
                num_heads=2,
                intermediate_dim=32,
                max_length=20,
                num_tf_features=3,
                num_outputs=1,
                use_tf_context=True,
                use_chromatin_guidance=False,
            ),
            rngs=rngs,
        )
        with_guidance = ContextualEpigenomicsOperator(
            ContextualEpigenomicsConfig(
                hidden_dim=16,
                num_layers=1,
                num_heads=2,
                intermediate_dim=32,
                max_length=20,
                num_tf_features=3,
                num_outputs=1,
                use_tf_context=True,
                use_chromatin_guidance=True,
                chromatin_guidance_weight=0.5,
            ),
            rngs=nnx.Rngs(42),
        )

        no_guidance_losses = compute_contextual_epigenomics_loss(without_guidance, data)
        guidance_losses = compute_contextual_epigenomics_loss(with_guidance, data)

        assert no_guidance_losses["chromatin_guidance"] == 0.0
        assert jnp.allclose(no_guidance_losses["total"], no_guidance_losses["supervised"])
        assert guidance_losses["chromatin_guidance"] > 0.0
        assert guidance_losses["total"] > guidance_losses["supervised"]

    def test_training_smoke_decreases_loss(self, rngs: nnx.Rngs) -> None:
        data = _make_batch()
        operator = ContextualEpigenomicsOperator(
            ContextualEpigenomicsConfig(
                hidden_dim=16,
                num_layers=1,
                num_heads=2,
                intermediate_dim=32,
                max_length=20,
                num_tf_features=3,
                num_outputs=1,
                use_tf_context=True,
                use_chromatin_guidance=True,
                chromatin_guidance_weight=0.25,
            ),
            rngs=rngs,
        )
        optimizer = nnx.Optimizer(operator, optax.adam(1e-2), wrt=nnx.Param)

        def loss_fn(model: ContextualEpigenomicsOperator) -> jnp.ndarray:
            return compute_contextual_epigenomics_loss(model, data)["total"]

        initial_loss = float(loss_fn(operator))
        for _ in range(20):
            loss, grads = nnx.value_and_grad(loss_fn)(operator)
            optimizer.update(operator, grads)

        final_loss = float(loss_fn(operator))

        assert float(loss) >= 0.0
        assert final_loss < initial_loss

    def test_multiclass_mode_emits_rank3_logits(self, rngs: nnx.Rngs) -> None:
        data = _make_batch(
            target_semantics="chromatin_state_id",
            num_outputs=3,
        )
        operator = ContextualEpigenomicsOperator(
            ContextualEpigenomicsConfig(
                hidden_dim=16,
                num_layers=1,
                num_heads=2,
                intermediate_dim=32,
                max_length=20,
                num_tf_features=3,
                num_outputs=3,
                use_tf_context=True,
                use_chromatin_guidance=True,
            ),
            rngs=rngs,
        )

        result = operator.apply(data, {}, None)[0]
        losses = compute_contextual_epigenomics_loss(operator, data)

        assert result["logits"].shape == (4, 20, 3)
        assert losses["supervised"] >= 0.0


class TestSyntheticContextualEpigenomicsDataset:
    """Tests for the shared synthetic contextual epigenomics source helper."""

    def test_build_dataset_supports_both_target_semantics(self) -> None:
        peak_data = build_synthetic_contextual_epigenomics_dataset(
            n_examples=2,
            sequence_length=12,
            num_tf_features=3,
            target_semantics="binary_peak_mask",
            num_output_classes=1,
        )
        state_data = build_synthetic_contextual_epigenomics_dataset(
            n_examples=2,
            sequence_length=12,
            num_tf_features=3,
            target_semantics="chromatin_state_id",
            num_output_classes=3,
        )

        assert peak_data["sequence"].shape == (2, 12, 4)
        assert peak_data["targets"].shape == (2, 12)
        assert state_data["targets"].shape == (2, 12)
        assert set(jnp.unique(peak_data["targets"]).tolist()).issubset({0, 1})
        assert set(jnp.unique(state_data["targets"]).tolist()).issubset({0, 1, 2})

    def test_invalid_target_semantics_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="target_semantics"):
            build_synthetic_contextual_epigenomics_dataset(
                n_examples=2,
                sequence_length=12,
                num_tf_features=3,
                target_semantics=cast(Any, "unsupported"),
                num_output_classes=2,
            )
