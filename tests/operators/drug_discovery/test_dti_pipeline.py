"""Tests for the shared differentiable DTI integration pipeline."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax import nnx

from diffbio.operators.foundation_models import decode_foundation_text
from diffbio.sources import DTISourceConfig, DavisDTISource, build_paired_dti_batch


def test_pipeline_builds_inputs_from_source_contract_and_scores_batch() -> None:
    """The DTI pipeline should reuse source contracts and encoder building blocks."""
    from diffbio.operators.drug_discovery import (
        DTIPipelineConfig,
        DifferentiableDTIPipeline,
        build_dti_pipeline_inputs,
    )

    data = DavisDTISource(DTISourceConfig(dataset_name="davis", split="train")).load()
    batch = build_paired_dti_batch(data, indices=np.array([0, 1, 2], dtype=np.int32))
    config = DTIPipelineConfig(
        protein_hidden_dim=8,
        protein_num_heads=2,
        protein_num_layers=1,
        drug_fingerprint_dim=8,
        drug_hidden_dim=8,
        pair_hidden_dim=12,
        max_protein_length=12,
    )

    input_data = build_dti_pipeline_inputs(batch, config=config)
    assert input_data["protein_one_hot"].shape == (3, 12, 20)
    assert input_data["protein_attention_mask"].shape == (3, 12)
    assert input_data["drug_graphs"]["node_features"].shape[0] == 3

    pipeline = DifferentiableDTIPipeline(config, rngs=nnx.Rngs(42))
    result, state, metadata = pipeline.apply(input_data, {"seen": True}, {"source": "test"})

    assert state == {"seen": True}
    assert metadata == {"source": "test"}
    assert result["scores"].shape == (3,)
    assert result["protein_embeddings"].shape == (3, config.protein_hidden_dim)
    assert result["drug_fingerprints"].shape == (3, config.drug_fingerprint_dim)
    assert result["pair_embeddings"].shape == (
        3,
        config.protein_hidden_dim + config.drug_fingerprint_dim,
    )
    assert result["dti_pipeline"]["integration_layer"] == "shared_dti_pipeline_v1"
    assert result["dti_pipeline"]["protein_encoder"]["operator"] == "TransformerSequenceEncoder"
    assert result["dti_pipeline"]["drug_encoder"]["operator"] == (
        "DifferentiableMolecularFingerprint"
    )
    assert decode_foundation_text(result["foundation_model"]["adapter_mode"]) == (
        "native_trainable"
    )
    assert decode_foundation_text(result["foundation_model"]["preprocessing_version"]) == (
        "protein_one_hot_v1"
    )


def test_pipeline_gradients_flow_through_drug_encoder_path() -> None:
    """Gradient evidence must be specific to the differentiable drug encoder."""
    from diffbio.operators.drug_discovery import (
        DTIPipelineConfig,
        DifferentiableDTIPipeline,
        build_dti_pipeline_inputs,
    )

    data = DavisDTISource(DTISourceConfig(dataset_name="davis", split="train")).load()
    batch = build_paired_dti_batch(data, indices=np.array([0, 1], dtype=np.int32))
    config = DTIPipelineConfig(
        protein_hidden_dim=8,
        protein_num_heads=2,
        protein_num_layers=1,
        drug_fingerprint_dim=8,
        drug_hidden_dim=8,
        pair_hidden_dim=12,
        max_protein_length=12,
    )
    input_data = build_dti_pipeline_inputs(batch, config=config)
    pipeline = DifferentiableDTIPipeline(config, rngs=nnx.Rngs(42))

    def loss_fn(model: DifferentiableDTIPipeline) -> jnp.ndarray:
        result, _, _ = model.apply(input_data, {}, None)
        return jnp.mean(jnp.square(result["scores"]))

    grads = nnx.grad(loss_fn)(pipeline)
    drug_grad_norm = _gradient_norm_for_path(grads, "drug_encoder")

    assert drug_grad_norm > 0.0


def _gradient_norm_for_path(tree: object, needle: str) -> float:
    """Compute gradient norm for NNX variables whose path contains ``needle``."""
    total_sq = 0.0
    for path, variable in nnx.iter_graph(tree):
        if needle not in ".".join(str(part) for part in path):
            continue
        if isinstance(variable, nnx.Param):
            value = variable[...]
        elif isinstance(variable, nnx.Variable):
            value = variable.get_value()
        else:
            continue
        if isinstance(value, jnp.ndarray):
            total_sq += float(jnp.sum(value**2))
    return total_sq**0.5
