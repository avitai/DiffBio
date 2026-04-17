"""Tests for shared foundation-model contracts."""

import pytest
from flax import nnx

from diffbio.operators.foundation_models import (
    AdapterMode,
    DifferentiableFoundationModel,
    FoundationArtifactSpec,
    FOUNDATION_BENCHMARK_COMPARISON_AXES,
    FoundationModelKind,
    PoolingStrategy,
    build_foundation_benchmark_metadata,
    TransformerSequenceEncoder,
    TransformerSequenceEncoderConfig,
    build_foundation_model_metadata,
    create_foundation_model,
    decode_foundation_text,
    get_foundation_model_cls,
)


class TestFoundationArtifactSpec:
    """Tests for artifact metadata specs and encoding."""

    def test_metadata_roundtrip(self) -> None:
        """Encoded metadata should decode back to the original values."""
        spec = FoundationArtifactSpec(
            model_family=FoundationModelKind.SEQUENCE_TRANSFORMER,
            artifact_id="diffbio.sequence.v1",
            preprocessing_version="one_hot_v1",
            adapter_mode=AdapterMode.FROZEN_ENCODER,
            pooling_strategy=PoolingStrategy.CLS,
        )

        metadata = build_foundation_model_metadata(spec)

        assert decode_foundation_text(metadata["model_family"]) == "sequence_transformer"
        assert decode_foundation_text(metadata["artifact_id"]) == "diffbio.sequence.v1"
        assert decode_foundation_text(metadata["preprocessing_version"]) == "one_hot_v1"
        assert decode_foundation_text(metadata["adapter_mode"]) == "frozen_encoder"
        assert decode_foundation_text(metadata["pooling_strategy"]) == "cls"

    def test_ascii_validation(self) -> None:
        """Artifact metadata should remain ASCII-only for jit-safe encoding."""
        with pytest.raises(ValueError, match="ASCII-only"):
            FoundationArtifactSpec(
                model_family=FoundationModelKind.SINGLE_CELL_TRANSFORMER,
                artifact_id="diffbio.β",
                preprocessing_version="counts_v1",
                adapter_mode=AdapterMode.NATIVE_TRAINABLE,
                pooling_strategy=PoolingStrategy.MEAN,
            )

    def test_benchmark_metadata_adds_dataset_and_task(self) -> None:
        """Benchmark-facing metadata should include dataset/task explicitly."""
        spec = FoundationArtifactSpec(
            model_family=FoundationModelKind.SEQUENCE_TRANSFORMER,
            artifact_id="diffbio.sequence.v1",
            preprocessing_version="one_hot_v1",
            adapter_mode=AdapterMode.FROZEN_ENCODER,
            pooling_strategy=PoolingStrategy.CLS,
        )

        benchmark_metadata = build_foundation_benchmark_metadata(
            build_foundation_model_metadata(spec),
            dataset="synthetic_genomics",
            task="promoter_classification",
        )

        assert benchmark_metadata["dataset"] == "synthetic_genomics"
        assert benchmark_metadata["task"] == "promoter_classification"
        assert benchmark_metadata["model_family"] == "sequence_transformer"
        assert benchmark_metadata["adapter_mode"] == "frozen_encoder"
        assert benchmark_metadata["artifact_id"] == "diffbio.sequence.v1"
        assert benchmark_metadata["preprocessing_version"] == "one_hot_v1"
        assert benchmark_metadata["pooling_strategy"] == "cls"

    def test_benchmark_metadata_rejects_missing_required_field(self) -> None:
        """Missing required artifact fields should fail contract normalization."""
        spec = FoundationArtifactSpec(
            model_family=FoundationModelKind.SEQUENCE_TRANSFORMER,
            artifact_id="diffbio.sequence.v1",
            preprocessing_version="one_hot_v1",
            adapter_mode=AdapterMode.FROZEN_ENCODER,
            pooling_strategy=PoolingStrategy.CLS,
        )
        metadata = build_foundation_model_metadata(spec)
        metadata.pop("artifact_id")

        with pytest.raises(ValueError, match="artifact_id"):
            build_foundation_benchmark_metadata(
                metadata,
                dataset="synthetic_genomics",
                task="promoter_classification",
            )

    def test_benchmark_comparison_axes_match_contract(self) -> None:
        """Comparison axes should stay aligned with the benchmark metadata contract."""
        assert list(FOUNDATION_BENCHMARK_COMPARISON_AXES) == [
            "dataset",
            "task",
            "model_family",
            "adapter_mode",
            "artifact_id",
            "preprocessing_version",
        ]


class TestFoundationRegistry:
    """Tests for the built-in foundation-model registry."""

    def test_builtin_registry_entries(self) -> None:
        """Built-in operators should be registered under their model families."""
        assert (
            get_foundation_model_cls(FoundationModelKind.SEQUENCE_TRANSFORMER)
            is TransformerSequenceEncoder
        )
        assert (
            get_foundation_model_cls(FoundationModelKind.SINGLE_CELL_TRANSFORMER)
            is DifferentiableFoundationModel
        )

    def test_factory_instantiates_registered_model(self) -> None:
        """The shared factory should instantiate registered operators."""
        config = TransformerSequenceEncoderConfig(
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            intermediate_dim=32,
            max_length=32,
            dropout_rate=0.0,
        )
        operator = create_foundation_model(
            FoundationModelKind.SEQUENCE_TRANSFORMER,
            config=config,
            rngs=nnx.Rngs(0),
        )
        assert isinstance(operator, TransformerSequenceEncoder)
