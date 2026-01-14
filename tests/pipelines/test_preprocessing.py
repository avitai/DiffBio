"""Tests for diffbio.pipelines.preprocessing module.

These tests define the expected behavior of the preprocessing pipeline
for differentiable bioinformatics workflows.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.pipelines.preprocessing import (
    PreprocessingPipeline,
    PreprocessingPipelineConfig,
    create_preprocessing_pipeline,
)


class TestPreprocessingPipelineConfig:
    """Tests for PreprocessingPipelineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PreprocessingPipelineConfig()
        assert config.read_length == 150
        assert config.adapter_sequence == "AGATCGGAAGAG"
        assert config.quality_threshold == 20.0
        assert config.enable_adapter_removal is True
        assert config.enable_duplicate_weighting is True
        assert config.enable_error_correction is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = PreprocessingPipelineConfig(
            read_length=100,
            quality_threshold=30.0,
            enable_adapter_removal=False,
        )
        assert config.read_length == 100
        assert config.quality_threshold == 30.0
        assert config.enable_adapter_removal is False


class TestPreprocessingPipeline:
    """Tests for PreprocessingPipeline."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def config(self):
        return PreprocessingPipelineConfig(
            read_length=50,
            error_correction_window=5,
            error_correction_hidden_dim=32,
        )

    @pytest.fixture
    def pipeline(self, config, rngs):
        return PreprocessingPipeline(config, rngs=rngs)

    @pytest.fixture
    def sample_data(self):
        """Create sample input data."""
        key = jax.random.key(0)
        k1, k2 = jax.random.split(key)

        # One-hot encoded reads (10 reads, 50 bp each)
        reads_logits = jax.random.normal(k1, (10, 50, 4))
        reads = jax.nn.softmax(reads_logits, axis=-1)

        # Quality scores (Phred scale, 0-40)
        quality = jax.random.uniform(k2, (10, 50), minval=10, maxval=40)

        return {
            "reads": reads,
            "quality": quality,
        }

    def test_initialization(self, config, rngs):
        """Test pipeline initialization."""
        pipeline = PreprocessingPipeline(config, rngs=rngs)
        assert pipeline is not None
        assert hasattr(pipeline, "quality_filter")

    def test_apply_output_structure(self, pipeline, sample_data):
        """Test that apply returns expected output structure."""
        result, state, metadata = pipeline.apply(sample_data, {}, None)

        # Should have original keys
        assert "reads" in result
        assert "quality" in result

        # Should have preprocessed outputs
        assert "preprocessed_reads" in result
        assert "preprocessed_quality" in result
        assert "read_weights" in result

    def test_apply_output_shapes(self, pipeline, sample_data):
        """Test output shapes match input shapes."""
        result, _, _ = pipeline.apply(sample_data, {}, None)

        assert result["preprocessed_reads"].shape == sample_data["reads"].shape
        assert result["preprocessed_quality"].shape == sample_data["quality"].shape
        assert result["read_weights"].shape == (sample_data["reads"].shape[0],)

    def test_apply_output_finite(self, pipeline, sample_data):
        """Test that outputs are finite."""
        result, _, _ = pipeline.apply(sample_data, {}, None)

        assert jnp.isfinite(result["preprocessed_reads"]).all()
        assert jnp.isfinite(result["preprocessed_quality"]).all()
        assert jnp.isfinite(result["read_weights"]).all()

    def test_read_weights_positive(self, pipeline, sample_data):
        """Test that read weights are positive."""
        result, _, _ = pipeline.apply(sample_data, {}, None)

        assert (result["read_weights"] >= 0).all()
        assert (result["read_weights"] <= 1).all()

    def test_gradient_flows(self, pipeline, sample_data):
        """Test that gradients flow through pipeline."""

        def loss_fn(pipe, reads, quality):
            data = {"reads": reads, "quality": quality}
            result, _, _ = pipe.apply(data, {}, None)
            return result["preprocessed_reads"].sum()

        grads = nnx.grad(loss_fn)(pipeline, sample_data["reads"], sample_data["quality"])
        assert grads is not None

    def test_disabled_components(self, rngs):
        """Test pipeline with disabled components."""
        config = PreprocessingPipelineConfig(
            read_length=50,
            enable_adapter_removal=False,
            enable_duplicate_weighting=False,
            enable_error_correction=False,
        )
        pipeline = PreprocessingPipeline(config, rngs=rngs)

        key = jax.random.key(0)
        reads = jax.nn.softmax(jax.random.normal(key, (5, 50, 4)), axis=-1)
        quality = jax.random.uniform(key, (5, 50), minval=10, maxval=40)
        data = {"reads": reads, "quality": quality}

        result, _, _ = pipeline.apply(data, {}, None)
        assert "preprocessed_reads" in result


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_pipeline_jit(self, rngs):
        """Test pipeline with JIT compilation."""
        config = PreprocessingPipelineConfig(
            read_length=50,
            error_correction_window=5,
            error_correction_hidden_dim=32,
        )
        pipeline = PreprocessingPipeline(config, rngs=rngs)

        @jax.jit
        def jit_apply(reads, quality):
            data = {"reads": reads, "quality": quality}
            result, _, _ = pipeline.apply(data, {}, None)
            return result["preprocessed_reads"]

        key = jax.random.key(0)
        reads = jax.nn.softmax(jax.random.normal(key, (5, 50, 4)), axis=-1)
        quality = jax.random.uniform(key, (5, 50), minval=10, maxval=40)

        result = jit_apply(reads, quality)
        assert jnp.isfinite(result).all()


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_preprocessing_pipeline(self):
        """Test factory function creates valid pipeline."""
        pipeline = create_preprocessing_pipeline(
            read_length=100,
            quality_threshold=25.0,
            seed=123,
        )
        assert pipeline is not None
        assert isinstance(pipeline, PreprocessingPipeline)

    def test_factory_with_disabled_components(self):
        """Test factory with disabled components."""
        pipeline = create_preprocessing_pipeline(
            read_length=100,
            enable_adapter_removal=False,
            enable_error_correction=False,
        )
        assert pipeline is not None
