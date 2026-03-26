"""Tests for the enhanced variant calling pipeline.

Following TDD principles, these tests define the expected behavior
of the EnhancedVariantCallingPipeline before implementation.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


class TestEnhancedVariantCallingPipelineConfig:
    """Tests for EnhancedVariantCallingPipelineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from diffbio.pipelines.enhanced_variant_calling import (
            EnhancedVariantCallingPipelineConfig,
        )

        config = EnhancedVariantCallingPipelineConfig()

        assert config.reference_length == 1000
        assert config.num_classes == 3
        assert config.pileup_window_size == 11
        assert config.enable_preprocessing is True
        assert config.enable_quality_recalibration is True

    def test_custom_config(self):
        """Test custom configuration values."""
        from diffbio.pipelines.enhanced_variant_calling import (
            EnhancedVariantCallingPipelineConfig,
        )

        config = EnhancedVariantCallingPipelineConfig(
            reference_length=500,
            num_classes=4,
            pileup_window_size=21,
            enable_preprocessing=False,
        )

        assert config.reference_length == 500
        assert config.num_classes == 4
        assert config.pileup_window_size == 21
        assert config.enable_preprocessing is False


class TestEnhancedVariantCallingPipeline:
    """Tests for EnhancedVariantCallingPipeline operator."""

    @pytest.fixture
    def config(self):
        """Provide default config."""
        from diffbio.pipelines.enhanced_variant_calling import (
            EnhancedVariantCallingPipelineConfig,
        )

        return EnhancedVariantCallingPipelineConfig(
            reference_length=100,
            num_classes=3,
            pileup_window_size=11,
            cnn_hidden_channels=(16, 32),
            cnn_fc_dims=(32, 16),
        )

    @pytest.fixture
    def pipeline(self, config, rngs):
        """Create pipeline instance."""
        from diffbio.pipelines.enhanced_variant_calling import (
            EnhancedVariantCallingPipeline,
        )

        return EnhancedVariantCallingPipeline(config, rngs=rngs)

    @pytest.fixture
    def sample_data(self, config):
        """Create sample variant calling data."""
        num_reads = 20
        read_length = 50
        reference_length = config.reference_length

        key = jax.random.key(42)

        # One-hot encoded reads (num_reads, read_length, 4)
        keys = jax.random.split(key, 3)
        base_indices = jax.random.randint(keys[0], (num_reads, read_length), 0, 4)
        reads = jax.nn.one_hot(base_indices, 4)

        # Read positions on reference
        positions = jax.random.randint(keys[1], (num_reads,), 0, reference_length - read_length)

        # Quality scores
        quality = jax.random.uniform(keys[2], (num_reads, read_length), minval=20, maxval=40)

        return {
            "reads": reads,
            "positions": positions,
            "quality": quality,
        }

    def test_initialization(self, config, rngs):
        """Test pipeline initialization."""
        from diffbio.pipelines.enhanced_variant_calling import (
            EnhancedVariantCallingPipeline,
        )

        pipeline = EnhancedVariantCallingPipeline(config, rngs=rngs)

        assert pipeline is not None
        assert hasattr(pipeline, "pileup")
        assert hasattr(pipeline, "cnn_classifier")
        assert hasattr(pipeline, "quality_recalibration")

    def test_initialization_without_preprocessing(self, rngs):
        """Test initialization with preprocessing disabled."""
        from diffbio.pipelines.enhanced_variant_calling import (
            EnhancedVariantCallingPipeline,
            EnhancedVariantCallingPipelineConfig,
        )

        config = EnhancedVariantCallingPipelineConfig(
            reference_length=100,
            enable_preprocessing=False,
        )
        pipeline = EnhancedVariantCallingPipeline(config, rngs=rngs)

        assert pipeline.quality_filter is None

    def test_initialization_without_quality_recalibration(self, rngs):
        """Test initialization with quality recalibration disabled."""
        from diffbio.pipelines.enhanced_variant_calling import (
            EnhancedVariantCallingPipeline,
            EnhancedVariantCallingPipelineConfig,
        )

        config = EnhancedVariantCallingPipelineConfig(
            reference_length=100,
            enable_quality_recalibration=False,
        )
        pipeline = EnhancedVariantCallingPipeline(config, rngs=rngs)

        assert pipeline.quality_recalibration is None

    def test_apply_full_pipeline(self, pipeline, sample_data):
        """Test full pipeline apply method."""
        result, state, metadata = pipeline.apply(sample_data, {}, None)

        # Check output keys
        assert "pileup" in result
        assert "logits" in result
        assert "probabilities" in result
        assert "quality_scores" in result
        assert "filter_weights" in result

        # Check shapes
        ref_len = pipeline.config.reference_length
        num_classes = pipeline.config.num_classes

        assert result["pileup"].shape == (ref_len, 4)
        assert result["logits"].shape == (ref_len, num_classes)
        assert result["probabilities"].shape == (ref_len, num_classes)
        assert result["quality_scores"].shape == (ref_len,)
        assert result["filter_weights"].shape == (ref_len,)

    def test_apply_without_quality_recalibration(self, sample_data, rngs):
        """Test pipeline without quality recalibration."""
        from diffbio.pipelines.enhanced_variant_calling import (
            EnhancedVariantCallingPipeline,
            EnhancedVariantCallingPipelineConfig,
        )

        config = EnhancedVariantCallingPipelineConfig(
            reference_length=100,
            enable_quality_recalibration=False,
            cnn_hidden_channels=(16, 32),
            cnn_fc_dims=(32, 16),
        )
        pipeline = EnhancedVariantCallingPipeline(config, rngs=rngs)

        result, _, _ = pipeline.apply(sample_data, {}, None)

        # Should still have core outputs
        assert "pileup" in result
        assert "probabilities" in result
        # Should NOT have quality recalibration outputs
        assert "quality_scores" not in result

    def test_probabilities_valid(self, pipeline, sample_data):
        """Test that probabilities are valid."""
        result, _, _ = pipeline.apply(sample_data, {}, None)

        probs = result["probabilities"]

        # Should be non-negative
        assert jnp.all(probs >= 0.0)

        # Should sum to 1 along class dimension
        row_sums = probs.sum(axis=-1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-5)

    def test_output_finite(self, pipeline, sample_data):
        """Test that all outputs are finite."""
        result, _, _ = pipeline.apply(sample_data, {}, None)

        for key in ["pileup", "logits", "probabilities", "quality_scores"]:
            assert jnp.all(jnp.isfinite(result[key])), f"{key} contains non-finite values"

    def test_preserves_original_data(self, pipeline, sample_data):
        """Test that original data is preserved in output."""
        result, _, _ = pipeline.apply(sample_data, {}, None)

        assert "reads" in result
        assert "positions" in result
        assert "quality" in result


class TestEnhancedVariantCallingDifferentiability:
    """Tests for gradient flow through the pipeline."""

    @pytest.fixture
    def config(self):
        """Provide config for gradient tests."""
        from diffbio.pipelines.enhanced_variant_calling import (
            EnhancedVariantCallingPipelineConfig,
        )

        return EnhancedVariantCallingPipelineConfig(
            reference_length=50,
            num_classes=3,
            pileup_window_size=5,
            cnn_hidden_channels=(8, 16),
            cnn_fc_dims=(16, 8),
            enable_preprocessing=False,  # Simpler for gradient test
        )

    def test_gradient_flow_through_pipeline(self, config, rngs):
        """Test that gradients flow through the full pipeline."""
        from diffbio.pipelines.enhanced_variant_calling import (
            EnhancedVariantCallingPipeline,
        )

        pipeline = EnhancedVariantCallingPipeline(config, rngs=rngs)
        pipeline.eval()  # Set to eval mode for deterministic dropout

        def loss_fn(pipe, reads, positions, quality):
            data = {
                "reads": reads,
                "positions": positions,
                "quality": quality,
            }
            result, _, _ = pipe.apply(data, {}, None)
            return result["probabilities"].sum()

        num_reads = 10
        read_length = 20
        ref_len = config.reference_length

        key = jax.random.key(0)
        keys = jax.random.split(key, 3)
        base_indices = jax.random.randint(keys[0], (num_reads, read_length), 0, 4)
        reads = jax.nn.one_hot(base_indices, 4)
        positions = jax.random.randint(keys[1], (num_reads,), 0, ref_len - read_length)
        quality = jax.random.uniform(keys[2], (num_reads, read_length), minval=20, maxval=40)

        grads = nnx.grad(loss_fn)(pipeline, reads, positions, quality)

        # Gradients should exist
        assert grads is not None

    def test_gradient_wrt_input_reads(self, config, rngs):
        """Test gradient with respect to input reads."""
        from diffbio.pipelines.enhanced_variant_calling import (
            EnhancedVariantCallingPipeline,
        )

        pipeline = EnhancedVariantCallingPipeline(config, rngs=rngs)
        pipeline.eval()  # Set to eval mode for deterministic dropout

        def loss_fn(reads, positions, quality):
            data = {
                "reads": reads,
                "positions": positions,
                "quality": quality,
            }
            result, _, _ = pipeline.apply(data, {}, None)
            return result["probabilities"].sum()

        num_reads = 10
        read_length = 20
        ref_len = config.reference_length

        key = jax.random.key(0)
        keys = jax.random.split(key, 3)
        reads = jax.random.uniform(keys[0], (num_reads, read_length, 4))
        positions = jax.random.randint(keys[1], (num_reads,), 0, ref_len - read_length)
        quality = jax.random.uniform(keys[2], (num_reads, read_length), minval=20, maxval=40)

        grad = jax.grad(loss_fn)(reads, positions, quality)

        assert grad.shape == reads.shape
        assert jnp.all(jnp.isfinite(grad))


class TestEnhancedVariantCallingJITCompatibility:
    """Tests for JIT compilation compatibility."""

    @pytest.fixture
    def config(self):
        """Provide config for JIT tests."""
        from diffbio.pipelines.enhanced_variant_calling import (
            EnhancedVariantCallingPipelineConfig,
        )

        return EnhancedVariantCallingPipelineConfig(
            reference_length=50,
            num_classes=3,
            pileup_window_size=5,
            cnn_hidden_channels=(8, 16),
            cnn_fc_dims=(16, 8),
        )

    def test_jit_apply(self, config, rngs):
        """Test JIT compilation of apply method."""
        from diffbio.pipelines.enhanced_variant_calling import (
            EnhancedVariantCallingPipeline,
        )

        pipeline = EnhancedVariantCallingPipeline(config, rngs=rngs)
        pipeline.eval()  # Set to eval mode for deterministic dropout

        @nnx.jit
        def jit_apply(reads, positions, quality):
            data = {
                "reads": reads,
                "positions": positions,
                "quality": quality,
            }
            result, _, _ = pipeline.apply(data, {}, None)
            return result["probabilities"]

        num_reads = 10
        read_length = 20
        ref_len = config.reference_length

        key = jax.random.key(0)
        keys = jax.random.split(key, 3)
        base_indices = jax.random.randint(keys[0], (num_reads, read_length), 0, 4)
        reads = jax.nn.one_hot(base_indices, 4)
        positions = jax.random.randint(keys[1], (num_reads,), 0, ref_len - read_length)
        quality = jax.random.uniform(keys[2], (num_reads, read_length), minval=20, maxval=40)

        # Should compile and run without error
        result = jit_apply(reads, positions, quality)
        assert result.shape == (ref_len, config.num_classes)


class TestCreateEnhancedVariantCallingPipeline:
    """Tests for factory function."""

    def test_create_pipeline_default(self):
        """Test factory function with defaults."""
        from diffbio.pipelines.enhanced_variant_calling import (
            create_enhanced_variant_calling_pipeline,
        )

        pipeline = create_enhanced_variant_calling_pipeline()
        assert pipeline is not None

    def test_create_pipeline_custom(self):
        """Test factory function with custom parameters."""
        from diffbio.pipelines.enhanced_variant_calling import (
            create_enhanced_variant_calling_pipeline,
        )

        pipeline = create_enhanced_variant_calling_pipeline(
            reference_length=500,
            num_classes=4,
            enable_quality_recalibration=False,
            seed=123,
        )

        assert pipeline.config.reference_length == 500
        assert pipeline.config.num_classes == 4
        assert pipeline.config.enable_quality_recalibration is False
