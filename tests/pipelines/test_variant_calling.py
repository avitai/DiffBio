"""Tests for the end-to-end variant calling pipeline."""

import jax
import jax.numpy as jnp
import pytest
from datarax.typing import Batch, Element

from diffbio.constants import ClassifierType
from diffbio.pipelines import (
    VariantCallingPipeline,
    VariantCallingPipelineConfig,
    create_cnn_variant_pipeline,
    create_variant_calling_pipeline,
)


class TestVariantCallingPipelineConfig:
    """Tests for pipeline configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VariantCallingPipelineConfig()
        assert config.reference_length == 100
        assert config.num_classes == 3
        assert config.quality_threshold == 20.0
        assert config.pileup_window_size == 11
        assert config.classifier_hidden_dim == 64
        assert config.use_quality_weights is True
        assert config.stochastic is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = VariantCallingPipelineConfig(
            reference_length=200,
            num_classes=4,
            quality_threshold=30.0,
        )
        assert config.reference_length == 200
        assert config.num_classes == 4
        assert config.quality_threshold == 30.0


class TestVariantCallingPipeline:
    """Tests for the variant calling pipeline."""

    @pytest.fixture
    def pipeline(self, rngs):
        config = VariantCallingPipelineConfig(
            reference_length=30,
            num_classes=3,
            pileup_window_size=5,
            classifier_hidden_dim=16,
        )
        pipeline = VariantCallingPipeline(config, rngs=rngs)
        pipeline.eval_mode()
        return pipeline

    @pytest.fixture
    def sample_data(self):
        """Create sample input data for a single element."""
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)

        num_reads = 5
        read_length = 10

        # Random reads (one-hot encoded)
        indices = jax.random.randint(k1, (num_reads, read_length), 0, 4)
        reads = jax.nn.one_hot(indices, 4)

        # Random positions (within reference bounds)
        positions = jax.random.randint(k2, (num_reads,), 0, 20)

        # Quality scores
        quality = jax.random.uniform(k3, (num_reads, read_length), minval=10.0, maxval=40.0)

        return {
            "reads": reads,
            "positions": positions,
            "quality": quality,
        }

    @pytest.fixture
    def sample_batch(self, sample_data):
        """Create a batch with multiple samples."""
        elements = []
        for i in range(3):
            # Slightly modify data for each element
            key = jax.random.PRNGKey(42 + i)
            k1, k2, k3 = jax.random.split(key, 3)

            indices = jax.random.randint(k1, (5, 10), 0, 4)
            reads = jax.nn.one_hot(indices, 4)
            positions = jax.random.randint(k2, (5,), 0, 20)
            quality = jax.random.uniform(k3, (5, 10), minval=10.0, maxval=40.0)

            data = {"reads": reads, "positions": positions, "quality": quality}
            state = {"sample_id": jnp.array(i)}
            elements.append(Element(data=data, state=state))

        return Batch(elements)

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initializes with correct components."""
        assert hasattr(pipeline, "quality_filter")
        assert hasattr(pipeline, "pileup")
        assert hasattr(pipeline, "classifier")
        assert pipeline.config.reference_length == 30
        assert pipeline.config.num_classes == 3

    def test_apply_single_element(self, pipeline, sample_data):
        """Test apply() processes a single element correctly."""
        state = {}
        result_data, result_state, _ = pipeline.apply(sample_data, state, None)

        # Check all output keys present
        assert "reads" in result_data  # Input preserved
        assert "positions" in result_data
        assert "quality" in result_data
        assert "filtered_reads" in result_data
        assert "filtered_quality" in result_data
        assert "pileup" in result_data
        assert "logits" in result_data
        assert "probabilities" in result_data

        # Check output shapes
        assert result_data["pileup"].shape == (30, 4)  # reference_length x 4
        assert result_data["logits"].shape == (30, 3)  # reference_length x num_classes
        assert result_data["probabilities"].shape == (30, 3)

    def test_apply_batch(self, pipeline, sample_batch):
        """Test apply_batch() processes entire batch."""
        result_batch = pipeline.apply_batch(sample_batch)

        # Verify batch structure
        assert result_batch.batch_size == 3

        # Verify output data
        result_data = result_batch.data.get_value()
        assert "pileup" in result_data
        assert "logits" in result_data
        assert "probabilities" in result_data

        # Verify batched shapes
        assert result_data["pileup"].shape == (3, 30, 4)
        assert result_data["logits"].shape == (3, 30, 3)
        assert result_data["probabilities"].shape == (3, 30, 3)

    def test_callable_interface(self, pipeline, sample_batch):
        """Test __call__ interface."""
        result_batch = pipeline(sample_batch)

        assert result_batch.batch_size == 3
        result_data = result_batch.data.get_value()
        assert "probabilities" in result_data

    def test_probabilities_are_valid(self, pipeline, sample_batch):
        """Test output probabilities are valid distributions."""
        result_batch = pipeline.apply_batch(sample_batch)
        result_data = result_batch.data.get_value()
        probs = result_data["probabilities"]

        # Each position should sum to 1
        sums = probs.sum(axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)

        # All probabilities should be non-negative
        assert jnp.all(probs >= 0)

    def test_pileup_is_valid_distribution(self, pipeline, sample_batch):
        """Test pileup outputs are valid distributions."""
        result_batch = pipeline.apply_batch(sample_batch)
        result_data = result_batch.data.get_value()
        pileup = result_data["pileup"]

        # Each position should sum to 1
        sums = pileup.sum(axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)

    def test_call_variants_method(self, pipeline, sample_batch):
        """Test the call_variants convenience method."""
        result = pipeline.call_variants(sample_batch)

        assert "predictions" in result
        assert "probabilities" in result
        assert "pileup" in result

        # Predictions should be class indices
        assert result["predictions"].shape == (3, 30)
        assert jnp.all(result["predictions"] >= 0)
        assert jnp.all(result["predictions"] < 3)


class TestVariantCallingPipelineJITCompatibility:
    """Tests for JIT compatibility of the variant calling pipeline."""

    @pytest.fixture
    def pipeline(self, rngs):
        config = VariantCallingPipelineConfig(
            reference_length=20,
            num_classes=3,
            pileup_window_size=5,
            classifier_hidden_dim=8,
        )
        pipeline = VariantCallingPipeline(config, rngs=rngs)
        pipeline.eval_mode()
        return pipeline

    def test_jit_apply(self, pipeline):
        """Test JIT compilation works for VariantCallingPipeline."""
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)

        indices = jax.random.randint(k1, (3, 8), 0, 4)
        reads = jax.nn.one_hot(indices, 4).astype(jnp.float32)
        positions = jax.random.randint(k2, (3,), 0, 10)
        quality = jax.random.uniform(k3, (3, 8), minval=10.0, maxval=40.0)

        @jax.jit
        def forward(r, pos, q):
            data = {"reads": r, "positions": pos, "quality": q}
            result_data, _, _ = pipeline.apply(data, {}, None)
            return result_data["logits"], result_data["probabilities"]

        logits, probs = forward(reads, positions, quality)
        assert logits.shape == (20, 3)
        assert probs.shape == (20, 3)
        assert jnp.all(jnp.isfinite(logits))
        assert jnp.all(jnp.isfinite(probs))

        # Second call should produce same result
        logits2, _ = forward(reads, positions, quality)
        assert jnp.allclose(logits, logits2)


class TestVariantCallingPipelineGradients:
    """Tests for gradient flow through the pipeline."""

    @pytest.fixture
    def pipeline(self, rngs):
        config = VariantCallingPipelineConfig(
            reference_length=20,
            num_classes=3,
            pileup_window_size=5,
            classifier_hidden_dim=8,
        )
        pipeline = VariantCallingPipeline(config, rngs=rngs)
        pipeline.eval_mode()
        return pipeline

    def test_gradient_through_pipeline(self, pipeline):
        """Test gradients flow through entire pipeline."""
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)

        indices = jax.random.randint(k1, (3, 8), 0, 4)
        reads = jax.nn.one_hot(indices, 4).astype(jnp.float32)
        positions = jax.random.randint(k2, (3,), 0, 10)
        quality = jax.random.uniform(k3, (3, 8), minval=10.0, maxval=40.0)

        def loss_fn(r):
            data = {"reads": r, "positions": positions, "quality": quality}
            result_data, _, _ = pipeline.apply(data, {}, None)
            # Sum logits as loss
            return jnp.sum(result_data["logits"])

        grad = jax.grad(loss_fn)(reads)

        assert grad is not None
        assert grad.shape == reads.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_through_batch(self, pipeline):
        """Test gradients flow through batch processing."""
        key = jax.random.PRNGKey(42)
        batch_size = 2
        num_reads = 3
        read_length = 8

        k1, k2, k3 = jax.random.split(key, 3)
        indices = jax.random.randint(k1, (batch_size, num_reads, read_length), 0, 4)
        reads = jax.nn.one_hot(indices, 4).astype(jnp.float32)
        positions = jax.random.randint(k2, (batch_size, num_reads), 0, 10)
        quality = jax.random.uniform(
            k3, (batch_size, num_reads, read_length), minval=10.0, maxval=40.0
        )

        def loss_fn(r):
            # Process each sample and sum losses
            def process_single(single_reads, single_positions, single_quality):
                data = {
                    "reads": single_reads,
                    "positions": single_positions,
                    "quality": single_quality,
                }
                result_data, _, _ = pipeline.apply(data, {}, None)
                return jnp.sum(result_data["logits"])

            losses = jax.vmap(process_single)(r, positions, quality)
            return jnp.sum(losses)

        grad = jax.grad(loss_fn)(reads)

        assert grad is not None
        assert grad.shape == reads.shape
        assert jnp.all(jnp.isfinite(grad))


class TestFactoryFunction:
    """Tests for the create_variant_calling_pipeline factory."""

    def test_factory_creates_pipeline(self):
        """Test factory creates valid pipeline."""
        pipeline = create_variant_calling_pipeline(
            reference_length=50,
            num_classes=3,
            seed=42,
        )

        assert isinstance(pipeline, VariantCallingPipeline)
        assert pipeline.config.reference_length == 50

    def test_factory_with_custom_params(self):
        """Test factory with custom parameters."""
        pipeline = create_variant_calling_pipeline(
            reference_length=100,
            num_classes=4,
            quality_threshold=25.0,
            hidden_dim=128,
            seed=123,
        )

        assert pipeline.config.reference_length == 100
        assert pipeline.config.num_classes == 4
        assert pipeline.config.quality_threshold == 25.0
        assert pipeline.config.classifier_hidden_dim == 128


class TestCNNVariantPipeline:
    """Tests for the CNN-based variant calling pipeline."""

    @pytest.fixture
    def cnn_pipeline(self, rngs):
        """Create a CNN variant calling pipeline."""
        config = VariantCallingPipelineConfig(
            reference_length=30,
            num_classes=3,
            pileup_window_size=11,  # Larger window for CNN
            classifier_type="cnn",
            cnn_hidden_channels=(16, 32),
            cnn_fc_dims=(32, 16),
            apply_pileup_softmax=False,  # Better for variant detection
        )
        pipeline = VariantCallingPipeline(config, rngs=rngs)
        pipeline.eval_mode()
        return pipeline

    @pytest.fixture
    def sample_data(self):
        """Create sample input data for CNN testing."""
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)

        num_reads = 10
        read_length = 15

        indices = jax.random.randint(k1, (num_reads, read_length), 0, 4)
        reads = jax.nn.one_hot(indices, 4)
        positions = jax.random.randint(k2, (num_reads,), 0, 15)
        quality = jax.random.uniform(k3, (num_reads, read_length), minval=10.0, maxval=40.0)

        return {
            "reads": reads,
            "positions": positions,
            "quality": quality,
        }

    def test_cnn_pipeline_initialization(self, cnn_pipeline):
        """Test CNN pipeline initializes with correct components."""
        assert hasattr(cnn_pipeline, "quality_filter")
        assert hasattr(cnn_pipeline, "pileup")
        assert hasattr(cnn_pipeline, "classifier")
        assert cnn_pipeline.config.classifier_type == ClassifierType.CNN

    def test_cnn_pipeline_apply(self, cnn_pipeline, sample_data):
        """Test CNN pipeline processes data correctly."""
        result_data, _, _ = cnn_pipeline.apply(sample_data, {}, None)

        # Check all output keys present
        assert "pileup" in result_data
        assert "logits" in result_data
        assert "probabilities" in result_data
        # CNN pipeline should also have coverage and quality
        assert "coverage" in result_data
        assert "mean_quality" in result_data

        # Check output shapes
        assert result_data["pileup"].shape == (30, 4)
        assert result_data["logits"].shape == (30, 3)
        assert result_data["probabilities"].shape == (30, 3)
        assert result_data["coverage"].shape == (30, 1)
        assert result_data["mean_quality"].shape == (30, 1)

    def test_cnn_pipeline_probabilities_valid(self, cnn_pipeline, sample_data):
        """Test CNN pipeline outputs valid probability distributions."""
        result_data, _, _ = cnn_pipeline.apply(sample_data, {}, None)
        probs = result_data["probabilities"]

        # Each position should sum to 1
        sums = probs.sum(axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)

        # All probabilities should be non-negative
        assert jnp.all(probs >= 0)

    def test_cnn_pipeline_gradient_flow(self, cnn_pipeline, sample_data):
        """Test gradients flow through CNN pipeline."""
        reads = sample_data["reads"]
        positions = sample_data["positions"]
        quality = sample_data["quality"]

        def loss_fn(r):
            data = {"reads": r, "positions": positions, "quality": quality}
            result_data, _, _ = cnn_pipeline.apply(data, {}, None)
            return jnp.sum(result_data["logits"])

        grad = jax.grad(loss_fn)(reads)

        assert grad is not None
        assert grad.shape == reads.shape
        assert jnp.all(jnp.isfinite(grad))


class TestCNNFactoryFunction:
    """Tests for the create_cnn_variant_pipeline factory."""

    def test_factory_creates_cnn_pipeline(self):
        """Test factory creates valid CNN pipeline."""
        pipeline = create_cnn_variant_pipeline(
            reference_length=50,
            num_classes=3,
            seed=42,
        )

        assert isinstance(pipeline, VariantCallingPipeline)
        assert pipeline.config.reference_length == 50
        assert pipeline.config.classifier_type == ClassifierType.CNN

    def test_factory_with_custom_params(self):
        """Test CNN factory with custom parameters."""
        pipeline = create_cnn_variant_pipeline(
            reference_length=100,
            num_classes=4,
            pileup_window_size=31,
            cnn_hidden_channels=(64, 128),
            cnn_fc_dims=(128, 64),
            seed=123,
        )

        assert pipeline.config.reference_length == 100
        assert pipeline.config.num_classes == 4
        assert pipeline.config.pileup_window_size == 31
        assert pipeline.config.classifier_type == ClassifierType.CNN
        # Softmax should be disabled for CNN (better for variant detection)
        assert pipeline.config.apply_pileup_softmax is False

    def test_mlp_vs_cnn_factory_difference(self):
        """Test that MLP and CNN factories create different pipelines."""
        mlp_pipeline = create_variant_calling_pipeline(
            reference_length=50,
            classifier_type="mlp",
            seed=42,
        )
        cnn_pipeline = create_cnn_variant_pipeline(
            reference_length=50,
            seed=42,
        )

        assert mlp_pipeline.config.classifier_type == ClassifierType.MLP
        assert cnn_pipeline.config.classifier_type == ClassifierType.CNN


class TestEdgeCases:
    """Edge case tests for variant calling pipeline."""

    def test_single_read(self, rngs):
        """Test pipeline handles single read."""
        config = VariantCallingPipelineConfig(
            reference_length=20,
            num_classes=3,
            pileup_window_size=5,
            classifier_hidden_dim=8,
        )
        pipeline = VariantCallingPipeline(config, rngs=rngs)
        pipeline.eval_mode()

        # Single read
        indices = jnp.array([[0, 1, 2, 3, 0]])  # 1 read, 5 bases
        reads = jax.nn.one_hot(indices, 4)
        positions = jnp.array([0])
        quality = jnp.ones((1, 5)) * 30.0

        data = {"reads": reads, "positions": positions, "quality": quality}
        result_data, _, _ = pipeline.apply(data, {}, None)

        assert result_data["logits"].shape == (20, 3)
        assert jnp.all(jnp.isfinite(result_data["logits"]))

    def test_minimal_reference_length(self, rngs):
        """Test pipeline with minimal reference length equal to window size."""
        window_size = 5
        config = VariantCallingPipelineConfig(
            reference_length=window_size,  # Same as window
            num_classes=3,
            pileup_window_size=window_size,
            classifier_hidden_dim=8,
        )
        pipeline = VariantCallingPipeline(config, rngs=rngs)
        pipeline.eval_mode()

        key = jax.random.PRNGKey(42)
        indices = jax.random.randint(key, (3, 5), 0, 4)
        reads = jax.nn.one_hot(indices, 4)
        positions = jnp.array([0, 0, 0])  # All start at 0
        quality = jnp.ones((3, 5)) * 30.0

        data = {"reads": reads, "positions": positions, "quality": quality}
        result_data, _, _ = pipeline.apply(data, {}, None)

        assert result_data["logits"].shape == (window_size, 3)
        assert jnp.all(jnp.isfinite(result_data["probabilities"]))

    def test_all_low_quality_reads(self, rngs):
        """Test pipeline handles all low quality reads."""
        config = VariantCallingPipelineConfig(
            reference_length=20,
            num_classes=3,
            pileup_window_size=5,
            quality_threshold=30.0,
            classifier_hidden_dim=8,
        )
        pipeline = VariantCallingPipeline(config, rngs=rngs)
        pipeline.eval_mode()

        key = jax.random.PRNGKey(42)
        indices = jax.random.randint(key, (3, 8), 0, 4)
        reads = jax.nn.one_hot(indices, 4)
        positions = jnp.array([0, 5, 10])
        # All quality scores below threshold
        quality = jnp.ones((3, 8)) * 5.0

        data = {"reads": reads, "positions": positions, "quality": quality}
        result_data, _, _ = pipeline.apply(data, {}, None)

        # Should still produce valid output
        assert jnp.all(jnp.isfinite(result_data["logits"]))
        sums = result_data["probabilities"].sum(axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)

    def test_reads_at_reference_end(self, rngs):
        """Test pipeline handles reads at reference end boundary."""
        ref_len = 20
        config = VariantCallingPipelineConfig(
            reference_length=ref_len,
            num_classes=3,
            pileup_window_size=5,
            classifier_hidden_dim=8,
        )
        pipeline = VariantCallingPipeline(config, rngs=rngs)
        pipeline.eval_mode()

        read_length = 8
        key = jax.random.PRNGKey(42)
        indices = jax.random.randint(key, (3, read_length), 0, 4)
        reads = jax.nn.one_hot(indices, 4)
        # Positions near end of reference
        positions = jnp.array([ref_len - 5, ref_len - 3, ref_len - 1])
        quality = jnp.ones((3, read_length)) * 30.0

        data = {"reads": reads, "positions": positions, "quality": quality}
        result_data, _, _ = pipeline.apply(data, {}, None)

        assert result_data["logits"].shape == (ref_len, 3)
        assert jnp.all(jnp.isfinite(result_data["logits"]))
