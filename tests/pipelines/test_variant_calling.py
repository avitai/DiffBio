"""Tests for the end-to-end variant calling pipeline."""

import jax
import jax.numpy as jnp
import pytest
from datarax.core.element_batch import Batch, Element
from flax import nnx

from diffbio.pipelines import (
    VariantCallingPipeline,
    VariantCallingPipelineConfig,
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
    def rngs(self):
        return nnx.Rngs(42)

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
        assert pipeline.pipeline_config.reference_length == 30
        assert pipeline.pipeline_config.num_classes == 3

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


class TestVariantCallingPipelineGradients:
    """Tests for gradient flow through the pipeline."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

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
        assert pipeline.pipeline_config.reference_length == 50

    def test_factory_with_custom_params(self):
        """Test factory with custom parameters."""
        pipeline = create_variant_calling_pipeline(
            reference_length=100,
            num_classes=4,
            quality_threshold=25.0,
            hidden_dim=128,
            seed=123,
        )

        assert pipeline.pipeline_config.reference_length == 100
        assert pipeline.pipeline_config.num_classes == 4
        assert pipeline.pipeline_config.quality_threshold == 25.0
        assert pipeline.pipeline_config.classifier_hidden_dim == 128
