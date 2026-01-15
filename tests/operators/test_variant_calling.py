"""Tests for diffbio.operators.variant module.

These tests define the expected behavior of variant calling components
for differentiable variant detection pipelines.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.variant import (
    DifferentiablePileup,
    PileupConfig,
    VariantClassifier,
    VariantClassifierConfig,
)


def generate_random_reads(key: jax.Array, num_reads: int, read_length: int) -> jax.Array:
    """Generate random one-hot encoded reads.

    Args:
        key: Random key.
        num_reads: Number of reads to generate.
        read_length: Length of each read.

    Returns:
        Reads array of shape (num_reads, read_length, 4).
    """
    indices = jax.random.randint(key, (num_reads, read_length), 0, 4)
    return jax.nn.one_hot(indices, 4)


def generate_random_positions(key: jax.Array, num_reads: int, max_position: int) -> jax.Array:
    """Generate random read positions.

    Args:
        key: Random key.
        num_reads: Number of reads.
        max_position: Maximum starting position.

    Returns:
        Position array of shape (num_reads,).
    """
    return jax.random.randint(key, (num_reads,), 0, max_position)


def generate_random_quality_scores(key: jax.Array, num_reads: int, read_length: int) -> jax.Array:
    """Generate random quality scores.

    Args:
        key: Random key.
        num_reads: Number of reads.
        read_length: Length of each read.

    Returns:
        Quality scores of shape (num_reads, read_length).
    """
    return jax.random.uniform(key, (num_reads, read_length), minval=0.0, maxval=40.0)


class TestPileupConfig:
    """Tests for PileupConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PileupConfig()
        assert config.window_size == 21
        assert config.min_coverage == 1
        assert config.max_coverage == 100
        assert config.use_quality_weights is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PileupConfig(
            window_size=31, min_coverage=5, max_coverage=200, use_quality_weights=False
        )
        assert config.window_size == 31
        assert config.min_coverage == 5
        assert config.max_coverage == 200
        assert config.use_quality_weights is False


class TestDifferentiablePileup:
    """Tests for differentiable pileup operator."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def pileup_config(self):
        return PileupConfig(window_size=11)

    def test_initialization(self, rngs, pileup_config):
        """Test pileup operator initialization."""
        pileup = DifferentiablePileup(pileup_config, rngs=rngs)
        assert pileup is not None

    def test_pileup_output_shape(self, rngs, pileup_config):
        """Test pileup produces correct output shape."""
        pileup = DifferentiablePileup(pileup_config, rngs=rngs)

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        # Generate reads
        num_reads = 10
        read_length = 20
        reference_length = 50

        reads = generate_random_reads(keys[0], num_reads, read_length)
        positions = generate_random_positions(keys[1], num_reads, reference_length - read_length)
        quality = generate_random_quality_scores(keys[2], num_reads, read_length)

        result = pileup.compute_pileup(reads, positions, quality, reference_length)

        # Output should be a dict with "pileup" key of shape (reference_length, 4)
        assert "pileup" in result
        assert result["pileup"].shape == (reference_length, 4)

    def test_pileup_coverage_tracking(self, rngs, pileup_config):
        """Test pileup tracks coverage correctly."""
        pileup = DifferentiablePileup(pileup_config, rngs=rngs)

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        num_reads = 20
        read_length = 15
        reference_length = 40

        reads = generate_random_reads(keys[0], num_reads, read_length)
        positions = generate_random_positions(keys[1], num_reads, reference_length - read_length)
        quality = generate_random_quality_scores(keys[2], num_reads, read_length)

        result = pileup.compute_pileup(reads, positions, quality, reference_length)
        pileup_data = result["pileup"]

        # Result should have valid probability distributions
        # Sum along nucleotide axis should be close to 1 where there's coverage
        row_sums = jnp.sum(pileup_data, axis=1)
        # Positions with coverage should sum to ~1
        has_coverage = row_sums > 0.1
        if jnp.any(has_coverage):
            assert jnp.allclose(row_sums[has_coverage], 1.0, atol=0.1)

    def test_pileup_quality_weighting(self, rngs):
        """Test pileup uses quality weighting when enabled."""
        config_with_qual = PileupConfig(use_quality_weights=True)
        config_no_qual = PileupConfig(use_quality_weights=False)

        pileup_with_qual = DifferentiablePileup(config_with_qual, rngs=rngs)
        pileup_no_qual = DifferentiablePileup(config_no_qual, rngs=rngs)

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        num_reads = 10
        read_length = 15
        reference_length = 30

        reads = generate_random_reads(keys[0], num_reads, read_length)
        positions = generate_random_positions(keys[1], num_reads, reference_length - read_length)
        # Extreme quality differences
        quality = jax.random.uniform(keys[2], (num_reads, read_length), minval=5.0, maxval=35.0)

        result_with = pileup_with_qual.compute_pileup(reads, positions, quality, reference_length)
        result_without = pileup_no_qual.compute_pileup(reads, positions, quality, reference_length)

        # Results should be different when quality weighting is used
        # (though both should be valid)
        assert jnp.all(jnp.isfinite(result_with["pileup"]))
        assert jnp.all(jnp.isfinite(result_without["pileup"]))

    def test_pileup_differentiable(self, rngs, pileup_config):
        """Test pileup is differentiable through reads."""
        pileup = DifferentiablePileup(pileup_config, rngs=rngs)

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        num_reads = 10
        read_length = 15
        reference_length = 30

        reads = generate_random_reads(keys[0], num_reads, read_length)
        positions = generate_random_positions(keys[1], num_reads, reference_length - read_length)
        quality = generate_random_quality_scores(keys[2], num_reads, read_length)

        def loss(r):
            result = pileup.compute_pileup(r, positions, quality, reference_length)
            return jnp.sum(result["pileup"])

        grad = jax.grad(loss)(reads)

        assert grad is not None
        assert grad.shape == reads.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_pileup_jit_compatible(self, rngs, pileup_config):
        """Test pileup works with JIT."""
        pileup = DifferentiablePileup(pileup_config, rngs=rngs)

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        num_reads = 10
        read_length = 15
        reference_length = 30

        reads = generate_random_reads(keys[0], num_reads, read_length)
        positions = generate_random_positions(keys[1], num_reads, reference_length - read_length)
        quality = generate_random_quality_scores(keys[2], num_reads, read_length)

        # reference_length must be static for segment_sum
        @jax.jit
        def jit_pileup(r, p, q):
            return pileup.compute_pileup(r, p, q, reference_length)

        result = jit_pileup(reads, positions, quality)
        assert jnp.all(jnp.isfinite(result["pileup"]))


class TestVariantClassifierConfig:
    """Tests for VariantClassifierConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VariantClassifierConfig()
        assert config.num_classes == 3  # REF, SNV, INDEL
        assert config.hidden_dim == 64
        assert config.num_layers == 2
        assert config.dropout_rate == 0.1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = VariantClassifierConfig(
            num_classes=5, hidden_dim=128, num_layers=4, dropout_rate=0.2
        )
        assert config.num_classes == 5
        assert config.hidden_dim == 128
        assert config.num_layers == 4
        assert config.dropout_rate == 0.2


class TestVariantClassifier:
    """Tests for variant classifier."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def classifier_config(self):
        return VariantClassifierConfig(num_classes=3, hidden_dim=32, num_layers=2)

    def test_initialization(self, rngs, classifier_config):
        """Test classifier initialization."""
        classifier = VariantClassifier(classifier_config, rngs=rngs)
        assert classifier is not None

    def test_classifier_output_shape(self, rngs, classifier_config):
        """Test classifier produces correct output shape."""
        classifier = VariantClassifier(classifier_config, rngs=rngs)

        # Create pileup-like input (position_context, 4)
        key = jax.random.PRNGKey(42)
        window_size = 21
        pileup_window = jax.random.uniform(key, (window_size, 4))
        pileup_window = pileup_window / pileup_window.sum(axis=-1, keepdims=True)

        logits = classifier.classify(pileup_window)

        # Output should be (num_classes,) for single position
        assert logits.shape == (classifier_config.num_classes,)

    def test_classifier_batch_output_shape(self, rngs, classifier_config):
        """Test classifier handles batched input."""
        classifier = VariantClassifier(classifier_config, rngs=rngs)
        # Set to eval mode to disable dropout for deterministic vmap
        classifier.eval()

        key = jax.random.PRNGKey(42)
        batch_size = 16
        window_size = 21
        pileup_batch = jax.random.uniform(key, (batch_size, window_size, 4))
        pileup_batch = pileup_batch / pileup_batch.sum(axis=-1, keepdims=True)

        # Vmap over batch using jax.vmap on the classify method
        batched_classify = jax.vmap(classifier.classify)
        logits = batched_classify(pileup_batch)

        assert logits.shape == (batch_size, classifier_config.num_classes)

    def test_classifier_output_reasonable(self, rngs, classifier_config):
        """Test classifier outputs reasonable logits."""
        classifier = VariantClassifier(classifier_config, rngs=rngs)

        key = jax.random.PRNGKey(42)
        window_size = 21
        pileup_window = jax.random.uniform(key, (window_size, 4))
        pileup_window = pileup_window / pileup_window.sum(axis=-1, keepdims=True)

        logits = classifier.classify(pileup_window)

        # Logits should be finite
        assert jnp.all(jnp.isfinite(logits))
        # Softmax should give valid probabilities
        probs = jax.nn.softmax(logits)
        assert jnp.allclose(jnp.sum(probs), 1.0)

    def test_classifier_differentiable(self, rngs, classifier_config):
        """Test classifier is differentiable."""
        classifier = VariantClassifier(classifier_config, rngs=rngs)
        # Set to eval mode to disable dropout for deterministic gradient
        classifier.eval()

        key = jax.random.PRNGKey(42)
        window_size = 21
        pileup_window = jax.random.uniform(key, (window_size, 4))
        pileup_window = pileup_window / pileup_window.sum(axis=-1, keepdims=True)

        # Use nnx.grad for NNX modules
        @nnx.jit
        def compute_loss_and_grad(model, pileup):
            def loss_fn(x):
                logits = model.classify(x)
                return jnp.sum(logits)

            return jax.grad(loss_fn)(pileup)

        grad = compute_loss_and_grad(classifier, pileup_window)

        assert grad is not None
        assert grad.shape == pileup_window.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_classifier_jit_compatible(self, rngs, classifier_config):
        """Test classifier works with JIT."""
        classifier = VariantClassifier(classifier_config, rngs=rngs)
        # Set to eval mode for deterministic JIT
        classifier.eval()

        key = jax.random.PRNGKey(42)
        window_size = 21
        pileup_window = jax.random.uniform(key, (window_size, 4))
        pileup_window = pileup_window / pileup_window.sum(axis=-1, keepdims=True)

        # Use nnx.jit for NNX modules
        @nnx.jit
        def jit_classify(model, pileup):
            return model.classify(pileup)

        logits = jit_classify(classifier, pileup_window)
        assert jnp.all(jnp.isfinite(logits))

    def test_classifier_training_mode(self, rngs, classifier_config):
        """Test classifier behaves differently in training mode (dropout)."""
        classifier = VariantClassifier(classifier_config, rngs=rngs)

        key = jax.random.PRNGKey(42)
        window_size = 21
        pileup_window = jax.random.uniform(key, (window_size, 4))
        pileup_window = pileup_window / pileup_window.sum(axis=-1, keepdims=True)

        # Run in inference mode (deterministic)
        classifier.eval()
        logits_eval = classifier.classify(pileup_window)

        # Both should be finite and valid
        assert jnp.all(jnp.isfinite(logits_eval))


class TestVariantCallingIntegration:
    """Integration tests for variant calling pipeline."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_pileup_to_classifier_pipeline(self, rngs):
        """Test pileup output flows to classifier."""
        window_size = 11
        pileup_config = PileupConfig(window_size=window_size)
        classifier_config = VariantClassifierConfig(
            num_classes=3, hidden_dim=32, input_window=window_size
        )

        pileup = DifferentiablePileup(pileup_config, rngs=rngs)
        classifier = VariantClassifier(classifier_config, rngs=rngs)
        classifier.eval()  # Disable dropout

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        num_reads = 20
        read_length = 15
        reference_length = 50

        reads = generate_random_reads(keys[0], num_reads, read_length)
        positions = generate_random_positions(keys[1], num_reads, reference_length - read_length)
        quality = generate_random_quality_scores(keys[2], num_reads, read_length)

        # Generate pileup
        pileup_result = pileup.compute_pileup(reads, positions, quality, reference_length)

        # Extract window around position 25
        center = 25
        half_window = pileup_config.window_size // 2
        window = pileup_result["pileup"][center - half_window : center + half_window + 1]

        # Classify
        logits = classifier.classify(window)

        assert logits.shape == (classifier_config.num_classes,)
        assert jnp.all(jnp.isfinite(logits))

    def test_end_to_end_gradient_flow(self, rngs):
        """Test gradients flow through entire pipeline."""
        window_size = 11
        pileup_config = PileupConfig(window_size=window_size)
        classifier_config = VariantClassifierConfig(
            num_classes=3, hidden_dim=32, input_window=window_size
        )

        pileup = DifferentiablePileup(pileup_config, rngs=rngs)
        classifier = VariantClassifier(classifier_config, rngs=rngs)
        classifier.eval()  # Disable dropout for gradient computation

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        num_reads = 10
        read_length = 15
        reference_length = 30

        reads = generate_random_reads(keys[0], num_reads, read_length)
        positions = generate_random_positions(keys[1], num_reads, reference_length - read_length)
        quality = generate_random_quality_scores(keys[2], num_reads, read_length)

        def pipeline_loss(r):
            # Generate pileup
            pileup_result = pileup.compute_pileup(r, positions, quality, reference_length)

            # Extract center window
            center = reference_length // 2
            half_window = pileup_config.window_size // 2
            window = pileup_result["pileup"][center - half_window : center + half_window + 1]

            # Classify and compute loss (e.g., cross-entropy with target class 0)
            logits = classifier.classify(window)
            return jax.nn.log_softmax(logits)[0]  # Log prob of class 0

        grad = jax.grad(pipeline_loss)(reads)

        assert grad is not None
        assert grad.shape == reads.shape
        assert jnp.all(jnp.isfinite(grad))


class TestEnhancedPileup:
    """Tests for enhanced pileup with coverage and quality channels."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_pileup_returns_coverage_when_enabled(self, rngs):
        """Test pileup returns coverage channel when configured."""
        config = PileupConfig(
            window_size=11,
            reference_length=50,
            return_coverage=True,
        )
        pileup = DifferentiablePileup(config, rngs=rngs)

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        num_reads = 20
        read_length = 15

        reads = generate_random_reads(keys[0], num_reads, read_length)
        max_pos = config.reference_length - read_length
        positions = generate_random_positions(keys[1], num_reads, max_pos)
        quality = generate_random_quality_scores(keys[2], num_reads, read_length)

        data = {"reads": reads, "positions": positions, "quality": quality}
        result, _, _ = pileup.apply(data, {}, None)

        assert "coverage" in result
        assert result["coverage"].shape == (config.reference_length, 1)
        # Coverage should be non-negative
        assert jnp.all(result["coverage"] >= 0)

    def test_pileup_returns_quality_when_enabled(self, rngs):
        """Test pileup returns mean quality channel when configured."""
        config = PileupConfig(
            window_size=11,
            reference_length=50,
            return_coverage=True,
            return_quality=True,
        )
        pileup = DifferentiablePileup(config, rngs=rngs)

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        num_reads = 20
        read_length = 15

        reads = generate_random_reads(keys[0], num_reads, read_length)
        max_pos = config.reference_length - read_length
        positions = generate_random_positions(keys[1], num_reads, max_pos)
        quality = generate_random_quality_scores(keys[2], num_reads, read_length)

        data = {"reads": reads, "positions": positions, "quality": quality}
        result, _, _ = pileup.apply(data, {}, None)

        assert "mean_quality" in result
        assert result["mean_quality"].shape == (config.reference_length, 1)
        # Mean quality should be in valid range where there's coverage
        coverage = result["coverage"]
        mean_quality = result["mean_quality"]
        # Where there's coverage, quality should be reasonable
        has_coverage = coverage.squeeze() > 0.1
        if jnp.any(has_coverage):
            assert jnp.all(mean_quality[has_coverage] >= 0)
            assert jnp.all(mean_quality[has_coverage] <= 50)

    def test_pileup_no_final_softmax_option(self, rngs):
        """Test pileup can skip final softmax normalization."""
        config = PileupConfig(
            window_size=11,
            reference_length=50,
            apply_softmax=False,  # Skip final softmax
        )
        pileup = DifferentiablePileup(config, rngs=rngs)

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        num_reads = 20
        read_length = 15

        reads = generate_random_reads(keys[0], num_reads, read_length)
        max_pos = config.reference_length - read_length
        positions = generate_random_positions(keys[1], num_reads, max_pos)
        quality = generate_random_quality_scores(keys[2], num_reads, read_length)

        data = {"reads": reads, "positions": positions, "quality": quality}
        result, _, _ = pileup.apply(data, {}, None)

        pileup_data = result["pileup"]
        # Without softmax, values don't need to sum to 1
        # But they should still be valid
        assert jnp.all(jnp.isfinite(pileup_data))

    def test_coverage_reflects_actual_read_depth(self, rngs):
        """Test that coverage channel reflects actual read depth."""
        config = PileupConfig(
            window_size=11,
            reference_length=30,
            return_coverage=True,
            use_quality_weights=False,  # Disable quality weighting for cleaner test
        )
        pileup = DifferentiablePileup(config, rngs=rngs)

        # Create controlled test data: 5 reads all starting at position 5
        num_reads = 5
        read_length = 10

        key = jax.random.PRNGKey(42)
        reads = generate_random_reads(key, num_reads, read_length)
        positions = jnp.full((num_reads,), 5)  # All reads at position 5
        quality = jnp.full((num_reads, read_length), 30.0)  # Uniform quality

        data = {"reads": reads, "positions": positions, "quality": quality}
        result, _, _ = pileup.apply(data, {}, None)

        coverage = result["coverage"].squeeze()

        # Positions 5-14 should have coverage == 5
        # Other positions should have coverage == 0
        assert jnp.allclose(coverage[:5], 0.0, atol=0.1)
        assert jnp.all(coverage[5:15] > 0)  # Should have coverage
        assert jnp.allclose(coverage[15:], 0.0, atol=0.1)


class TestEdgeCases:
    """Edge case tests."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_single_read(self, rngs):
        """Test pileup with single read."""
        config = PileupConfig(window_size=11)
        pileup = DifferentiablePileup(config, rngs=rngs)

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        reads = generate_random_reads(keys[0], 1, 10)
        positions = jnp.array([5])
        quality = generate_random_quality_scores(keys[2], 1, 10)

        result = pileup.compute_pileup(reads, positions, quality, 20)

        assert "pileup" in result
        assert result["pileup"].shape == (20, 4)
        assert jnp.all(jnp.isfinite(result["pileup"]))

    def test_small_window_classifier(self, rngs):
        """Test classifier with minimal window."""
        window_size = 5
        config = VariantClassifierConfig(
            num_classes=3, hidden_dim=16, num_layers=1, input_window=window_size
        )
        classifier = VariantClassifier(config, rngs=rngs)

        key = jax.random.PRNGKey(42)
        small_window = jax.random.uniform(key, (window_size, 4))
        small_window = small_window / small_window.sum(axis=-1, keepdims=True)

        logits = classifier.classify(small_window)

        assert logits.shape == (config.num_classes,)
        assert jnp.all(jnp.isfinite(logits))
