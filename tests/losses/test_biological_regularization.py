"""Tests for diffbio.losses.biological_regularization module.

These tests define the expected behavior of biological regularization losses
that help prevent adversarial optimization of differentiable bioinformatics
components.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.losses.biological_regularization import (
    BiologicalPlausibilityLoss,
    BiologicalRegularizationConfig,
    GapPatternRegularization,
    GCContentRegularization,
    SequenceComplexityLoss,
)


class TestBiologicalRegularizationConfig:
    """Tests for BiologicalRegularizationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BiologicalRegularizationConfig()
        assert config.gc_content_weight == 1.0
        assert config.gap_pattern_weight == 1.0
        assert config.complexity_weight == 1.0
        assert config.target_gc_content == 0.5
        assert config.target_gc_tolerance == 0.2

    def test_custom_config(self):
        """Test custom configuration values."""
        config = BiologicalRegularizationConfig(
            gc_content_weight=2.0,
            gap_pattern_weight=0.5,
            target_gc_content=0.4,
        )
        assert config.gc_content_weight == 2.0
        assert config.gap_pattern_weight == 0.5
        assert config.target_gc_content == 0.4


class TestGCContentRegularization:
    """Tests for GC content regularization loss."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_initialization(self, rngs):
        """Test GC content regularization initialization."""
        loss = GCContentRegularization(target_gc=0.5, tolerance=0.2, rngs=rngs)
        assert loss is not None
        assert float(loss.target_gc[...]) == 0.5

    def test_optimal_gc_content_low_loss(self, rngs):
        """Test that sequences with target GC content have low loss."""
        loss = GCContentRegularization(target_gc=0.5, tolerance=0.2, rngs=rngs)

        # Create sequence with 50% GC content (2 G/C, 2 A/T)
        # One-hot: A=0, C=1, G=2, T=3
        seq = jnp.array(
            [
                [1, 0, 0, 0],  # A
                [0, 1, 0, 0],  # C
                [0, 0, 1, 0],  # G
                [0, 0, 0, 1],  # T
            ],
            dtype=jnp.float32,
        )

        loss_value = loss(seq)
        # Should be very low since GC content is exactly 0.5
        assert loss_value < 0.1

    def test_extreme_gc_content_high_loss(self, rngs):
        """Test that sequences with extreme GC content have high loss."""
        loss = GCContentRegularization(target_gc=0.5, tolerance=0.2, rngs=rngs)

        # Create sequence with 100% GC content (all G)
        seq = jnp.array(
            [
                [0, 0, 1, 0],  # G
                [0, 0, 1, 0],  # G
                [0, 0, 1, 0],  # G
                [0, 0, 1, 0],  # G
            ],
            dtype=jnp.float32,
        )

        loss_value = loss(seq)
        # GC content is 1.0, target is 0.5, tolerance is 0.2
        # Excess = |1.0 - 0.5| - 0.2 = 0.3, squared = 0.09
        # Should be non-trivial (> 0) since GC content exceeds tolerance
        assert loss_value > 0.05

    def test_gc_loss_differentiable(self, rngs):
        """Test that GC content loss is differentiable."""
        loss_fn = GCContentRegularization(target_gc=0.5, tolerance=0.2, rngs=rngs)

        # Random soft sequence
        seq = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(0), (10, 4)))

        def loss(s):
            return loss_fn(s)

        grad = jax.grad(loss)(seq)
        assert grad is not None
        assert grad.shape == seq.shape


class TestGapPatternRegularization:
    """Tests for gap pattern regularization loss."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_initialization(self, rngs):
        """Test gap pattern regularization initialization."""
        loss = GapPatternRegularization(max_gap_length=10, rngs=rngs)
        assert loss is not None

    def test_no_gaps_low_loss(self, rngs):
        """Test that sequences with no gaps have low gap pattern loss."""
        loss = GapPatternRegularization(max_gap_length=10, rngs=rngs)

        # Alignment with no gaps (high alignment weights)
        alignment_weights = jnp.array(
            [
                [0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.05, 0.9],
            ],
            dtype=jnp.float32,
        )

        loss_value = loss(alignment_weights)
        # Low loss for well-aligned sequences
        assert loss_value < 1.0

    def test_many_gaps_high_loss(self, rngs):
        """Test that alignments with many gaps have high loss."""
        loss = GapPatternRegularization(max_gap_length=10, rngs=rngs)

        # Alignment with gaps (low alignment weights, sparse pattern)
        alignment_weights = jnp.array(
            [
                [0.1, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.1],
            ],
            dtype=jnp.float32,
        )

        loss_value = loss(alignment_weights)
        # Higher loss for gappy alignments
        assert loss_value > 0.0

    def test_gap_loss_differentiable(self, rngs):
        """Test that gap pattern loss is differentiable."""
        loss_fn = GapPatternRegularization(max_gap_length=10, rngs=rngs)

        alignment = jax.nn.softmax(
            jax.random.normal(jax.random.PRNGKey(0), (5, 5)), axis=-1
        )

        def loss(a):
            return loss_fn(a)

        grad = jax.grad(loss)(alignment)
        assert grad is not None
        assert grad.shape == alignment.shape


class TestSequenceComplexityLoss:
    """Tests for sequence complexity regularization."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_initialization(self, rngs):
        """Test sequence complexity loss initialization."""
        loss = SequenceComplexityLoss(min_entropy=1.0, rngs=rngs)
        assert loss is not None

    def test_uniform_sequence_high_complexity(self, rngs):
        """Test that uniform distribution has high complexity (low loss)."""
        loss = SequenceComplexityLoss(min_entropy=1.0, rngs=rngs)

        # Uniform soft sequence (all positions equally likely)
        seq = jnp.ones((10, 4)) / 4.0

        loss_value = loss(seq)
        # High complexity (entropy ~2.0) should have low loss
        assert loss_value < 0.5

    def test_degenerate_sequence_low_complexity(self, rngs):
        """Test that highly biased sequence has low complexity (high loss)."""
        loss = SequenceComplexityLoss(min_entropy=1.0, rngs=rngs)

        # Very low entropy sequence (almost all A)
        seq = jnp.array(
            [
                [0.97, 0.01, 0.01, 0.01],
                [0.97, 0.01, 0.01, 0.01],
                [0.97, 0.01, 0.01, 0.01],
                [0.97, 0.01, 0.01, 0.01],
            ],
            dtype=jnp.float32,
        )

        loss_value = loss(seq)
        # Low complexity should have higher loss
        assert loss_value > 0.5

    def test_complexity_loss_differentiable(self, rngs):
        """Test that complexity loss is differentiable."""
        loss_fn = SequenceComplexityLoss(min_entropy=1.0, rngs=rngs)

        seq = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(0), (10, 4)))

        def loss(s):
            return loss_fn(s)

        grad = jax.grad(loss)(seq)
        assert grad is not None
        assert grad.shape == seq.shape


class TestBiologicalPlausibilityLoss:
    """Tests for the combined biological plausibility loss."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_initialization(self, rngs):
        """Test combined loss initialization."""
        config = BiologicalRegularizationConfig()
        loss = BiologicalPlausibilityLoss(config, rngs=rngs)
        assert loss is not None

    def test_plausible_sequence_low_loss(self, rngs):
        """Test that biologically plausible sequences have low combined loss."""
        config = BiologicalRegularizationConfig(
            gc_content_weight=1.0,
            complexity_weight=1.0,
        )
        loss_fn = BiologicalPlausibilityLoss(config, rngs=rngs)

        # Biologically plausible sequence: balanced GC, good complexity
        # Approximately 50% GC, uniform-ish distribution
        seq = jnp.array(
            [
                [0.4, 0.2, 0.2, 0.2],  # Slightly A-biased
                [0.2, 0.4, 0.2, 0.2],  # Slightly C-biased
                [0.2, 0.2, 0.4, 0.2],  # Slightly G-biased
                [0.2, 0.2, 0.2, 0.4],  # Slightly T-biased
            ],
            dtype=jnp.float32,
        )

        loss_value = loss_fn(seq)
        # Plausible sequences should have reasonable loss
        assert loss_value < 2.0

    def test_combined_loss_differentiable(self, rngs):
        """Test that combined loss is differentiable."""
        config = BiologicalRegularizationConfig()
        loss_fn = BiologicalPlausibilityLoss(config, rngs=rngs)

        seq = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(0), (10, 4)))

        def loss(s):
            return loss_fn(s)

        grad = jax.grad(loss)(seq)
        assert grad is not None
        assert grad.shape == seq.shape

    def test_weight_effects(self, rngs):
        """Test that weights affect the loss components."""
        # High GC weight
        config_gc = BiologicalRegularizationConfig(
            gc_content_weight=10.0,
            complexity_weight=0.0,
        )
        loss_gc = BiologicalPlausibilityLoss(config_gc, rngs=rngs)

        # High complexity weight
        config_complexity = BiologicalRegularizationConfig(
            gc_content_weight=0.0,
            complexity_weight=10.0,
        )
        loss_complexity = BiologicalPlausibilityLoss(config_complexity, rngs=rngs)

        # Sequence with extreme GC (all G) but uniform complexity
        seq = jnp.array(
            [
                [0.0, 0.0, 1.0, 0.0],  # G
                [0.0, 0.0, 1.0, 0.0],  # G
                [0.0, 0.0, 1.0, 0.0],  # G
                [0.0, 0.0, 1.0, 0.0],  # G
            ],
            dtype=jnp.float32,
        )

        gc_loss = loss_gc(seq)

        # GC-weighted loss should be much higher (extreme GC)
        assert gc_loss > 0.0
        # Complexity loss also works (just verify no error)
        _ = loss_complexity(seq)


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_gc_loss_jit_compatible(self, rngs):
        """Test GC content loss works with JIT."""
        loss_fn = GCContentRegularization(target_gc=0.5, tolerance=0.2, rngs=rngs)
        seq = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(0), (10, 4)))

        @jax.jit
        def jit_loss(s):
            return loss_fn(s)

        result = jit_loss(seq)
        assert result is not None

    def test_combined_loss_jit_compatible(self, rngs):
        """Test combined loss works with JIT."""
        config = BiologicalRegularizationConfig()
        loss_fn = BiologicalPlausibilityLoss(config, rngs=rngs)
        seq = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(0), (10, 4)))

        @jax.jit
        def jit_loss(s):
            return loss_fn(s)

        result = jit_loss(seq)
        assert result is not None
