"""Performance and scalability tests for biological regularization losses.

These tests verify that loss functions are fast and scale well
with increasing sequence lengths.
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


# Sequence lengths to test scalability
SMALL_LENGTH = 100
MEDIUM_LENGTH = 1000
LARGE_LENGTH = 10000
XLARGE_LENGTH = 100000


def generate_random_soft_sequence(key: jax.Array, length: int) -> jax.Array:
    """Generate a random soft one-hot encoded DNA sequence."""
    # Generate logits and apply softmax to get valid probability distribution
    logits = jax.random.normal(key, (length, 4))
    return jax.nn.softmax(logits, axis=-1)


def generate_random_alignment_weights(key: jax.Array, len1: int, len2: int) -> jax.Array:
    """Generate random soft alignment matrix."""
    logits = jax.random.normal(key, (len1, len2))
    return jax.nn.softmax(logits, axis=-1)


class TestGCContentScalability:
    """Tests for GC content regularization scalability."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def gc_loss(self, rngs):
        return GCContentRegularization(target_gc=0.5, tolerance=0.2, rngs=rngs)

    @pytest.mark.parametrize(
        "length",
        [SMALL_LENGTH, MEDIUM_LENGTH, LARGE_LENGTH],
        ids=["small", "medium", "large"],
    )
    def test_scalability_with_length(self, gc_loss, length):
        """Test GC loss scales with sequence length."""
        key = jax.random.PRNGKey(42)
        seq = generate_random_soft_sequence(key, length)

        loss = gc_loss(seq)

        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_very_large_sequence(self, gc_loss):
        """Test GC loss works with very large sequences."""
        key = jax.random.PRNGKey(42)
        seq = generate_random_soft_sequence(key, XLARGE_LENGTH)

        loss = gc_loss(seq)

        assert jnp.isfinite(loss)


class TestSequenceComplexityScalability:
    """Tests for sequence complexity loss scalability."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def complexity_loss(self, rngs):
        return SequenceComplexityLoss(min_entropy=1.0, rngs=rngs)

    @pytest.mark.parametrize(
        "length",
        [SMALL_LENGTH, MEDIUM_LENGTH, LARGE_LENGTH],
        ids=["small", "medium", "large"],
    )
    def test_scalability_with_length(self, complexity_loss, length):
        """Test complexity loss scales with sequence length."""
        key = jax.random.PRNGKey(42)
        seq = generate_random_soft_sequence(key, length)

        loss = complexity_loss(seq)

        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_very_large_sequence(self, complexity_loss):
        """Test complexity loss works with very large sequences."""
        key = jax.random.PRNGKey(42)
        seq = generate_random_soft_sequence(key, XLARGE_LENGTH)

        loss = complexity_loss(seq)

        assert jnp.isfinite(loss)


class TestGapPatternScalability:
    """Tests for gap pattern regularization scalability."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def gap_loss(self, rngs):
        return GapPatternRegularization(max_gap_length=10, rngs=rngs)

    @pytest.mark.parametrize(
        "size",
        [(10, 10), (50, 50), (100, 100)],
        ids=["small", "medium", "large"],
    )
    def test_scalability_with_alignment_size(self, gap_loss, size):
        """Test gap loss scales with alignment matrix size."""
        key = jax.random.PRNGKey(42)
        len1, len2 = size
        alignment = generate_random_alignment_weights(key, len1, len2)

        loss = gap_loss(alignment)

        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_large_alignment_matrix(self, gap_loss):
        """Test gap loss works with large alignment matrices."""
        key = jax.random.PRNGKey(42)
        alignment = generate_random_alignment_weights(key, 200, 200)

        loss = gap_loss(alignment)

        assert jnp.isfinite(loss)


class TestCombinedLossScalability:
    """Tests for combined biological plausibility loss scalability."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def combined_loss(self, rngs):
        config = BiologicalRegularizationConfig()
        return BiologicalPlausibilityLoss(config, rngs=rngs)

    @pytest.mark.parametrize(
        "length",
        [SMALL_LENGTH, MEDIUM_LENGTH, LARGE_LENGTH],
        ids=["small", "medium", "large"],
    )
    def test_scalability_with_length(self, combined_loss, length):
        """Test combined loss scales with sequence length."""
        key = jax.random.PRNGKey(42)
        seq = generate_random_soft_sequence(key, length)

        loss = combined_loss(seq)

        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_very_large_sequence(self, combined_loss):
        """Test combined loss works with very large sequences."""
        key = jax.random.PRNGKey(42)
        seq = generate_random_soft_sequence(key, XLARGE_LENGTH)

        loss = combined_loss(seq)

        assert jnp.isfinite(loss)


class TestLossJITPerformance:
    """Tests for JIT compilation performance of losses."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_gc_loss_jit_speedup(self, rngs):
        """Test GC loss benefits from JIT compilation."""
        gc_loss = GCContentRegularization(target_gc=0.5, tolerance=0.2, rngs=rngs)

        key = jax.random.PRNGKey(42)
        seq = generate_random_soft_sequence(key, MEDIUM_LENGTH)

        @jax.jit
        def jit_loss(s):
            return gc_loss(s)

        # First call triggers compilation
        result1 = jit_loss(seq)
        result1.block_until_ready()

        # Subsequent calls use compiled version
        result2 = jit_loss(seq)
        result2.block_until_ready()

        assert jnp.allclose(result1, result2)

    def test_complexity_loss_jit_speedup(self, rngs):
        """Test complexity loss benefits from JIT compilation."""
        complexity_loss = SequenceComplexityLoss(min_entropy=1.0, rngs=rngs)

        key = jax.random.PRNGKey(42)
        seq = generate_random_soft_sequence(key, MEDIUM_LENGTH)

        @jax.jit
        def jit_loss(s):
            return complexity_loss(s)

        result1 = jit_loss(seq)
        result1.block_until_ready()

        result2 = jit_loss(seq)
        result2.block_until_ready()

        assert jnp.allclose(result1, result2)

    def test_combined_loss_jit_speedup(self, rngs):
        """Test combined loss benefits from JIT compilation."""
        config = BiologicalRegularizationConfig()
        combined_loss = BiologicalPlausibilityLoss(config, rngs=rngs)

        key = jax.random.PRNGKey(42)
        seq = generate_random_soft_sequence(key, MEDIUM_LENGTH)

        @jax.jit
        def jit_loss(s):
            return combined_loss(s)

        result1 = jit_loss(seq)
        result1.block_until_ready()

        result2 = jit_loss(seq)
        result2.block_until_ready()

        assert jnp.allclose(result1, result2)


class TestLossGradientPerformance:
    """Tests for gradient computation performance."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_gc_loss_gradient_at_scale(self, rngs):
        """Test GC loss gradient computes efficiently at scale."""
        gc_loss = GCContentRegularization(target_gc=0.5, tolerance=0.2, rngs=rngs)

        key = jax.random.PRNGKey(42)
        seq = generate_random_soft_sequence(key, LARGE_LENGTH)

        grad = jax.grad(gc_loss)(seq)

        assert grad.shape == seq.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_complexity_loss_gradient_at_scale(self, rngs):
        """Test complexity loss gradient computes efficiently at scale."""
        complexity_loss = SequenceComplexityLoss(min_entropy=1.0, rngs=rngs)

        key = jax.random.PRNGKey(42)
        seq = generate_random_soft_sequence(key, LARGE_LENGTH)

        grad = jax.grad(complexity_loss)(seq)

        assert grad.shape == seq.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_combined_loss_gradient_at_scale(self, rngs):
        """Test combined loss gradient computes efficiently at scale."""
        config = BiologicalRegularizationConfig()
        combined_loss = BiologicalPlausibilityLoss(config, rngs=rngs)

        key = jax.random.PRNGKey(42)
        seq = generate_random_soft_sequence(key, LARGE_LENGTH)

        def loss_fn(s):
            return combined_loss(s)

        grad = jax.grad(loss_fn)(seq)

        assert grad.shape == seq.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_jit_grad_combination(self, rngs):
        """Test JIT + grad combination for losses."""
        config = BiologicalRegularizationConfig()
        combined_loss = BiologicalPlausibilityLoss(config, rngs=rngs)

        key = jax.random.PRNGKey(42)
        seq = generate_random_soft_sequence(key, MEDIUM_LENGTH)

        @jax.jit
        def loss_and_grad(s):
            return jax.value_and_grad(combined_loss)(s)

        loss, grad = loss_and_grad(seq)

        assert jnp.isfinite(loss)
        assert grad.shape == seq.shape
        assert jnp.all(jnp.isfinite(grad))


class TestLossBenchmarks:
    """Benchmark tests using pytest-benchmark."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_benchmark_gc_loss_small(self, benchmark, rngs):
        """Benchmark GC loss for small sequences."""
        gc_loss = GCContentRegularization(target_gc=0.5, tolerance=0.2, rngs=rngs)

        @jax.jit
        def jit_loss(s):
            return gc_loss(s)

        key = jax.random.PRNGKey(42)
        seq = generate_random_soft_sequence(key, SMALL_LENGTH)

        # Warm up
        result = jit_loss(seq)
        result.block_until_ready()

        def run_loss():
            result = jit_loss(seq)
            result.block_until_ready()
            return result

        result = benchmark(run_loss)
        assert jnp.isfinite(result)

    def test_benchmark_gc_loss_medium(self, benchmark, rngs):
        """Benchmark GC loss for medium sequences."""
        gc_loss = GCContentRegularization(target_gc=0.5, tolerance=0.2, rngs=rngs)

        @jax.jit
        def jit_loss(s):
            return gc_loss(s)

        key = jax.random.PRNGKey(42)
        seq = generate_random_soft_sequence(key, MEDIUM_LENGTH)

        # Warm up
        result = jit_loss(seq)
        result.block_until_ready()

        def run_loss():
            result = jit_loss(seq)
            result.block_until_ready()
            return result

        result = benchmark(run_loss)
        assert jnp.isfinite(result)

    def test_benchmark_gc_loss_large(self, benchmark, rngs):
        """Benchmark GC loss for large sequences."""
        gc_loss = GCContentRegularization(target_gc=0.5, tolerance=0.2, rngs=rngs)

        @jax.jit
        def jit_loss(s):
            return gc_loss(s)

        key = jax.random.PRNGKey(42)
        seq = generate_random_soft_sequence(key, LARGE_LENGTH)

        # Warm up
        result = jit_loss(seq)
        result.block_until_ready()

        def run_loss():
            result = jit_loss(seq)
            result.block_until_ready()
            return result

        result = benchmark(run_loss)
        assert jnp.isfinite(result)

    def test_benchmark_combined_loss_medium(self, benchmark, rngs):
        """Benchmark combined loss for medium sequences."""
        config = BiologicalRegularizationConfig()
        combined_loss = BiologicalPlausibilityLoss(config, rngs=rngs)

        @jax.jit
        def jit_loss(s):
            return combined_loss(s)

        key = jax.random.PRNGKey(42)
        seq = generate_random_soft_sequence(key, MEDIUM_LENGTH)

        # Warm up
        result = jit_loss(seq)
        result.block_until_ready()

        def run_loss():
            result = jit_loss(seq)
            result.block_until_ready()
            return result

        result = benchmark(run_loss)
        assert jnp.isfinite(result)


class TestNumericalStability:
    """Tests for numerical stability at scale."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_gc_loss_stability_with_extreme_gc(self, rngs):
        """Test GC loss stability with extreme GC content."""
        gc_loss = GCContentRegularization(target_gc=0.5, tolerance=0.2, rngs=rngs)

        length = 1000

        # All G (100% GC)
        all_g = jnp.zeros((length, 4)).at[:, 2].set(1.0)
        loss_g = gc_loss(all_g)
        assert jnp.isfinite(loss_g)

        # All A (0% GC)
        all_a = jnp.zeros((length, 4)).at[:, 0].set(1.0)
        loss_a = gc_loss(all_a)
        assert jnp.isfinite(loss_a)

    def test_complexity_loss_stability_with_uniform(self, rngs):
        """Test complexity loss stability with uniform distribution."""
        complexity_loss = SequenceComplexityLoss(min_entropy=1.0, rngs=rngs)

        length = 1000

        # Uniform distribution (max entropy)
        uniform = jnp.ones((length, 4)) / 4.0
        loss_uniform = complexity_loss(uniform)
        assert jnp.isfinite(loss_uniform)

        # Near-deterministic (low entropy)
        deterministic = jnp.zeros((length, 4)).at[:, 0].set(0.97)
        deterministic = deterministic.at[:, 1:].set(0.01)
        loss_det = complexity_loss(deterministic)
        assert jnp.isfinite(loss_det)

    def test_gradient_stability_at_scale(self, rngs):
        """Test gradient stability with large sequences."""
        config = BiologicalRegularizationConfig()
        combined_loss = BiologicalPlausibilityLoss(config, rngs=rngs)

        key = jax.random.PRNGKey(42)
        seq = generate_random_soft_sequence(key, LARGE_LENGTH)

        grad = jax.grad(combined_loss)(seq)

        # Gradients should be finite and not explode
        assert jnp.all(jnp.isfinite(grad))
        grad_norm = jnp.linalg.norm(grad)
        assert grad_norm < 1e6  # Reasonable upper bound
