"""Performance and scalability tests for quality filter operator.

These tests verify that the quality filter is fast and scales well
with increasing sequence lengths.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.quality_filter import (
    DifferentiableQualityFilter,
    QualityFilterConfig,
)


# Sequence lengths to test scalability
SMALL_LENGTH = 100
MEDIUM_LENGTH = 1000
LARGE_LENGTH = 10000
XLARGE_LENGTH = 100000


def generate_random_sequence_data(key: jax.Array, length: int) -> dict[str, jax.Array]:
    """Generate random one-hot encoded DNA sequence with quality scores."""
    keys = jax.random.split(key, 2)
    indices = jax.random.randint(keys[0], (length,), 0, 4)
    sequence = jax.nn.one_hot(indices, 4)
    # Quality scores between 0 and 40 (typical Phred range)
    quality_scores = jax.random.uniform(keys[1], (length,), minval=0.0, maxval=40.0)
    return {"sequence": sequence, "quality_scores": quality_scores}


class TestQualityFilterScalability:
    """Tests for quality filter scalability with different sequence lengths."""

    @pytest.fixture
    def quality_filter(self, rngs):
        config = QualityFilterConfig(initial_threshold=20.0)
        return DifferentiableQualityFilter(config, rngs=rngs)

    @pytest.mark.parametrize(
        "length",
        [SMALL_LENGTH, MEDIUM_LENGTH, LARGE_LENGTH],
        ids=["small", "medium", "large"],
    )
    def test_scalability_with_length(self, quality_filter, length):
        """Test quality filter scales with sequence length."""
        key = jax.random.PRNGKey(42)
        data = generate_random_sequence_data(key, length)
        state = {}

        transformed, new_state, metadata = quality_filter.apply(data, state, None, None)

        # Verify correct shapes
        assert transformed["sequence"].shape == (length, 4)
        assert transformed["quality_scores"].shape == (length,)

    def test_very_large_sequence(self, quality_filter):
        """Test quality filter works with very large sequences."""
        key = jax.random.PRNGKey(42)
        data = generate_random_sequence_data(key, XLARGE_LENGTH)
        state = {}

        transformed, _, _ = quality_filter.apply(data, state, None, None)

        assert transformed["sequence"].shape == (XLARGE_LENGTH, 4)
        # Values should be finite
        assert jnp.all(jnp.isfinite(transformed["sequence"]))


class TestQualityFilterJITPerformance:
    """Tests for JIT compilation performance."""

    def test_jit_compilation_works(self, rngs):
        """Test that JIT compilation works for quality filter."""
        config = QualityFilterConfig(initial_threshold=20.0)
        quality_filter = DifferentiableQualityFilter(config, rngs=rngs)

        key = jax.random.PRNGKey(42)
        data = generate_random_sequence_data(key, MEDIUM_LENGTH)
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return quality_filter.apply(data, state, None, None)

        # First call triggers compilation
        result1, _, _ = jit_apply(data, state)
        result1["sequence"].block_until_ready()

        # Second call uses compiled version
        result2, _, _ = jit_apply(data, state)
        result2["sequence"].block_until_ready()

        # Results should be identical
        assert jnp.allclose(result1["sequence"], result2["sequence"])

    def test_jit_with_different_sizes(self, rngs):
        """Test JIT with different sequence sizes."""
        config = QualityFilterConfig(initial_threshold=20.0)
        quality_filter = DifferentiableQualityFilter(config, rngs=rngs)

        @jax.jit
        def jit_apply(data, state):
            return quality_filter.apply(data, state, None, None)

        key = jax.random.PRNGKey(42)

        for length in [100, 500, 1000]:
            data = generate_random_sequence_data(key, length)
            state = {}

            result, _, _ = jit_apply(data, state)
            assert result["sequence"].shape == (length, 4)
            assert jnp.all(jnp.isfinite(result["sequence"]))


class TestQualityFilterGradientPerformance:
    """Tests for gradient computation performance."""

    def test_gradient_computes_for_large_sequences(self, rngs):
        """Test gradients compute efficiently for large sequences."""
        config = QualityFilterConfig(initial_threshold=20.0)
        quality_filter = DifferentiableQualityFilter(config, rngs=rngs)

        key = jax.random.PRNGKey(42)
        data = generate_random_sequence_data(key, LARGE_LENGTH)
        state = {}

        def loss_fn(seq):
            d = {"sequence": seq, "quality_scores": data["quality_scores"]}
            transformed, _, _ = quality_filter.apply(d, state, None, None)
            return jnp.sum(transformed["sequence"])

        grad = jax.grad(loss_fn)(data["sequence"])

        assert grad.shape == data["sequence"].shape
        assert jnp.all(jnp.isfinite(grad))

    def test_jit_grad_combination(self, rngs):
        """Test JIT + grad combination works efficiently."""
        config = QualityFilterConfig(initial_threshold=20.0)
        quality_filter = DifferentiableQualityFilter(config, rngs=rngs)

        key = jax.random.PRNGKey(42)
        data = generate_random_sequence_data(key, MEDIUM_LENGTH)
        state = {}

        @jax.jit
        def loss_and_grad(seq, quality):
            def loss_fn(s):
                d = {"sequence": s, "quality_scores": quality}
                transformed, _, _ = quality_filter.apply(d, state, None, None)
                return jnp.sum(transformed["sequence"])

            return jax.value_and_grad(loss_fn)(seq)

        loss, grad = loss_and_grad(data["sequence"], data["quality_scores"])

        assert jnp.isfinite(loss)
        assert grad.shape == data["sequence"].shape
        assert jnp.all(jnp.isfinite(grad))


class TestQualityFilterBenchmarks:
    """Benchmark tests using pytest-benchmark."""

    @pytest.fixture
    def quality_filter(self, rngs):
        config = QualityFilterConfig(initial_threshold=20.0)
        return DifferentiableQualityFilter(config, rngs=rngs)

    @pytest.fixture
    def jit_apply(self, quality_filter):
        """Return JIT-compiled apply function."""

        @jax.jit
        def apply(data, state):
            return quality_filter.apply(data, state, None, None)

        return apply

    def test_benchmark_small_filter(self, benchmark, jit_apply):
        """Benchmark small sequence filtering."""
        key = jax.random.PRNGKey(42)
        data = generate_random_sequence_data(key, SMALL_LENGTH)
        state = {}

        # Warm up JIT
        result, _, _ = jit_apply(data, state)
        result["sequence"].block_until_ready()

        def run_filter():
            result, _, _ = jit_apply(data, state)
            result["sequence"].block_until_ready()
            return result

        result = benchmark(run_filter)
        assert jnp.all(jnp.isfinite(result["sequence"]))

    def test_benchmark_medium_filter(self, benchmark, jit_apply):
        """Benchmark medium sequence filtering."""
        key = jax.random.PRNGKey(42)
        data = generate_random_sequence_data(key, MEDIUM_LENGTH)
        state = {}

        # Warm up JIT
        result, _, _ = jit_apply(data, state)
        result["sequence"].block_until_ready()

        def run_filter():
            result, _, _ = jit_apply(data, state)
            result["sequence"].block_until_ready()
            return result

        result = benchmark(run_filter)
        assert jnp.all(jnp.isfinite(result["sequence"]))

    def test_benchmark_large_filter(self, benchmark, jit_apply):
        """Benchmark large sequence filtering."""
        key = jax.random.PRNGKey(42)
        data = generate_random_sequence_data(key, LARGE_LENGTH)
        state = {}

        # Warm up JIT
        result, _, _ = jit_apply(data, state)
        result["sequence"].block_until_ready()

        def run_filter():
            result, _, _ = jit_apply(data, state)
            result["sequence"].block_until_ready()
            return result

        result = benchmark(run_filter)
        assert jnp.all(jnp.isfinite(result["sequence"]))


class TestNumericalStability:
    """Tests for numerical stability at scale."""

    def test_stability_with_extreme_quality_scores(self, rngs):
        """Test stability with extreme quality score values."""
        config = QualityFilterConfig(initial_threshold=20.0)
        quality_filter = DifferentiableQualityFilter(config, rngs=rngs)

        key = jax.random.PRNGKey(42)
        length = 1000

        # Create sequence with extreme quality scores
        indices = jax.random.randint(key, (length,), 0, 4)
        sequence = jax.nn.one_hot(indices, 4)

        # Very low quality scores (near 0)
        low_quality = jnp.zeros(length)
        data_low = {"sequence": sequence, "quality_scores": low_quality}
        result_low, _, _ = quality_filter.apply(data_low, {}, None, None)
        assert jnp.all(jnp.isfinite(result_low["sequence"]))

        # Very high quality scores
        high_quality = jnp.ones(length) * 50.0
        data_high = {"sequence": sequence, "quality_scores": high_quality}
        result_high, _, _ = quality_filter.apply(data_high, {}, None, None)
        assert jnp.all(jnp.isfinite(result_high["sequence"]))

    def test_gradient_stability_with_large_sequences(self, rngs):
        """Test gradient stability with large sequences."""
        config = QualityFilterConfig(initial_threshold=20.0)
        quality_filter = DifferentiableQualityFilter(config, rngs=rngs)

        key = jax.random.PRNGKey(42)
        data = generate_random_sequence_data(key, LARGE_LENGTH)
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return jnp.sum(transformed["sequence"])

        loss, grads = loss_fn(quality_filter)

        assert jnp.isfinite(loss)
        # Check gradient for threshold parameter
        assert hasattr(grads, "threshold")
        assert jnp.isfinite(grads.threshold[...])
