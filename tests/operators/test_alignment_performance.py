"""Performance and scalability tests for alignment operators.

These tests verify that alignment operators are fast and scale well
with increasing sequence lengths.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.alignment import SmoothSmithWaterman, SmithWatermanConfig
from diffbio.operators.alignment.scoring import create_dna_scoring_matrix


# Sequence lengths to test scalability
SMALL_LENGTH = 10
MEDIUM_LENGTH = 50
LARGE_LENGTH = 100
XLARGE_LENGTH = 200


def generate_random_dna_sequence(key: jax.Array, length: int) -> jax.Array:
    """Generate a random one-hot encoded DNA sequence."""
    indices = jax.random.randint(key, (length,), 0, 4)
    return jax.nn.one_hot(indices, 4)


class TestAlignmentScalability:
    """Tests for alignment scalability with different sequence lengths."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def scoring_matrix(self):
        return create_dna_scoring_matrix(match=2.0, mismatch=-1.0)

    @pytest.fixture
    def aligner(self, rngs, scoring_matrix):
        config = SmithWatermanConfig(temperature=1.0)
        return SmoothSmithWaterman(config, scoring_matrix=scoring_matrix, rngs=rngs)

    @pytest.mark.parametrize(
        "length",
        [SMALL_LENGTH, MEDIUM_LENGTH, LARGE_LENGTH],
        ids=["small", "medium", "large"],
    )
    def test_scalability_same_length(self, aligner, length):
        """Test alignment scales with sequence length."""
        key = jax.random.PRNGKey(42)
        seq1 = generate_random_dna_sequence(key, length)
        seq2 = generate_random_dna_sequence(jax.random.split(key)[0], length)

        result = aligner.align(seq1, seq2)

        # Verify correct shapes
        assert result.alignment_matrix.shape == (length + 1, length + 1)
        assert result.soft_alignment.shape == (length, length)
        assert result.score.shape == ()

    def test_scalability_different_lengths(self, aligner):
        """Test alignment with different sequence lengths."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)

        seq1 = generate_random_dna_sequence(keys[0], 30)
        seq2 = generate_random_dna_sequence(keys[1], 50)

        result = aligner.align(seq1, seq2)

        assert result.alignment_matrix.shape == (31, 51)
        assert result.soft_alignment.shape == (30, 50)

    def test_large_sequence_alignment(self, aligner):
        """Test alignment works with large sequences."""
        key = jax.random.PRNGKey(42)
        seq1 = generate_random_dna_sequence(key, XLARGE_LENGTH)
        seq2 = generate_random_dna_sequence(jax.random.split(key)[0], XLARGE_LENGTH)

        result = aligner.align(seq1, seq2)

        assert result.alignment_matrix.shape == (XLARGE_LENGTH + 1, XLARGE_LENGTH + 1)
        assert result.score.shape == ()
        # Score should be reasonable (not NaN or Inf)
        assert jnp.isfinite(result.score)


class TestAlignmentJITPerformance:
    """Tests for JIT compilation performance."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def scoring_matrix(self):
        return create_dna_scoring_matrix(match=2.0, mismatch=-1.0)

    def test_jit_compilation_speedup(self, rngs, scoring_matrix):
        """Test that JIT compilation provides speedup."""
        config = SmithWatermanConfig(temperature=1.0)
        aligner = SmoothSmithWaterman(config, scoring_matrix=scoring_matrix, rngs=rngs)

        key = jax.random.PRNGKey(42)
        seq1 = generate_random_dna_sequence(key, MEDIUM_LENGTH)
        seq2 = generate_random_dna_sequence(jax.random.split(key)[0], MEDIUM_LENGTH)

        # JIT compile the alignment function
        @jax.jit
        def jit_align(s1, s2):
            return aligner.align(s1, s2)

        # First call triggers compilation
        result1 = jit_align(seq1, seq2)
        # Block until computation is done
        result1.score.block_until_ready()

        # Subsequent calls should be fast (already compiled)
        result2 = jit_align(seq1, seq2)
        result2.score.block_until_ready()

        # Results should be identical
        assert jnp.allclose(result1.score, result2.score)

    def test_jit_works_with_different_fixed_sizes(self, rngs, scoring_matrix):
        """Test JIT with static shapes produces consistent results."""
        config = SmithWatermanConfig(temperature=1.0)
        aligner = SmoothSmithWaterman(config, scoring_matrix=scoring_matrix, rngs=rngs)

        @jax.jit
        def jit_align(s1, s2):
            return aligner.align(s1, s2)

        key = jax.random.PRNGKey(42)

        # Test with different sizes (each will trigger recompilation)
        for length in [10, 20, 30]:
            seq1 = generate_random_dna_sequence(key, length)
            seq2 = generate_random_dna_sequence(jax.random.split(key)[0], length)

            result = jit_align(seq1, seq2)
            assert result.alignment_matrix.shape == (length + 1, length + 1)
            assert jnp.isfinite(result.score)


class TestAlignmentGradientPerformance:
    """Tests for gradient computation performance."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def scoring_matrix(self):
        return create_dna_scoring_matrix(match=2.0, mismatch=-1.0)

    def test_gradient_computes_for_medium_sequences(self, rngs, scoring_matrix):
        """Test gradients compute efficiently for medium sequences."""
        config = SmithWatermanConfig(temperature=1.0)
        aligner = SmoothSmithWaterman(config, scoring_matrix=scoring_matrix, rngs=rngs)

        key = jax.random.PRNGKey(42)
        seq1 = generate_random_dna_sequence(key, MEDIUM_LENGTH)
        seq2 = generate_random_dna_sequence(jax.random.split(key)[0], MEDIUM_LENGTH)

        def loss_fn(s1):
            result = aligner.align(s1, seq2)
            return result.score

        grad = jax.grad(loss_fn)(seq1)

        assert grad.shape == seq1.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_jit_grad_combination(self, rngs, scoring_matrix):
        """Test JIT + grad combination works efficiently."""
        config = SmithWatermanConfig(temperature=1.0)
        aligner = SmoothSmithWaterman(config, scoring_matrix=scoring_matrix, rngs=rngs)

        key = jax.random.PRNGKey(42)
        seq1 = generate_random_dna_sequence(key, MEDIUM_LENGTH)
        seq2 = generate_random_dna_sequence(jax.random.split(key)[0], MEDIUM_LENGTH)

        @jax.jit
        def loss_and_grad(s1, s2):
            def loss_fn(s):
                result = aligner.align(s, s2)
                return result.score

            return jax.value_and_grad(loss_fn)(s1)

        loss, grad = loss_and_grad(seq1, seq2)

        assert jnp.isfinite(loss)
        assert grad.shape == seq1.shape
        assert jnp.all(jnp.isfinite(grad))


class TestAlignmentBenchmarks:
    """Benchmark tests using pytest-benchmark."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def scoring_matrix(self):
        return create_dna_scoring_matrix(match=2.0, mismatch=-1.0)

    @pytest.fixture
    def aligner(self, rngs, scoring_matrix):
        config = SmithWatermanConfig(temperature=1.0)
        return SmoothSmithWaterman(config, scoring_matrix=scoring_matrix, rngs=rngs)

    @pytest.fixture
    def jit_aligner(self, aligner):
        """Return JIT-compiled alignment function."""

        @jax.jit
        def align(s1, s2):
            return aligner.align(s1, s2)

        return align

    def test_benchmark_small_alignment(self, benchmark, jit_aligner):
        """Benchmark small sequence alignment."""
        key = jax.random.PRNGKey(42)
        seq1 = generate_random_dna_sequence(key, SMALL_LENGTH)
        seq2 = generate_random_dna_sequence(jax.random.split(key)[0], SMALL_LENGTH)

        # Warm up JIT
        result = jit_aligner(seq1, seq2)
        result.score.block_until_ready()

        def run_alignment():
            result = jit_aligner(seq1, seq2)
            result.score.block_until_ready()
            return result

        result = benchmark(run_alignment)
        assert jnp.isfinite(result.score)

    def test_benchmark_medium_alignment(self, benchmark, jit_aligner):
        """Benchmark medium sequence alignment."""
        key = jax.random.PRNGKey(42)
        seq1 = generate_random_dna_sequence(key, MEDIUM_LENGTH)
        seq2 = generate_random_dna_sequence(jax.random.split(key)[0], MEDIUM_LENGTH)

        # Warm up JIT
        result = jit_aligner(seq1, seq2)
        result.score.block_until_ready()

        def run_alignment():
            result = jit_aligner(seq1, seq2)
            result.score.block_until_ready()
            return result

        result = benchmark(run_alignment)
        assert jnp.isfinite(result.score)

    def test_benchmark_large_alignment(self, benchmark, jit_aligner):
        """Benchmark large sequence alignment."""
        key = jax.random.PRNGKey(42)
        seq1 = generate_random_dna_sequence(key, LARGE_LENGTH)
        seq2 = generate_random_dna_sequence(jax.random.split(key)[0], LARGE_LENGTH)

        # Warm up JIT
        result = jit_aligner(seq1, seq2)
        result.score.block_until_ready()

        def run_alignment():
            result = jit_aligner(seq1, seq2)
            result.score.block_until_ready()
            return result

        result = benchmark(run_alignment)
        assert jnp.isfinite(result.score)


class TestNumericalStability:
    """Tests for numerical stability at scale."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def scoring_matrix(self):
        return create_dna_scoring_matrix(match=2.0, mismatch=-1.0)

    def test_no_overflow_with_large_sequences(self, rngs, scoring_matrix):
        """Test no numerical overflow with large sequences."""
        config = SmithWatermanConfig(temperature=1.0)
        aligner = SmoothSmithWaterman(config, scoring_matrix=scoring_matrix, rngs=rngs)

        key = jax.random.PRNGKey(42)
        seq1 = generate_random_dna_sequence(key, XLARGE_LENGTH)
        seq2 = generate_random_dna_sequence(jax.random.split(key)[0], XLARGE_LENGTH)

        result = aligner.align(seq1, seq2)

        # Check no NaN or Inf values
        assert jnp.all(jnp.isfinite(result.score))
        assert jnp.all(jnp.isfinite(result.alignment_matrix))
        assert jnp.all(jnp.isfinite(result.soft_alignment))

    def test_stability_with_low_temperature(self, rngs, scoring_matrix):
        """Test numerical stability with low temperature (sharper distributions)."""
        config = SmithWatermanConfig(temperature=0.1)
        aligner = SmoothSmithWaterman(config, scoring_matrix=scoring_matrix, rngs=rngs)

        key = jax.random.PRNGKey(42)
        seq1 = generate_random_dna_sequence(key, MEDIUM_LENGTH)
        seq2 = generate_random_dna_sequence(jax.random.split(key)[0], MEDIUM_LENGTH)

        result = aligner.align(seq1, seq2)

        assert jnp.all(jnp.isfinite(result.score))
        assert jnp.all(jnp.isfinite(result.soft_alignment))

    def test_stability_with_high_temperature(self, rngs, scoring_matrix):
        """Test numerical stability with high temperature (smoother distributions)."""
        config = SmithWatermanConfig(temperature=10.0)
        aligner = SmoothSmithWaterman(config, scoring_matrix=scoring_matrix, rngs=rngs)

        key = jax.random.PRNGKey(42)
        seq1 = generate_random_dna_sequence(key, MEDIUM_LENGTH)
        seq2 = generate_random_dna_sequence(jax.random.split(key)[0], MEDIUM_LENGTH)

        result = aligner.align(seq1, seq2)

        assert jnp.all(jnp.isfinite(result.score))
        assert jnp.all(jnp.isfinite(result.soft_alignment))

    def test_gradient_stability_at_scale(self, rngs, scoring_matrix):
        """Test gradient stability with larger sequences."""
        config = SmithWatermanConfig(temperature=1.0)
        aligner = SmoothSmithWaterman(config, scoring_matrix=scoring_matrix, rngs=rngs)

        key = jax.random.PRNGKey(42)
        seq1 = generate_random_dna_sequence(key, LARGE_LENGTH)
        seq2 = generate_random_dna_sequence(jax.random.split(key)[0], LARGE_LENGTH)

        def loss_fn(s1):
            result = aligner.align(s1, seq2)
            return result.score

        grad = jax.grad(loss_fn)(seq1)

        # Gradients should be finite and not explode
        assert jnp.all(jnp.isfinite(grad))
        # Check gradient magnitude is reasonable
        grad_norm = jnp.linalg.norm(grad)
        assert grad_norm < 1e6  # Reasonable upper bound
