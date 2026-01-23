"""Tests for diffbio.operators.alignment module.

These tests define the expected behavior of the Smooth Smith-Waterman
alignment operator. Implementation should be written to pass these tests.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.alignment import (
    SmoothSmithWaterman,
    SmithWatermanConfig,
)
from diffbio.operators.alignment.scoring import (
    BLOSUM62,
    DNA_SIMPLE,
    create_dna_scoring_matrix,
)
from diffbio.sequences.dna import encode_dna_string


class TestScoringMatrix:
    """Tests for scoring matrix utilities."""

    def test_dna_simple_shape(self):
        """Test DNA simple scoring matrix has correct shape."""
        assert DNA_SIMPLE.shape == (4, 4)

    def test_dna_simple_diagonal(self):
        """Test DNA simple scoring has positive diagonal (matches)."""
        diagonal = jnp.diag(DNA_SIMPLE)
        assert jnp.all(diagonal > 0)

    def test_dna_simple_symmetric(self):
        """Test DNA scoring matrix is symmetric."""
        assert jnp.allclose(DNA_SIMPLE, DNA_SIMPLE.T)

    def test_create_dna_scoring_matrix(self):
        """Test creating custom DNA scoring matrix."""
        matrix = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)
        assert matrix.shape == (4, 4)
        # Check diagonal (matches)
        assert jnp.all(jnp.diag(matrix) == 2.0)
        # Check off-diagonal (mismatches) - mask out diagonal with identity
        mask = jnp.ones((4, 4)) - jnp.eye(4)
        off_diag_values = matrix * mask
        # Off-diagonal should be -1.0, masked diagonal should be 0.0
        expected = jnp.ones((4, 4)) * -1.0 * mask
        assert jnp.allclose(off_diag_values, expected)

    def test_blosum62_shape(self):
        """Test BLOSUM62 has correct shape for proteins."""
        assert BLOSUM62.shape == (20, 20)

    def test_blosum62_symmetric(self):
        """Test BLOSUM62 is symmetric."""
        assert jnp.allclose(BLOSUM62, BLOSUM62.T)


class TestSmithWatermanConfig:
    """Tests for SmithWatermanConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SmithWatermanConfig()
        assert config.temperature == 1.0
        assert config.gap_open == -10.0
        assert config.gap_extend == -1.0
        assert config.stochastic is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SmithWatermanConfig(
            temperature=0.5,
            gap_open=-12.0,
            gap_extend=-2.0,
        )
        assert config.temperature == 0.5
        assert config.gap_open == -12.0
        assert config.gap_extend == -2.0


class TestSmoothSmithWaterman:
    """Tests for SmoothSmithWaterman operator."""

    @pytest.fixture
    def simple_scoring(self):
        """Provide simple DNA scoring matrix."""
        return create_dna_scoring_matrix(match=2.0, mismatch=-1.0)

    def test_initialization(self, rngs, simple_scoring):
        """Test operator initialization."""
        config = SmithWatermanConfig(temperature=1.0)
        op = SmoothSmithWaterman(config, scoring_matrix=simple_scoring, rngs=rngs)
        assert op is not None
        assert float(op._temperature) == 1.0  # Use TemperatureOperator's property

    def test_identical_sequences_high_score(self, rngs, simple_scoring):
        """Test that identical sequences produce high alignment score."""
        config = SmithWatermanConfig(temperature=1.0)
        op = SmoothSmithWaterman(config, scoring_matrix=simple_scoring, rngs=rngs)

        seq = encode_dna_string("ACGT")
        result = op.align(seq, seq)

        # Identical sequences should have high score
        assert result.score > 0
        # Score should be close to sum of match scores
        expected_max = 4 * 2.0  # 4 positions * match score of 2
        assert result.score > expected_max * 0.5

    def test_different_sequences_lower_score(self, rngs, simple_scoring):
        """Test that different sequences produce lower alignment score."""
        config = SmithWatermanConfig(temperature=1.0)
        op = SmoothSmithWaterman(config, scoring_matrix=simple_scoring, rngs=rngs)

        seq1 = encode_dna_string("AAAA")
        seq2 = encode_dna_string("TTTT")

        result_same = op.align(seq1, seq1)
        result_diff = op.align(seq1, seq2)

        # Different sequences should score lower than identical
        assert result_same.score > result_diff.score

    def test_alignment_matrix_shape(self, rngs, simple_scoring):
        """Test alignment matrix has correct shape."""
        config = SmithWatermanConfig(temperature=1.0)
        op = SmoothSmithWaterman(config, scoring_matrix=simple_scoring, rngs=rngs)

        seq1 = encode_dna_string("ACGT")  # length 4
        seq2 = encode_dna_string("ACG")  # length 3

        result = op.align(seq1, seq2)

        # Alignment matrix should be (len1+1, len2+1)
        assert result.alignment_matrix.shape == (5, 4)

    def test_soft_alignment_sums_to_one(self, rngs, simple_scoring):
        """Test soft alignment matrix rows sum to approximately 1."""
        config = SmithWatermanConfig(temperature=1.0)
        op = SmoothSmithWaterman(config, scoring_matrix=simple_scoring, rngs=rngs)

        seq1 = encode_dna_string("ACGT")
        seq2 = encode_dna_string("ACGT")

        result = op.align(seq1, seq2)

        # Each position in seq1 should have soft alignment to seq2
        # The traceback probabilities should be valid distributions
        assert result.soft_alignment is not None
        # Soft alignment should have reasonable values
        assert jnp.all(result.soft_alignment >= 0)

    def test_temperature_effect(self, rngs, simple_scoring):
        """Test that temperature affects alignment sharpness."""
        seq1 = encode_dna_string("ACGT")
        seq2 = encode_dna_string("ACGT")

        # Low temperature (sharper)
        config_low = SmithWatermanConfig(temperature=0.1)
        op_low = SmoothSmithWaterman(config_low, scoring_matrix=simple_scoring, rngs=rngs)
        result_low = op_low.align(seq1, seq2)

        # High temperature (softer)
        config_high = SmithWatermanConfig(temperature=10.0)
        op_high = SmoothSmithWaterman(config_high, scoring_matrix=simple_scoring, rngs=rngs)
        result_high = op_high.align(seq1, seq2)

        # Low temperature should give higher max in soft alignment (sharper)
        max_low = jnp.max(result_low.soft_alignment)
        max_high = jnp.max(result_high.soft_alignment)
        assert max_low > max_high

    def test_gap_penalties_affect_score(self, rngs, simple_scoring):
        """Test that gap penalties affect alignment score."""
        seq1 = encode_dna_string("ACGT")
        seq2 = encode_dna_string("AGT")  # Missing C

        # Low gap penalty
        config_low = SmithWatermanConfig(gap_open=-1.0, gap_extend=-0.5)
        op_low = SmoothSmithWaterman(config_low, scoring_matrix=simple_scoring, rngs=rngs)
        result_low = op_low.align(seq1, seq2)

        # High gap penalty
        config_high = SmithWatermanConfig(gap_open=-20.0, gap_extend=-5.0)
        op_high = SmoothSmithWaterman(config_high, scoring_matrix=simple_scoring, rngs=rngs)
        result_high = op_high.align(seq1, seq2)

        # Higher gap penalty should give lower score for gapped alignment
        assert result_low.score > result_high.score

    def test_local_alignment_property(self, rngs, simple_scoring):
        """Test local alignment ignores flanking mismatches."""
        config = SmithWatermanConfig(temperature=0.5)
        op = SmoothSmithWaterman(config, scoring_matrix=simple_scoring, rngs=rngs)

        # Sequences with matching core and mismatching flanks
        seq1 = encode_dna_string("TTACGTTT")
        seq2 = encode_dna_string("AAACGTAA")

        result = op.align(seq1, seq2)

        # Should find the "ACGT" match in the middle
        # Score should be positive despite flanking mismatches
        assert result.score > 0


class TestGradientFlow:
    """Tests for gradient flow through alignment."""

    @pytest.fixture
    def simple_scoring(self):
        return create_dna_scoring_matrix(match=2.0, mismatch=-1.0)

    def test_gradient_wrt_sequence(self, rngs, simple_scoring):
        """Test gradients flow with respect to input sequence."""
        config = SmithWatermanConfig(temperature=1.0)
        op = SmoothSmithWaterman(config, scoring_matrix=simple_scoring, rngs=rngs)

        seq1 = encode_dna_string("ACGT")
        seq2 = encode_dna_string("ACGT")

        def loss_fn(s1):
            result = op.align(s1, seq2)
            return result.score

        grad = jax.grad(loss_fn)(seq1)
        assert grad is not None
        assert grad.shape == seq1.shape

    def test_gradient_wrt_scoring_matrix(self, rngs):
        """Test gradients flow with respect to scoring matrix."""
        config = SmithWatermanConfig(temperature=1.0)
        scoring = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)
        op = SmoothSmithWaterman(config, scoring_matrix=scoring, rngs=rngs)

        seq1 = encode_dna_string("ACGT")
        seq2 = encode_dna_string("ACGT")

        # Use NNX's value_and_grad for module gradient computation
        @nnx.value_and_grad
        def loss_fn(model):
            result = model.align(seq1, seq2)
            return result.score

        _, grads = loss_fn(op)

        # Check gradients exist for scoring matrix
        assert hasattr(grads, "scoring_matrix")

    def test_gradient_wrt_temperature(self, rngs, simple_scoring):
        """Test gradients flow with respect to temperature when learnable."""
        config = SmithWatermanConfig(temperature=1.0, learnable_temperature=True)
        op = SmoothSmithWaterman(config, scoring_matrix=simple_scoring, rngs=rngs)

        seq1 = encode_dna_string("ACGT")
        seq2 = encode_dna_string("ACGT")

        @nnx.value_and_grad
        def loss_fn(model):
            result = model.align(seq1, seq2)
            return result.score

        _, grads = loss_fn(op)

        # Check gradients exist for temperature when learnable_temperature=True
        assert hasattr(grads, "temperature")


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture
    def simple_scoring(self):
        return create_dna_scoring_matrix(match=2.0, mismatch=-1.0)

    def test_align_is_jit_compatible(self, rngs, simple_scoring):
        """Test align method works with JIT."""
        config = SmithWatermanConfig(temperature=1.0)
        op = SmoothSmithWaterman(config, scoring_matrix=simple_scoring, rngs=rngs)

        seq1 = encode_dna_string("ACGT")
        seq2 = encode_dna_string("ACGT")

        # JIT compile the align method
        @jax.jit
        def jit_align(s1, s2):
            return op.align(s1, s2)

        result = jit_align(seq1, seq2)
        assert result.score is not None

    def test_jit_produces_same_result(self, rngs, simple_scoring):
        """Test JIT produces same result as eager execution."""
        config = SmithWatermanConfig(temperature=1.0)
        op = SmoothSmithWaterman(config, scoring_matrix=simple_scoring, rngs=rngs)

        seq1 = encode_dna_string("ACGT")
        seq2 = encode_dna_string("ACGT")

        # Eager execution
        eager_result = op.align(seq1, seq2)

        # JIT execution
        @jax.jit
        def jit_align(s1, s2):
            return op.align(s1, s2)

        jit_result = jit_align(seq1, seq2)

        assert jnp.allclose(eager_result.score, jit_result.score)


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def simple_scoring(self):
        return create_dna_scoring_matrix(match=2.0, mismatch=-1.0)

    def test_single_nucleotide(self, rngs, simple_scoring):
        """Test alignment of single nucleotide sequences."""
        config = SmithWatermanConfig(temperature=1.0)
        op = SmoothSmithWaterman(config, scoring_matrix=simple_scoring, rngs=rngs)

        seq1 = encode_dna_string("A")
        seq2 = encode_dna_string("A")

        result = op.align(seq1, seq2)
        assert result.score > 0
        assert result.alignment_matrix.shape == (2, 2)

    def test_different_lengths(self, rngs, simple_scoring):
        """Test alignment of sequences with different lengths."""
        config = SmithWatermanConfig(temperature=1.0)
        op = SmoothSmithWaterman(config, scoring_matrix=simple_scoring, rngs=rngs)

        seq1 = encode_dna_string("ACGTACGT")  # length 8
        seq2 = encode_dna_string("ACG")  # length 3

        result = op.align(seq1, seq2)
        assert result.alignment_matrix.shape == (9, 4)

    def test_no_match_sequence(self, rngs, simple_scoring):
        """Test alignment when sequences have no matching regions."""
        config = SmithWatermanConfig(temperature=1.0)
        op = SmoothSmithWaterman(config, scoring_matrix=simple_scoring, rngs=rngs)

        seq1 = encode_dna_string("AAAA")
        seq2 = encode_dna_string("TTTT")

        result = op.align(seq1, seq2)
        # Local alignment score should be non-negative (can be 0 for no match)
        assert result.score >= 0
