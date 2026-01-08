"""Tests for diffbio.losses.alignment_losses module.

These tests define the expected behavior of alignment loss functions
for training differentiable alignment models.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.losses.alignment_losses import (
    AlignmentConsistencyLoss,
    AlignmentScoreLoss,
    SoftEditDistanceLoss,
)
from diffbio.operators.alignment import SmoothSmithWaterman, SmithWatermanConfig
from diffbio.operators.alignment.scoring import create_dna_scoring_matrix


def generate_random_sequence(key: jax.Array, length: int) -> jax.Array:
    """Generate a random one-hot encoded DNA sequence."""
    indices = jax.random.randint(key, (length,), 0, 4)
    return jax.nn.one_hot(indices, 4)


def generate_soft_alignment(key: jax.Array, len1: int, len2: int) -> jax.Array:
    """Generate a random soft alignment matrix."""
    logits = jax.random.normal(key, (len1, len2))
    return jax.nn.softmax(logits, axis=-1)


class TestAlignmentScoreLoss:
    """Tests for alignment score loss."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_initialization(self, rngs):
        """Test alignment score loss initialization."""
        loss = AlignmentScoreLoss(rngs=rngs)
        assert loss is not None

    def test_identical_sequences_low_loss(self, rngs):
        """Test that identical sequences produce low alignment loss."""
        loss_fn = AlignmentScoreLoss(rngs=rngs)

        # Create identical sequences
        seq = jnp.array(
            [
                [1, 0, 0, 0],  # A
                [0, 1, 0, 0],  # C
                [0, 0, 1, 0],  # G
                [0, 0, 0, 1],  # T
            ],
            dtype=jnp.float32,
        )

        # Perfect diagonal alignment
        alignment = jnp.eye(4)

        loss = loss_fn(seq, seq, alignment)

        # Identical aligned sequences should have low loss
        assert loss < 1.0

    def test_different_sequences_higher_loss(self, rngs):
        """Test that different sequences produce higher alignment loss."""
        loss_fn = AlignmentScoreLoss(rngs=rngs)

        seq1 = jnp.array(
            [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],  # All A
            dtype=jnp.float32,
        )
        seq2 = jnp.array(
            [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],  # All T
            dtype=jnp.float32,
        )

        alignment = jnp.eye(4)

        loss_diff = loss_fn(seq1, seq2, alignment)
        loss_same = loss_fn(seq1, seq1, alignment)

        # Different sequences should have higher loss
        assert loss_diff > loss_same

    def test_alignment_score_differentiable(self, rngs):
        """Test that alignment score loss is differentiable."""
        loss_fn = AlignmentScoreLoss(rngs=rngs)

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        seq1 = generate_random_sequence(keys[0], 10)
        seq2 = generate_random_sequence(keys[1], 10)
        alignment = generate_soft_alignment(keys[2], 10, 10)

        def loss(s1):
            return loss_fn(s1, seq2, alignment)

        grad = jax.grad(loss)(seq1)
        assert grad is not None
        assert grad.shape == seq1.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_jit_compatible(self, rngs):
        """Test alignment score loss works with JIT."""
        loss_fn = AlignmentScoreLoss(rngs=rngs)

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        seq1 = generate_random_sequence(keys[0], 10)
        seq2 = generate_random_sequence(keys[1], 10)
        alignment = generate_soft_alignment(keys[2], 10, 10)

        @jax.jit
        def jit_loss(s1, s2, a):
            return loss_fn(s1, s2, a)

        loss = jit_loss(seq1, seq2, alignment)
        assert jnp.isfinite(loss)


class TestSoftEditDistanceLoss:
    """Tests for soft edit distance loss."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_initialization(self, rngs):
        """Test soft edit distance loss initialization."""
        loss = SoftEditDistanceLoss(rngs=rngs)
        assert loss is not None

    def test_identical_sequences_zero_distance(self, rngs):
        """Test that identical sequences have near-zero edit distance."""
        loss_fn = SoftEditDistanceLoss(rngs=rngs)

        seq = jnp.array(
            [
                [1, 0, 0, 0],  # A
                [0, 1, 0, 0],  # C
                [0, 0, 1, 0],  # G
                [0, 0, 0, 1],  # T
            ],
            dtype=jnp.float32,
        )

        distance = loss_fn(seq, seq)

        # Identical sequences should have very low edit distance
        assert distance < 0.5

    def test_completely_different_sequences_high_distance(self, rngs):
        """Test that completely different sequences have high edit distance."""
        loss_fn = SoftEditDistanceLoss(rngs=rngs)

        seq1 = jnp.array(
            [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],  # All A
            dtype=jnp.float32,
        )
        seq2 = jnp.array(
            [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],  # All T
            dtype=jnp.float32,
        )

        distance = loss_fn(seq1, seq2)

        # Completely different sequences should have high edit distance
        assert distance > 1.0

    def test_edit_distance_symmetric(self, rngs):
        """Test that edit distance is symmetric."""
        loss_fn = SoftEditDistanceLoss(rngs=rngs)

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        seq1 = generate_random_sequence(keys[0], 10)
        seq2 = generate_random_sequence(keys[1], 10)

        dist_12 = loss_fn(seq1, seq2)
        dist_21 = loss_fn(seq2, seq1)

        # Edit distance should be approximately symmetric
        assert jnp.allclose(dist_12, dist_21, atol=0.1)

    def test_edit_distance_differentiable(self, rngs):
        """Test that edit distance is differentiable."""
        loss_fn = SoftEditDistanceLoss(rngs=rngs)

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        seq1 = generate_random_sequence(keys[0], 10)
        seq2 = generate_random_sequence(keys[1], 10)

        def loss(s1):
            return loss_fn(s1, seq2)

        grad = jax.grad(loss)(seq1)
        assert grad is not None
        assert grad.shape == seq1.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_edit_distance_scales_with_length(self, rngs):
        """Test that edit distance scales appropriately with sequence length."""
        loss_fn = SoftEditDistanceLoss(normalize=True, rngs=rngs)

        key = jax.random.PRNGKey(42)

        # Short sequences
        seq1_short = generate_random_sequence(key, 5)
        seq2_short = generate_random_sequence(jax.random.split(key)[0], 5)
        dist_short = loss_fn(seq1_short, seq2_short)

        # Long sequences
        seq1_long = generate_random_sequence(key, 50)
        seq2_long = generate_random_sequence(jax.random.split(key)[0], 50)
        dist_long = loss_fn(seq1_long, seq2_long)

        # Normalized distance should be similar regardless of length
        assert jnp.abs(dist_short - dist_long) < 1.0

    def test_jit_compatible(self, rngs):
        """Test edit distance works with JIT."""
        loss_fn = SoftEditDistanceLoss(rngs=rngs)

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        seq1 = generate_random_sequence(keys[0], 10)
        seq2 = generate_random_sequence(keys[1], 10)

        @jax.jit
        def jit_loss(s1, s2):
            return loss_fn(s1, s2)

        loss = jit_loss(seq1, seq2)
        assert jnp.isfinite(loss)


class TestAlignmentConsistencyLoss:
    """Tests for alignment consistency loss."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_initialization(self, rngs):
        """Test alignment consistency loss initialization."""
        loss = AlignmentConsistencyLoss(rngs=rngs)
        assert loss is not None

    def test_consistent_alignments_low_loss(self, rngs):
        """Test that consistent alignments produce low loss."""
        loss_fn = AlignmentConsistencyLoss(rngs=rngs)

        # Create consistent alignments (A->B, B->C consistent with A->C)
        # Perfect diagonal alignments are consistent
        align_ab = jnp.eye(4)
        align_bc = jnp.eye(4)
        align_ac = jnp.eye(4)

        loss = loss_fn(align_ab, align_bc, align_ac)

        # Consistent alignments should have low loss
        assert loss < 0.5

    def test_inconsistent_alignments_high_loss(self, rngs):
        """Test that inconsistent alignments produce high loss."""
        loss_fn = AlignmentConsistencyLoss(rngs=rngs)

        # Create inconsistent alignments
        align_ab = jnp.eye(4)  # A[i] -> B[i]
        align_bc = jnp.eye(4)  # B[i] -> C[i]
        # Inconsistent: A->C should be diagonal but it's anti-diagonal
        align_ac = jnp.fliplr(jnp.eye(4))

        loss = loss_fn(align_ab, align_bc, align_ac)

        # Inconsistent alignments should have higher loss
        assert loss > 0.1

    def test_consistency_differentiable(self, rngs):
        """Test that consistency loss is differentiable."""
        loss_fn = AlignmentConsistencyLoss(rngs=rngs)

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        align_ab = generate_soft_alignment(keys[0], 10, 10)
        align_bc = generate_soft_alignment(keys[1], 10, 10)
        align_ac = generate_soft_alignment(keys[2], 10, 10)

        def loss(a_ab):
            return loss_fn(a_ab, align_bc, align_ac)

        grad = jax.grad(loss)(align_ab)
        assert grad is not None
        assert grad.shape == align_ab.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_jit_compatible(self, rngs):
        """Test consistency loss works with JIT."""
        loss_fn = AlignmentConsistencyLoss(rngs=rngs)

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        align_ab = generate_soft_alignment(keys[0], 10, 10)
        align_bc = generate_soft_alignment(keys[1], 10, 10)
        align_ac = generate_soft_alignment(keys[2], 10, 10)

        @jax.jit
        def jit_loss(a_ab, a_bc, a_ac):
            return loss_fn(a_ab, a_bc, a_ac)

        loss = jit_loss(align_ab, align_bc, align_ac)
        assert jnp.isfinite(loss)


class TestIntegrationWithAligner:
    """Tests for integration with SmoothSmithWaterman aligner."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def aligner(self, rngs):
        config = SmithWatermanConfig(temperature=1.0)
        scoring = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)
        return SmoothSmithWaterman(config, scoring_matrix=scoring, rngs=rngs)

    def test_alignment_score_loss_with_aligner(self, rngs, aligner):
        """Test alignment score loss integrates with aligner output."""
        loss_fn = AlignmentScoreLoss(rngs=rngs)

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        seq1 = generate_random_sequence(keys[0], 10)
        seq2 = generate_random_sequence(keys[1], 10)

        # Get alignment from aligner
        result = aligner.align(seq1, seq2)

        # Use soft alignment as input to loss
        loss = loss_fn(seq1, seq2, result.soft_alignment)

        assert jnp.isfinite(loss)

    def test_end_to_end_gradient_flow(self, rngs, aligner):
        """Test gradients flow through aligner to loss."""
        loss_fn = AlignmentScoreLoss(rngs=rngs)

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        seq1 = generate_random_sequence(keys[0], 10)
        seq2 = generate_random_sequence(keys[1], 10)

        def end_to_end_loss(s1):
            result = aligner.align(s1, seq2)
            return loss_fn(s1, seq2, result.soft_alignment)

        grad = jax.grad(end_to_end_loss)(seq1)

        assert grad is not None
        assert grad.shape == seq1.shape
        assert jnp.all(jnp.isfinite(grad))


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_single_position_sequences(self, rngs):
        """Test losses work with single position sequences."""
        score_loss = AlignmentScoreLoss(rngs=rngs)
        edit_loss = SoftEditDistanceLoss(rngs=rngs)

        seq = jnp.array([[1, 0, 0, 0]], dtype=jnp.float32)  # Single A
        alignment = jnp.array([[1.0]])

        score = score_loss(seq, seq, alignment)
        edit = edit_loss(seq, seq)

        assert jnp.isfinite(score)
        assert jnp.isfinite(edit)

    def test_different_length_sequences(self, rngs):
        """Test edit distance works with different length sequences."""
        loss_fn = SoftEditDistanceLoss(rngs=rngs)

        seq1 = generate_random_sequence(jax.random.PRNGKey(1), 5)
        seq2 = generate_random_sequence(jax.random.PRNGKey(2), 10)

        distance = loss_fn(seq1, seq2)

        assert jnp.isfinite(distance)
        # Longer sequence should contribute to larger distance
        assert distance > 0

    def test_uniform_soft_sequences(self, rngs):
        """Test losses work with uniform (uncertain) sequences."""
        score_loss = AlignmentScoreLoss(rngs=rngs)

        # Uniform distribution (maximum uncertainty)
        seq = jnp.ones((5, 4)) / 4.0
        alignment = jnp.ones((5, 5)) / 5.0

        loss = score_loss(seq, seq, alignment)

        assert jnp.isfinite(loss)
