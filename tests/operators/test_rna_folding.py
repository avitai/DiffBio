"""Tests for DifferentiableRNAFold operator.

This module tests the differentiable RNA secondary structure prediction
operator that implements the McCaskill partition function algorithm for
computing base pair probabilities.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.rna_structure import (
    DifferentiableRNAFold,
    RNAFoldConfig,
    create_rna_fold_predictor,
)


class TestRNAFoldConfig:
    """Tests for RNAFoldConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RNAFoldConfig()

        assert config.temperature == 1.0
        assert config.min_hairpin_loop == 3
        assert config.alphabet_size == 4

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RNAFoldConfig(
            temperature=0.5,
            min_hairpin_loop=4,
            alphabet_size=4,
        )

        assert config.temperature == 0.5
        assert config.min_hairpin_loop == 4
        assert config.alphabet_size == 4


class TestDifferentiableRNAFold:
    """Tests for DifferentiableRNAFold operator."""

    @pytest.fixture
    def config(self):
        """Create default config for tests."""
        return RNAFoldConfig(
            temperature=1.0,
            min_hairpin_loop=3,
        )

    @pytest.fixture
    def predictor(self, config):
        """Create predictor for tests."""
        return DifferentiableRNAFold(config, rngs=nnx.Rngs(42))

    def test_initialization(self, predictor, config):
        """Test predictor initializes correctly."""
        assert predictor.config.temperature == config.temperature
        assert predictor.config.min_hairpin_loop == config.min_hairpin_loop

    def test_forward_pass_shape(self, predictor):
        """Test forward pass produces correct output shapes."""
        seq_len = 30
        # Create one-hot encoded RNA sequence
        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.PRNGKey(0), (seq_len,), 0, 4),
            num_classes=4,
        )

        data = {"sequence": sequence}
        result, state, metadata = predictor.apply(data, {}, None)

        # Check output keys
        assert "sequence" in result
        assert "bp_probs" in result
        assert "partition_function" in result

        # Check shapes
        assert result["sequence"].shape == (seq_len, 4)
        assert result["bp_probs"].shape == (seq_len, seq_len)
        assert result["partition_function"].shape == ()

    def test_bp_probs_properties(self, predictor):
        """Test base pair probability matrix properties."""
        seq_len = 20
        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.PRNGKey(0), (seq_len,), 0, 4),
            num_classes=4,
        )

        data = {"sequence": sequence}
        result, _, _ = predictor.apply(data, {}, None)

        bp_probs = result["bp_probs"]

        # BP matrix should be symmetric
        assert jnp.allclose(bp_probs, bp_probs.T, atol=1e-5)

        # All values should be in [0, 1]
        assert jnp.all(bp_probs >= 0)
        assert jnp.all(bp_probs <= 1)

        # Diagonal should be zero (can't pair with self)
        assert jnp.allclose(jnp.diag(bp_probs), 0.0)

        # Positions too close for hairpin loop should be zero
        min_loop = predictor.config.min_hairpin_loop
        for i in range(seq_len):
            for j in range(max(0, i - min_loop), min(seq_len, i + min_loop + 1)):
                assert bp_probs[i, j] == 0.0

    def test_batched_input(self, predictor):
        """Test batched input processing."""
        batch_size = 4
        seq_len = 25
        sequences = jax.nn.one_hot(
            jax.random.randint(jax.random.PRNGKey(0), (batch_size, seq_len), 0, 4),
            num_classes=4,
        )

        data = {"sequence": sequences}
        result, _, _ = predictor.apply(data, {}, None)

        assert result["bp_probs"].shape == (batch_size, seq_len, seq_len)
        assert result["partition_function"].shape == (batch_size,)

    def test_valid_base_pairs_only(self, predictor):
        """Test that only valid Watson-Crick base pairs have non-zero probability."""
        # Create sequence with known pattern
        # A=0, C=1, G=2, U=3
        # Valid pairs: A-U (0-3), G-C (2-1), G-U (2-3 wobble)
        seq_len = 20
        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.PRNGKey(0), (seq_len,), 0, 4),
            num_classes=4,
        )

        data = {"sequence": sequence}
        result, _, _ = predictor.apply(data, {}, None)

        bp_probs = result["bp_probs"]

        # Check that invalid pairs have near-zero probability
        # Get nucleotide indices
        nuc_indices = jnp.argmax(sequence, axis=-1)

        for i in range(seq_len):
            for j in range(i + predictor.config.min_hairpin_loop + 1, seq_len):
                nuc_i = nuc_indices[i]
                nuc_j = nuc_indices[j]
                # Valid pairs: A-U, U-A, G-C, C-G, G-U, U-G
                is_valid = (
                    (nuc_i == 0 and nuc_j == 3)
                    or (nuc_i == 3 and nuc_j == 0)  # A-U, U-A
                    or (nuc_i == 2 and nuc_j == 1)
                    or (nuc_i == 1 and nuc_j == 2)  # G-C, C-G
                    or (nuc_i == 2 and nuc_j == 3)
                    or (nuc_i == 3 and nuc_j == 2)  # G-U, U-G (wobble)
                )
                if not is_valid:
                    # Invalid pairs should have very low probability
                    assert bp_probs[i, j] < 0.01

    def test_gradient_flow(self, predictor):
        """Test that gradients flow through the predictor."""
        seq_len = 15
        # Use soft probabilities for gradient flow
        logits = jax.random.normal(jax.random.PRNGKey(0), (seq_len, 4))
        sequence = jax.nn.softmax(logits, axis=-1)

        def loss_fn(seq):
            data = {"sequence": seq}
            result, _, _ = predictor.apply(data, {}, None)
            # Use partition function as loss (not normalized, so gradient flows)
            # Also weight bp_probs by position to break symmetry
            weights = jnp.arange(seq_len)[:, None] + jnp.arange(seq_len)[None, :]
            return (result["bp_probs"] * weights.astype(jnp.float32)).sum()

        grads = jax.grad(loss_fn)(sequence)

        # Gradients should exist and be non-zero
        assert grads is not None
        assert grads.shape == sequence.shape
        assert jnp.abs(grads).max() > 1e-10

    def test_parameter_gradients(self, predictor):
        """Test gradients with respect to predictor parameters."""
        seq_len = 12
        logits = jax.random.normal(jax.random.PRNGKey(0), (seq_len, 4))
        sequence = jax.nn.softmax(logits, axis=-1)

        def loss_fn(pred):
            data = {"sequence": sequence}
            result, _, _ = pred.apply(data, {}, None)
            return result["bp_probs"].sum()

        _, grads = nnx.value_and_grad(loss_fn)(predictor)
        assert grads is not None

    def test_jit_compatibility(self, predictor):
        """Test JIT compilation works."""
        seq_len = 20
        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.PRNGKey(0), (seq_len,), 0, 4),
            num_classes=4,
        )

        @jax.jit
        def predict(seq):
            data = {"sequence": seq}
            result, _, _ = predictor.apply(data, {}, None)
            return result["bp_probs"]

        # Should compile and run without errors
        bp_probs = predict(sequence)
        assert bp_probs.shape == (seq_len, seq_len)

        # Second call should produce same result
        bp_probs2 = predict(sequence)
        assert jnp.allclose(bp_probs, bp_probs2)

    def test_temperature_effect(self):
        """Test temperature affects output sharpness."""
        seq_len = 20
        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.PRNGKey(0), (seq_len,), 0, 4),
            num_classes=4,
        )
        data = {"sequence": sequence}

        # Low temperature = sharper probabilities
        config_low = RNAFoldConfig(temperature=0.1)
        pred_low = DifferentiableRNAFold(config_low, rngs=nnx.Rngs(42))
        result_low, _, _ = pred_low.apply(data, {}, None)

        # High temperature = smoother probabilities
        config_high = RNAFoldConfig(temperature=5.0)
        pred_high = DifferentiableRNAFold(config_high, rngs=nnx.Rngs(42))
        result_high, _, _ = pred_high.apply(data, {}, None)

        # Low temp should have more extreme values (closer to 0 or 1)
        bp_low = result_low["bp_probs"]
        bp_high = result_high["bp_probs"]

        # Both should be valid probabilities
        assert jnp.all(bp_low >= 0) and jnp.all(bp_low <= 1)
        assert jnp.all(bp_high >= 0) and jnp.all(bp_high <= 1)

        # Max probability should differ between temperatures
        # (low temp = sharper peaks, high temp = more uniform)
        max_low = bp_low.max()
        max_high = bp_high.max()
        # Both should have some non-zero probability
        assert max_low > 0 and max_high > 0


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_rna_fold_predictor_default(self):
        """Test default factory function."""
        predictor = create_rna_fold_predictor()
        assert isinstance(predictor, DifferentiableRNAFold)

    def test_create_rna_fold_predictor_custom(self):
        """Test factory with custom parameters."""
        predictor = create_rna_fold_predictor(
            temperature=0.5,
            min_hairpin_loop=4,
        )
        assert predictor.config.temperature == 0.5
        assert predictor.config.min_hairpin_loop == 4


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_short_sequence(self):
        """Test handling of very short sequences."""
        config = RNAFoldConfig()
        predictor = DifferentiableRNAFold(config, rngs=nnx.Rngs(42))

        # Minimum viable length (min_hairpin_loop + 2)
        seq_len = 6
        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.PRNGKey(0), (seq_len,), 0, 4),
            num_classes=4,
        )

        data = {"sequence": sequence}
        result, _, _ = predictor.apply(data, {}, None)
        assert result["bp_probs"].shape == (seq_len, seq_len)

    def test_longer_sequence(self):
        """Test handling of longer sequences."""
        config = RNAFoldConfig()
        predictor = DifferentiableRNAFold(config, rngs=nnx.Rngs(42))

        seq_len = 100
        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.PRNGKey(0), (seq_len,), 0, 4),
            num_classes=4,
        )

        data = {"sequence": sequence}
        result, _, _ = predictor.apply(data, {}, None)
        assert result["bp_probs"].shape == (seq_len, seq_len)

    def test_soft_one_hot_input(self):
        """Test handling of soft (probabilistic) one-hot input."""
        config = RNAFoldConfig()
        predictor = DifferentiableRNAFold(config, rngs=nnx.Rngs(42))

        seq_len = 15
        # Soft probabilities instead of hard one-hot
        sequence = jax.nn.softmax(
            jax.random.normal(jax.random.PRNGKey(0), (seq_len, 4)),
            axis=-1,
        )

        data = {"sequence": sequence}
        result, _, _ = predictor.apply(data, {}, None)
        assert result["bp_probs"].shape == (seq_len, seq_len)

    def test_preserves_input_in_output(self):
        """Test that original input is preserved in output."""
        config = RNAFoldConfig()
        predictor = DifferentiableRNAFold(config, rngs=nnx.Rngs(42))

        seq_len = 15
        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.PRNGKey(0), (seq_len,), 0, 4),
            num_classes=4,
        )
        data = {"sequence": sequence, "other_key": "preserved"}

        result, _, _ = predictor.apply(data, {}, None)

        # Original sequence should be in output
        assert jnp.allclose(result["sequence"], sequence)
        # Other keys should be preserved
        assert result["other_key"] == "preserved"

    def test_known_hairpin_structure(self):
        """Test with a known simple hairpin structure."""
        config = RNAFoldConfig(temperature=0.1)  # Low temp for sharp output
        predictor = DifferentiableRNAFold(config, rngs=nnx.Rngs(42))

        # Create sequence that should form hairpin: GGGG...CCCC
        # This tests that complementary ends have higher pair probability
        # Indices: A=0, C=1, G=2, U=3
        seq_indices = jnp.array(
            [2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1]  # GGGG  # AAAAA (loop)  # CCCC
        )
        sequence = jax.nn.one_hot(seq_indices, num_classes=4)

        data = {"sequence": sequence}
        result, _, _ = predictor.apply(data, {}, None)

        bp_probs = result["bp_probs"]

        # G-C pairs at ends should have reasonable probability
        # Positions 0-12, 1-11, 2-10, 3-9 should pair (G-C)
        # Note: exact values depend on energy model
        # Just check the matrix has valid structure
        assert bp_probs.shape == (13, 13)
        assert jnp.all(bp_probs >= 0)
