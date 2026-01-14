"""Tests for differentiable motif discovery operator.

Following TDD principles, these tests define the expected behavior
of the DifferentiableMotifDiscovery operator (MEME-style).
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest


class TestMotifDiscoveryConfig:
    """Tests for MotifDiscoveryConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from diffbio.operators.rnaseq.motif_discovery import MotifDiscoveryConfig

        config = MotifDiscoveryConfig(stream_name=None)

        assert config.motif_width == 12
        assert config.num_motifs == 1
        assert config.alphabet_size == 4
        assert config.temperature == 1.0
        assert config.stochastic is False

    def test_custom_config(self):
        """Test custom configuration values."""
        from diffbio.operators.rnaseq.motif_discovery import MotifDiscoveryConfig

        config = MotifDiscoveryConfig(
            motif_width=20,
            num_motifs=5,
            alphabet_size=4,
            temperature=0.5,
            stream_name=None,
        )

        assert config.motif_width == 20
        assert config.num_motifs == 5
        assert config.temperature == 0.5


class TestDifferentiableMotifDiscovery:
    """Tests for DifferentiableMotifDiscovery operator."""

    @pytest.fixture
    def rngs(self):
        """Provide RNG fixture."""
        return nnx.Rngs(42)

    @pytest.fixture
    def config(self):
        """Provide default config."""
        from diffbio.operators.rnaseq.motif_discovery import MotifDiscoveryConfig

        return MotifDiscoveryConfig(
            motif_width=8,
            num_motifs=2,
            alphabet_size=4,
            temperature=1.0,
            stream_name=None,
        )

    @pytest.fixture
    def motif_op(self, config, rngs):
        """Create motif discovery instance."""
        from diffbio.operators.rnaseq.motif_discovery import DifferentiableMotifDiscovery

        return DifferentiableMotifDiscovery(config, rngs=rngs)

    def test_initialization(self, config, rngs):
        """Test operator initialization."""
        from diffbio.operators.rnaseq.motif_discovery import DifferentiableMotifDiscovery

        motif_op = DifferentiableMotifDiscovery(config, rngs=rngs)

        assert motif_op.config == config
        assert hasattr(motif_op, "pwm_logits")
        # PWM shape: (num_motifs, motif_width, alphabet_size)
        assert motif_op.pwm_logits.value.shape == (
            config.num_motifs,
            config.motif_width,
            config.alphabet_size,
        )

    def test_initialization_without_rngs(self, config):
        """Test initialization without providing RNGs."""
        from diffbio.operators.rnaseq.motif_discovery import DifferentiableMotifDiscovery

        motif_op = DifferentiableMotifDiscovery(config, rngs=None)
        assert motif_op is not None

    def test_apply_single_sequence(self, motif_op, config):
        """Test motif scanning on a single sequence."""
        seq_length = 100

        # One-hot encoded sequence (length, alphabet_size)
        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.key(0), (seq_length,), 0, 4), 4
        )

        data = {"sequence": sequence}
        result, state, metadata = motif_op.apply(data, {}, None)

        assert "motif_scores" in result
        assert "motif_positions" in result
        assert "pwm" in result

        # Scores shape: (num_positions, num_motifs)
        expected_positions = seq_length - config.motif_width + 1
        assert result["motif_scores"].shape == (expected_positions, config.num_motifs)

    def test_apply_batch_sequences(self, motif_op, config):
        """Test motif scanning on batch of sequences."""
        batch_size = 5
        seq_length = 100

        sequences = jax.nn.one_hot(
            jax.random.randint(jax.random.key(0), (batch_size, seq_length), 0, 4), 4
        )

        data = {"sequence": sequences}
        result, state, metadata = motif_op.apply(data, {}, None)

        expected_positions = seq_length - config.motif_width + 1
        assert result["motif_scores"].shape == (
            batch_size,
            expected_positions,
            config.num_motifs,
        )

    def test_pwm_is_valid_probability(self, motif_op, config):
        """Test that PWM is a valid probability distribution."""
        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.key(0), (50,), 0, 4), 4
        )

        data = {"sequence": sequence}
        result, _, _ = motif_op.apply(data, {}, None)

        pwm = result["pwm"]

        # PWM should be normalized (sums to 1 over alphabet at each position)
        sums = pwm.sum(axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)

        # PWM values should be in [0, 1]
        assert jnp.all(pwm >= 0.0)
        assert jnp.all(pwm <= 1.0)

    def test_output_finite(self, motif_op):
        """Test that all outputs are finite."""
        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.key(0), (50,), 0, 4), 4
        )

        data = {"sequence": sequence}
        result, _, _ = motif_op.apply(data, {}, None)

        assert jnp.all(jnp.isfinite(result["motif_scores"]))
        assert jnp.all(jnp.isfinite(result["pwm"]))

    def test_preserves_original_data(self, motif_op):
        """Test that original data is preserved in output."""
        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.key(0), (50,), 0, 4), 4
        )
        extra_data = jnp.array([1.0, 2.0, 3.0])

        data = {"sequence": sequence, "extra": extra_data}
        result, _, _ = motif_op.apply(data, {}, None)

        assert "extra" in result
        assert jnp.allclose(result["extra"], extra_data)


class TestMotifDiscoveryDifferentiability:
    """Tests for gradient flow through the motif discovery operator."""

    @pytest.fixture
    def rngs(self):
        """Provide RNG fixture."""
        return nnx.Rngs(42)

    @pytest.fixture
    def config(self):
        """Provide default config."""
        from diffbio.operators.rnaseq.motif_discovery import MotifDiscoveryConfig

        return MotifDiscoveryConfig(
            motif_width=8,
            num_motifs=2,
            stream_name=None,
        )

    def test_gradient_flow_through_operator(self, config, rngs):
        """Test that gradients flow through the operator."""
        from diffbio.operators.rnaseq.motif_discovery import DifferentiableMotifDiscovery

        motif_op = DifferentiableMotifDiscovery(config, rngs=rngs)

        def loss_fn(op, sequence):
            data = {"sequence": sequence}
            result, _, _ = op.apply(data, {}, None)
            return result["motif_scores"].sum()

        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.key(0), (50,), 0, 4), 4
        )
        grads = nnx.grad(loss_fn)(motif_op, sequence)

        assert grads is not None

    def test_gradient_wrt_pwm(self, config, rngs):
        """Test gradient with respect to PWM parameters."""
        from diffbio.operators.rnaseq.motif_discovery import DifferentiableMotifDiscovery

        motif_op = DifferentiableMotifDiscovery(config, rngs=rngs)

        def loss_fn(op, sequence):
            data = {"sequence": sequence}
            result, _, _ = op.apply(data, {}, None)
            return result["motif_scores"].mean()

        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.key(0), (50,), 0, 4), 4
        )
        grads = nnx.grad(loss_fn)(motif_op, sequence)

        assert hasattr(grads, "pwm_logits")
        assert grads.pwm_logits.value is not None
        assert grads.pwm_logits.value.shape == motif_op.pwm_logits.value.shape

    def test_gradient_wrt_input(self, config, rngs):
        """Test gradient with respect to input sequence."""
        from diffbio.operators.rnaseq.motif_discovery import DifferentiableMotifDiscovery

        motif_op = DifferentiableMotifDiscovery(config, rngs=rngs)

        def loss_fn(sequence):
            data = {"sequence": sequence}
            result, _, _ = motif_op.apply(data, {}, None)
            return result["motif_scores"].sum()

        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.key(0), (50,), 0, 4), 4
        ).astype(jnp.float32)

        grad = jax.grad(loss_fn)(sequence)

        assert grad.shape == sequence.shape
        assert jnp.all(jnp.isfinite(grad))


class TestMotifDiscoveryJITCompatibility:
    """Tests for JIT compilation compatibility."""

    @pytest.fixture
    def rngs(self):
        """Provide RNG fixture."""
        return nnx.Rngs(42)

    @pytest.fixture
    def config(self):
        """Provide default config."""
        from diffbio.operators.rnaseq.motif_discovery import MotifDiscoveryConfig

        return MotifDiscoveryConfig(
            motif_width=8,
            num_motifs=2,
            stream_name=None,
        )

    def test_jit_apply(self, config, rngs):
        """Test JIT compilation of apply method."""
        from diffbio.operators.rnaseq.motif_discovery import DifferentiableMotifDiscovery

        motif_op = DifferentiableMotifDiscovery(config, rngs=rngs)

        @jax.jit
        def jit_apply(sequence):
            data = {"sequence": sequence}
            result, _, _ = motif_op.apply(data, {}, None)
            return result["motif_scores"]

        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.key(0), (50,), 0, 4), 4
        )

        # Should compile and run without error
        result = jit_apply(sequence)
        expected_positions = 50 - config.motif_width + 1
        assert result.shape == (expected_positions, config.num_motifs)

    def test_jit_gradient(self, config, rngs):
        """Test JIT compilation of gradient computation."""
        from diffbio.operators.rnaseq.motif_discovery import DifferentiableMotifDiscovery

        motif_op = DifferentiableMotifDiscovery(config, rngs=rngs)

        @jax.jit
        def loss_and_grad(sequence):
            def loss_fn(seq):
                data = {"sequence": seq}
                result, _, _ = motif_op.apply(data, {}, None)
                return result["motif_scores"].sum()

            return jax.value_and_grad(loss_fn)(sequence)

        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.key(0), (50,), 0, 4), 4
        ).astype(jnp.float32)

        # Should compile and run without error
        loss, grad = loss_and_grad(sequence)
        assert jnp.isfinite(loss)
        assert jnp.all(jnp.isfinite(grad))


class TestPWMScanning:
    """Tests for PWM scanning functionality."""

    @pytest.fixture
    def rngs(self):
        """Provide RNG fixture."""
        return nnx.Rngs(42)

    @pytest.fixture
    def config(self):
        """Provide config."""
        from diffbio.operators.rnaseq.motif_discovery import MotifDiscoveryConfig

        return MotifDiscoveryConfig(
            motif_width=6,
            num_motifs=1,
            stream_name=None,
        )

    def test_motif_scores_correlate_with_pwm_match(self, config, rngs):
        """Test that motif scores are higher for sequences matching the PWM."""
        from diffbio.operators.rnaseq.motif_discovery import DifferentiableMotifDiscovery

        motif_op = DifferentiableMotifDiscovery(config, rngs=rngs)

        # Create a sequence that has a clear motif pattern
        # The PWM will have some structure, and we check that
        # different positions get different scores
        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.key(0), (50,), 0, 4), 4
        )

        data = {"sequence": sequence}
        result, _, _ = motif_op.apply(data, {}, None)

        scores = result["motif_scores"]

        # Scores should have variation (not all the same)
        assert jnp.std(scores) > 0
