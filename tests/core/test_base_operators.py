"""Tests for base operator classes.

Following TDD: These tests define the expected behavior for domain-specific
base operator classes that provide shared functionality.
"""

import jax
import jax.numpy as jnp
import pytest
from dataclasses import dataclass
from datarax.core.config import OperatorConfig
from flax import nnx

from diffbio.constants import (
    DEFAULT_EDGE_FEATURES,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_HMM_EMISSIONS,
    DEFAULT_HMM_STATES,
    DEFAULT_LATENT_DIM,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_NODE_FEATURES,
    DEFAULT_NUM_HEADS,
    DEFAULT_TEMPERATURE,
    DNA_ALPHABET_SIZE,
    EPSILON,
)


# =============================================================================
# Mock Configs for Testing
# =============================================================================


@dataclass(frozen=True)
class MockTemperatureConfig(OperatorConfig):
    """Mock config for TemperatureOperator tests."""

    temperature: float = DEFAULT_TEMPERATURE
    learnable_temperature: bool = False


@dataclass(frozen=True)
class MockSequenceConfig(OperatorConfig):
    """Mock config for SequenceOperator tests."""

    alphabet_size: int = DNA_ALPHABET_SIZE
    max_length: int = DEFAULT_MAX_SEQ_LENGTH


@dataclass(frozen=True)
class MockEncoderDecoderConfig(OperatorConfig):
    """Mock config for EncoderDecoderOperator tests."""

    latent_dim: int = DEFAULT_LATENT_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM


@dataclass(frozen=True)
class MockGraphConfig(OperatorConfig):
    """Mock config for GraphOperator tests."""

    node_features: int = DEFAULT_NODE_FEATURES
    edge_features: int = DEFAULT_EDGE_FEATURES
    num_heads: int = DEFAULT_NUM_HEADS


@dataclass(frozen=True)
class MockHMMConfig(OperatorConfig):
    """Mock config for HMMOperator tests."""

    num_states: int = DEFAULT_HMM_STATES
    num_emissions: int = DEFAULT_HMM_EMISSIONS
    temperature: float = DEFAULT_TEMPERATURE


# =============================================================================
# TemperatureOperator Tests
# =============================================================================


class TestTemperatureOperator:
    """Tests for TemperatureOperator base class."""

    def test_init(self, rngs):
        """Test TemperatureOperator initialization."""
        from diffbio.core.base_operators import TemperatureOperator

        op = TemperatureOperator(MockTemperatureConfig(), rngs=rngs)
        assert op is not None

    def test_soft_max_method(self, rngs):
        """Test soft_max method uses logsumexp."""
        from diffbio.core.base_operators import TemperatureOperator

        op = TemperatureOperator(MockTemperatureConfig(), rngs=rngs)
        values = jnp.array([1.0, 5.0, 2.0])
        result = op.soft_max(values)

        # Result should be close to but >= max
        hard_max = jnp.max(values)
        assert float(result) >= float(hard_max)
        # With default temperature, result should be within 2*temperature of max
        assert float(result) < float(hard_max) + DEFAULT_TEMPERATURE * 2

    def test_temperature_learnable(self, rngs):
        """Test temperature parameter is learnable."""
        from diffbio.core.base_operators import TemperatureOperator

        @dataclass(frozen=True)
        class LearnableConfig(OperatorConfig):
            temperature: float = DEFAULT_TEMPERATURE
            learnable_temperature: bool = True

        op = TemperatureOperator(LearnableConfig(), rngs=rngs)

        # Should be able to compute gradients w.r.t. temperature
        def loss_fn(op):
            values = jnp.array([1.0, 5.0, 2.0])
            return op.soft_max(values)

        grads = nnx.grad(loss_fn)(op)
        assert grads is not None


# =============================================================================
# SequenceOperator Tests
# =============================================================================


class TestSequenceOperator:
    """Tests for SequenceOperator base class."""

    def test_init(self, rngs):
        """Test SequenceOperator initialization."""
        from diffbio.core.base_operators import SequenceOperator

        op = SequenceOperator(MockSequenceConfig(), rngs=rngs)
        assert op is not None
        assert op.alphabet_size == DNA_ALPHABET_SIZE

    def test_validate_sequence(self, rngs):
        """Test sequence validation method."""
        from diffbio.core.base_operators import SequenceOperator

        op = SequenceOperator(MockSequenceConfig(), rngs=rngs)

        # Valid sequence (one-hot encoded)
        seq_length = 3
        valid_seq = jnp.eye(DNA_ALPHABET_SIZE)[:seq_length]
        assert op.validate_sequence(valid_seq)

        # Invalid shape (wrong alphabet size)
        wrong_alphabet = DNA_ALPHABET_SIZE + 1
        invalid_seq = jnp.ones((seq_length, wrong_alphabet))
        assert not op.validate_sequence(invalid_seq)

    def test_normalize_sequence(self, rngs):
        """Test sequence normalization to sum to 1."""
        from diffbio.core.base_operators import SequenceOperator

        op = SequenceOperator(MockSequenceConfig(), rngs=rngs)

        # Unnormalized sequence
        unnorm = jnp.array([[1.0, 2.0, 1.0, 0.5], [0.5, 0.5, 2.0, 1.0]])
        normalized = op.normalize_sequence(unnorm)

        # Each row should sum to 1
        row_sums = jnp.sum(normalized, axis=-1)
        assert jnp.allclose(row_sums, 1.0)


# =============================================================================
# EncoderDecoderOperator Tests
# =============================================================================


class TestEncoderDecoderOperator:
    """Tests for EncoderDecoderOperator (VAE pattern)."""

    def test_init(self, rngs):
        """Test EncoderDecoderOperator initialization."""
        from diffbio.core.base_operators import EncoderDecoderOperator

        op = EncoderDecoderOperator(MockEncoderDecoderConfig(), rngs=rngs)
        assert op is not None
        assert hasattr(op, "latent_dim")

    def test_reparameterize(self, rngs):
        """Test reparameterization trick."""
        from diffbio.core.base_operators import EncoderDecoderOperator

        op = EncoderDecoderOperator(MockEncoderDecoderConfig(), rngs=rngs)

        batch_size = 5
        mean = jnp.zeros((batch_size, DEFAULT_LATENT_DIM))
        log_var = jnp.zeros((batch_size, DEFAULT_LATENT_DIM))  # variance = 1

        z = op.reparameterize(mean, log_var)

        assert z.shape == (batch_size, DEFAULT_LATENT_DIM)
        # Should be close to mean when variance is 1
        assert jnp.abs(jnp.mean(z)) < 1.0

    def test_kl_divergence(self, rngs):
        """Test KL divergence computation."""
        from diffbio.core.base_operators import EncoderDecoderOperator

        op = EncoderDecoderOperator(MockEncoderDecoderConfig(), rngs=rngs)

        # Standard normal -> KL should be 0
        batch_size = 5
        mean = jnp.zeros((batch_size, DEFAULT_LATENT_DIM))
        log_var = jnp.zeros((batch_size, DEFAULT_LATENT_DIM))  # variance = 1

        kl = op.kl_divergence(mean, log_var)

        assert kl.shape == ()
        # KL divergence from N(0,1) to N(0,1) should be close to 0
        assert jnp.abs(kl) < EPSILON * 10


# =============================================================================
# GraphOperator Tests
# =============================================================================


class TestGraphOperator:
    """Tests for GraphOperator base class."""

    def test_init(self, rngs):
        """Test GraphOperator initialization."""
        from diffbio.core.base_operators import GraphOperator

        op = GraphOperator(MockGraphConfig(), rngs=rngs)
        assert op is not None
        assert hasattr(op, "node_features")

    def test_scatter_aggregate(self, rngs):
        """Test scatter aggregation methods."""
        from diffbio.core.base_operators import GraphOperator

        @dataclass(frozen=True)
        class SmallGraphConfig(OperatorConfig):
            node_features: int = 4
            edge_features: int = 2
            num_heads: int = 1

        op = GraphOperator(SmallGraphConfig(), rngs=rngs)

        # Test scatter_sum
        feature_dim = 2
        messages = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        indices = jnp.array([0, 0, 1])
        num_nodes = 3

        result = op.scatter_aggregate(messages, indices, num_nodes, "sum")

        assert result.shape == (num_nodes, feature_dim)
        # Node 0 should have sum of messages 0 and 1
        assert jnp.allclose(result[0], jnp.array([4.0, 6.0]))
        # Node 1 should have message 2
        assert jnp.allclose(result[1], jnp.array([5.0, 6.0]))


# =============================================================================
# HMMOperator Tests
# =============================================================================


class TestHMMOperator:
    """Tests for HMMOperator base class."""

    def test_init(self, rngs):
        """Test HMMOperator initialization."""
        from diffbio.core.base_operators import HMMOperator

        op = HMMOperator(MockHMMConfig(), rngs=rngs)
        assert op is not None
        assert hasattr(op, "num_states")

    def test_forward_pass(self, rngs):
        """Test forward algorithm."""
        from diffbio.core.base_operators import HMMOperator

        op = HMMOperator(MockHMMConfig(), rngs=rngs)

        # Observations must be valid indices into emission matrix
        observations = jnp.array([0, 1, 2, 3, 0, 1])
        log_prob = op.forward_pass(observations)

        assert log_prob.shape == ()
        assert log_prob < 0  # Log probability is negative

    def test_forward_backward_posteriors(self, rngs):
        """Test forward-backward gives valid posteriors."""
        from diffbio.core.base_operators import HMMOperator

        op = HMMOperator(MockHMMConfig(), rngs=rngs)

        seq_length = 4
        observations = jnp.array([0, 1, 2, 3])
        posteriors = op.forward_backward_posteriors(observations)

        assert posteriors.shape == (seq_length, DEFAULT_HMM_STATES)
        # Each position should sum to 1
        assert jnp.allclose(jnp.sum(posteriors, axis=-1), 1.0, atol=1e-5)


# =============================================================================
# Composition Tests
# =============================================================================


class TestOperatorComposition:
    """Test that base operators compose correctly."""

    def test_temperature_with_sequence(self, rngs):
        """Test combining Temperature and Sequence operators."""
        from diffbio.core.base_operators import SequenceOperator, TemperatureOperator

        @dataclass(frozen=True)
        class AlignmentConfig(OperatorConfig):
            temperature: float = DEFAULT_TEMPERATURE
            alphabet_size: int = DNA_ALPHABET_SIZE
            max_length: int = DEFAULT_MAX_SEQ_LENGTH

        class TestAligner(TemperatureOperator, SequenceOperator):
            def __init__(self, config, *, rngs=None):
                TemperatureOperator.__init__(self, config, rngs=rngs)
                SequenceOperator.__init__(self, config, rngs=rngs)

            def apply(self, data, state, metadata, random_params=None, stats=None):
                return data, state, metadata

        aligner = TestAligner(AlignmentConfig(), rngs=rngs)
        assert hasattr(aligner, "soft_max")
        assert hasattr(aligner, "validate_sequence")


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases for base operators."""

    def test_low_temperature(self, rngs):
        """Test behavior with very low temperature (approaches hard max)."""
        from diffbio.core.base_operators import TemperatureOperator

        low_temp = 0.001

        @dataclass(frozen=True)
        class LowTempConfig(OperatorConfig):
            temperature: float = low_temp

        op = TemperatureOperator(LowTempConfig(), rngs=rngs)
        values = jnp.array([1.0, 5.0, 2.0])
        result = op.soft_max(values)

        # Very low temp should be very close to hard max
        hard_max = jnp.max(values)
        assert jnp.abs(result - hard_max) < low_temp

    def test_empty_sequence(self, rngs):
        """Test operators handle empty sequences."""
        from diffbio.core.base_operators import SequenceOperator

        op = SequenceOperator(MockSequenceConfig(), rngs=rngs)

        empty_seq = jnp.zeros((0, DNA_ALPHABET_SIZE))
        assert op.validate_sequence(empty_seq)

    def test_jit_compatibility(self, rngs):
        """Test base operators work with JIT."""
        from diffbio.core.base_operators import TemperatureOperator

        op = TemperatureOperator(MockTemperatureConfig(), rngs=rngs)

        @jax.jit
        def compute(values):
            return op.soft_max(values)

        values = jnp.array([1.0, 5.0, 2.0])
        result = compute(values)
        assert jnp.isfinite(result)


# =============================================================================
# Mathematical Verification Tests
# =============================================================================


class TestMathematicalVerification:
    """Tests verifying mathematical correctness of base operators."""

    def test_soft_max_formula(self, rngs):
        """Verify soft_max matches logsumexp formula."""
        from diffbio.core.base_operators import TemperatureOperator

        temperature = 0.5

        @dataclass(frozen=True)
        class Config(OperatorConfig):
            temperature: float = 0.5  # Use literal, not variable reference

        op = TemperatureOperator(Config(), rngs=rngs)
        values = jnp.array([1.0, 2.0, 3.0])

        # Manual: T * log(sum(exp(x/T)))
        expected = temperature * jax.scipy.special.logsumexp(values / temperature)
        result = op.soft_max(values)

        assert jnp.allclose(result, expected, atol=1e-5)

    def test_soft_max_approaches_max(self, rngs):
        """Verify soft_max approaches hard max as temperature -> 0."""
        from diffbio.core.base_operators import TemperatureOperator

        @dataclass(frozen=True)
        class LowTempConfig(OperatorConfig):
            temperature: float = 0.0001

        op = TemperatureOperator(LowTempConfig(), rngs=rngs)
        values = jnp.array([1.0, 5.0, 2.0, 3.0])

        result = op.soft_max(values)
        hard_max = jnp.max(values)

        assert jnp.abs(result - hard_max) < 0.001

    def test_soft_argmax_formula(self, rngs):
        """Verify soft_argmax is weighted sum of positions."""
        from diffbio.core.base_operators import TemperatureOperator

        temperature = 1.0

        @dataclass(frozen=True)
        class Config(OperatorConfig):
            temperature: float = 1.0  # Use literal, not variable reference

        op = TemperatureOperator(Config(), rngs=rngs)
        logits = jnp.array([1.0, 2.0, 3.0, 4.0])

        # Manual: sum(softmax(x/T) * positions)
        weights = jax.nn.softmax(logits / temperature)
        positions = jnp.arange(len(logits), dtype=logits.dtype)
        expected = jnp.sum(weights * positions)

        result = op.soft_argmax(logits)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_kl_divergence_formula(self, rngs):
        """Verify KL divergence follows the closed-form formula for Gaussians."""
        from diffbio.core.base_operators import EncoderDecoderOperator

        op = EncoderDecoderOperator(MockEncoderDecoderConfig(), rngs=rngs)

        # Non-standard normal to verify formula
        mean = jnp.array([[1.0, 2.0], [0.5, -0.5]])
        log_var = jnp.array([[0.5, -0.5], [1.0, 0.0]])

        # Manual: -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
        expected = -0.5 * jnp.sum(1 + log_var - mean**2 - jnp.exp(log_var))

        result = op.kl_divergence(mean, log_var)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_reparameterize_statistics(self, rngs):
        """Verify reparameterization produces samples with correct statistics."""
        from diffbio.core.base_operators import EncoderDecoderOperator

        op = EncoderDecoderOperator(MockEncoderDecoderConfig(), rngs=rngs)

        # Sample many times with mean=1, var=0.25 (std=0.5)
        target_mean = 1.0
        target_std = 0.5
        mean = jnp.full((1000, DEFAULT_LATENT_DIM), target_mean)
        log_var = jnp.full((1000, DEFAULT_LATENT_DIM), jnp.log(target_std**2))

        # Create operator with different seeds for each sample
        samples = op.reparameterize(mean, log_var)

        # Check statistics (note: single draw so variability expected)
        sample_mean = jnp.mean(samples)
        # Mean should be approximately target_mean
        assert jnp.abs(sample_mean - target_mean) < 0.5

    def test_hmm_forward_log_space(self, rngs):
        """Verify HMM forward pass works correctly in log space."""
        from diffbio.core.base_operators import HMMOperator

        op = HMMOperator(MockHMMConfig(), rngs=rngs)
        observations = jnp.array([0, 1, 2])

        log_prob = op.forward_pass(observations)

        # Log probability should be negative
        assert log_prob < 0
        # Should be finite
        assert jnp.isfinite(log_prob)

    def test_hmm_posteriors_bayes_rule(self, rngs):
        """Verify HMM posteriors satisfy Bayes rule properties."""
        from diffbio.core.base_operators import HMMOperator

        op = HMMOperator(MockHMMConfig(), rngs=rngs)
        observations = jnp.array([0, 1, 2, 3])

        posteriors = op.forward_backward_posteriors(observations)

        # All posteriors should be non-negative
        assert jnp.all(posteriors >= 0)
        # All posteriors should be <= 1
        assert jnp.all(posteriors <= 1.0)
        # Each row should sum to 1
        row_sums = jnp.sum(posteriors, axis=-1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-5)

    def test_scatter_aggregate_sum(self, rngs):
        """Verify scatter_aggregate sum is mathematically correct."""
        from diffbio.core.base_operators import GraphOperator

        op = GraphOperator(MockGraphConfig(), rngs=rngs)

        messages = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        indices = jnp.array([0, 0, 1, 2])
        num_nodes = 4

        result = op.scatter_aggregate(messages, indices, num_nodes, "sum")

        # Node 0: sum of messages 0 and 1
        assert jnp.allclose(result[0], jnp.array([4.0, 6.0]))
        # Node 1: message 2
        assert jnp.allclose(result[1], jnp.array([5.0, 6.0]))
        # Node 2: message 3
        assert jnp.allclose(result[2], jnp.array([7.0, 8.0]))
        # Node 3: no messages
        assert jnp.allclose(result[3], jnp.array([0.0, 0.0]))

    def test_scatter_aggregate_mean(self, rngs):
        """Verify scatter_aggregate mean is mathematically correct."""
        from diffbio.core.base_operators import GraphOperator

        op = GraphOperator(MockGraphConfig(), rngs=rngs)

        messages = jnp.array([[2.0, 4.0], [4.0, 8.0]])
        indices = jnp.array([0, 0])
        num_nodes = 2

        result = op.scatter_aggregate(messages, indices, num_nodes, "mean")

        # Node 0: mean of messages 0 and 1
        assert jnp.allclose(result[0], jnp.array([3.0, 6.0]))


# =============================================================================
# Flax NNX Compatibility Tests
# =============================================================================


class TestFlaxNNXCompatibility:
    """Tests verifying Flax NNX compatibility."""

    def test_nnx_jit_with_operator(self, rngs):
        """Test base operators work with nnx.jit."""
        from diffbio.core.base_operators import TemperatureOperator

        op = TemperatureOperator(MockTemperatureConfig(), rngs=rngs)

        @nnx.jit
        def compute(operator, values):
            return operator.soft_max(values)

        values = jnp.array([1.0, 5.0, 2.0])
        result = compute(op, values)
        assert jnp.isfinite(result)

    def test_nnx_grad_with_learnable_temperature(self, rngs):
        """Test nnx.grad works with learnable temperature parameter."""
        from diffbio.core.base_operators import TemperatureOperator

        @dataclass(frozen=True)
        class LearnableConfig(OperatorConfig):
            temperature: float = DEFAULT_TEMPERATURE
            learnable_temperature: bool = True

        op = TemperatureOperator(LearnableConfig(), rngs=rngs)

        def loss_fn(operator, values):
            return operator.soft_max(values)

        values = jnp.array([1.0, 5.0, 2.0])
        grads = nnx.grad(loss_fn)(op, values)

        assert grads is not None

    def test_nnx_value_and_grad_encoder_decoder(self, rngs):
        """Test nnx.value_and_grad with EncoderDecoderOperator."""
        from diffbio.core.base_operators import EncoderDecoderOperator

        op = EncoderDecoderOperator(MockEncoderDecoderConfig(), rngs=rngs)

        def loss_fn(operator, mean, log_var):
            z = operator.reparameterize(mean, log_var)
            kl = operator.kl_divergence(mean, log_var)
            return jnp.sum(z) + kl

        mean = jnp.zeros((2, DEFAULT_LATENT_DIM))
        log_var = jnp.zeros((2, DEFAULT_LATENT_DIM))

        loss, grads = nnx.value_and_grad(loss_fn)(op, mean, log_var)

        assert jnp.isfinite(loss)
        assert grads is not None

    def test_nnx_split_merge_hmm(self, rngs):
        """Test nnx.split and nnx.merge work with HMMOperator."""
        from diffbio.core.base_operators import HMMOperator

        op = HMMOperator(MockHMMConfig(), rngs=rngs)

        # Split into graph def and state
        graphdef, state = nnx.split(op)

        # Merge back
        restored = nnx.merge(graphdef, state)

        # Should produce same output
        observations = jnp.array([0, 1, 2, 3])
        result1 = op.forward_pass(observations)
        result2 = restored.forward_pass(observations)

        assert jnp.allclose(result1, result2)

    def test_vmap_with_sequence_operator(self, rngs):
        """Test jax.vmap works with SequenceOperator methods."""
        from diffbio.core.base_operators import SequenceOperator

        op = SequenceOperator(MockSequenceConfig(), rngs=rngs)

        # Batch of sequences (batch, length, alphabet)
        batch_size = 5
        seq_length = 10
        batch_seqs = jax.random.normal(
            jax.random.PRNGKey(42), (batch_size, seq_length, DNA_ALPHABET_SIZE)
        )

        # vmap normalize over batch
        batch_normalized = jax.vmap(op.normalize_sequence)(batch_seqs)

        assert batch_normalized.shape == (batch_size, seq_length, DNA_ALPHABET_SIZE)
        # Each position should sum to 1
        assert jnp.allclose(
            jnp.sum(batch_normalized, axis=-1), jnp.ones((batch_size, seq_length)), atol=1e-5
        )


# =============================================================================
# Scalability Tests
# =============================================================================


class TestScalability:
    """Tests verifying scalability of base operators."""

    @pytest.mark.parametrize("size", [100, 500, 1000])
    def test_soft_max_scales(self, rngs, size):
        """Test soft_max handles various input sizes."""
        from diffbio.core.base_operators import TemperatureOperator

        op = TemperatureOperator(MockTemperatureConfig(), rngs=rngs)
        values = jax.random.normal(jax.random.PRNGKey(42), (size,))

        result = op.soft_max(values)

        assert jnp.isfinite(result)
        assert result >= jnp.max(values)

    @pytest.mark.parametrize("seq_len", [100, 500, 1000])
    def test_sequence_normalization_scales(self, rngs, seq_len):
        """Test sequence normalization handles various lengths."""
        from diffbio.core.base_operators import SequenceOperator

        op = SequenceOperator(MockSequenceConfig(), rngs=rngs)
        sequence = jax.random.normal(jax.random.PRNGKey(42), (seq_len, DNA_ALPHABET_SIZE))

        normalized = op.normalize_sequence(sequence)

        assert normalized.shape == (seq_len, DNA_ALPHABET_SIZE)
        assert jnp.allclose(jnp.sum(normalized, axis=-1), 1.0, atol=1e-5)

    @pytest.mark.parametrize("batch_size", [10, 50, 100])
    def test_reparameterize_batched(self, rngs, batch_size):
        """Test reparameterization handles various batch sizes."""
        from diffbio.core.base_operators import EncoderDecoderOperator

        op = EncoderDecoderOperator(MockEncoderDecoderConfig(), rngs=rngs)
        mean = jax.random.normal(jax.random.PRNGKey(42), (batch_size, DEFAULT_LATENT_DIM))
        log_var = jax.random.normal(jax.random.PRNGKey(43), (batch_size, DEFAULT_LATENT_DIM))

        z = op.reparameterize(mean, log_var)

        assert z.shape == (batch_size, DEFAULT_LATENT_DIM)
        assert jnp.all(jnp.isfinite(z))

    @pytest.mark.parametrize("num_messages", [100, 500, 1000])
    def test_scatter_aggregate_scales(self, rngs, num_messages):
        """Test scatter aggregation handles various message counts."""
        from diffbio.core.base_operators import GraphOperator

        op = GraphOperator(MockGraphConfig(), rngs=rngs)

        num_nodes = 50
        feature_dim = 16
        messages = jax.random.normal(jax.random.PRNGKey(42), (num_messages, feature_dim))
        indices = jax.random.randint(jax.random.PRNGKey(43), (num_messages,), 0, num_nodes)

        result = op.scatter_aggregate(messages, indices, num_nodes, "sum")

        assert result.shape == (num_nodes, feature_dim)
        assert jnp.all(jnp.isfinite(result))

    @pytest.mark.parametrize("seq_len", [50, 100, 200])
    def test_hmm_forward_scales(self, rngs, seq_len):
        """Test HMM forward pass handles various sequence lengths."""
        from diffbio.core.base_operators import HMMOperator

        op = HMMOperator(MockHMMConfig(), rngs=rngs)
        observations = jax.random.randint(
            jax.random.PRNGKey(42), (seq_len,), 0, DEFAULT_HMM_EMISSIONS
        )

        log_prob = op.forward_pass(observations)

        assert jnp.isfinite(log_prob)
        assert log_prob < 0  # Log probability should be negative

    @pytest.mark.parametrize("seq_len", [50, 100, 200])
    def test_hmm_forward_backward_scales(self, rngs, seq_len):
        """Test HMM forward-backward handles various sequence lengths."""
        from diffbio.core.base_operators import HMMOperator

        op = HMMOperator(MockHMMConfig(), rngs=rngs)
        observations = jax.random.randint(
            jax.random.PRNGKey(42), (seq_len,), 0, DEFAULT_HMM_EMISSIONS
        )

        posteriors = op.forward_backward_posteriors(observations)

        assert posteriors.shape == (seq_len, DEFAULT_HMM_STATES)
        assert jnp.allclose(jnp.sum(posteriors, axis=-1), 1.0, atol=1e-5)

    def test_global_pool_with_batched_graphs(self, rngs):
        """Test global pooling handles batched graph inputs."""
        from diffbio.core.base_operators import GraphOperator

        op = GraphOperator(MockGraphConfig(), rngs=rngs)

        # Simulate 3 graphs with 10, 15, 20 nodes packed together
        total_nodes = 45
        feature_dim = 16
        node_features = jax.random.normal(jax.random.PRNGKey(42), (total_nodes, feature_dim))
        # Batch assignment
        batch = jnp.concatenate(
            [
                jnp.zeros(10, dtype=jnp.int32),
                jnp.ones(15, dtype=jnp.int32),
                jnp.full(20, 2, dtype=jnp.int32),
            ]
        )

        result = op.global_pool(node_features, batch, aggregation="mean")

        assert result.shape == (3, feature_dim)
        assert jnp.all(jnp.isfinite(result))

    def test_elbo_loss_batched(self, rngs):
        """Test ELBO loss computation with batched inputs."""
        from diffbio.core.base_operators import EncoderDecoderOperator

        op = EncoderDecoderOperator(MockEncoderDecoderConfig(), rngs=rngs)

        batch_size = 32
        recon_loss = jnp.array(100.0)  # Scalar reconstruction loss
        mean = jax.random.normal(jax.random.PRNGKey(42), (batch_size, DEFAULT_LATENT_DIM))
        log_var = jax.random.normal(jax.random.PRNGKey(43), (batch_size, DEFAULT_LATENT_DIM))

        elbo = op.elbo_loss(recon_loss, mean, log_var, beta=1.0)

        assert jnp.isfinite(elbo)
        assert elbo.shape == ()  # Scalar output
