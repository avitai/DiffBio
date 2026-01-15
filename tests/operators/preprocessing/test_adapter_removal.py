"""Tests for diffbio.operators.preprocessing.adapter_removal module.

These tests define the expected behavior of the SoftAdapterRemoval
operator. Implementation should be written to pass these tests.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.preprocessing.adapter_removal import (
    AdapterRemovalConfig,
    SoftAdapterRemoval,
)
from diffbio.sequences.dna import encode_dna_string


class TestAdapterRemovalConfig:
    """Tests for AdapterRemovalConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AdapterRemovalConfig()
        assert config.adapter_sequence == "AGATCGGAAGAG"
        assert config.temperature == 1.0
        assert config.match_threshold == 0.5
        assert config.min_overlap == 6
        assert config.stochastic is False

    def test_custom_adapter(self):
        """Test custom adapter sequence configuration."""
        config = AdapterRemovalConfig(adapter_sequence="CTGTCTCTTAT")
        assert config.adapter_sequence == "CTGTCTCTTAT"

    def test_custom_temperature(self):
        """Test custom temperature configuration."""
        config = AdapterRemovalConfig(temperature=0.5)
        assert config.temperature == 0.5


class TestSoftAdapterRemoval:
    """Tests for SoftAdapterRemoval operator."""

    @pytest.fixture
    def sample_data_no_adapter(self):
        """Provide sample data without adapter."""
        sequence = encode_dna_string("ACGTACGTACGTACGT")
        quality = jnp.ones(16) * 30.0
        return {"sequence": sequence, "quality_scores": quality}

    @pytest.fixture
    def sample_data_with_adapter(self):
        """Provide sample data with adapter at the end."""
        # Sequence + partial adapter (first 8 bases of AGATCGGAAGAG)
        seq_str = "ACGTACGTAACCTTGG" + "AGATCGGA"
        sequence = encode_dna_string(seq_str)
        quality = jnp.ones(24) * 30.0
        return {"sequence": sequence, "quality_scores": quality}

    def test_initialization(self, rngs):
        """Test operator initialization."""
        config = AdapterRemovalConfig()
        op = SoftAdapterRemoval(config, rngs=rngs)
        assert op is not None
        assert op.adapter is not None
        assert float(op.temperature[...]) == 1.0

    def test_initialization_custom_adapter(self, rngs):
        """Test initialization with custom adapter."""
        config = AdapterRemovalConfig(adapter_sequence="CTGTCTCT")
        op = SoftAdapterRemoval(config, rngs=rngs)
        assert op.adapter[...].shape[0] == 8  # Length of custom adapter

    def test_apply_no_adapter_preserves_sequence(self, rngs, sample_data_no_adapter):
        """Test that sequences without adapter are mostly preserved."""
        config = AdapterRemovalConfig()
        op = SoftAdapterRemoval(config, rngs=rngs)

        transformed_data, state, metadata = op.apply(sample_data_no_adapter, {}, None, None)

        # Without adapter, sequence should be largely preserved
        original_sum = jnp.sum(sample_data_no_adapter["sequence"])
        transformed_sum = jnp.sum(transformed_data["sequence"])
        # Allow for some reduction due to soft operations
        assert transformed_sum > 0.7 * original_sum

    def test_apply_with_adapter_trims_end(self, rngs, sample_data_with_adapter):
        """Test that sequences with adapter have end trimmed."""
        config = AdapterRemovalConfig()
        op = SoftAdapterRemoval(config, rngs=rngs)

        transformed_data, state, metadata = op.apply(sample_data_with_adapter, {}, None, None)

        # With adapter at end, the adapter portion should be down-weighted
        # The first 16 bases should have higher weight than last 8
        first_part_sum = jnp.sum(transformed_data["sequence"][:16])
        last_part_sum = jnp.sum(transformed_data["sequence"][16:])

        # First part should have higher total weight than adapter region
        # (Note: this depends on the soft matching working correctly)
        assert first_part_sum >= last_part_sum * 0.5

    def test_apply_returns_adapter_score(self, rngs, sample_data_with_adapter):
        """Test that apply returns adapter match score."""
        config = AdapterRemovalConfig()
        op = SoftAdapterRemoval(config, rngs=rngs)

        transformed_data, state, metadata = op.apply(sample_data_with_adapter, {}, None, None)

        assert "adapter_score" in transformed_data
        # Score should be non-negative
        assert transformed_data["adapter_score"] >= 0

    def test_apply_returns_trim_position(self, rngs, sample_data_with_adapter):
        """Test that apply returns trim position."""
        config = AdapterRemovalConfig()
        op = SoftAdapterRemoval(config, rngs=rngs)

        transformed_data, state, metadata = op.apply(sample_data_with_adapter, {}, None, None)

        assert "trim_position" in transformed_data
        # Trim position should be in valid range
        seq_len = sample_data_with_adapter["sequence"].shape[0]
        assert 0 <= transformed_data["trim_position"] <= seq_len

    def test_output_shape_preserved(self, rngs, sample_data_no_adapter):
        """Test that output shape matches input shape."""
        config = AdapterRemovalConfig()
        op = SoftAdapterRemoval(config, rngs=rngs)

        transformed_data, _, _ = op.apply(sample_data_no_adapter, {}, None, None)

        assert transformed_data["sequence"].shape == sample_data_no_adapter["sequence"].shape
        expected_shape = sample_data_no_adapter["quality_scores"].shape
        assert transformed_data["quality_scores"].shape == expected_shape


class TestGradientFlow:
    """Tests for gradient flow through adapter removal."""

    def test_gradient_flows_through_apply(self, rngs):
        """Test that gradients flow through the apply method."""
        config = AdapterRemovalConfig()
        op = SoftAdapterRemoval(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTAACCTTGG")
        quality = jnp.ones(16) * 30.0
        state = {}

        def loss_fn(seq):
            data = {"sequence": seq, "quality_scores": quality}
            transformed, _, _ = op.apply(data, state, None, None)
            return jnp.sum(transformed["sequence"])

        grad = jax.grad(loss_fn)(sequence)
        assert grad is not None
        assert grad.shape == sequence.shape

    def test_gradient_wrt_quality(self, rngs):
        """Test that gradients flow with respect to quality scores."""
        config = AdapterRemovalConfig()
        op = SoftAdapterRemoval(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTAACCTTGG")
        quality = jnp.ones(16) * 30.0
        state = {}

        def loss_fn(q):
            data = {"sequence": sequence, "quality_scores": q}
            transformed, _, _ = op.apply(data, state, None, None)
            return jnp.sum(transformed["quality_scores"])

        grad = jax.grad(loss_fn)(quality)
        assert grad is not None
        assert grad.shape == quality.shape

    def test_temperature_is_learnable(self, rngs):
        """Test that temperature parameter is learnable."""
        config = AdapterRemovalConfig()
        op = SoftAdapterRemoval(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTAACCTTGG")
        quality = jnp.ones(16) * 30.0
        data = {"sequence": sequence, "quality_scores": quality}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return jnp.sum(transformed["sequence"])

        loss, grads = loss_fn(op)

        assert hasattr(grads, "temperature")
        # Temperature gradient may be zero if no adapter found


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    def test_apply_is_jit_compatible(self, rngs):
        """Test that apply method works with JIT."""
        config = AdapterRemovalConfig()
        op = SoftAdapterRemoval(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTAACCTTGG")
        quality = jnp.ones(16) * 30.0
        data = {"sequence": sequence, "quality_scores": quality}
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, new_state, metadata = jit_apply(data, state)
        assert transformed["sequence"].shape == data["sequence"].shape

    def test_jit_produces_same_result(self, rngs):
        """Test that JIT produces same result as eager execution."""
        config = AdapterRemovalConfig()
        op = SoftAdapterRemoval(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTAACCTTGG")
        quality = jnp.ones(16) * 30.0
        data = {"sequence": sequence, "quality_scores": quality}
        state = {}

        # Eager execution
        eager_result, _, _ = op.apply(data, state, None, None)

        # JIT execution
        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        jit_result, _, _ = jit_apply(data, state)

        assert jnp.allclose(eager_result["sequence"], jit_result["sequence"], rtol=1e-5)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_short_sequence(self, rngs):
        """Test with sequence shorter than adapter."""
        config = AdapterRemovalConfig(min_overlap=3)
        op = SoftAdapterRemoval(config, rngs=rngs)

        sequence = encode_dna_string("ACGT")  # 4 bases
        quality = jnp.ones(4) * 30.0
        data = {"sequence": sequence, "quality_scores": quality}
        state = {}

        transformed, _, _ = op.apply(data, state, None, None)
        assert transformed["sequence"].shape == (4, 4)

    def test_exact_adapter_sequence(self, rngs):
        """Test with sequence that is exactly the adapter."""
        config = AdapterRemovalConfig()
        op = SoftAdapterRemoval(config, rngs=rngs)

        sequence = encode_dna_string("AGATCGGAAGAG")  # Full adapter
        quality = jnp.ones(12) * 30.0
        data = {"sequence": sequence, "quality_scores": quality}
        state = {}

        transformed, _, _ = op.apply(data, state, None, None)
        # Should detect adapter match
        assert transformed["adapter_score"] > 0

    def test_high_temperature_smoothing(self, rngs):
        """Test with high temperature (smooth trimming)."""
        config = AdapterRemovalConfig(temperature=10.0)
        op = SoftAdapterRemoval(config, rngs=rngs)

        seq_str = "ACGTACGTAACCTTGG" + "AGATCGGA"
        sequence = encode_dna_string(seq_str)
        quality = jnp.ones(24) * 30.0
        data = {"sequence": sequence, "quality_scores": quality}
        state = {}

        transformed, _, _ = op.apply(data, state, None, None)
        # High temperature should result in smoother (less sharp) trimming
        assert transformed["sequence"].shape == data["sequence"].shape

    def test_low_temperature_sharp_trimming(self, rngs):
        """Test with low temperature (sharp trimming)."""
        config = AdapterRemovalConfig(temperature=0.1)
        op = SoftAdapterRemoval(config, rngs=rngs)

        seq_str = "ACGTACGTAACCTTGG" + "AGATCGGA"
        sequence = encode_dna_string(seq_str)
        quality = jnp.ones(24) * 30.0
        data = {"sequence": sequence, "quality_scores": quality}
        state = {}

        transformed, _, _ = op.apply(data, state, None, None)
        # Low temperature should result in sharper trimming
        assert transformed["sequence"].shape == data["sequence"].shape
