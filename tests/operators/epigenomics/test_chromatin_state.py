"""Tests for differentiable chromatin state annotation operator.

Following TDD principles, these tests define the expected behavior
of the ChromatinStateAnnotator operator (ChromHMM-style).
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest


class TestChromatinStateConfig:
    """Tests for ChromatinStateConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from diffbio.operators.epigenomics.chromatin_state import ChromatinStateConfig

        config = ChromatinStateConfig(stream_name=None)

        assert config.num_states == 15
        assert config.num_marks == 6
        assert config.temperature == 1.0
        assert config.stochastic is False

    def test_custom_config(self):
        """Test custom configuration values."""
        from diffbio.operators.epigenomics.chromatin_state import ChromatinStateConfig

        config = ChromatinStateConfig(
            num_states=10,
            num_marks=4,
            temperature=0.5,
            stream_name=None,
        )

        assert config.num_states == 10
        assert config.num_marks == 4
        assert config.temperature == 0.5


class TestChromatinStateAnnotator:
    """Tests for ChromatinStateAnnotator operator."""

    @pytest.fixture
    def config(self):
        """Provide default config."""
        from diffbio.operators.epigenomics.chromatin_state import ChromatinStateConfig

        return ChromatinStateConfig(
            num_states=8,
            num_marks=4,
            temperature=1.0,
            stream_name=None,
        )

    @pytest.fixture
    def annotator(self, config, rngs):
        """Create annotator instance."""
        from diffbio.operators.epigenomics.chromatin_state import ChromatinStateAnnotator

        return ChromatinStateAnnotator(config, rngs=rngs)

    def test_initialization(self, config, rngs):
        """Test operator initialization."""
        from diffbio.operators.epigenomics.chromatin_state import ChromatinStateAnnotator

        annotator = ChromatinStateAnnotator(config, rngs=rngs)

        assert annotator.config == config
        assert hasattr(annotator, "transition_logits")
        assert hasattr(annotator, "emission_logits")
        assert hasattr(annotator, "initial_logits")

    def test_initialization_without_rngs(self, config):
        """Test initialization without providing RNGs."""
        from diffbio.operators.epigenomics.chromatin_state import ChromatinStateAnnotator

        annotator = ChromatinStateAnnotator(config, rngs=None)
        assert annotator is not None

    def test_apply_single_sequence(self, annotator, config):
        """Test apply with single sequence input."""
        length = 100
        num_marks = config.num_marks

        # Create synthetic histone mark signals
        marks = jax.random.normal(jax.random.key(0), (length, num_marks))

        data = {"histone_marks": marks}
        result, state, metadata = annotator.apply(data, {}, None)

        # Check output keys
        assert "state_probabilities" in result
        assert "viterbi_path" in result
        assert "state_posteriors" in result
        assert "histone_marks" in result

        # Check shapes
        assert result["state_probabilities"].shape == (length, config.num_states)
        assert result["viterbi_path"].shape == (length,)

    def test_apply_batch_input(self, annotator, config):
        """Test apply with batched input."""
        batch_size = 4
        length = 100
        num_marks = config.num_marks

        marks = jax.random.normal(jax.random.key(0), (batch_size, length, num_marks))

        data = {"histone_marks": marks}
        result, state, metadata = annotator.apply(data, {}, None)

        # Check shapes (batched)
        assert result["state_probabilities"].shape == (
            batch_size,
            length,
            config.num_states,
        )
        assert result["viterbi_path"].shape == (batch_size, length)

    def test_state_probabilities_sum_to_one(self, annotator, config):
        """Test that state probabilities sum to 1 at each position."""
        length = 100
        marks = jax.random.normal(jax.random.key(0), (length, config.num_marks))

        data = {"histone_marks": marks}
        result, _, _ = annotator.apply(data, {}, None)

        state_probs = result["state_probabilities"]
        sums = state_probs.sum(axis=-1)

        assert jnp.allclose(sums, 1.0, atol=1e-5)

    def test_state_probabilities_range(self, annotator, config):
        """Test that state probabilities are in [0, 1]."""
        marks = jax.random.normal(jax.random.key(0), (100, config.num_marks))

        data = {"histone_marks": marks}
        result, _, _ = annotator.apply(data, {}, None)

        state_probs = result["state_probabilities"]
        assert jnp.all(state_probs >= 0.0)
        assert jnp.all(state_probs <= 1.0)

    def test_output_finite(self, annotator, config):
        """Test that all outputs are finite."""
        marks = jax.random.normal(jax.random.key(0), (100, config.num_marks))

        data = {"histone_marks": marks}
        result, _, _ = annotator.apply(data, {}, None)

        for key in ["state_probabilities", "state_posteriors"]:
            assert jnp.all(jnp.isfinite(result[key])), f"{key} contains non-finite values"

    def test_preserves_original_data(self, annotator, config):
        """Test that original data is preserved in output."""
        marks = jax.random.normal(jax.random.key(0), (100, config.num_marks))
        extra_data = jnp.array([1.0, 2.0, 3.0])

        data = {"histone_marks": marks, "extra": extra_data}
        result, _, _ = annotator.apply(data, {}, None)

        assert "extra" in result
        assert jnp.allclose(result["extra"], extra_data)


class TestChromatinStateDifferentiability:
    """Tests for gradient flow through the chromatin state annotator."""

    @pytest.fixture
    def config(self):
        """Provide default config."""
        from diffbio.operators.epigenomics.chromatin_state import ChromatinStateConfig

        return ChromatinStateConfig(
            num_states=8,
            num_marks=4,
            temperature=1.0,
            stream_name=None,
        )

    def test_gradient_flow_through_operator(self, config, rngs):
        """Test that gradients flow through the operator."""
        from diffbio.operators.epigenomics.chromatin_state import ChromatinStateAnnotator

        annotator = ChromatinStateAnnotator(config, rngs=rngs)

        def loss_fn(op, marks):
            data = {"histone_marks": marks}
            result, _, _ = op.apply(data, {}, None)
            return result["state_probabilities"].sum()

        marks = jax.random.normal(jax.random.key(0), (50, config.num_marks))
        grads = nnx.grad(loss_fn)(annotator, marks)

        assert grads is not None

    def test_gradient_wrt_transitions(self, config, rngs):
        """Test gradient with respect to transition parameters."""
        from diffbio.operators.epigenomics.chromatin_state import ChromatinStateAnnotator

        annotator = ChromatinStateAnnotator(config, rngs=rngs)

        def loss_fn(op, marks):
            data = {"histone_marks": marks}
            result, _, _ = op.apply(data, {}, None)
            return result["state_probabilities"].mean()

        marks = jax.random.normal(jax.random.key(0), (50, config.num_marks))
        grads = nnx.grad(loss_fn)(annotator, marks)

        assert hasattr(grads, "transition_logits")
        assert grads.transition_logits[...] is not None

    def test_gradient_wrt_emissions(self, config, rngs):
        """Test gradient with respect to emission parameters."""
        from diffbio.operators.epigenomics.chromatin_state import ChromatinStateAnnotator

        annotator = ChromatinStateAnnotator(config, rngs=rngs)

        def loss_fn(op, marks):
            data = {"histone_marks": marks}
            result, _, _ = op.apply(data, {}, None)
            return result["state_probabilities"].mean()

        marks = jax.random.normal(jax.random.key(0), (50, config.num_marks))
        grads = nnx.grad(loss_fn)(annotator, marks)

        assert hasattr(grads, "emission_logits")
        assert grads.emission_logits[...] is not None

    def test_gradient_wrt_input(self, config, rngs):
        """Test gradient with respect to input histone marks."""
        from diffbio.operators.epigenomics.chromatin_state import ChromatinStateAnnotator

        annotator = ChromatinStateAnnotator(config, rngs=rngs)

        def loss_fn(marks):
            data = {"histone_marks": marks}
            result, _, _ = annotator.apply(data, {}, None)
            return result["state_probabilities"].sum()

        marks = jax.random.normal(jax.random.key(0), (50, config.num_marks))
        grad = jax.grad(loss_fn)(marks)

        assert grad.shape == marks.shape
        assert jnp.all(jnp.isfinite(grad))


class TestChromatinStateJITCompatibility:
    """Tests for JIT compilation compatibility."""

    @pytest.fixture
    def config(self):
        """Provide default config."""
        from diffbio.operators.epigenomics.chromatin_state import ChromatinStateConfig

        return ChromatinStateConfig(
            num_states=8,
            num_marks=4,
            stream_name=None,
        )

    def test_jit_apply(self, config, rngs):
        """Test JIT compilation of apply method."""
        from diffbio.operators.epigenomics.chromatin_state import ChromatinStateAnnotator

        annotator = ChromatinStateAnnotator(config, rngs=rngs)

        @jax.jit
        def jit_apply(marks):
            data = {"histone_marks": marks}
            result, _, _ = annotator.apply(data, {}, None)
            return result["state_probabilities"]

        marks = jax.random.normal(jax.random.key(0), (50, config.num_marks))

        # Should compile and run without error
        result = jit_apply(marks)
        assert result.shape == (50, config.num_states)

    def test_jit_gradient(self, config, rngs):
        """Test JIT compilation of gradient computation."""
        from diffbio.operators.epigenomics.chromatin_state import ChromatinStateAnnotator

        annotator = ChromatinStateAnnotator(config, rngs=rngs)

        @jax.jit
        def loss_and_grad(marks):
            def loss_fn(m):
                data = {"histone_marks": m}
                result, _, _ = annotator.apply(data, {}, None)
                return result["state_probabilities"].sum()

            return jax.value_and_grad(loss_fn)(marks)

        marks = jax.random.normal(jax.random.key(0), (50, config.num_marks))

        # Should compile and run without error
        loss, grad = loss_and_grad(marks)
        assert jnp.isfinite(loss)
        assert jnp.all(jnp.isfinite(grad))


class TestChromatinStateHMMBehavior:
    """Tests for HMM-specific behavior."""

    @pytest.fixture
    def config(self):
        """Provide config."""
        from diffbio.operators.epigenomics.chromatin_state import ChromatinStateConfig

        return ChromatinStateConfig(
            num_states=4,
            num_marks=3,
            temperature=1.0,
            stream_name=None,
        )

    def test_forward_algorithm_numeric_stability(self, config, rngs):
        """Test numeric stability of forward algorithm with long sequences."""
        from diffbio.operators.epigenomics.chromatin_state import ChromatinStateAnnotator

        annotator = ChromatinStateAnnotator(config, rngs=rngs)

        # Long sequence that could cause underflow without log-space computation
        length = 1000
        marks = jax.random.normal(jax.random.key(0), (length, config.num_marks))

        data = {"histone_marks": marks}
        result, _, _ = annotator.apply(data, {}, None)

        # All outputs should be finite
        assert jnp.all(jnp.isfinite(result["state_probabilities"]))
        assert jnp.all(jnp.isfinite(result["state_posteriors"]))

    def test_viterbi_path_valid_states(self, config, rngs):
        """Test that Viterbi path contains valid state indices."""
        from diffbio.operators.epigenomics.chromatin_state import ChromatinStateAnnotator

        annotator = ChromatinStateAnnotator(config, rngs=rngs)

        marks = jax.random.normal(jax.random.key(0), (100, config.num_marks))

        data = {"histone_marks": marks}
        result, _, _ = annotator.apply(data, {}, None)

        # Soft Viterbi produces probabilities, check they're valid
        viterbi = result["viterbi_path"]
        assert viterbi.shape == (100,)
        # For soft Viterbi, values represent most likely state (as float)
        # They should be in valid range
        assert jnp.all(viterbi >= 0)
        assert jnp.all(viterbi < config.num_states)
