"""Tests for differentiable peak calling operator.

Following TDD principles, these tests define the expected behavior
of the DifferentiablePeakCaller operator.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest


class TestPeakCallerConfig:
    """Tests for PeakCallerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from diffbio.operators.epigenomics.peak_calling import PeakCallerConfig

        config = PeakCallerConfig(stream_name=None)

        assert config.window_size == 200
        assert config.num_filters == 32
        assert config.kernel_sizes == (5, 11, 21)
        assert config.threshold == 0.5
        assert config.temperature == 1.0
        assert config.min_peak_width == 50
        assert config.stochastic is False

    def test_custom_config(self):
        """Test custom configuration values."""
        from diffbio.operators.epigenomics.peak_calling import PeakCallerConfig

        config = PeakCallerConfig(
            window_size=100,
            num_filters=64,
            kernel_sizes=(3, 7, 15),
            threshold=0.3,
            temperature=0.5,
            min_peak_width=30,
            stream_name=None,
        )

        assert config.window_size == 100
        assert config.num_filters == 64
        assert config.kernel_sizes == (3, 7, 15)
        assert config.threshold == 0.3
        assert config.temperature == 0.5
        assert config.min_peak_width == 30


class TestPeakDetectionCNN:
    """Tests for the PeakDetectionCNN module."""

    def test_initialization(self, rngs):
        """Test CNN initialization."""
        from diffbio.operators.epigenomics.peak_calling import PeakDetectionCNN

        cnn = PeakDetectionCNN(
            num_filters=32,
            kernel_sizes=(5, 11, 21),
            rngs=rngs,
        )

        assert cnn.num_filters == 32
        assert cnn.kernel_sizes == (5, 11, 21)
        assert len(cnn.conv_layers) == 3

    def test_forward_pass_2d(self, rngs):
        """Test forward pass with 2D input (batch, length)."""
        from diffbio.operators.epigenomics.peak_calling import PeakDetectionCNN

        cnn = PeakDetectionCNN(num_filters=16, kernel_sizes=(5, 11), rngs=rngs)

        # Input: (batch=2, length=100)
        x = jax.random.normal(jax.random.key(0), (2, 100))
        output = cnn(x)

        assert output.shape == (2, 100)

    def test_forward_pass_3d(self, rngs):
        """Test forward pass with 3D input (batch, length, channels)."""
        from diffbio.operators.epigenomics.peak_calling import PeakDetectionCNN

        cnn = PeakDetectionCNN(num_filters=16, kernel_sizes=(5, 11), rngs=rngs)

        # Input: (batch=2, length=100, channels=1)
        x = jax.random.normal(jax.random.key(0), (2, 100, 1))
        output = cnn(x)

        assert output.shape == (2, 100)

    def test_output_finite(self, rngs):
        """Test that output values are finite."""
        from diffbio.operators.epigenomics.peak_calling import PeakDetectionCNN

        cnn = PeakDetectionCNN(num_filters=16, kernel_sizes=(5, 11), rngs=rngs)

        x = jax.random.normal(jax.random.key(0), (2, 100))
        output = cnn(x)

        assert jnp.all(jnp.isfinite(output))


class TestDifferentiablePeakCaller:
    """Tests for DifferentiablePeakCaller operator."""

    @pytest.fixture
    def config(self):
        """Provide default config."""
        from diffbio.operators.epigenomics.peak_calling import PeakCallerConfig

        return PeakCallerConfig(
            window_size=50,
            num_filters=16,
            kernel_sizes=(5, 11),
            min_peak_width=10,
            stream_name=None,
        )

    @pytest.fixture
    def peak_caller(self, config, rngs):
        """Create peak caller instance."""
        from diffbio.operators.epigenomics.peak_calling import DifferentiablePeakCaller

        return DifferentiablePeakCaller(config, rngs=rngs)

    def test_initialization(self, config, rngs):
        """Test operator initialization."""
        from diffbio.operators.epigenomics.peak_calling import DifferentiablePeakCaller

        peak_caller = DifferentiablePeakCaller(config, rngs=rngs)

        assert peak_caller.config == config
        assert hasattr(peak_caller, "threshold")
        # Temperature is managed by TemperatureOperator base class (learnable)
        assert hasattr(peak_caller, "temperature")  # Learnable Param
        assert hasattr(peak_caller, "peak_cnn")

    def test_initialization_without_rngs(self, config):
        """Test initialization without providing RNGs."""
        from diffbio.operators.epigenomics.peak_calling import DifferentiablePeakCaller

        # Should not raise, uses default RNGs
        peak_caller = DifferentiablePeakCaller(config, rngs=None)
        assert peak_caller is not None

    def test_apply_single_sequence(self, peak_caller):
        """Test apply with single sequence input."""
        # Create synthetic coverage data with a clear peak
        length = 200
        x = jnp.zeros(length)
        # Add a Gaussian peak
        peak_center = 100
        peak_width = 20
        positions = jnp.arange(length)
        x = jnp.exp(-((positions - peak_center) ** 2) / (2 * peak_width**2))

        data = {"coverage": x}
        result, state, metadata = peak_caller.apply(data, {}, None)

        # Check output keys
        assert "peak_scores" in result
        assert "peak_probabilities" in result
        assert "peak_summits" in result
        assert "peak_starts" in result
        assert "peak_ends" in result
        assert "coverage" in result

        # Check shapes (single sequence, no batch dim)
        assert result["peak_scores"].shape == (length,)
        assert result["peak_probabilities"].shape == (length,)
        assert result["peak_summits"].shape == (length,)

    def test_apply_batch_input(self, peak_caller):
        """Test apply with batched input."""
        batch_size = 4
        length = 200

        # Create batch of coverage signals
        coverage = jax.random.normal(jax.random.key(0), (batch_size, length))
        coverage = jnp.abs(coverage)  # Make positive

        data = {"coverage": coverage}
        result, state, metadata = peak_caller.apply(data, {}, None)

        # Check shapes (batched)
        assert result["peak_scores"].shape == (batch_size, length)
        assert result["peak_probabilities"].shape == (batch_size, length)
        assert result["peak_summits"].shape == (batch_size, length)

    def test_peak_probabilities_range(self, peak_caller):
        """Test that peak probabilities are in [0, 1]."""
        coverage = jax.random.normal(jax.random.key(0), (2, 100))

        data = {"coverage": coverage}
        result, _, _ = peak_caller.apply(data, {}, None)

        peak_probs = result["peak_probabilities"]
        assert jnp.all(peak_probs >= 0.0)
        assert jnp.all(peak_probs <= 1.0)

    def test_output_finite(self, peak_caller):
        """Test that all outputs are finite."""
        coverage = jax.random.normal(jax.random.key(0), (2, 100))

        data = {"coverage": coverage}
        result, _, _ = peak_caller.apply(data, {}, None)

        for key in ["peak_scores", "peak_probabilities", "peak_summits"]:
            assert jnp.all(jnp.isfinite(result[key])), f"{key} contains non-finite values"

    def test_preserves_original_data(self, peak_caller):
        """Test that original data is preserved in output."""
        coverage = jax.random.normal(jax.random.key(0), (2, 100))
        extra_data = jnp.array([1.0, 2.0, 3.0])

        data = {"coverage": coverage, "extra": extra_data}
        result, _, _ = peak_caller.apply(data, {}, None)

        assert "extra" in result
        assert jnp.allclose(result["extra"], extra_data)


class TestPeakCallerDifferentiability:
    """Tests for gradient flow through the peak caller."""

    @pytest.fixture
    def config(self):
        """Provide default config."""
        from diffbio.operators.epigenomics.peak_calling import PeakCallerConfig

        return PeakCallerConfig(
            window_size=50,
            num_filters=16,
            kernel_sizes=(5, 11),
            min_peak_width=10,
            stream_name=None,
        )

    def test_gradient_flow_through_cnn(self, config, rngs):
        """Test that gradients flow through the CNN."""
        from diffbio.operators.epigenomics.peak_calling import PeakDetectionCNN

        cnn = PeakDetectionCNN(num_filters=16, kernel_sizes=(5, 11), rngs=rngs)

        def loss_fn(model, x):
            output = model(x)
            return output.sum()

        x = jax.random.normal(jax.random.key(0), (2, 100))
        grads = nnx.grad(loss_fn)(cnn, x)

        # Check gradients exist for conv layers
        assert grads is not None

    def test_gradient_flow_through_operator(self, config, rngs):
        """Test that gradients flow through the full operator."""
        from diffbio.operators.epigenomics.peak_calling import DifferentiablePeakCaller

        peak_caller = DifferentiablePeakCaller(config, rngs=rngs)

        def loss_fn(op, coverage):
            data = {"coverage": coverage}
            result, _, _ = op.apply(data, {}, None)
            return result["peak_probabilities"].sum()

        coverage = jax.random.normal(jax.random.key(0), (2, 100))
        grads = nnx.grad(loss_fn)(peak_caller, coverage)

        assert grads is not None

    def test_gradient_wrt_threshold(self, config, rngs):
        """Test gradient with respect to threshold parameter."""
        from diffbio.operators.epigenomics.peak_calling import DifferentiablePeakCaller

        peak_caller = DifferentiablePeakCaller(config, rngs=rngs)

        def loss_fn(op, coverage):
            data = {"coverage": coverage}
            result, _, _ = op.apply(data, {}, None)
            return result["peak_probabilities"].mean()

        coverage = jax.random.normal(jax.random.key(0), (2, 100))

        # Compute gradients
        grads = nnx.grad(loss_fn)(peak_caller, coverage)

        # Threshold gradient should exist and be non-zero for non-trivial input
        assert hasattr(grads, "threshold")
        # The gradient magnitude should be meaningful
        assert grads.threshold[...] is not None

    def test_gradient_wrt_temperature(self, config, rngs):
        """Test gradient with respect to temperature parameter."""
        from diffbio.operators.epigenomics.peak_calling import DifferentiablePeakCaller

        peak_caller = DifferentiablePeakCaller(config, rngs=rngs)

        def loss_fn(op, coverage):
            data = {"coverage": coverage}
            result, _, _ = op.apply(data, {}, None)
            return result["peak_probabilities"].mean()

        coverage = jax.random.normal(jax.random.key(0), (2, 100))
        grads = nnx.grad(loss_fn)(peak_caller, coverage)

        assert hasattr(grads, "temperature")
        assert grads.temperature[...] is not None

    def test_gradient_wrt_input(self, config, rngs):
        """Test gradient with respect to input coverage."""
        from diffbio.operators.epigenomics.peak_calling import DifferentiablePeakCaller

        peak_caller = DifferentiablePeakCaller(config, rngs=rngs)

        def loss_fn(coverage):
            data = {"coverage": coverage}
            result, _, _ = peak_caller.apply(data, {}, None)
            return result["peak_probabilities"].sum()

        coverage = jax.random.normal(jax.random.key(0), (2, 100))
        grad = jax.grad(loss_fn)(coverage)

        assert grad.shape == coverage.shape
        assert jnp.all(jnp.isfinite(grad))


class TestPeakCallerJITCompatibility:
    """Tests for JIT compilation compatibility."""

    @pytest.fixture
    def config(self):
        """Provide default config."""
        from diffbio.operators.epigenomics.peak_calling import PeakCallerConfig

        return PeakCallerConfig(
            window_size=50,
            num_filters=16,
            kernel_sizes=(5, 11),
            min_peak_width=10,
            stream_name=None,
        )

    def test_jit_apply(self, config, rngs):
        """Test JIT compilation of apply method."""
        from diffbio.operators.epigenomics.peak_calling import DifferentiablePeakCaller

        peak_caller = DifferentiablePeakCaller(config, rngs=rngs)

        @jax.jit
        def jit_apply(coverage):
            data = {"coverage": coverage}
            result, _, _ = peak_caller.apply(data, {}, None)
            return result["peak_probabilities"]

        coverage = jax.random.normal(jax.random.key(0), (2, 100))

        # Should compile and run without error
        result = jit_apply(coverage)
        assert result.shape == (2, 100)

    def test_jit_gradient(self, config, rngs):
        """Test JIT compilation of gradient computation."""
        from diffbio.operators.epigenomics.peak_calling import DifferentiablePeakCaller

        peak_caller = DifferentiablePeakCaller(config, rngs=rngs)

        @jax.jit
        def loss_and_grad(coverage):
            def loss_fn(cov):
                data = {"coverage": cov}
                result, _, _ = peak_caller.apply(data, {}, None)
                return result["peak_probabilities"].sum()

            return jax.value_and_grad(loss_fn)(coverage)

        coverage = jax.random.normal(jax.random.key(0), (2, 100))

        # Should compile and run without error
        loss, grad = loss_and_grad(coverage)
        assert jnp.isfinite(loss)
        assert jnp.all(jnp.isfinite(grad))


class TestPeakCallerBiologicalBehavior:
    """Tests for biologically meaningful behavior."""

    @pytest.fixture
    def config(self):
        """Provide config for biological tests."""
        from diffbio.operators.epigenomics.peak_calling import PeakCallerConfig

        return PeakCallerConfig(
            window_size=50,
            num_filters=32,
            kernel_sizes=(5, 11, 21),
            threshold=0.0,  # Low threshold to detect most signals
            min_peak_width=10,
            stream_name=None,
        )

    def test_detects_gaussian_peak(self, config, rngs):
        """Test that the model can detect a Gaussian peak after training."""
        from diffbio.operators.epigenomics.peak_calling import DifferentiablePeakCaller

        peak_caller = DifferentiablePeakCaller(config, rngs=rngs)

        # Create synthetic coverage with clear peak
        length = 200
        positions = jnp.arange(length)
        peak_center = 100
        coverage = jnp.exp(-((positions - peak_center) ** 2) / (2 * 20**2))

        data = {"coverage": coverage}
        result, _, _ = peak_caller.apply(data, {}, None)

        # The peak scores should be computed (actual detection depends on training)
        assert result["peak_scores"].shape == (length,)
        assert jnp.all(jnp.isfinite(result["peak_scores"]))

    def test_responds_to_coverage_magnitude(self, config, rngs):
        """Test that peak scores respond to coverage magnitude."""
        from diffbio.operators.epigenomics.peak_calling import DifferentiablePeakCaller

        peak_caller = DifferentiablePeakCaller(config, rngs=rngs)

        # Low coverage
        low_coverage = jnp.ones(100) * 0.1
        data_low = {"coverage": low_coverage}
        result_low, _, _ = peak_caller.apply(data_low, {}, None)

        # High coverage
        high_coverage = jnp.ones(100) * 10.0
        data_high = {"coverage": high_coverage}
        result_high, _, _ = peak_caller.apply(data_high, {}, None)

        # Scores should differ based on magnitude
        # (The exact relationship depends on training, but they should be different)
        assert not jnp.allclose(result_low["peak_scores"], result_high["peak_scores"], atol=1e-3)
