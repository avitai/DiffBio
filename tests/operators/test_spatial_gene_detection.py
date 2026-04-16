"""Tests for spatial gene detection operators."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.multiomics import (
    DifferentiableSpatialGeneDetector,
    SpatialGeneDetectorConfig,
    create_spatial_gene_detector,
)


class TestSpatialGeneDetectorConfig:
    """Test configuration validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SpatialGeneDetectorConfig()
        assert config.n_genes == 2000
        assert config.lengthscale == 1.0
        assert config.variance == 1.0
        assert config.noise_variance == 0.1
        assert config.n_inducing_points == 100
        assert config.stochastic is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = SpatialGeneDetectorConfig(
            n_genes=500,
            lengthscale=2.0,
            n_inducing_points=50,
        )
        assert config.n_genes == 500
        assert config.lengthscale == 2.0
        assert config.n_inducing_points == 50

    def test_invalid_gene_count_rejected(self):
        """Gene count must be positive."""
        with pytest.raises(ValueError, match="n_genes"):
            SpatialGeneDetectorConfig(n_genes=0)

    def test_invalid_hidden_dims_rejected(self):
        """Hidden dimensions must all be positive."""
        with pytest.raises(ValueError, match="hidden_dims"):
            SpatialGeneDetectorConfig(hidden_dims=[32, 0])

    def test_invalid_pvalue_threshold_rejected(self):
        """P-value threshold must lie in [0, 1]."""
        with pytest.raises(ValueError, match="pvalue_threshold"):
            SpatialGeneDetectorConfig(pvalue_threshold=1.5)


class TestSpatialGeneDetectorBasic:
    """Test basic functionality."""

    @pytest.fixture
    def detector(self):
        """Create a test detector."""
        config = SpatialGeneDetectorConfig(
            n_genes=100,
            n_inducing_points=20,
            hidden_dims=[32],
        )
        return DifferentiableSpatialGeneDetector(config, rngs=nnx.Rngs(42))

    @pytest.fixture
    def sample_data(self):
        """Create sample spatial transcriptomics data."""
        n_spots = 50
        n_genes = 100

        key = jax.random.key(0)
        k1, k2, k3 = jax.random.split(key, 3)

        # Spatial coordinates (2D)
        coords = jax.random.uniform(k1, (n_spots, 2), minval=0, maxval=10)
        # Gene expression counts
        expression = jax.random.poisson(k2, lam=10.0, shape=(n_spots, n_genes)).astype(jnp.float32)
        # Total counts per spot (for normalization)
        total_counts = jnp.sum(expression, axis=1)

        return {
            "spatial_coords": coords,
            "expression": expression,
            "total_counts": total_counts,
        }

    def test_forward_pass(self, detector, sample_data):
        """Test forward pass produces expected outputs."""
        result, state, metadata = detector.apply(sample_data, {}, None)

        n_spots = sample_data["spatial_coords"].shape[0]
        n_genes = detector.config.n_genes

        # Check output keys
        assert "spatial_variance" in result
        assert "spatial_pvalues" in result
        assert "is_spatial" in result
        assert "smoothed_expression" in result

        # Check shapes
        assert result["spatial_variance"].shape == (n_genes,)
        assert result["spatial_pvalues"].shape == (n_genes,)
        assert result["is_spatial"].shape == (n_genes,)
        assert result["smoothed_expression"].shape == (n_spots, n_genes)

    def test_spatial_variance_positive(self, detector, sample_data):
        """Test that spatial variance is positive."""
        result, _, _ = detector.apply(sample_data, {}, None)
        assert jnp.all(result["spatial_variance"] >= 0)

    def test_pvalues_in_range(self, detector, sample_data):
        """Test that p-values are in [0, 1]."""
        result, _, _ = detector.apply(sample_data, {}, None)
        pvalues = result["spatial_pvalues"]
        assert jnp.all(pvalues >= 0)
        assert jnp.all(pvalues <= 1)

    def test_is_spatial_soft(self, detector, sample_data):
        """Test that is_spatial is soft (in [0, 1])."""
        result, _, _ = detector.apply(sample_data, {}, None)
        is_spatial = result["is_spatial"]
        assert jnp.all(is_spatial >= 0)
        assert jnp.all(is_spatial <= 1)


class TestSpatialGeneDetectorDifferentiability:
    """Test gradient computation."""

    @pytest.fixture
    def detector(self):
        """Create a test detector."""
        config = SpatialGeneDetectorConfig(
            n_genes=50,
            n_inducing_points=10,
            hidden_dims=[16],
        )
        return DifferentiableSpatialGeneDetector(config, rngs=nnx.Rngs(42))

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        n_spots = 30
        n_genes = 50
        key = jax.random.key(0)
        k1, k2 = jax.random.split(key)

        return {
            "spatial_coords": jax.random.uniform(k1, (n_spots, 2), minval=0, maxval=10),
            "expression": jax.random.poisson(k2, lam=10.0, shape=(n_spots, n_genes)).astype(
                jnp.float32
            ),
            "total_counts": jnp.ones(n_spots) * n_genes * 10,
        }

    def test_gradient_flow(self, detector, sample_data):
        """Test that gradients flow through the operator."""

        @nnx.value_and_grad
        def loss_fn(model):
            result, _, _ = model.apply(sample_data, {}, None)
            # Maximize spatial variance detection
            return -result["spatial_variance"].mean()

        _, grads = loss_fn(detector)
        assert grads is not None

        # Check gradients are not NaN
        grad_leaves = jax.tree.leaves(grads)
        for leaf in grad_leaves:
            if hasattr(leaf, "shape"):
                assert not jnp.any(jnp.isnan(leaf)), "Gradient contains NaN"

    def test_kernel_parameter_gradients(self, detector, sample_data):
        """Test gradients flow to kernel parameters."""

        @nnx.value_and_grad
        def loss_fn(model):
            result, _, _ = model.apply(sample_data, {}, None)
            return result["smoothed_expression"].mean()

        _, grads = loss_fn(detector)

        assert hasattr(grads, "kernel_state")
        assert grads.kernel_state.log_lengthscale is not None


class TestSpatialGeneDetectorJIT:
    """Test JIT compilation."""

    def test_jit_compilation(self):
        """Test that forward pass can be JIT compiled."""
        config = SpatialGeneDetectorConfig(
            n_genes=50,
            n_inducing_points=10,
            hidden_dims=[16],
        )
        detector = DifferentiableSpatialGeneDetector(config, rngs=nnx.Rngs(42))

        n_spots = 30
        n_genes = 50
        key = jax.random.key(0)
        k1, k2 = jax.random.split(key)
        data = {
            "spatial_coords": jax.random.uniform(k1, (n_spots, 2), minval=0, maxval=10),
            "expression": jax.random.poisson(k2, lam=10.0, shape=(n_spots, n_genes)).astype(
                jnp.float32
            ),
            "total_counts": jnp.ones(n_spots) * n_genes * 10,
        }

        @jax.jit
        def forward(model, data):
            result, _, _ = model.apply(data, {}, None)
            return result["spatial_variance"]

        # Should not raise
        result = forward(detector, data)
        assert result.shape == (n_genes,)


class TestSpatialGeneDetectorKernel:
    """Test kernel computations."""

    @pytest.fixture
    def detector(self):
        """Create a test detector."""
        config = SpatialGeneDetectorConfig(
            n_genes=10,
            n_inducing_points=5,
            lengthscale=1.0,
            variance=1.0,
        )
        return DifferentiableSpatialGeneDetector(config, rngs=nnx.Rngs(42))

    def test_kernel_positive_definite(self, detector):
        """Test that kernel matrix is positive semi-definite."""
        key = jax.random.key(0)
        coords = jax.random.uniform(key, (10, 2))

        # Compute kernel matrix
        K = detector.compute_kernel(coords, coords)

        # Check symmetry
        assert jnp.allclose(K, K.T, atol=1e-5)

        # Check positive semi-definite (all eigenvalues >= 0)
        eigenvalues = jnp.linalg.eigvalsh(K)
        assert jnp.all(eigenvalues >= -1e-5), "Kernel should be positive semi-definite"

    def test_kernel_diagonal_equals_variance(self, detector):
        """Test that kernel diagonal equals variance parameter."""
        key = jax.random.key(0)
        coords = jax.random.uniform(key, (10, 2))

        K = detector.compute_kernel(coords, coords)
        diagonal = jnp.diag(K)

        expected_variance = detector.config.variance
        assert jnp.allclose(diagonal, expected_variance, atol=1e-5)

    def test_lengthscale_effect(self):
        """Test that lengthscale affects kernel decay."""
        key = jax.random.key(0)
        coords = jax.random.uniform(key, (10, 2))

        # Short lengthscale
        config_short = SpatialGeneDetectorConfig(n_genes=10, lengthscale=0.1)
        detector_short = DifferentiableSpatialGeneDetector(config_short, rngs=nnx.Rngs(42))
        K_short = detector_short.compute_kernel(coords, coords)

        # Long lengthscale
        config_long = SpatialGeneDetectorConfig(n_genes=10, lengthscale=10.0)
        detector_long = DifferentiableSpatialGeneDetector(config_long, rngs=nnx.Rngs(42))
        K_long = detector_long.compute_kernel(coords, coords)

        # Longer lengthscale should have higher off-diagonal values
        # (slower decay with distance)
        off_diag_short = jnp.mean(K_short - jnp.diag(jnp.diag(K_short)))
        off_diag_long = jnp.mean(K_long - jnp.diag(jnp.diag(K_long)))
        assert off_diag_long > off_diag_short


class TestSpatialGeneDetectorFactory:
    """Test factory function."""

    def test_create_spatial_gene_detector(self):
        """Test factory function creates valid detector."""
        detector = create_spatial_gene_detector(
            n_genes=200,
            n_inducing_points=30,
            lengthscale=2.0,
            seed=123,
        )

        assert isinstance(detector, DifferentiableSpatialGeneDetector)
        assert detector.config.n_genes == 200
        assert detector.config.n_inducing_points == 30
        assert detector.config.lengthscale == 2.0

    def test_create_spatial_gene_detector_defaults(self):
        """Test factory with default values."""
        detector = create_spatial_gene_detector()

        assert detector.config.n_genes == 2000
        assert detector.config.n_inducing_points == 100
        assert detector.config.lengthscale == 1.0
