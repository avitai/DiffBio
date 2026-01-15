"""Tests for diffbio.operators.statistical.nb_glm module.

These tests define the expected behavior of the DifferentiableNBGLM
operator for differential expression analysis. Implementation should
be written to pass these tests.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.statistical.nb_glm import (
    DifferentiableNBGLM,
    NBGLMConfig,
)


class TestNBGLMConfig:
    """Tests for NBGLMConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NBGLMConfig()
        assert config.n_features == 2000
        assert config.n_covariates == 2
        assert config.estimate_dispersion is True
        assert config.stochastic is False

    def test_custom_features(self):
        """Test custom number of features."""
        config = NBGLMConfig(n_features=5000)
        assert config.n_features == 5000

    def test_custom_covariates(self):
        """Test custom number of covariates."""
        config = NBGLMConfig(n_covariates=5)
        assert config.n_covariates == 5


class TestDifferentiableNBGLM:
    """Tests for DifferentiableNBGLM operator."""

    @pytest.fixture
    def rngs(self):
        """Provide RNGs for operator initialization."""
        return nnx.Rngs(42)

    @pytest.fixture
    def sample_data(self):
        """Provide sample count and design matrix data."""
        # Counts for single sample, 100 genes
        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        # Design matrix row (e.g., intercept + treatment indicator)
        design = jnp.array([1.0, 0.0])  # Control sample
        size_factor = jnp.array(1.0)
        return {"counts": counts, "design": design, "size_factor": size_factor}

    @pytest.fixture
    def batch_data(self):
        """Provide batch of count data with design matrix."""
        n_samples = 8
        n_genes = 100
        n_covariates = 2

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(n_samples, n_genes)).astype(
            jnp.float32
        )
        # Design matrix: intercept + treatment (half control, half treatment)
        design = jnp.zeros((n_samples, n_covariates))
        design = design.at[:, 0].set(1.0)  # Intercept
        design = design.at[4:, 1].set(1.0)  # Treatment for last 4 samples
        size_factors = jnp.ones(n_samples)
        return {
            "counts": counts,
            "design": design,
            "size_factors": size_factors,
        }

    def test_initialization(self, rngs):
        """Test operator initialization."""
        config = NBGLMConfig(n_features=100, n_covariates=2)
        op = DifferentiableNBGLM(config, rngs=rngs)
        assert op is not None
        assert op.n_features == 100
        assert op.n_covariates == 2

    def test_coefficients_shape(self, rngs):
        """Test that coefficients have correct shape."""
        config = NBGLMConfig(n_features=100, n_covariates=3)
        op = DifferentiableNBGLM(config, rngs=rngs)

        beta = op.get_coefficients()
        # Shape: (n_covariates, n_features)
        assert beta.shape == (3, 100)

    def test_dispersion_shape(self, rngs):
        """Test that dispersion parameters have correct shape."""
        config = NBGLMConfig(n_features=100, estimate_dispersion=True)
        op = DifferentiableNBGLM(config, rngs=rngs)

        dispersion = op.get_dispersion()
        # Shape: (n_features,)
        assert dispersion.shape == (100,)

    def test_dispersion_positive(self, rngs):
        """Test that dispersion is positive."""
        config = NBGLMConfig(n_features=100)
        op = DifferentiableNBGLM(config, rngs=rngs)

        dispersion = op.get_dispersion()
        assert jnp.all(dispersion > 0)

    def test_predict_mean_shape(self, rngs, sample_data):
        """Test that predicted mean has correct shape."""
        config = NBGLMConfig(n_features=100, n_covariates=2)
        op = DifferentiableNBGLM(config, rngs=rngs)

        mean = op.predict_mean(sample_data["design"], sample_data["size_factor"])
        assert mean.shape == (100,)

    def test_predict_mean_positive(self, rngs, sample_data):
        """Test that predicted mean is positive."""
        config = NBGLMConfig(n_features=100, n_covariates=2)
        op = DifferentiableNBGLM(config, rngs=rngs)

        mean = op.predict_mean(sample_data["design"], sample_data["size_factor"])
        assert jnp.all(mean > 0)

    def test_negative_binomial_log_prob(self, rngs, sample_data):
        """Test that NB log probability is computable."""
        config = NBGLMConfig(n_features=100, n_covariates=2)
        op = DifferentiableNBGLM(config, rngs=rngs)

        log_prob = op.negative_binomial_log_prob(
            sample_data["counts"], sample_data["design"], sample_data["size_factor"]
        )

        assert log_prob.shape == ()
        assert jnp.isfinite(log_prob)

    def test_apply_returns_log_likelihood(self, rngs, sample_data):
        """Test that apply returns log likelihood."""
        config = NBGLMConfig(n_features=100, n_covariates=2)
        op = DifferentiableNBGLM(config, rngs=rngs)

        transformed_data, state, metadata = op.apply(sample_data, {}, None, None)

        assert "log_likelihood" in transformed_data
        assert jnp.isfinite(transformed_data["log_likelihood"])

    def test_apply_returns_predicted_mean(self, rngs, sample_data):
        """Test that apply returns predicted mean."""
        config = NBGLMConfig(n_features=100, n_covariates=2)
        op = DifferentiableNBGLM(config, rngs=rngs)

        transformed_data, _, _ = op.apply(sample_data, {}, None, None)

        assert "predicted_mean" in transformed_data
        assert transformed_data["predicted_mean"].shape == (100,)

    def test_apply_preserves_counts(self, rngs, sample_data):
        """Test that apply preserves original counts."""
        config = NBGLMConfig(n_features=100, n_covariates=2)
        op = DifferentiableNBGLM(config, rngs=rngs)

        transformed_data, _, _ = op.apply(sample_data, {}, None, None)

        assert "counts" in transformed_data
        assert jnp.allclose(transformed_data["counts"], sample_data["counts"])


class TestBatchProcessing:
    """Tests for batch processing."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_batch_log_likelihood(self, rngs):
        """Test batch log likelihood computation."""
        config = NBGLMConfig(n_features=100, n_covariates=2)
        op = DifferentiableNBGLM(config, rngs=rngs)

        n_samples = 8
        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(n_samples, 100)).astype(
            jnp.float32
        )
        design = jnp.zeros((n_samples, 2))
        design = design.at[:, 0].set(1.0)
        design = design.at[4:, 1].set(1.0)
        size_factors = jnp.ones(n_samples)

        log_likelihood = op.batch_log_likelihood(counts, design, size_factors)

        assert log_likelihood.shape == ()
        assert jnp.isfinite(log_likelihood)


class TestGradientFlow:
    """Tests for gradient flow through NB GLM."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_gradient_flows_through_log_prob(self, rngs):
        """Test that gradients flow through log probability."""
        config = NBGLMConfig(n_features=100, n_covariates=2)
        op = DifferentiableNBGLM(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        design = jnp.array([1.0, 1.0])
        size_factor = jnp.array(1.0)

        def loss_fn(c):
            return op.negative_binomial_log_prob(c, design, size_factor)

        grad = jax.grad(loss_fn)(counts)
        assert grad is not None
        assert grad.shape == counts.shape

    def test_coefficients_are_learnable(self, rngs):
        """Test that coefficient parameters are learnable."""
        config = NBGLMConfig(n_features=100, n_covariates=2)
        op = DifferentiableNBGLM(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        design = jnp.array([1.0, 1.0])
        size_factor = jnp.array(1.0)
        data = {"counts": counts, "design": design, "size_factor": size_factor}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return -transformed["log_likelihood"]  # Negative for minimization

        loss, grads = loss_fn(op)

        assert hasattr(grads, "beta")

    def test_dispersion_is_learnable(self, rngs):
        """Test that dispersion parameters are learnable."""
        config = NBGLMConfig(n_features=100, n_covariates=2, estimate_dispersion=True)
        op = DifferentiableNBGLM(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        design = jnp.array([1.0, 1.0])
        size_factor = jnp.array(1.0)
        data = {"counts": counts, "design": design, "size_factor": size_factor}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return -transformed["log_likelihood"]

        loss, grads = loss_fn(op)

        assert hasattr(grads, "log_dispersion")


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_apply_is_jit_compatible(self, rngs):
        """Test that apply method works with JIT."""
        config = NBGLMConfig(n_features=100, n_covariates=2)
        op = DifferentiableNBGLM(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        design = jnp.array([1.0, 1.0])
        size_factor = jnp.array(1.0)
        data = {"counts": counts, "design": design, "size_factor": size_factor}
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, new_state, metadata = jit_apply(data, state)
        assert jnp.isfinite(transformed["log_likelihood"])

    def test_predict_is_jit_compatible(self, rngs):
        """Test that predict_mean works with JIT."""
        config = NBGLMConfig(n_features=100, n_covariates=2)
        op = DifferentiableNBGLM(config, rngs=rngs)

        design = jnp.array([1.0, 1.0])
        size_factor = jnp.array(1.0)

        @jax.jit
        def jit_predict(design, size_factor):
            return op.predict_mean(design, size_factor)

        mean = jit_predict(design, size_factor)
        assert mean.shape == (100,)


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_zero_counts(self, rngs):
        """Test with zero counts."""
        config = NBGLMConfig(n_features=100, n_covariates=2)
        op = DifferentiableNBGLM(config, rngs=rngs)

        counts = jnp.zeros(100)
        design = jnp.array([1.0, 0.0])
        size_factor = jnp.array(1.0)
        data = {"counts": counts, "design": design, "size_factor": size_factor}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["log_likelihood"])

    def test_high_counts(self, rngs):
        """Test with high counts."""
        config = NBGLMConfig(n_features=100, n_covariates=2)
        op = DifferentiableNBGLM(config, rngs=rngs)

        counts = jnp.ones(100) * 10000.0
        design = jnp.array([1.0, 0.0])
        size_factor = jnp.array(10.0)
        data = {"counts": counts, "design": design, "size_factor": size_factor}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["log_likelihood"])

    def test_single_covariate(self, rngs):
        """Test with single covariate (intercept only)."""
        config = NBGLMConfig(n_features=100, n_covariates=1)
        op = DifferentiableNBGLM(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        design = jnp.array([1.0])
        size_factor = jnp.array(1.0)
        data = {"counts": counts, "design": design, "size_factor": size_factor}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["log_likelihood"])

    def test_many_covariates(self, rngs):
        """Test with many covariates."""
        config = NBGLMConfig(n_features=100, n_covariates=10)
        op = DifferentiableNBGLM(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        design = jnp.ones(10)
        size_factor = jnp.array(1.0)
        data = {"counts": counts, "design": design, "size_factor": size_factor}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["log_likelihood"])

    def test_small_size_factor(self, rngs):
        """Test with small size factor."""
        config = NBGLMConfig(n_features=100, n_covariates=2)
        op = DifferentiableNBGLM(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        design = jnp.array([1.0, 0.0])
        size_factor = jnp.array(0.1)
        data = {"counts": counts, "design": design, "size_factor": size_factor}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["log_likelihood"])
