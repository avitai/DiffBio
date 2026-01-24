"""Tests for DifferentiableAncestryEstimator operator.

This module tests the Neural ADMIXTURE-style ancestry estimation operator
that uses an autoencoder to estimate population ancestry proportions.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


class TestAncestryEstimatorConfig:
    """Tests for AncestryEstimatorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from diffbio.operators.population import AncestryEstimatorConfig

        config = AncestryEstimatorConfig()
        assert config.n_snps == 10000
        assert config.n_populations == 5
        assert config.hidden_dims == (128, 64)
        assert config.temperature == 1.0
        assert config.dropout_rate == 0.1

    def test_custom_config(self):
        """Test custom configuration values."""
        from diffbio.operators.population import AncestryEstimatorConfig

        config = AncestryEstimatorConfig(
            n_snps=50000,
            n_populations=10,
            hidden_dims=(256, 128, 64),
            temperature=0.5,
            dropout_rate=0.2,
        )
        assert config.n_snps == 50000
        assert config.n_populations == 10
        assert config.hidden_dims == (256, 128, 64)
        assert config.temperature == 0.5
        assert config.dropout_rate == 0.2


class TestAncestryEstimator:
    """Tests for DifferentiableAncestryEstimator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        from diffbio.operators.population import AncestryEstimatorConfig

        return AncestryEstimatorConfig(
            n_snps=100,
            n_populations=3,
            hidden_dims=(32, 16),
            temperature=1.0,
            dropout_rate=0.1,
        )

    @pytest.fixture
    def estimator(self, config):
        """Create test estimator."""
        from diffbio.operators.population import DifferentiableAncestryEstimator

        return DifferentiableAncestryEstimator(config, rngs=nnx.Rngs(42))

    @pytest.fixture
    def sample_data(self, config):
        """Create sample genotype data."""
        n_samples = 20
        n_snps = config.n_snps

        # Genotype matrix (0, 1, 2 encoding as floats)
        key = jax.random.PRNGKey(42)
        genotypes = jax.random.randint(key, (n_samples, n_snps), 0, 3).astype(jnp.float32)

        return {"genotypes": genotypes}

    def test_output_shapes(self, estimator, sample_data, config):
        """Test output tensor shapes."""
        result, _, _ = estimator.apply(sample_data, {}, None)

        n_samples = sample_data["genotypes"].shape[0]

        # Ancestry proportions
        assert "ancestry_proportions" in result
        assert result["ancestry_proportions"].shape == (n_samples, config.n_populations)

        # Reconstructed genotypes
        assert "reconstructed" in result
        assert result["reconstructed"].shape == sample_data["genotypes"].shape

        # Latent representation
        assert "latent" in result
        assert result["latent"].shape[0] == n_samples

    def test_ancestry_proportions_sum_to_one(self, estimator, sample_data):
        """Test that ancestry proportions sum to 1 for each sample."""
        result, _, _ = estimator.apply(sample_data, {}, None)

        proportions = result["ancestry_proportions"]
        sums = jnp.sum(proportions, axis=-1)

        assert jnp.allclose(sums, 1.0, atol=1e-5)

    def test_ancestry_proportions_non_negative(self, estimator, sample_data):
        """Test that ancestry proportions are non-negative."""
        result, _, _ = estimator.apply(sample_data, {}, None)

        proportions = result["ancestry_proportions"]

        assert jnp.all(proportions >= 0)

    def test_output_finite(self, estimator, sample_data):
        """Test all outputs are finite."""
        result, _, _ = estimator.apply(sample_data, {}, None)

        assert jnp.all(jnp.isfinite(result["ancestry_proportions"]))
        assert jnp.all(jnp.isfinite(result["reconstructed"]))
        assert jnp.all(jnp.isfinite(result["latent"]))

    def test_preserves_input_data(self, estimator, sample_data):
        """Test that input data is preserved in output."""
        result, _, _ = estimator.apply(sample_data, {}, None)

        assert "genotypes" in result
        assert jnp.array_equal(result["genotypes"], sample_data["genotypes"])

    def test_temperature_affects_sharpness(self, config, sample_data):
        """Test that temperature controls sharpness of proportions."""
        from diffbio.operators.population import (
            AncestryEstimatorConfig,
            DifferentiableAncestryEstimator,
        )

        # High temperature (softer)
        config_high = AncestryEstimatorConfig(
            n_snps=config.n_snps,
            n_populations=config.n_populations,
            hidden_dims=config.hidden_dims,
            temperature=5.0,
            dropout_rate=0.0,
        )
        estimator_high = DifferentiableAncestryEstimator(config_high, rngs=nnx.Rngs(42))
        estimator_high.eval()
        result_high, _, _ = estimator_high.apply(sample_data, {}, None)

        # Low temperature (sharper)
        config_low = AncestryEstimatorConfig(
            n_snps=config.n_snps,
            n_populations=config.n_populations,
            hidden_dims=config.hidden_dims,
            temperature=0.1,
            dropout_rate=0.0,
        )
        estimator_low = DifferentiableAncestryEstimator(config_low, rngs=nnx.Rngs(42))
        estimator_low.eval()
        result_low, _, _ = estimator_low.apply(sample_data, {}, None)

        # Lower temperature should have higher max proportions (sharper)
        max_high = jnp.max(result_high["ancestry_proportions"], axis=-1).mean()
        max_low = jnp.max(result_low["ancestry_proportions"], axis=-1).mean()

        assert max_low > max_high

    def test_train_eval_mode(self, estimator, sample_data):
        """Test train and eval mode switching."""
        # Eval mode should be deterministic
        estimator.eval()
        result1, _, _ = estimator.apply(sample_data, {}, None)
        result2, _, _ = estimator.apply(sample_data, {}, None)

        assert jnp.allclose(result1["ancestry_proportions"], result2["ancestry_proportions"])

    def test_different_batch_sizes(self, config):
        """Test with different batch sizes."""
        from diffbio.operators.population import DifferentiableAncestryEstimator

        estimator = DifferentiableAncestryEstimator(config, rngs=nnx.Rngs(42))
        estimator.eval()

        for batch_size in [1, 10, 50]:
            genotypes = jax.random.randint(
                jax.random.PRNGKey(batch_size), (batch_size, config.n_snps), 0, 3
            ).astype(jnp.float32)

            result, _, _ = estimator.apply({"genotypes": genotypes}, {}, None)

            assert result["ancestry_proportions"].shape == (
                batch_size,
                config.n_populations,
            )


class TestAncestryEstimatorDifferentiability:
    """Tests for gradient flow through ancestry estimator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        from diffbio.operators.population import AncestryEstimatorConfig

        return AncestryEstimatorConfig(
            n_snps=50,
            n_populations=3,
            hidden_dims=(16, 8),
            temperature=1.0,
            dropout_rate=0.0,
        )

    @pytest.fixture
    def estimator(self, config):
        """Create test estimator."""
        from diffbio.operators.population import DifferentiableAncestryEstimator

        estimator = DifferentiableAncestryEstimator(config, rngs=nnx.Rngs(42))
        estimator.eval()
        return estimator

    @pytest.fixture
    def sample_data(self, config):
        """Create sample data."""
        n_samples = 10
        genotypes = jax.random.randint(
            jax.random.PRNGKey(42), (n_samples, config.n_snps), 0, 3
        ).astype(jnp.float32)

        return {"genotypes": genotypes}

    def test_gradient_flow(self, estimator, sample_data):
        """Test gradients flow through the estimator."""

        @nnx.value_and_grad
        def loss_fn(model):
            result, _, _ = model.apply(sample_data, {}, None)
            return result["ancestry_proportions"].mean()

        loss, grads = loss_fn(estimator)

        # Check gradients exist
        assert grads is not None

        # Check loss is finite
        assert jnp.isfinite(loss)

    def test_gradient_wrt_reconstruction(self, estimator, sample_data):
        """Test gradients for reconstruction loss."""

        @nnx.value_and_grad
        def loss_fn(model):
            result, _, _ = model.apply(sample_data, {}, None)
            # Reconstruction loss
            return jnp.mean((result["reconstructed"] - sample_data["genotypes"]) ** 2)

        loss, grads = loss_fn(estimator)

        assert jnp.isfinite(loss)
        assert grads is not None

    def test_jit_compilation(self, estimator, sample_data):
        """Test JIT compilation works."""

        @jax.jit
        def forward(model, data):
            result, _, _ = model.apply(data, {}, None)
            return result["ancestry_proportions"]

        result = forward(estimator, sample_data)
        assert result.shape[1] == 3  # n_populations


class TestAncestryEstimatorFactory:
    """Tests for create_ancestry_estimator factory function."""

    def test_factory_creates_estimator(self):
        """Test factory function creates working estimator."""
        from diffbio.operators.population import create_ancestry_estimator

        estimator = create_ancestry_estimator(
            n_snps=100,
            n_populations=5,
        )

        genotypes = jax.random.randint(jax.random.PRNGKey(42), (10, 100), 0, 3).astype(jnp.float32)

        result, _, _ = estimator.apply({"genotypes": genotypes}, {}, None)

        assert result["ancestry_proportions"].shape == (10, 5)

    def test_factory_with_custom_params(self):
        """Test factory with custom parameters."""
        from diffbio.operators.population import create_ancestry_estimator

        estimator = create_ancestry_estimator(
            n_snps=500,
            n_populations=10,
            hidden_dims=(64, 32),
            temperature=0.5,
        )

        genotypes = jax.random.randint(jax.random.PRNGKey(42), (5, 500), 0, 3).astype(jnp.float32)

        result, _, _ = estimator.apply({"genotypes": genotypes}, {}, None)

        assert result["ancestry_proportions"].shape == (5, 10)
