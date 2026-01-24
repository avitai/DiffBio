"""Tests for differentiable spectral similarity operators (MS2DeepScore-style).

These operators implement Siamese neural networks for predicting molecular
structural similarity from tandem mass spectra (MS/MS).
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


class TestSpectralSimilarityConfig:
    """Tests for SpectralSimilarityConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from diffbio.operators.metabolomics import SpectralSimilarityConfig

        config = SpectralSimilarityConfig()

        assert config.n_bins == 1000
        assert config.embedding_dim == 200
        assert config.hidden_dims == (512, 256)
        assert config.dropout_rate == 0.2
        assert config.min_mz == 0.0
        assert config.max_mz == 1000.0

    def test_custom_config(self):
        """Test custom configuration values."""
        from diffbio.operators.metabolomics import SpectralSimilarityConfig

        config = SpectralSimilarityConfig(
            n_bins=2000,
            embedding_dim=128,
            hidden_dims=(256, 128, 64),
            dropout_rate=0.1,
            min_mz=50.0,
            max_mz=2000.0,
        )

        assert config.n_bins == 2000
        assert config.embedding_dim == 128
        assert config.hidden_dims == (256, 128, 64)
        assert config.dropout_rate == 0.1
        assert config.min_mz == 50.0
        assert config.max_mz == 2000.0


class TestDifferentiableSpectralSimilarity:
    """Tests for DifferentiableSpectralSimilarity operator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        from diffbio.operators.metabolomics import SpectralSimilarityConfig

        return SpectralSimilarityConfig(
            n_bins=500,
            embedding_dim=64,
            hidden_dims=(128, 64),
            dropout_rate=0.1,
        )

    @pytest.fixture
    def operator(self, config):
        """Create operator instance."""
        from diffbio.operators.metabolomics import DifferentiableSpectralSimilarity

        return DifferentiableSpectralSimilarity(config, rngs=nnx.Rngs(42))

    def test_initialization(self, operator, config):
        """Test operator initialization."""
        assert operator is not None
        assert operator.config == config

    def test_embedding_output_shape(self, operator):
        """Test that embedding has correct shape."""
        n_spectra = 10
        spectra = jax.random.uniform(jax.random.PRNGKey(0), (n_spectra, 500))

        data = {"spectra": spectra}
        result, state, metadata = operator.apply(data, {}, None)

        assert "embeddings" in result
        assert result["embeddings"].shape == (n_spectra, 64)

    def test_similarity_output_range(self, operator):
        """Test that similarity scores are in valid range."""
        n_pairs = 5
        spectra_a = jax.random.uniform(jax.random.PRNGKey(0), (n_pairs, 500))
        spectra_b = jax.random.uniform(jax.random.PRNGKey(1), (n_pairs, 500))

        data = {"spectra_a": spectra_a, "spectra_b": spectra_b}
        result, state, metadata = operator.apply(data, {}, None)

        assert "similarity_scores" in result
        assert result["similarity_scores"].shape == (n_pairs,)
        # Cosine similarity should be in [-1, 1]
        assert jnp.all(result["similarity_scores"] >= -1.0)
        assert jnp.all(result["similarity_scores"] <= 1.0)

    def test_identical_spectra_high_similarity(self, operator):
        """Test that identical spectra have high similarity."""
        spectra = jax.random.uniform(jax.random.PRNGKey(0), (5, 500))

        data = {"spectra_a": spectra, "spectra_b": spectra}
        operator.eval()  # Disable dropout
        result, _, _ = operator.apply(data, {}, None)

        # Identical spectra should have very high similarity
        assert jnp.all(result["similarity_scores"] > 0.99)

    def test_single_spectrum_embedding(self, operator):
        """Test embedding of single spectrum."""
        spectrum = jax.random.uniform(jax.random.PRNGKey(0), (1, 500))

        data = {"spectra": spectrum}
        result, _, _ = operator.apply(data, {}, None)

        assert result["embeddings"].shape == (1, 64)

    def test_batch_processing(self, operator):
        """Test processing of large batch."""
        n_spectra = 100
        spectra = jax.random.uniform(jax.random.PRNGKey(0), (n_spectra, 500))

        data = {"spectra": spectra}
        result, _, _ = operator.apply(data, {}, None)

        assert result["embeddings"].shape == (n_spectra, 64)

    def test_gradient_flow(self, operator):
        """Test that gradients flow through the operator."""
        spectra_a = jax.random.uniform(jax.random.PRNGKey(0), (5, 500))
        spectra_b = jax.random.uniform(jax.random.PRNGKey(1), (5, 500))

        @nnx.value_and_grad
        def loss_fn(model):
            data = {"spectra_a": spectra_a, "spectra_b": spectra_b}
            result, _, _ = model.apply(data, {}, None)
            return result["similarity_scores"].mean()

        loss, grads = loss_fn(operator)

        assert loss is not None
        assert grads is not None

    def test_jit_compilation(self, operator):
        """Test JIT compilation with nnx.jit."""
        spectra = jax.random.uniform(jax.random.PRNGKey(0), (10, 500))

        @nnx.jit
        def apply_jit(model, data):
            return model.apply(data, {}, None)

        data = {"spectra": spectra}
        result, _, _ = apply_jit(operator, data)

        assert result["embeddings"].shape == (10, 64)

    def test_train_eval_modes(self, operator):
        """Test train and eval mode switching."""
        spectra = jax.random.uniform(jax.random.PRNGKey(0), (10, 500))
        data = {"spectra": spectra}

        # Eval mode should be deterministic
        operator.eval()
        result_eval1, _, _ = operator.apply(data, {}, None)
        result_eval2, _, _ = operator.apply(data, {}, None)
        assert jnp.allclose(result_eval1["embeddings"], result_eval2["embeddings"])

        # Train mode with dropout may have variation (but small with low dropout rate)
        operator.train()
        # Just verify it runs without error
        result_train, _, _ = operator.apply(data, {}, None)
        assert result_train["embeddings"].shape == (10, 64)


class TestSpectrumBinning:
    """Tests for spectrum binning utilities."""

    def test_bin_spectrum(self):
        """Test spectrum binning from m/z and intensity pairs."""
        from diffbio.operators.metabolomics import bin_spectrum

        # Create test spectrum: m/z values and intensities
        mz_values = jnp.array([100.0, 200.0, 300.0, 400.0, 500.0])
        intensities = jnp.array([0.5, 1.0, 0.3, 0.8, 0.6])

        binned = bin_spectrum(
            mz_values, intensities, n_bins=100, min_mz=0.0, max_mz=1000.0
        )

        assert binned.shape == (100,)
        # Bins 10, 20, 30, 40, 50 should have values (100/10=10, etc.)
        assert binned[10] > 0  # 100.0 m/z
        assert binned[20] > 0  # 200.0 m/z

    def test_bin_spectrum_normalization(self):
        """Test that binned spectrum is normalized."""
        from diffbio.operators.metabolomics import bin_spectrum

        mz_values = jnp.array([100.0, 200.0, 300.0])
        intensities = jnp.array([1.0, 2.0, 3.0])

        binned = bin_spectrum(mz_values, intensities, n_bins=100, normalize=True)

        # Maximum intensity should be 1.0 after normalization
        assert jnp.isclose(jnp.max(binned), 1.0, atol=1e-5)


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_spectral_similarity(self):
        """Test factory function creates operator."""
        from diffbio.operators.metabolomics import create_spectral_similarity

        operator = create_spectral_similarity(
            n_bins=500, embedding_dim=128, hidden_dims=(256, 128)
        )

        assert operator is not None

        # Test basic operation
        spectra = jax.random.uniform(jax.random.PRNGKey(0), (5, 500))
        result, _, _ = operator.apply({"spectra": spectra}, {}, None)
        assert result["embeddings"].shape == (5, 128)

    def test_create_with_defaults(self):
        """Test factory with default parameters."""
        from diffbio.operators.metabolomics import create_spectral_similarity

        operator = create_spectral_similarity()

        assert operator is not None
        assert operator.config.n_bins == 1000
        assert operator.config.embedding_dim == 200


class TestMetabolomicsModuleImports:
    """Tests for metabolomics module imports."""

    def test_import_from_package(self):
        """Test imports from metabolomics package."""
        from diffbio.operators.metabolomics import (
            DifferentiableSpectralSimilarity,
            SpectralSimilarityConfig,
            bin_spectrum,
            create_spectral_similarity,
        )

        assert DifferentiableSpectralSimilarity is not None
        assert SpectralSimilarityConfig is not None
        assert create_spectral_similarity is not None
        assert bin_spectrum is not None

    def test_import_from_operators(self):
        """Test metabolomics module is accessible from operators."""
        from diffbio.operators import metabolomics

        assert metabolomics is not None
        assert hasattr(metabolomics, "DifferentiableSpectralSimilarity")
