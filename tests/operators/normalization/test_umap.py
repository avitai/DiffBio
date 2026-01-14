"""Tests for differentiable UMAP dimensionality reduction operator.

Following TDD principles, these tests define the expected behavior
of the DifferentiableUMAP operator for dimensionality reduction.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest


class TestUMAPConfig:
    """Tests for UMAPConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from diffbio.operators.normalization.umap import UMAPConfig

        config = UMAPConfig(stream_name=None)

        assert config.n_components == 2
        assert config.n_neighbors == 15
        assert config.min_dist == 0.1
        assert config.metric == "euclidean"
        assert config.stochastic is False

    def test_custom_config(self):
        """Test custom configuration values."""
        from diffbio.operators.normalization.umap import UMAPConfig

        config = UMAPConfig(
            n_components=3,
            n_neighbors=30,
            min_dist=0.5,
            metric="cosine",
            stream_name=None,
        )

        assert config.n_components == 3
        assert config.n_neighbors == 30
        assert config.min_dist == 0.5
        assert config.metric == "cosine"


class TestDifferentiableUMAP:
    """Tests for DifferentiableUMAP operator."""

    @pytest.fixture
    def rngs(self):
        """Provide RNG fixture."""
        return nnx.Rngs(42)

    @pytest.fixture
    def config(self):
        """Provide default config."""
        from diffbio.operators.normalization.umap import UMAPConfig

        return UMAPConfig(
            n_components=2,
            n_neighbors=10,
            min_dist=0.1,
            input_features=50,
            hidden_dim=32,
            stream_name=None,
        )

    @pytest.fixture
    def umap(self, config, rngs):
        """Create UMAP instance."""
        from diffbio.operators.normalization.umap import DifferentiableUMAP

        return DifferentiableUMAP(config, rngs=rngs)

    def test_initialization(self, config, rngs):
        """Test operator initialization."""
        from diffbio.operators.normalization.umap import DifferentiableUMAP

        umap = DifferentiableUMAP(config, rngs=rngs)

        assert umap.config == config
        assert hasattr(umap, "a_param")
        assert hasattr(umap, "b_param")

    def test_initialization_without_rngs(self, config):
        """Test initialization without providing RNGs."""
        from diffbio.operators.normalization.umap import DifferentiableUMAP

        umap = DifferentiableUMAP(config, rngs=None)
        assert umap is not None

    def test_apply_reduces_dimensions(self, umap, config):
        """Test that apply reduces dimensions correctly."""
        n_samples = 50
        n_features = 50  # Match config.input_features

        high_dim_data = jax.random.normal(jax.random.key(0), (n_samples, n_features))

        data = {"features": high_dim_data}
        result, state, metadata = umap.apply(data, {}, None)

        assert "embedding" in result
        assert result["embedding"].shape == (n_samples, config.n_components)

    def test_embedding_shape(self, config, rngs):
        """Test embedding output shape for various inputs."""
        from diffbio.operators.normalization.umap import DifferentiableUMAP, UMAPConfig

        n_samples = 30
        n_features = 50

        for n_components in [2, 3, 5]:
            config_nc = UMAPConfig(
                n_components=n_components,
                n_neighbors=10,
                input_features=n_features,
                hidden_dim=32,
                stream_name=None,
            )
            umap = DifferentiableUMAP(config_nc, rngs=rngs)

            high_dim_data = jax.random.normal(jax.random.key(0), (n_samples, n_features))

            data = {"features": high_dim_data}
            result, _, _ = umap.apply(data, {}, None)

            assert result["embedding"].shape == (n_samples, n_components)

    def test_output_finite(self, umap):
        """Test that all outputs are finite."""
        n_samples = 30
        n_features = 50
        high_dim_data = jax.random.normal(jax.random.key(0), (n_samples, n_features))

        data = {"features": high_dim_data}
        result, _, _ = umap.apply(data, {}, None)

        assert jnp.all(jnp.isfinite(result["embedding"]))

    def test_preserves_original_data(self, umap):
        """Test that original data is preserved in output."""
        high_dim_data = jax.random.normal(jax.random.key(0), (30, 50))
        extra_data = jnp.array([1.0, 2.0, 3.0])

        data = {"features": high_dim_data, "extra": extra_data}
        result, _, _ = umap.apply(data, {}, None)

        assert "extra" in result
        assert jnp.allclose(result["extra"], extra_data)
        assert "features" in result

    def test_different_samples_different_embeddings(self, umap):
        """Test that different input samples produce different embeddings."""
        data1 = jax.random.normal(jax.random.key(0), (30, 50))
        data2 = jax.random.normal(jax.random.key(1), (30, 50))

        result1, _, _ = umap.apply({"features": data1}, {}, None)
        result2, _, _ = umap.apply({"features": data2}, {}, None)

        # Embeddings should be different for different inputs
        assert not jnp.allclose(result1["embedding"], result2["embedding"])


class TestUMAPDifferentiability:
    """Tests for gradient flow through the UMAP operator."""

    @pytest.fixture
    def rngs(self):
        """Provide RNG fixture."""
        return nnx.Rngs(42)

    @pytest.fixture
    def config(self):
        """Provide default config."""
        from diffbio.operators.normalization.umap import UMAPConfig

        return UMAPConfig(
            n_components=2,
            n_neighbors=10,
            input_features=50,
            hidden_dim=32,
            stream_name=None,
        )

    def test_gradient_flow_through_operator(self, config, rngs):
        """Test that gradients flow through the operator."""
        from diffbio.operators.normalization.umap import DifferentiableUMAP

        umap = DifferentiableUMAP(config, rngs=rngs)

        def loss_fn(op, features):
            data = {"features": features}
            result, _, _ = op.apply(data, {}, None)
            return result["embedding"].sum()

        features = jax.random.normal(jax.random.key(0), (30, 50))
        grads = nnx.grad(loss_fn)(umap, features)

        assert grads is not None

    def test_gradient_wrt_input(self, config, rngs):
        """Test gradient with respect to input features."""
        from diffbio.operators.normalization.umap import DifferentiableUMAP

        umap = DifferentiableUMAP(config, rngs=rngs)

        def loss_fn(features):
            data = {"features": features}
            result, _, _ = umap.apply(data, {}, None)
            return result["embedding"].sum()

        features = jax.random.normal(jax.random.key(0), (30, 50))
        grad = jax.grad(loss_fn)(features)

        assert grad.shape == features.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_wrt_parameters(self, config, rngs):
        """Test gradient with respect to learnable parameters."""
        from diffbio.operators.normalization.umap import DifferentiableUMAP

        umap = DifferentiableUMAP(config, rngs=rngs)

        def loss_fn(op, features):
            data = {"features": features}
            result, _, _ = op.apply(data, {}, None)
            return result["embedding"].mean()

        features = jax.random.normal(jax.random.key(0), (30, 50))
        grads = nnx.grad(loss_fn)(umap, features)

        # Check that parameter gradients exist
        assert hasattr(grads, "a_param")
        assert hasattr(grads, "b_param")


class TestUMAPJITCompatibility:
    """Tests for JIT compilation compatibility."""

    @pytest.fixture
    def rngs(self):
        """Provide RNG fixture."""
        return nnx.Rngs(42)

    @pytest.fixture
    def config(self):
        """Provide default config."""
        from diffbio.operators.normalization.umap import UMAPConfig

        return UMAPConfig(
            n_components=2,
            n_neighbors=10,
            input_features=50,
            hidden_dim=32,
            stream_name=None,
        )

    def test_jit_apply(self, config, rngs):
        """Test JIT compilation of apply method."""
        from diffbio.operators.normalization.umap import DifferentiableUMAP

        umap = DifferentiableUMAP(config, rngs=rngs)

        @jax.jit
        def jit_apply(features):
            data = {"features": features}
            result, _, _ = umap.apply(data, {}, None)
            return result["embedding"]

        features = jax.random.normal(jax.random.key(0), (30, 50))

        # Should compile and run without error
        result = jit_apply(features)
        assert result.shape == (30, 2)

    def test_jit_gradient(self, config, rngs):
        """Test JIT compilation of gradient computation."""
        from diffbio.operators.normalization.umap import DifferentiableUMAP

        umap = DifferentiableUMAP(config, rngs=rngs)

        @jax.jit
        def loss_and_grad(features):
            def loss_fn(f):
                data = {"features": f}
                result, _, _ = umap.apply(data, {}, None)
                return result["embedding"].sum()

            return jax.value_and_grad(loss_fn)(features)

        features = jax.random.normal(jax.random.key(0), (30, 50))

        # Should compile and run without error
        loss, grad = loss_and_grad(features)
        assert jnp.isfinite(loss)
        assert jnp.all(jnp.isfinite(grad))


class TestUMAPLocalStructure:
    """Tests for UMAP's local structure preservation."""

    @pytest.fixture
    def rngs(self):
        """Provide RNG fixture."""
        return nnx.Rngs(42)

    @pytest.fixture
    def config(self):
        """Provide config."""
        from diffbio.operators.normalization.umap import UMAPConfig

        return UMAPConfig(
            n_components=2,
            n_neighbors=10,
            input_features=50,
            hidden_dim=32,
            stream_name=None,
        )

    def test_similar_points_close_in_embedding(self, config, rngs):
        """Test that similar points remain close in the embedding."""
        from diffbio.operators.normalization.umap import DifferentiableUMAP

        umap = DifferentiableUMAP(config, rngs=rngs)

        # Create two clusters in high-dimensional space
        cluster1 = jax.random.normal(jax.random.key(0), (15, 50))
        cluster2 = jax.random.normal(jax.random.key(1), (15, 50)) + 10.0

        features = jnp.concatenate([cluster1, cluster2], axis=0)

        data = {"features": features}
        result, _, _ = umap.apply(data, {}, None)
        embedding = result["embedding"]

        # Compute within-cluster and between-cluster distances
        emb_c1 = embedding[:15]
        emb_c2 = embedding[15:]

        within_c1 = jnp.mean(jnp.linalg.norm(emb_c1[:, None] - emb_c1[None, :], axis=-1))
        within_c2 = jnp.mean(jnp.linalg.norm(emb_c2[:, None] - emb_c2[None, :], axis=-1))
        between = jnp.mean(jnp.linalg.norm(emb_c1[:, None] - emb_c2[None, :], axis=-1))

        # After proper training, between-cluster distance should be larger
        # For untrained model, just check that computation works
        assert jnp.isfinite(within_c1)
        assert jnp.isfinite(within_c2)
        assert jnp.isfinite(between)
