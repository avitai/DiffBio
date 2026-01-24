"""Tests for DifferentiableCRISPRScorer operator.

This module tests the DeepCRISPR-style CRISPR guide RNA scoring operator
that uses a CNN to predict on-target efficiency.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


class TestCRISPRScorerConfig:
    """Tests for CRISPRScorerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from diffbio.operators.crispr import CRISPRScorerConfig

        config = CRISPRScorerConfig()
        assert config.guide_length == 23
        assert config.alphabet_size == 4
        assert config.hidden_channels == (64, 128, 256)
        assert config.fc_dims == (256, 128)
        assert config.dropout_rate == 0.2

    def test_custom_config(self):
        """Test custom configuration values."""
        from diffbio.operators.crispr import CRISPRScorerConfig

        config = CRISPRScorerConfig(
            guide_length=30,
            hidden_channels=(32, 64),
            fc_dims=(128, 64),
            dropout_rate=0.1,
        )
        assert config.guide_length == 30
        assert config.hidden_channels == (32, 64)
        assert config.fc_dims == (128, 64)
        assert config.dropout_rate == 0.1


class TestCRISPRScorer:
    """Tests for DifferentiableCRISPRScorer."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        from diffbio.operators.crispr import CRISPRScorerConfig

        return CRISPRScorerConfig(
            guide_length=23,
            hidden_channels=(32, 64),
            fc_dims=(64, 32),
            dropout_rate=0.1,
        )

    @pytest.fixture
    def scorer(self, config):
        """Create test scorer."""
        from diffbio.operators.crispr import DifferentiableCRISPRScorer

        scorer = DifferentiableCRISPRScorer(config, rngs=nnx.Rngs(42))
        scorer.eval()
        return scorer

    @pytest.fixture
    def sample_data(self, config):
        """Create sample guide RNA data."""
        n_guides = 16
        guide_length = config.guide_length

        # One-hot encoded guide sequences
        key = jax.random.PRNGKey(42)
        indices = jax.random.randint(key, (n_guides, guide_length), 0, 4)
        guides = jax.nn.one_hot(indices, 4)

        return {"guides": guides}

    def test_output_shapes(self, scorer, sample_data, config):
        """Test output tensor shapes."""
        result, _, _ = scorer.apply(sample_data, {}, None)

        n_guides = sample_data["guides"].shape[0]

        # Efficiency scores
        assert "efficiency_scores" in result
        assert result["efficiency_scores"].shape == (n_guides,)

        # Feature maps (from CNN)
        assert "features" in result
        assert result["features"].shape[0] == n_guides

    def test_efficiency_scores_bounded(self, scorer, sample_data):
        """Test that efficiency scores are in [0, 1] range."""
        result, _, _ = scorer.apply(sample_data, {}, None)

        scores = result["efficiency_scores"]

        assert jnp.all(scores >= 0.0)
        assert jnp.all(scores <= 1.0)

    def test_output_finite(self, scorer, sample_data):
        """Test all outputs are finite."""
        result, _, _ = scorer.apply(sample_data, {}, None)

        assert jnp.all(jnp.isfinite(result["efficiency_scores"]))
        assert jnp.all(jnp.isfinite(result["features"]))

    def test_preserves_input_data(self, scorer, sample_data):
        """Test that input data is preserved in output."""
        result, _, _ = scorer.apply(sample_data, {}, None)

        assert "guides" in result
        assert jnp.array_equal(result["guides"], sample_data["guides"])

    def test_train_eval_mode(self, config):
        """Test train and eval mode switching."""
        from diffbio.operators.crispr import DifferentiableCRISPRScorer

        scorer = DifferentiableCRISPRScorer(config, rngs=nnx.Rngs(42))

        # Create data
        key = jax.random.PRNGKey(0)
        indices = jax.random.randint(key, (8, config.guide_length), 0, 4)
        guides = jax.nn.one_hot(indices, 4)
        data = {"guides": guides}

        # Eval mode should be deterministic
        scorer.eval()
        result1, _, _ = scorer.apply(data, {}, None)
        result2, _, _ = scorer.apply(data, {}, None)

        assert jnp.allclose(result1["efficiency_scores"], result2["efficiency_scores"])

    def test_different_batch_sizes(self, config):
        """Test with different batch sizes."""
        from diffbio.operators.crispr import DifferentiableCRISPRScorer

        scorer = DifferentiableCRISPRScorer(config, rngs=nnx.Rngs(42))
        scorer.eval()

        for batch_size in [1, 8, 32]:
            key = jax.random.PRNGKey(batch_size)
            indices = jax.random.randint(key, (batch_size, config.guide_length), 0, 4)
            guides = jax.nn.one_hot(indices, 4)

            result, _, _ = scorer.apply({"guides": guides}, {}, None)

            assert result["efficiency_scores"].shape == (batch_size,)


class TestCRISPRScorerDifferentiability:
    """Tests for gradient flow through CRISPR scorer."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        from diffbio.operators.crispr import CRISPRScorerConfig

        return CRISPRScorerConfig(
            guide_length=20,
            hidden_channels=(16, 32),
            fc_dims=(32, 16),
            dropout_rate=0.0,
        )

    @pytest.fixture
    def scorer(self, config):
        """Create test scorer."""
        from diffbio.operators.crispr import DifferentiableCRISPRScorer

        scorer = DifferentiableCRISPRScorer(config, rngs=nnx.Rngs(42))
        scorer.eval()
        return scorer

    @pytest.fixture
    def sample_data(self, config):
        """Create sample data."""
        n_guides = 8
        key = jax.random.PRNGKey(42)
        indices = jax.random.randint(key, (n_guides, config.guide_length), 0, 4)
        guides = jax.nn.one_hot(indices, 4)

        return {"guides": guides}

    def test_gradient_flow(self, scorer, sample_data):
        """Test gradients flow through the scorer."""

        @nnx.value_and_grad
        def loss_fn(model):
            result, _, _ = model.apply(sample_data, {}, None)
            return result["efficiency_scores"].mean()

        loss, grads = loss_fn(scorer)

        # Check gradients exist
        assert grads is not None

        # Check loss is finite
        assert jnp.isfinite(loss)

    def test_gradient_wrt_target(self, scorer, sample_data):
        """Test gradients for regression loss."""
        target_scores = jnp.ones(sample_data["guides"].shape[0]) * 0.8

        @nnx.value_and_grad
        def loss_fn(model):
            result, _, _ = model.apply(sample_data, {}, None)
            return jnp.mean((result["efficiency_scores"] - target_scores) ** 2)

        loss, grads = loss_fn(scorer)

        assert jnp.isfinite(loss)
        assert grads is not None

    def test_jit_compilation(self, scorer, sample_data):
        """Test JIT compilation works."""

        @jax.jit
        def forward(model, data):
            result, _, _ = model.apply(data, {}, None)
            return result["efficiency_scores"]

        result = forward(scorer, sample_data)
        assert result.shape == (8,)


class TestCRISPRScorerFactory:
    """Tests for create_crispr_scorer factory function."""

    def test_factory_creates_scorer(self):
        """Test factory function creates working scorer."""
        from diffbio.operators.crispr import create_crispr_scorer

        scorer = create_crispr_scorer(guide_length=23)

        key = jax.random.PRNGKey(42)
        indices = jax.random.randint(key, (10, 23), 0, 4)
        guides = jax.nn.one_hot(indices, 4)

        result, _, _ = scorer.apply({"guides": guides}, {}, None)

        assert result["efficiency_scores"].shape == (10,)

    def test_factory_with_custom_params(self):
        """Test factory with custom parameters."""
        from diffbio.operators.crispr import create_crispr_scorer

        scorer = create_crispr_scorer(
            guide_length=30,
            hidden_channels=(64, 128),
            fc_dims=(128, 64),
        )

        key = jax.random.PRNGKey(42)
        indices = jax.random.randint(key, (5, 30), 0, 4)
        guides = jax.nn.one_hot(indices, 4)

        result, _, _ = scorer.apply({"guides": guides}, {}, None)

        assert result["efficiency_scores"].shape == (5,)
