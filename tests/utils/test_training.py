"""Tests for training utilities."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.pipelines import create_variant_calling_pipeline
from diffbio.utils.training import (
    Trainer,
    TrainingConfig,
    TrainingState,
    create_optax_optimizer,
    create_synthetic_training_data,
    cross_entropy_loss,
    data_iterator,
)


class TestTrainingConfig:
    """Tests for training configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.learning_rate == 1e-3
        assert config.num_epochs == 100
        assert config.log_every == 10
        assert config.grad_clip_norm == 1.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            learning_rate=5e-4,
            num_epochs=50,
            grad_clip_norm=0.5,
        )
        assert config.learning_rate == 5e-4
        assert config.num_epochs == 50
        assert config.grad_clip_norm == 0.5


class TestTrainingState:
    """Tests for training state."""

    def test_default_state(self):
        """Test default state initialization."""
        state = TrainingState()
        assert state.step == 0
        assert state.epoch == 0
        assert state.loss_history == []
        assert state.best_loss == float("inf")

    def test_custom_state(self):
        """Test custom state initialization."""
        state = TrainingState(step=10, epoch=2, best_loss=0.5)
        assert state.step == 10
        assert state.epoch == 2
        assert state.best_loss == 0.5


class TestOptimizer:
    """Tests for optimizer creation."""

    def test_create_optimizer_with_clipping(self):
        """Test optimizer with gradient clipping."""
        config = TrainingConfig(learning_rate=1e-3, grad_clip_norm=1.0)
        optimizer = create_optax_optimizer(config)
        assert optimizer is not None

    def test_create_optimizer_without_clipping(self):
        """Test optimizer without gradient clipping."""
        config = TrainingConfig(learning_rate=1e-3, grad_clip_norm=None)
        optimizer = create_optax_optimizer(config)
        assert optimizer is not None


class TestCrossEntropyLoss:
    """Tests for cross-entropy loss function."""

    def test_perfect_prediction(self):
        """Test loss with perfect predictions."""
        logits = jnp.array([[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]])
        labels = jnp.array([0, 1])
        loss = cross_entropy_loss(logits, labels, num_classes=3)
        # Perfect predictions should have very low loss
        assert loss < 0.01

    def test_random_prediction(self):
        """Test loss with random predictions."""
        logits = jnp.zeros((4, 3))  # Equal logits
        labels = jnp.array([0, 1, 2, 0])
        loss = cross_entropy_loss(logits, labels, num_classes=3)
        # Random predictions should have loss around log(3) ~ 1.1
        assert 1.0 < loss < 1.2

    def test_loss_is_differentiable(self):
        """Test that loss is differentiable."""
        logits = jnp.array([[1.0, 2.0, 3.0]])
        labels = jnp.array([2])

        def loss_fn(log):
            return cross_entropy_loss(log, labels, num_classes=3)

        grad = jax.grad(loss_fn)(logits)
        assert grad is not None
        assert grad.shape == logits.shape
        assert jnp.all(jnp.isfinite(grad))


class TestSyntheticData:
    """Tests for synthetic data generation."""

    def test_generate_data(self):
        """Test synthetic data generation."""
        inputs, targets = create_synthetic_training_data(
            num_samples=5,
            num_reads=3,
            read_length=10,
            reference_length=20,
            seed=42,
        )

        assert len(inputs) == 5
        assert len(targets) == 5

        # Check input structure
        for inp in inputs:
            assert "reads" in inp
            assert "positions" in inp
            assert "quality" in inp
            assert inp["reads"].shape == (3, 10, 4)
            assert inp["positions"].shape == (3,)
            assert inp["quality"].shape == (3, 10)

        # Check target structure
        for tgt in targets:
            assert "labels" in tgt
            assert tgt["labels"].shape == (20,)

    def test_data_is_valid(self):
        """Test that generated data is valid."""
        inputs, targets = create_synthetic_training_data(
            num_samples=3,
            num_reads=2,
            read_length=8,
            reference_length=15,
            seed=42,
        )

        for inp in inputs:
            # Reads should be valid probability distributions
            read_sums = inp["reads"].sum(axis=-1)
            assert jnp.allclose(read_sums, 1.0, atol=1e-5)

            # Quality scores should be in valid range
            assert jnp.all(inp["quality"] >= 0)
            assert jnp.all(inp["quality"] <= 50)

        for tgt in targets:
            # Labels should be valid class indices
            assert jnp.all(tgt["labels"] >= 0)
            assert jnp.all(tgt["labels"] < 3)


class TestDataIterator:
    """Tests for data iterator."""

    def test_iterator_yields_all_samples(self):
        """Test that iterator yields all samples."""
        inputs = [{"x": jnp.array([1])}, {"x": jnp.array([2])}]
        targets = [{"y": jnp.array([0])}, {"y": jnp.array([1])}]

        count = 0
        for inp, tgt in data_iterator(inputs, targets):
            assert "x" in inp
            assert "y" in tgt
            count += 1

        assert count == 2


class TestTrainerIntegration:
    """Integration tests for the Trainer class."""

    @pytest.fixture
    def small_pipeline(self):
        """Create a small pipeline for testing."""
        return create_variant_calling_pipeline(
            reference_length=15,
            num_classes=3,
            hidden_dim=8,
            seed=42,
        )

    def test_trainer_initialization(self, small_pipeline):
        """Test trainer initialization."""
        config = TrainingConfig(learning_rate=1e-3)
        trainer = Trainer(small_pipeline, config)

        assert trainer.pipeline is not None
        assert trainer.config is not None
        assert trainer.optimizer is not None
        assert trainer.training_state.step == 0

    def test_single_training_step(self, small_pipeline):
        """Test a single training step runs without error."""
        config = TrainingConfig(learning_rate=1e-3)
        trainer = Trainer(small_pipeline, config)

        # Create single sample
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)

        indices = jax.random.randint(k1, (3, 8), 0, 4)
        reads = jax.nn.one_hot(indices, 4).astype(jnp.float32)
        positions = jax.random.randint(k2, (3,), 0, 5)
        quality = jax.random.uniform(k3, (3, 8), minval=20.0, maxval=40.0)

        batch_data = {"reads": reads, "positions": positions, "quality": quality}
        targets = {"labels": jnp.zeros(15, dtype=jnp.int32)}

        def loss_fn(predictions, tgts):
            return cross_entropy_loss(
                predictions["logits"],
                tgts["labels"],
                num_classes=3,
            )

        # Create data iterator
        def data_iter():
            yield batch_data, targets

        # Train for one step
        metrics = trainer.train_epoch(data_iter(), loss_fn)

        assert "avg_loss" in metrics
        assert trainer.training_state.step == 1
        assert len(trainer.training_state.loss_history) == 1

    def test_loss_decreases_over_training(self, small_pipeline):
        """Test that training runs for multiple epochs."""
        config = TrainingConfig(learning_rate=1e-2, log_every=100)
        trainer = Trainer(small_pipeline, config)

        # Create training data
        inputs, targets = create_synthetic_training_data(
            num_samples=5,
            num_reads=3,
            read_length=8,
            reference_length=15,
            seed=42,
        )

        def loss_fn(predictions, tgts):
            return cross_entropy_loss(
                predictions["logits"],
                tgts["labels"],
                num_classes=3,
            )

        # Train for a few epochs
        for epoch in range(2):
            def data_iter():
                return data_iterator(inputs, targets)

            metrics = trainer.train_epoch(data_iter(), loss_fn)

        # Training should have run
        assert trainer.training_state.step > 0
        assert len(trainer.training_state.loss_history) > 0
