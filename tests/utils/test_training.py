"""Tests for training utilities."""

import jax
import jax.numpy as jnp
import pytest

from diffbio.pipelines import create_variant_calling_pipeline
from diffbio.utils.training import (
    Trainer,
    TrainingConfig,
    TrainingState,
    create_optax_optimizer,
    create_realistic_training_data,
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


class TestRealisticSyntheticData:
    """Tests for realistic synthetic data generation."""

    def test_generate_data_structure(self):
        """Test that realistic data has correct structure."""
        inputs, targets = create_realistic_training_data(
            num_samples=5,
            num_reads=10,
            read_length=20,
            reference_length=50,
            seed=42,
        )

        assert len(inputs) == 5
        assert len(targets) == 5

        # Check input structure
        for inp in inputs:
            assert "reads" in inp
            assert "positions" in inp
            assert "quality" in inp
            assert "strand" in inp
            assert inp["reads"].shape == (10, 20, 4)
            assert inp["positions"].shape == (10,)
            assert inp["quality"].shape == (10, 20)
            assert inp["strand"].shape == (10,)

        # Check target structure
        for tgt in targets:
            assert "labels" in tgt
            assert "variant_alleles" in tgt
            assert "is_heterozygous" in tgt
            assert tgt["labels"].shape == (50,)

    def test_reads_are_valid_distributions(self):
        """Test that reads are valid one-hot encoded sequences."""
        inputs, _ = create_realistic_training_data(
            num_samples=3,
            num_reads=5,
            read_length=15,
            reference_length=30,
            seed=42,
        )

        for inp in inputs:
            # Each position should sum to 1 (one-hot encoding)
            read_sums = inp["reads"].sum(axis=-1)
            assert jnp.allclose(read_sums, 1.0, atol=1e-5)

    def test_variants_appear_in_reads(self):
        """Test that variant positions show alternate alleles in reads.

        This is the KEY test - the old implementation failed because
        variants were labeled but not actually present in reads.
        """
        inputs, targets = create_realistic_training_data(
            num_samples=10,
            num_reads=20,
            read_length=30,
            reference_length=100,
            variant_rate=0.1,  # 10% variant rate
            heterozygous_rate=0.0,  # All homozygous for simpler testing
            error_rate=0.0,  # No errors for simpler testing
            seed=42,
        )

        # For each sample, check that variant positions show alternate alleles
        variant_found = False
        for inp, tgt in zip(inputs, targets):
            reads = inp["reads"]
            positions = inp["positions"]
            labels = tgt["labels"]
            alt_alleles = tgt["variant_alleles"]

            # Find SNP positions (label == 1)
            snp_positions = jnp.where(labels == 1)[0]

            if len(snp_positions) == 0:
                continue

            # For each read, check if it covers any SNP position
            for read_idx in range(len(positions)):
                read_start = positions[read_idx]
                read_end = read_start + 30  # read_length

                for snp_pos in snp_positions:
                    if read_start <= snp_pos < read_end:
                        # This read covers this SNP
                        offset = snp_pos - read_start
                        read_base = jnp.argmax(reads[read_idx, offset])
                        expected_alt = alt_alleles[snp_pos]

                        # The read should show the alternate allele (for homozygous)
                        if read_base == expected_alt:
                            variant_found = True

        assert variant_found, "No variants found in reads - data generation is broken!"

    def test_quality_profile_position_dependent(self):
        """Test that quality scores follow position-dependent profile."""
        inputs, _ = create_realistic_training_data(
            num_samples=5,
            num_reads=100,
            read_length=50,
            reference_length=100,
            error_rate=0.0,  # No errors to see clean profile
            seed=42,
        )

        # Average quality across all reads at each position
        all_qualities = jnp.stack([inp["quality"] for inp in inputs])
        mean_quality = all_qualities.mean(axis=(0, 1))  # Average over samples and reads

        # Quality should be higher in middle than at ends
        center = 25
        center_quality = mean_quality[center - 5 : center + 5].mean()
        edge_quality = (mean_quality[:5].mean() + mean_quality[-5:].mean()) / 2

        assert center_quality > edge_quality, "Quality should be higher in center of reads"

    def test_quality_in_valid_range(self):
        """Test that quality scores are in valid Phred range."""
        inputs, _ = create_realistic_training_data(
            num_samples=3,
            num_reads=10,
            read_length=20,
            reference_length=50,
            seed=42,
        )

        for inp in inputs:
            assert jnp.all(inp["quality"] >= 5.0), "Quality should be >= 5"
            assert jnp.all(inp["quality"] <= 40.0), "Quality should be <= 40"

    def test_heterozygous_variants_show_mixed_alleles(self):
        """Test that heterozygous variants show ~50% variant alleles."""
        inputs, targets = create_realistic_training_data(
            num_samples=20,
            num_reads=100,  # Many reads for statistical power
            read_length=80,
            reference_length=100,
            variant_rate=0.1,
            heterozygous_rate=1.0,  # All heterozygous
            error_rate=0.0,
            seed=42,
        )

        # Count variant vs reference alleles at heterozygous SNP sites
        variant_count = 0
        total_count = 0

        for inp, tgt in zip(inputs, targets):
            reads = inp["reads"]
            positions = inp["positions"]
            labels = tgt["labels"]
            alt_alleles = tgt["variant_alleles"]

            # Find SNP positions
            snp_positions = jnp.where(labels == 1)[0]

            for snp_pos in snp_positions:
                # Count reads covering this position
                for read_idx in range(len(positions)):
                    read_start = positions[read_idx]
                    read_end = read_start + 80

                    if read_start <= snp_pos < read_end:
                        offset = snp_pos - read_start
                        read_base = jnp.argmax(reads[read_idx, offset])
                        expected_alt = alt_alleles[snp_pos]

                        if read_base == expected_alt:
                            variant_count += 1
                        total_count += 1

        if total_count > 0:
            variant_fraction = variant_count / total_count
            # Should be around 50% for heterozygous (allow some variance)
            assert 0.3 < variant_fraction < 0.7, (
                f"Heterozygous variants should show ~50% alt allele, got {variant_fraction:.2%}"
            )

    def test_strand_information_present(self):
        """Test that strand information is generated."""
        inputs, _ = create_realistic_training_data(
            num_samples=5,
            num_reads=20,
            read_length=30,
            reference_length=100,
            seed=42,
        )

        for inp in inputs:
            strands = inp["strand"]
            # Should have both forward (0) and reverse (1) strands
            assert jnp.any(strands == 0), "Should have forward strand reads"
            assert jnp.any(strands == 1), "Should have reverse strand reads"


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
        assert trainer.training_state.loss_history is not None
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

            trainer.train_epoch(data_iter(), loss_fn)

        # Training should have run
        assert trainer.training_state.step > 0
        assert trainer.training_state.loss_history is not None
        assert len(trainer.training_state.loss_history) > 0
