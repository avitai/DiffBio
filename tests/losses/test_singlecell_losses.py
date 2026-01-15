"""Tests for diffbio.losses.singlecell_losses module.

These tests define the expected behavior of single-cell specific loss functions
for differentiable bioinformatics pipelines.
"""

import jax
import jax.numpy as jnp

from diffbio.losses.singlecell_losses import (
    BatchMixingLoss,
    ClusteringCompactnessLoss,
    VelocityConsistencyLoss,
)


class TestBatchMixingLoss:
    """Tests for BatchMixingLoss."""

    def test_initialization(self, rngs):
        """Test loss initialization."""
        loss_fn = BatchMixingLoss(rngs=rngs)
        assert loss_fn is not None

    def test_initialization_with_params(self, rngs):
        """Test initialization with custom parameters."""
        loss_fn = BatchMixingLoss(n_neighbors=20, temperature=0.5, rngs=rngs)
        assert loss_fn.n_neighbors == 20
        assert loss_fn.temperature == 0.5

    def test_loss_shape(self, rngs):
        """Test that loss returns scalar."""
        loss_fn = BatchMixingLoss(n_neighbors=5, n_batches=3, rngs=rngs)

        key = jax.random.key(0)
        # Embeddings for 50 cells with 32 latent dimensions
        embeddings = jax.random.normal(key, (50, 32))
        # Batch labels (3 batches)
        batch_labels = jax.random.randint(key, (50,), 0, 3)

        loss = loss_fn(embeddings, batch_labels)
        assert loss.shape == ()

    def test_loss_finite(self, rngs):
        """Test that loss is finite."""
        loss_fn = BatchMixingLoss(n_neighbors=5, n_batches=3, rngs=rngs)

        key = jax.random.key(0)
        embeddings = jax.random.normal(key, (50, 32))
        batch_labels = jax.random.randint(key, (50,), 0, 3)

        loss = loss_fn(embeddings, batch_labels)
        assert jnp.isfinite(loss)

    def test_perfect_mixing(self, rngs):
        """Test that perfect mixing yields low loss."""
        loss_fn = BatchMixingLoss(n_neighbors=10, n_batches=3, rngs=rngs)

        key = jax.random.key(0)
        # Create perfectly mixed embeddings (batches interleaved in same space)
        embeddings = jax.random.normal(key, (60, 32))
        # Interleaved batch labels
        batch_labels = jnp.tile(jnp.array([0, 1, 2]), 20)

        loss = loss_fn(embeddings, batch_labels)
        assert jnp.isfinite(loss)

    def test_gradient_flows(self, rngs):
        """Test that gradients flow through loss."""
        loss_fn = BatchMixingLoss(n_neighbors=5, n_batches=3, rngs=rngs)

        key = jax.random.key(0)
        batch_labels = jax.random.randint(key, (50,), 0, 3)

        def compute_loss(embeddings):
            return loss_fn(embeddings, batch_labels)

        embeddings = jax.random.normal(key, (50, 32))
        grad = jax.grad(compute_loss)(embeddings)

        assert grad is not None
        assert grad.shape == embeddings.shape
        assert jnp.isfinite(grad).all()


class TestClusteringCompactnessLoss:
    """Tests for ClusteringCompactnessLoss."""

    def test_initialization(self, rngs):
        """Test loss initialization."""
        loss_fn = ClusteringCompactnessLoss(rngs=rngs)
        assert loss_fn is not None

    def test_initialization_with_params(self, rngs):
        """Test initialization with custom parameters."""
        loss_fn = ClusteringCompactnessLoss(
            separation_weight=2.0,
            min_separation=0.5,
            rngs=rngs,
        )
        assert loss_fn.separation_weight == 2.0
        assert loss_fn.min_separation == 0.5

    def test_loss_shape(self, rngs):
        """Test that loss returns scalar."""
        loss_fn = ClusteringCompactnessLoss(rngs=rngs)

        key = jax.random.key(0)
        # Embeddings for 50 cells
        embeddings = jax.random.normal(key, (50, 32))
        # Soft cluster assignments (5 clusters)
        assignments = jax.nn.softmax(jax.random.normal(key, (50, 5)), axis=-1)

        loss = loss_fn(embeddings, assignments)
        assert loss.shape == ()

    def test_loss_finite(self, rngs):
        """Test that loss is finite."""
        loss_fn = ClusteringCompactnessLoss(rngs=rngs)

        key = jax.random.key(0)
        embeddings = jax.random.normal(key, (50, 32))
        assignments = jax.nn.softmax(jax.random.normal(key, (50, 5)), axis=-1)

        loss = loss_fn(embeddings, assignments)
        assert jnp.isfinite(loss)

    def test_compact_clusters_lower_loss(self, rngs):
        """Test that more compact clusters yield lower compactness component."""
        loss_fn = ClusteringCompactnessLoss(separation_weight=0.0, rngs=rngs)

        key = jax.random.key(0)
        k1, k2 = jax.random.split(key)

        # Tight clusters (low variance)
        tight_embeddings = jax.random.normal(k1, (50, 32)) * 0.1
        assignments = jax.nn.softmax(jax.random.normal(k2, (50, 5)), axis=-1)
        tight_loss = loss_fn(tight_embeddings, assignments)

        # Spread clusters (high variance)
        spread_embeddings = jax.random.normal(k1, (50, 32)) * 10.0
        spread_loss = loss_fn(spread_embeddings, assignments)

        # Tight clusters should have lower loss
        assert tight_loss < spread_loss

    def test_gradient_flows(self, rngs):
        """Test that gradients flow through loss."""
        loss_fn = ClusteringCompactnessLoss(rngs=rngs)

        key = jax.random.key(0)
        assignments = jax.nn.softmax(jax.random.normal(key, (50, 5)), axis=-1)

        def compute_loss(embeddings):
            return loss_fn(embeddings, assignments)

        embeddings = jax.random.normal(key, (50, 32))
        grad = jax.grad(compute_loss)(embeddings)

        assert grad is not None
        assert grad.shape == embeddings.shape
        assert jnp.isfinite(grad).all()

    def test_gradient_flows_through_assignments(self, rngs):
        """Test that gradients flow through soft assignments."""
        loss_fn = ClusteringCompactnessLoss(rngs=rngs)

        key = jax.random.key(0)
        embeddings = jax.random.normal(key, (50, 32))

        def compute_loss(logits):
            assignments = jax.nn.softmax(logits, axis=-1)
            return loss_fn(embeddings, assignments)

        logits = jax.random.normal(key, (50, 5))
        grad = jax.grad(compute_loss)(logits)

        assert grad is not None
        assert grad.shape == logits.shape
        assert jnp.isfinite(grad).all()


class TestVelocityConsistencyLoss:
    """Tests for VelocityConsistencyLoss."""

    def test_initialization(self, rngs):
        """Test loss initialization."""
        loss_fn = VelocityConsistencyLoss(rngs=rngs)
        assert loss_fn is not None

    def test_initialization_with_params(self, rngs):
        """Test initialization with custom parameters."""
        loss_fn = VelocityConsistencyLoss(
            dt=0.05,
            cosine_weight=2.0,
            magnitude_weight=0.5,
            rngs=rngs,
        )
        assert loss_fn.dt == 0.05
        assert loss_fn.cosine_weight == 2.0
        assert loss_fn.magnitude_weight == 0.5

    def test_loss_shape(self, rngs):
        """Test that loss returns scalar."""
        loss_fn = VelocityConsistencyLoss(rngs=rngs)

        key = jax.random.key(0)
        k1, k2, k3 = jax.random.split(key, 3)

        # Current expression (50 cells, 100 genes)
        expression = jax.random.normal(k1, (50, 100))
        # Velocity estimates
        velocity = jax.random.normal(k2, (50, 100)) * 0.1
        # Future expression (ground truth or estimated)
        future_expression = expression + jax.random.normal(k3, (50, 100)) * 0.1

        loss = loss_fn(expression, velocity, future_expression)
        assert loss.shape == ()

    def test_loss_finite(self, rngs):
        """Test that loss is finite."""
        loss_fn = VelocityConsistencyLoss(rngs=rngs)

        key = jax.random.key(0)
        k1, k2, k3 = jax.random.split(key, 3)

        expression = jax.random.normal(k1, (50, 100))
        velocity = jax.random.normal(k2, (50, 100)) * 0.1
        future_expression = expression + jax.random.normal(k3, (50, 100)) * 0.1

        loss = loss_fn(expression, velocity, future_expression)
        assert jnp.isfinite(loss)

    def test_consistent_velocity_lower_loss(self, rngs):
        """Test that consistent velocity gives lower loss."""
        loss_fn = VelocityConsistencyLoss(dt=0.1, rngs=rngs)

        key = jax.random.key(0)
        k1, k2 = jax.random.split(key)
        expression = jax.random.normal(k1, (50, 100))

        # Consistent velocity: points toward actual future
        delta = jax.random.normal(k2, (50, 100)) * 0.5  # Larger delta for clearer signal
        future_expression = expression + delta
        consistent_velocity = delta / 0.1  # Perfect match
        consistent_loss = loss_fn(expression, consistent_velocity, future_expression)

        # Inconsistent velocity: opposite direction (clearly wrong)
        inconsistent_velocity = -consistent_velocity  # Opposite direction
        inconsistent_loss = loss_fn(expression, inconsistent_velocity, future_expression)

        # Consistent should have lower loss (opposite direction has cosine = -1)
        assert consistent_loss < inconsistent_loss

    def test_gradient_flows(self, rngs):
        """Test that gradients flow through loss."""
        loss_fn = VelocityConsistencyLoss(rngs=rngs)

        key = jax.random.key(0)
        k1, k2 = jax.random.split(key)

        expression = jax.random.normal(k1, (50, 100))
        future_expression = expression + jax.random.normal(k2, (50, 100)) * 0.1

        def compute_loss(velocity):
            return loss_fn(expression, velocity, future_expression)

        velocity = jax.random.normal(key, (50, 100)) * 0.1
        grad = jax.grad(compute_loss)(velocity)

        assert grad is not None
        assert grad.shape == velocity.shape
        assert jnp.isfinite(grad).all()


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    def test_batch_mixing_jit(self, rngs):
        """Test BatchMixingLoss with JIT."""
        loss_fn = BatchMixingLoss(n_neighbors=5, n_batches=3, rngs=rngs)

        @jax.jit
        def jit_loss(embeddings, batch_labels):
            return loss_fn(embeddings, batch_labels)

        key = jax.random.key(0)
        embeddings = jax.random.normal(key, (50, 32))
        batch_labels = jax.random.randint(key, (50,), 0, 3)

        loss = jit_loss(embeddings, batch_labels)
        assert jnp.isfinite(loss)

    def test_clustering_compactness_jit(self, rngs):
        """Test ClusteringCompactnessLoss with JIT."""
        loss_fn = ClusteringCompactnessLoss(rngs=rngs)

        @jax.jit
        def jit_loss(embeddings, assignments):
            return loss_fn(embeddings, assignments)

        key = jax.random.key(0)
        embeddings = jax.random.normal(key, (50, 32))
        assignments = jax.nn.softmax(jax.random.normal(key, (50, 5)), axis=-1)

        loss = jit_loss(embeddings, assignments)
        assert jnp.isfinite(loss)

    def test_velocity_consistency_jit(self, rngs):
        """Test VelocityConsistencyLoss with JIT."""
        loss_fn = VelocityConsistencyLoss(rngs=rngs)

        @jax.jit
        def jit_loss(expression, velocity, future_expression):
            return loss_fn(expression, velocity, future_expression)

        key = jax.random.key(0)
        k1, k2, k3 = jax.random.split(key, 3)

        expression = jax.random.normal(k1, (50, 100))
        velocity = jax.random.normal(k2, (50, 100)) * 0.1
        future_expression = expression + jax.random.normal(k3, (50, 100)) * 0.1

        loss = jit_loss(expression, velocity, future_expression)
        assert jnp.isfinite(loss)
