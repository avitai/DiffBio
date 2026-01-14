"""Training utilities for differentiable bioinformatics pipelines.

This module provides training loops and utilities for end-to-end gradient-based
optimization of DiffBio pipelines using Flax NNX patterns.
"""

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float


@dataclass
class TrainingConfig:
    """Configuration for training loop.

    Attributes:
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        log_every: Log metrics every N steps
        grad_clip_norm: Maximum gradient norm (None to disable)
    """

    learning_rate: float = 1e-3
    num_epochs: int = 100
    log_every: int = 10
    grad_clip_norm: float | None = 1.0


@dataclass
class TrainingState:
    """State maintained during training.

    Attributes:
        step: Current training step
        epoch: Current epoch
        loss_history: List of loss values
        best_loss: Best loss seen so far
    """

    step: int = 0
    epoch: int = 0
    loss_history: list[float] | None = None
    best_loss: float = float("inf")

    def __post_init__(self):
        if self.loss_history is None:
            self.loss_history = []


def create_optax_optimizer(
    config: TrainingConfig,
) -> optax.GradientTransformation:
    """Create optax optimizer with optional gradient clipping.

    Args:
        config: Training configuration

    Returns:
        Optax optimizer
    """
    transforms = []

    if config.grad_clip_norm is not None:
        transforms.append(optax.clip_by_global_norm(config.grad_clip_norm))

    transforms.append(optax.adam(config.learning_rate))

    return optax.chain(*transforms)


def cross_entropy_loss(
    logits: Float[Array, "... num_classes"],
    labels: Float[Array, "..."],
    num_classes: int = 3,
) -> Float[Array, ""]:
    """Compute cross-entropy loss for variant classification.

    Args:
        logits: Raw model predictions
        labels: Integer class labels
        num_classes: Number of classes

    Returns:
        Scalar loss value
    """
    one_hot_labels = jax.nn.one_hot(labels.astype(jnp.int32), num_classes)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(one_hot_labels * log_probs, axis=-1))


class Trainer:
    """Training loop for DiffBio pipelines using Flax NNX patterns.

    This class handles the training loop using NNX's stateful approach:
    - Uses nnx.Optimizer for automatic parameter updates
    - Uses @nnx.jit for JIT compilation with state management
    - Supports gradient clipping and metric logging

    Example:
        >>> pipeline = create_variant_calling_pipeline(reference_length=100)
        >>> trainer = Trainer(pipeline, TrainingConfig(learning_rate=1e-3))
        >>>
        >>> # Define loss function
        >>> def loss_fn(predictions, targets):
        ...     return cross_entropy_loss(
        ...         predictions["logits"],
        ...         targets["labels"],
        ...     )
        >>>
        >>> # Train
        >>> trainer.train(data_iterator_fn, loss_fn)
        >>> trained_pipeline = trainer.pipeline
    """

    def __init__(
        self,
        pipeline: OperatorModule,
        config: TrainingConfig,
    ):
        """Initialize trainer.

        Args:
            pipeline: Pipeline to train
            config: Training configuration
        """
        self.pipeline = pipeline
        self.config = config

        # Create NNX optimizer (holds mutable reference to model)
        optax_opt = create_optax_optimizer(config)
        self.optimizer = nnx.Optimizer(pipeline, optax_opt, wrt=nnx.Param)

        # Training state
        self.training_state = TrainingState()

    def _create_train_step(
        self,
        loss_fn: Callable[[dict[str, Array], dict[str, Array]], Float[Array, ""]],
    ):
        """Create JIT-compiled training step using NNX patterns.

        Args:
            loss_fn: Loss function taking (predictions, targets) and returning scalar

        Returns:
            JIT-compiled training step function
        """

        @nnx.jit
        def train_step(
            model: OperatorModule,
            optimizer: nnx.Optimizer,
            batch_data: dict[str, Array],
            targets: dict[str, Array],
        ) -> tuple[Float[Array, ""], dict[str, Any]]:
            """Single training step with NNX state management.

            Args:
                model: The pipeline model
                optimizer: NNX optimizer
                batch_data: Input batch data
                targets: Target labels

            Returns:
                Tuple of (loss, metrics)
            """

            def compute_loss(model_inner: OperatorModule):
                # Apply pipeline
                result_data, _, _ = model_inner.apply(batch_data, {}, None)
                # Compute loss
                loss = loss_fn(result_data, targets)
                return loss

            # Compute loss and gradients
            loss, grads = nnx.value_and_grad(compute_loss)(model)

            # Update parameters in-place via optimizer
            # As of Flax 0.11.0, update requires both model and grads
            optimizer.update(model, grads)

            # Compute gradient norm for metrics
            grad_leaves = jax.tree.leaves(nnx.state(grads, nnx.Param))
            if grad_leaves:
                grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in grad_leaves))
            else:
                grad_norm = jnp.array(0.0)

            metrics = {
                "loss": loss,
                "grad_norm": grad_norm,
            }

            return loss, metrics

        return train_step

    def train_epoch(
        self,
        data_iterator,
        loss_fn: Callable,
    ) -> dict[str, float]:
        """Train for one epoch.

        Args:
            data_iterator: Iterator yielding (batch_data, targets) tuples
            loss_fn: Loss function

        Returns:
            Dict of epoch metrics
        """
        epoch_losses = []
        train_step = self._create_train_step(loss_fn)

        for batch_data, targets in data_iterator:
            # Run training step (updates model in-place via optimizer)
            loss, metrics = train_step(
                self.pipeline,
                self.optimizer,
                batch_data,
                targets,
            )

            epoch_losses.append(float(loss))
            self.training_state.step += 1
            self.training_state.loss_history.append(float(loss))

            # Log progress
            if self.training_state.step % self.config.log_every == 0:
                print(
                    f"Step {self.training_state.step}: "
                    f"loss={float(loss):.4f}, "
                    f"grad_norm={float(metrics['grad_norm']):.4f}"
                )

        # Update best loss
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        if avg_loss < self.training_state.best_loss:
            self.training_state.best_loss = avg_loss

        return {
            "avg_loss": avg_loss,
            "min_loss": min(epoch_losses) if epoch_losses else 0,
            "max_loss": max(epoch_losses) if epoch_losses else 0,
        }

    def train(
        self,
        data_iterator_fn: Callable,
        loss_fn: Callable,
    ) -> None:
        """Run full training loop.

        After training, the pipeline is updated in-place with trained parameters.
        Access via trainer.pipeline.

        Args:
            data_iterator_fn: Function that returns a fresh data iterator
            loss_fn: Loss function
        """
        # Set pipeline to training mode
        if hasattr(self.pipeline, "set_training"):
            self.pipeline.set_training(True)
        elif hasattr(self.pipeline, "train_mode"):
            self.pipeline.train_mode()

        for epoch in range(self.config.num_epochs):
            self.training_state.epoch = epoch
            data_iterator = data_iterator_fn()

            metrics = self.train_epoch(data_iterator, loss_fn)

            print(f"Epoch {epoch + 1}/{self.config.num_epochs}: avg_loss={metrics['avg_loss']:.4f}")

        # Set back to eval mode
        if hasattr(self.pipeline, "set_training"):
            self.pipeline.set_training(False)
        elif hasattr(self.pipeline, "eval_mode"):
            self.pipeline.eval_mode()


def create_synthetic_training_data(
    num_samples: int = 100,
    num_reads: int = 10,
    read_length: int = 50,
    reference_length: int = 100,
    variant_rate: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Array]], list[dict[str, Array]]]:
    """Create synthetic training data for variant calling.

    Generates reads with simulated variants for training.

    Args:
        num_samples: Number of samples to generate
        num_reads: Number of reads per sample
        read_length: Length of each read
        reference_length: Length of reference sequence
        variant_rate: Probability of variant at each position
        seed: Random seed

    Returns:
        Tuple of (inputs, targets) where:
        - inputs: List of dicts with reads, positions, quality
        - targets: List of dicts with labels (0=ref, 1=snp, 2=indel)
    """
    key = jax.random.PRNGKey(seed)
    inputs = []
    targets = []

    for i in range(num_samples):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)

        # Generate reference sequence
        ref_indices = jax.random.randint(k1, (reference_length,), 0, 4)
        ref_one_hot = jax.nn.one_hot(ref_indices, 4)

        # Generate variant labels
        variant_mask = jax.random.uniform(k2, (reference_length,)) < variant_rate
        labels = jnp.where(
            variant_mask,
            jax.random.randint(k3, (reference_length,), 1, 3),  # SNP or indel
            0,  # Reference
        )

        # Generate reads from reference (with variants)
        positions = jax.random.randint(k4, (num_reads,), 0, reference_length - read_length)

        # Extract read segments
        def get_read(pos):
            segment = jax.lax.dynamic_slice(ref_one_hot, (pos, 0), (read_length, 4))
            return segment

        reads = jax.vmap(get_read)(positions)

        # Add noise to reads at variant positions
        key, k5 = jax.random.split(key)
        noise = jax.random.normal(k5, reads.shape) * 0.1
        reads = reads + noise
        reads = jax.nn.softmax(reads, axis=-1)  # Renormalize

        # Generate quality scores
        key, k6 = jax.random.split(key)
        quality = jax.random.uniform(k6, (num_reads, read_length), minval=20.0, maxval=40.0)

        inputs.append(
            {
                "reads": reads,
                "positions": positions,
                "quality": quality,
            }
        )
        targets.append(
            {
                "labels": labels,
            }
        )

    return inputs, targets


def data_iterator(
    inputs: list[dict[str, Array]],
    targets: list[dict[str, Array]],
    batch_size: int = 1,
):
    """Create an iterator over training data.

    Args:
        inputs: List of input dicts
        targets: List of target dicts
        batch_size: Batch size (currently only supports 1)

    Yields:
        Tuples of (batch_data, targets)
    """
    # For simplicity, yield one sample at a time
    yield from zip(inputs, targets)


# Backwards compatibility alias
create_optimizer = create_optax_optimizer
