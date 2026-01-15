"""Training utilities for differentiable bioinformatics pipelines.

This module provides training loops and utilities for end-to-end gradient-based
optimization of DiffBio pipelines using Flax NNX patterns.
"""

from collections.abc import Iterator
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
        data_iterator: Iterator[tuple[dict[str, Array], dict[str, Array]]],
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
) -> Iterator[tuple[dict[str, Array], dict[str, Array]]]:
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


def create_realistic_training_data(
    num_samples: int = 100,
    num_reads: int = 20,
    read_length: int = 50,
    reference_length: int = 100,
    variant_rate: float = 0.05,
    heterozygous_rate: float = 0.5,
    error_rate: float = 0.01,
    seed: int = 42,
) -> tuple[list[dict[str, Array]], list[dict[str, Array]]]:
    """Create realistic synthetic training data for variant calling.

    Unlike `create_synthetic_training_data`, this function generates reads that
    actually contain variants at the labeled positions, making it possible for
    models to learn meaningful patterns.

    Features:
    - SNP simulation: Substitutes reference bases with alternate alleles
    - Heterozygous/homozygous modeling: Controls allele frequency in reads
    - Quality modeling: Position-dependent quality (higher in read center)
    - Sequencing errors: Random substitutions with low quality scores
    - Strand information: Assigns reads to forward/reverse strands

    Args:
        num_samples: Number of samples to generate
        num_reads: Number of reads per sample
        read_length: Length of each read
        reference_length: Length of reference sequence
        variant_rate: Probability of variant at each position (default 0.05)
        heterozygous_rate: Fraction of variants that are heterozygous (default 0.5)
        error_rate: Probability of sequencing error per base (default 0.01)
        seed: Random seed

    Returns:
        Tuple of (inputs, targets) where:
        - inputs: List of dicts with reads, positions, quality, strand
        - targets: List of dicts with labels (0=ref, 1=snp, 2=indel),
                   variant_alleles, zygosity
    """
    key = jax.random.PRNGKey(seed)
    inputs = []
    targets = []

    for _ in range(num_samples):
        key, *keys = jax.random.split(key, 9)
        k_ref, k_var, k_type, k_pos, k_het, k_alt, k_read_var, k_strand = keys

        # Generate reference sequence (A=0, C=1, G=2, T=3)
        ref_indices = jax.random.randint(k_ref, (reference_length,), 0, 4)
        ref_one_hot = jax.nn.one_hot(ref_indices, 4)

        # Generate variant positions and types
        variant_mask = jax.random.uniform(k_var, (reference_length,)) < variant_rate
        # Variant types: 1=SNP, 2=deletion (simplified - no insertions for now)
        # Focus on SNPs (type 1) for better learning signal
        variant_types = jnp.where(
            variant_mask,
            jnp.where(
                jax.random.uniform(k_type, (reference_length,)) < 0.9,
                1,  # SNP (90%)
                2,  # Deletion (10%)
            ),
            0,  # Reference
        )

        # Determine zygosity for each variant (0=hom, 1=het)
        is_heterozygous = jax.random.uniform(k_het, (reference_length,)) < heterozygous_rate

        # Generate alternate alleles for SNPs (different from reference)
        # For each position, pick a random base that's different from reference
        alt_offsets = jax.random.randint(k_alt, (reference_length,), 1, 4)
        alt_indices = (ref_indices + alt_offsets) % 4
        alt_one_hot = jax.nn.one_hot(alt_indices, 4)

        # Generate read positions
        positions = jax.random.randint(k_pos, (num_reads,), 0, reference_length - read_length)

        # Determine which reads carry variant allele (for heterozygous sites)
        # For homozygous variants: all reads show variant
        # For heterozygous variants: ~50% of reads show variant
        read_shows_variant = jax.random.uniform(k_read_var, (num_reads,)) < 0.5

        # Generate strand assignments (0=forward, 1=reverse)
        strands = jax.random.randint(k_strand, (num_reads,), 0, 2)

        # Build reads with actual variants
        def build_read(read_idx):
            pos = positions[read_idx]
            shows_var = read_shows_variant[read_idx]

            # Get reference segment for this read
            def get_base_at(offset):
                ref_pos = pos + offset
                ref_base = ref_one_hot[ref_pos]
                alt_base = alt_one_hot[ref_pos]

                # Check if this position is a variant
                is_var = variant_types[ref_pos] == 1  # SNP
                is_het = is_heterozygous[ref_pos]

                # Use variant allele if:
                # - Position is a variant AND
                # - (homozygous OR (heterozygous AND this read shows variant))
                use_variant = is_var & (~is_het | shows_var)

                return jnp.where(use_variant, alt_base, ref_base)

            # Build read base by base
            read_bases = jax.vmap(get_base_at)(jnp.arange(read_length))
            return read_bases

        reads = jax.vmap(build_read)(jnp.arange(num_reads))

        # Add sequencing errors (random substitutions at error_rate)
        key, k_err_pos, k_err_base = jax.random.split(key, 3)
        error_mask = jax.random.uniform(k_err_pos, (num_reads, read_length)) < error_rate

        # Generate random error bases
        error_offsets = jax.random.randint(k_err_base, (num_reads, read_length), 1, 4)
        current_bases = jnp.argmax(reads, axis=-1)  # Get current base indices
        error_bases = (current_bases + error_offsets) % 4
        error_one_hot = jax.nn.one_hot(error_bases, 4)

        # Apply errors
        reads = jnp.where(error_mask[..., None], error_one_hot, reads)

        # Generate quality scores with position-dependent profile
        # Quality is higher in middle of read, lower at ends
        key, k_qual_noise = jax.random.split(key, 2)

        # Position-dependent base quality (parabolic profile)
        pos_in_read = jnp.arange(read_length)
        center = read_length / 2
        # Quality ranges from 25 at ends to 35 in middle
        base_quality = 35.0 - 10.0 * ((pos_in_read - center) / center) ** 2
        base_quality = jnp.broadcast_to(base_quality, (num_reads, read_length))

        # Add random variation
        quality_noise = jax.random.uniform(
            k_qual_noise,
            (num_reads, read_length),
            minval=-3.0,
            maxval=3.0,
        )
        quality = base_quality + quality_noise

        # Lower quality at error positions
        quality = jnp.where(error_mask, jnp.minimum(quality, 15.0), quality)

        # Slightly lower quality near variant positions in reads
        def check_variant_overlap(read_idx):
            pos = positions[read_idx]

            # Get variant status for each position in this read
            def is_var_at(offset):
                ref_pos = pos + offset
                return variant_types[ref_pos] > 0

            return jax.vmap(is_var_at)(jnp.arange(read_length))

        variant_in_read = jax.vmap(check_variant_overlap)(jnp.arange(num_reads))
        quality = jnp.where(variant_in_read, quality - 2.0, quality)

        # Clamp quality to valid range
        quality = jnp.clip(quality, 5.0, 40.0)

        inputs.append(
            {
                "reads": reads,
                "positions": positions,
                "quality": quality,
                "strand": strands,
            }
        )
        targets.append(
            {
                "labels": variant_types,
                "variant_alleles": alt_indices,
                "is_heterozygous": is_heterozygous.astype(jnp.int32),
            }
        )

    return inputs, targets


# Backwards compatibility alias
create_optimizer = create_optax_optimizer
