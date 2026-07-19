"""Joint optimization of the single-cell preprocessing-to-annotation pipeline.

:func:`fit_jointly` unfreezes the whole :class:`JointPreprocessingPipeline` and
co-optimizes the preprocessing parameters (normalization pseudocount and depth,
the highly-variable-gene weights) together with the annotation probe against the
label loss -- the joint-optimization moat. The optimizer is built with the shared
artifex factory and the classification loss is balanced against a gene-weight
sparsity penalty with an opifex ``GradNormBalancer`` (via the DiffBio loss-
balancing seam), so the two objectives are weighted by their training rates rather
than a hand-tuned coefficient.

The DiffBio ``LossBalancingMixin`` seam is a *stateless* combiner -- it builds a
fresh balancer per call and never updates its weights -- so it cannot express the
cross-step adaptive GradNorm this training loop needs. This function therefore
composes ``GradNormBalancer`` directly in the persistent training-loop pattern
(``update_weights`` across steps); it does not reinvent the balancer, only uses it
statefully. The step needs the per-loss gradients for the GradNorm norms anyway, so
the combined update gradient is formed as the weighted sum of those per-loss
gradients (gradients are linear), avoiding a redundant third backward pass.

The frozen mode from the pipeline itself is untouched: joint training is opt-in
through this function and leaves the pipeline's ``apply`` behavior unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from artifex.generative_models.core.configuration.optimizer_config import OptimizerConfig
from artifex.generative_models.training.optimizers.factory import create_optimizer
from flax import nnx
from jax.typing import ArrayLike
from opifex.core.physics.gradnorm import GradNormBalancer, GradNormConfig

from diffbio.losses.singlecell_losses import gene_weight_sparsity_loss
from diffbio.pipelines.joint_preprocessing import JointPreprocessingPipeline
from diffbio.utils.training import cross_entropy_loss

_NUM_LOSSES = 2


@dataclass(frozen=True, kw_only=True, slots=True)
class JointTrainingConfig:
    """Configuration for :func:`fit_jointly`.

    Attributes:
        n_steps: Number of full-batch joint-optimization steps.
        learning_rate: Learning rate for the pipeline optimizer.
        grad_clip_norm: Global-norm gradient clip for the pipeline optimizer.
        gradnorm_alpha: GradNorm asymmetry parameter balancing the two losses.
        seed: Seed for the GradNorm balancer initialization.
    """

    n_steps: int = 200
    learning_rate: float = 1.0e-2
    grad_clip_norm: float = 1.0
    gradnorm_alpha: float = 1.5
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the configuration, failing fast on non-positive sizes.

        Raises:
            ValueError: If ``n_steps`` or ``learning_rate`` is not positive.
        """
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.grad_clip_norm <= 0.0:
            raise ValueError(f"grad_clip_norm must be positive, got {self.grad_clip_norm}")


@dataclass(frozen=True, slots=True)
class JointTrainingResult:
    """Outcome of a :func:`fit_jointly` run.

    Attributes:
        loss_history: Classification loss after each step.
        final_loss_weights: Final GradNorm weights for (classification, sparsity).
    """

    loss_history: tuple[float, ...]
    final_loss_weights: tuple[float, ...]


def _classification_loss(
    pipeline: JointPreprocessingPipeline,
    counts: jnp.ndarray,
    labels: jnp.ndarray,
    n_classes: int,
) -> jnp.ndarray:
    """Cross-entropy of the pipeline's predicted logits against the labels."""
    output, _, _ = pipeline.apply({"counts": counts}, {}, None)
    return cross_entropy_loss(output["logits"], labels, num_classes=n_classes)


def _global_norm(grads: nnx.State) -> jnp.ndarray:
    """Euclidean norm of all gradient leaves in a state tree."""
    return jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(grads)))


def fit_jointly(
    pipeline: JointPreprocessingPipeline,
    counts: ArrayLike,
    labels: ArrayLike,
    *,
    config: JointTrainingConfig | None = None,
) -> JointTrainingResult:
    """Jointly optimize the preprocessing pipeline and probe against the label loss.

    Unfreezes every learnable parameter of ``pipeline`` (normalization pseudocount
    and depth, SoftHVG gene weights, probe head) and trains them as one
    differentiable graph. The classification loss is balanced against a gene-weight
    sparsity penalty by an opifex ``GradNormBalancer``, and updates are applied with
    the artifex optimizer factory. The pipeline is trained in place.

    Args:
        pipeline: The pipeline to optimize (mutated in place).
        counts: Raw ``(n_cells, n_genes)`` count matrix.
        labels: ``(n_cells,)`` integer cell-type labels.
        config: Joint-training configuration; defaults are used when ``None``.

    Returns:
        A :class:`JointTrainingResult` with the per-step classification loss and the
        final GradNorm loss weights.
    """
    config = config or JointTrainingConfig()
    counts = jnp.asarray(counts)
    labels = jnp.asarray(labels)
    n_classes = int(pipeline.config.n_classes)

    optimizer = nnx.Optimizer(
        pipeline,
        create_optimizer(
            OptimizerConfig(
                name="diffbio_joint",
                optimizer_type="adam",
                learning_rate=config.learning_rate,
                gradient_clip_norm=config.grad_clip_norm,
            )
        ),
        wrt=nnx.Param,
    )
    balancer = GradNormBalancer(
        num_losses=_NUM_LOSSES,
        config=GradNormConfig(alpha=config.gradnorm_alpha),
        rngs=nnx.Rngs(config.seed),
    )

    def classification(model: JointPreprocessingPipeline) -> jnp.ndarray:
        return _classification_loss(model, counts, labels, n_classes)

    def sparsity(model: JointPreprocessingPipeline) -> jnp.ndarray:
        return gene_weight_sparsity_loss(model.soft_hvg.gene_weights[...])

    initial_losses = jnp.stack([classification(pipeline), sparsity(pipeline)])

    @nnx.jit
    def train_step(
        model: JointPreprocessingPipeline,
        opt: nnx.Optimizer,
        gradnorm: GradNormBalancer,
    ) -> jnp.ndarray:
        classification_loss, classification_grad = nnx.value_and_grad(classification)(model)
        sparsity_loss, sparsity_grad = nnx.value_and_grad(sparsity)(model)
        current = jnp.stack([classification_loss, sparsity_loss])
        grad_norms = jnp.stack([_global_norm(classification_grad), _global_norm(sparsity_grad)])
        gradnorm.update_weights(grad_norms, current, initial_losses)

        weights = gradnorm.weights
        combined_grad = jax.tree.map(
            lambda cls_g, sparse_g: weights[0] * cls_g + weights[1] * sparse_g,
            classification_grad,
            sparsity_grad,
        )
        opt.update(model, combined_grad)
        return classification_loss

    loss_history: list[float] = []
    for _ in range(config.n_steps):
        loss_history.append(float(train_step(pipeline, optimizer, balancer)))

    return JointTrainingResult(
        loss_history=tuple(loss_history),
        final_loss_weights=tuple(np.asarray(balancer.weights).tolist()),
    )
