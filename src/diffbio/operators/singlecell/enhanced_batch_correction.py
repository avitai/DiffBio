"""Enhanced differentiable batch correction operators using MMD and WGAN.

This module provides two neural-network-based batch correction strategies:

- **DifferentiableMMDBatchCorrection**: Autoencoder with Maximum Mean Discrepancy
  (MMD) regularisation that penalises distributional differences between batches
  in latent space.
- **DifferentiableWGANBatchCorrection**: Adversarial autoencoder with a Wasserstein
  GAN discriminator that learns batch-invariant latent representations through
  gradient reversal.

Design references:
- scGPT gradient reversal for adversarial batch correction.
- scVI batch encoding as one-hot or embedding.
- scGPT domain-specific batch normalisation.

Both operators inherit from ``OperatorModule`` and follow the standard
``apply(data, state, metadata)`` interface for pipeline composability.
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from artifex.generative_models.core.base import MLP
from artifex.generative_models.core.losses.adversarial import (
    wasserstein_discriminator_loss,
    wasserstein_generator_loss,
)
from artifex.generative_models.core.losses.base import reduce_loss
from artifex.generative_models.core.losses.divergence import maximum_mean_discrepancy
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, Int, PyTree

from diffbio.operators._loss_balancing import LossBalancingMixin
from diffbio.utils.nn_utils import ensure_rngs

logger = logging.getLogger(__name__)

__all__ = [
    "DifferentiableMMDBatchCorrection",
    "DifferentiableWGANBatchCorrection",
    "MMDBatchCorrectionConfig",
    "WGANBatchCorrectionConfig",
]


# ---------------------------------------------------------------------------
# Gradient reversal primitive (JAX custom_vjp)
# ---------------------------------------------------------------------------


@jax.custom_vjp
def _gradient_reversal(x: jax.Array, scale: float) -> jax.Array:
    """Pass-through forward, negate gradients backward."""
    return x


def _gradient_reversal_fwd(x: jax.Array, scale: float) -> tuple[jax.Array, float]:
    """Forward pass for gradient reversal."""
    return x, scale


def _gradient_reversal_bwd(scale: float, grad: jax.Array) -> tuple[jax.Array, None]:
    """Backward pass: negate and scale the gradient."""
    return (-scale * grad, None)


_gradient_reversal.defvjp(_gradient_reversal_fwd, _gradient_reversal_bwd)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MMDBatchCorrectionConfig(OperatorConfig):
    """Configuration for MMD-based batch correction.

    Attributes:
        n_genes: Number of input genes (features).
        hidden_dim: Width of hidden layers in the autoencoder.
        latent_dim: Dimensionality of the latent space.
        kernel_bandwidth: Bandwidth for the RBF kernel in the MMD loss.
        use_gradnorm: Whether to use GradNormBalancer for multi-task loss
            balancing between reconstruction and MMD losses.
    """

    n_genes: int = 2000
    hidden_dim: int = 128
    latent_dim: int = 64
    kernel_bandwidth: float = 1.0
    use_gradnorm: bool = False


@dataclass(frozen=True)
class WGANBatchCorrectionConfig(OperatorConfig):
    """Configuration for WGAN-based batch correction.

    Attributes:
        n_genes: Number of input genes (features).
        hidden_dim: Width of hidden layers in the generator autoencoder.
        latent_dim: Dimensionality of the latent space.
        discriminator_hidden_dim: Width of hidden layers in the discriminator.
        use_gradnorm: Whether to use GradNormBalancer for multi-task loss
            balancing between generator and discriminator losses.
    """

    n_genes: int = 2000
    hidden_dim: int = 128
    latent_dim: int = 64
    discriminator_hidden_dim: int = 64
    use_gradnorm: bool = False


# ---------------------------------------------------------------------------
# MMD batch correction
# ---------------------------------------------------------------------------


class DifferentiableMMDBatchCorrection(LossBalancingMixin, OperatorModule):
    """Autoencoder batch correction with MMD regularisation.

    Architecture:
        Encoder MLP maps gene expression to a latent representation, and a
        decoder MLP reconstructs the expression from that latent.  The MMD
        loss penalises distributional mismatch between batches in latent
        space so the learned representation becomes batch-invariant.

    Loss:
        ``reconstruction_mse + mmd(latent_batch_0, latent_batch_1, ...)``

    Args:
        config: MMDBatchCorrectionConfig with model hyper-parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = MMDBatchCorrectionConfig(n_genes=2000)
        >>> op = DifferentiableMMDBatchCorrection(config, rngs=nnx.Rngs(0))
        >>> result, _, _ = op.apply(data, {}, None)
    """

    def __init__(
        self,
        config: MMDBatchCorrectionConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialise encoder and decoder MLPs.

        Args:
            config: Operator configuration.
            rngs: Random number generators for weight initialisation.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        rngs = ensure_rngs(rngs)

        self.encoder = MLP(
            hidden_dims=[config.hidden_dim, config.hidden_dim, config.latent_dim],
            in_features=config.n_genes,
            activation="relu",
            use_batch_norm=False,
            rngs=rngs,
        )

        self.decoder = MLP(
            hidden_dims=[config.hidden_dim, config.hidden_dim, config.n_genes],
            in_features=config.latent_dim,
            activation="relu",
            use_batch_norm=False,
            rngs=rngs,
        )

    # -- helpers --------------------------------------------------------------

    def _encode(
        self, expression: Float[Array, "n_cells n_genes"]
    ) -> Float[Array, "n_cells latent_dim"]:
        """Map expression to latent space.

        Args:
            expression: Input gene expression matrix.

        Returns:
            Latent representation.
        """
        latent: jax.Array = self.encoder(expression)
        return latent

    def _decode(
        self, latent: Float[Array, "n_cells latent_dim"]
    ) -> Float[Array, "n_cells n_genes"]:
        """Reconstruct expression from latent space.

        Args:
            latent: Latent representation.

        Returns:
            Reconstructed gene expression.
        """
        reconstruction: jax.Array = self.decoder(latent)
        return reconstruction

    def _compute_pairwise_mmd(
        self,
        latent: Float[Array, "n_cells latent_dim"],
        batch_labels: Int[Array, "n_cells"],
    ) -> Float[Array, ""]:
        """Compute MMD between batch-0 and non-batch-0 cells in latent space.

        Uses masked mean embeddings to remain fully JIT-compatible (no boolean
        indexing).  For multi-batch data the comparison is batch-0 vs the rest,
        which encourages all batches to align in latent space.  When only one
        batch is present the two groups are identical and MMD is near zero.

        Args:
            latent: Latent representations for all cells.
            batch_labels: Integer batch assignments per cell.

        Returns:
            Scalar MMD loss.
        """
        n_cells = latent.shape[0]

        # Soft masks (JIT-safe: no boolean indexing)
        is_batch0 = (batch_labels == 0).astype(jnp.float32)  # (n_cells,)
        is_other = 1.0 - is_batch0

        n_batch0 = jnp.maximum(is_batch0.sum(), 1.0)
        n_other = jnp.maximum(is_other.sum(), 1.0)

        # Weighted latent: zero out cells not in the group, then stack as
        # (1, n_cells, latent_dim) for the MMD function which expects
        # [batch, samples, features].
        latent_batch0 = latent * is_batch0[:, None]  # (n_cells, d)
        latent_other = latent * is_other[:, None]

        # Normalize so the masked-out zeros don't bias the kernel.
        # Scale the active entries up by n_cells / n_active so that the
        # effective sample still fills the (n_cells, d) array.
        latent_batch0 = latent_batch0 * (n_cells / n_batch0)
        latent_other = latent_other * (n_cells / n_other)

        return maximum_mean_discrepancy(
            latent_batch0[None, ...],  # (1, n_cells, d)
            latent_other[None, ...],  # (1, n_cells, d)
            kernel_type="rbf",
            kernel_bandwidth=self.config.kernel_bandwidth,
            reduction="mean",
        )

    # -- apply ----------------------------------------------------------------

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Encode, decode, and compute MMD + reconstruction losses.

        Args:
            data: Dictionary containing:
                - ``"expression"``: Gene expression matrix ``(n_cells, n_genes)``
                - ``"batch_labels"``: Integer batch assignments ``(n_cells,)``
            state: Pipeline state (passed through unchanged).
            metadata: Pipeline metadata (passed through unchanged).
            random_params: Unused.
            stats: Unused.

        Returns:
            Tuple of ``(result, state, metadata)`` where *result* contains:
                - ``"expression"``: Original expression
                - ``"batch_labels"``: Original batch labels
                - ``"corrected_expression"``: Decoded (corrected) expression
                - ``"latent"``: Latent representation
                - ``"mmd_loss"``: Scalar MMD loss between batches
                - ``"reconstruction_loss"``: Scalar MSE reconstruction loss
        """
        expression = data["expression"]
        batch_labels = data["batch_labels"]

        # Forward pass
        latent = self._encode(expression)
        reconstructed = self._decode(latent)

        # Losses
        reconstruction_loss = reduce_loss(
            (reconstructed - expression) ** 2,
            reduction="mean",
        )
        mmd_loss = self._compute_pairwise_mmd(latent, batch_labels)

        result = {
            **data,
            "corrected_expression": reconstructed,
            "latent": latent,
            "mmd_loss": mmd_loss,
            "reconstruction_loss": reconstruction_loss,
        }
        return result, state, metadata


# ---------------------------------------------------------------------------
# WGAN batch correction
# ---------------------------------------------------------------------------


class DifferentiableWGANBatchCorrection(LossBalancingMixin, OperatorModule):
    """Adversarial autoencoder batch correction with Wasserstein GAN loss.

    Architecture:
        An encoder (generator) maps gene expression to a batch-invariant
        latent space and a decoder reconstructs the expression.  A separate
        discriminator tries to predict the batch label from the latent
        representation.  Gradient reversal (a la scGPT) ensures the encoder
        learns to *fool* the discriminator, yielding batch-invariant latents.

    Losses:
        - ``generator_loss``: Wasserstein generator loss (encoder wants to
          fool the discriminator) plus reconstruction MSE.
        - ``discriminator_loss``: Wasserstein discriminator/critic loss.

    Args:
        config: WGANBatchCorrectionConfig with model hyper-parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = WGANBatchCorrectionConfig(n_genes=2000)
        >>> op = DifferentiableWGANBatchCorrection(config, rngs=nnx.Rngs(0))
        >>> result, _, _ = op.apply(data, {}, None)
    """

    def __init__(
        self,
        config: WGANBatchCorrectionConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialise encoder, decoder, and discriminator MLPs.

        Args:
            config: Operator configuration.
            rngs: Random number generators for weight initialisation.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        rngs = ensure_rngs(rngs)

        self.encoder = MLP(
            hidden_dims=[config.hidden_dim, config.hidden_dim, config.latent_dim],
            in_features=config.n_genes,
            activation="relu",
            use_batch_norm=False,
            rngs=rngs,
        )

        self.decoder = MLP(
            hidden_dims=[config.hidden_dim, config.hidden_dim, config.n_genes],
            in_features=config.latent_dim,
            activation="relu",
            use_batch_norm=False,
            rngs=rngs,
        )

        self.discriminator = MLP(
            hidden_dims=[
                config.discriminator_hidden_dim,
                config.discriminator_hidden_dim,
                1,
            ],
            in_features=config.latent_dim,
            activation="relu",
            use_batch_norm=False,
            rngs=rngs,
        )

    # -- helpers --------------------------------------------------------------

    def _encode(
        self, expression: Float[Array, "n_cells n_genes"]
    ) -> Float[Array, "n_cells latent_dim"]:
        """Map expression to latent space.

        Args:
            expression: Input gene expression matrix.

        Returns:
            Latent representation.
        """
        latent: jax.Array = self.encoder(expression)
        return latent

    def _decode(
        self, latent: Float[Array, "n_cells latent_dim"]
    ) -> Float[Array, "n_cells n_genes"]:
        """Reconstruct expression from latent space.

        Args:
            latent: Latent representation.

        Returns:
            Reconstructed gene expression.
        """
        reconstruction: jax.Array = self.decoder(latent)
        return reconstruction

    def _discriminate(self, latent: Float[Array, "n_cells latent_dim"]) -> Float[Array, "n_cells"]:
        """Compute discriminator (critic) scores from latent representations.

        Args:
            latent: Latent representation.

        Returns:
            Per-cell scalar critic scores.
        """
        scores: jax.Array = self.discriminator(latent)
        return scores.squeeze(-1)  # (n_cells,)

    # -- apply ----------------------------------------------------------------

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Encode, decode, and compute adversarial + reconstruction losses.

        The discriminator receives latent codes through a gradient reversal
        layer so that encoder gradients push toward batch invariance while
        discriminator gradients push toward better batch classification.

        Args:
            data: Dictionary containing:
                - ``"expression"``: Gene expression matrix ``(n_cells, n_genes)``
                - ``"batch_labels"``: Integer batch assignments ``(n_cells,)``
            state: Pipeline state (passed through unchanged).
            metadata: Pipeline metadata (passed through unchanged).
            random_params: Unused.
            stats: Unused.

        Returns:
            Tuple of ``(result, state, metadata)`` where *result* contains:
                - ``"expression"``: Original expression
                - ``"batch_labels"``: Original batch labels
                - ``"corrected_expression"``: Decoded (corrected) expression
                - ``"latent"``: Latent representation
                - ``"discriminator_scores"``: Per-cell critic scores
                - ``"generator_loss"``: Scalar Wasserstein generator loss
                - ``"discriminator_loss"``: Scalar Wasserstein discriminator loss
        """
        expression = data["expression"]
        batch_labels = data["batch_labels"]

        # Encode -> decode
        latent = self._encode(expression)
        reconstructed = self._decode(latent)

        # Discriminator path: gradient reversal so encoder learns to fool it
        latent_reversed = _gradient_reversal(latent, 1.0)
        disc_scores = self._discriminate(latent_reversed)

        # Reconstruction loss
        reconstruction_loss = reduce_loss(
            (reconstructed - expression) ** 2,
            reduction="mean",
        )

        # Identify "real" (batch 0) and "fake" (batch != 0) for WGAN framing.
        # The discriminator tries to distinguish batch 0 from the rest.
        # We compute masked means manually to stay JIT-compatible (no boolean
        # indexing), then delegate to the library Wasserstein loss functions.
        is_batch0 = (batch_labels == 0).astype(jnp.float32)
        is_other = 1.0 - is_batch0
        n_real = jnp.maximum(is_batch0.sum(), 1.0)
        n_fake = jnp.maximum(is_other.sum(), 1.0)

        # Masked scores: one scalar per group (batch-mean critic output)
        real_mean_score = (disc_scores * is_batch0).sum() / n_real
        fake_mean_score = (disc_scores * is_other).sum() / n_fake

        # Use library Wasserstein losses (each expects 1-D scores)
        generator_loss = wasserstein_generator_loss(fake_mean_score[None]) + reconstruction_loss
        discriminator_loss = wasserstein_discriminator_loss(
            real_mean_score[None],
            fake_mean_score[None],
        )

        result = {
            **data,
            "corrected_expression": reconstructed,
            "latent": latent,
            "discriminator_scores": disc_scores,
            "generator_loss": generator_loss,
            "discriminator_loss": discriminator_loss,
        }
        return result, state, metadata


# ---------------------------------------------------------------------------
# Internal MLP container (nnx.Module so it appears in the grad tree)
# ---------------------------------------------------------------------------
