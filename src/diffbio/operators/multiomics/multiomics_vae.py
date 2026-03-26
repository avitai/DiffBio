"""Multi-omics VAE with Product-of-Experts fusion.

This module implements a differentiable multi-omics variational autoencoder
that jointly integrates data from multiple modalities (e.g., RNA-seq, ATAC-seq,
protein). The model learns a shared latent space via Product-of-Experts (PoE)
fusion of per-modality posterior distributions, following the approach
described in MULTIVI (Ashuach et al., 2023).

Key algorithm:
    1. Per-modality encoders map counts to (mu_m, logvar_m).
    2. PoE fuses posteriors:
       precision_joint = sum(1 / sigma_m^2)
       mu_joint = (sum mu_m / sigma_m^2) / precision_joint
    3. Reparameterised sample z ~ N(mu_joint, 1/precision_joint).
    4. Per-modality decoders reconstruct counts from z.
    5. ELBO = sum_m w_m * recon_loss_m + KL(q(z) || N(0,I)).
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from artifex.generative_models.core.losses.base import reduce_loss
from artifex.generative_models.core.losses.divergence import gaussian_kl_divergence
from datarax.core.config import OperatorConfig
from flax import nnx
from jaxtyping import Array, Float, PyTree

from opifex.core.physics.gradnorm import GradNormBalancer

from diffbio.constants import EPSILON
from diffbio.core.base_operators import EncoderDecoderOperator
from diffbio.utils.nn_utils import ensure_rngs

logger = logging.getLogger(__name__)

# Canonical key names for the two most common modalities.
_DEFAULT_MODALITY_KEYS = ("rna", "atac")


@dataclass(frozen=True)
class MultiOmicsVAEConfig(OperatorConfig):
    """Configuration for DifferentiableMultiOmicsVAE.

    Attributes:
        modality_dims: Feature dimension for each modality.
        latent_dim: Shared latent space dimension.
        hidden_dim: Hidden layer width for all encoders / decoders.
        modality_weight_mode: How reconstruction losses are weighted.
            'equal' gives uniform weight; 'learnable' uses softmax over
            a learnable log-weight vector.
    """

    modality_dims: list[int] = field(default_factory=lambda: [2000, 500])
    latent_dim: int = 10
    hidden_dim: int = 64
    modality_weight_mode: str = "equal"
    use_gradnorm: bool = False

    def __post_init__(self) -> None:
        """Set stochastic defaults for VAE sampling and validate."""
        object.__setattr__(self, "stochastic", True)
        if self.stream_name is None:
            object.__setattr__(self, "stream_name", "sample")
        super().__post_init__()


# -------------------------------------------------------------------
# Encoder / Decoder sub-modules
# -------------------------------------------------------------------


class _ModalityEncoder(nnx.Module):
    """Two-layer MLP encoder for a single modality.

    Maps log1p-transformed count vectors to a hidden representation that
    is then projected to (mu, logvar) by external heads.

    Attributes:
        layers: List of linear layers.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise the encoder.

        Args:
            in_features: Input dimension (number of features for this modality).
            hidden_dim: Width of hidden layers.
            rngs: Flax NNX random number generators.
        """
        super().__init__()
        self.layers = nnx.List(
            [
                nnx.Linear(in_features, hidden_dim, rngs=rngs),
                nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            ]
        )

    def __call__(self, x: Float[Array, "batch features"]) -> Float[Array, "batch hidden"]:
        """Forward pass through encoder layers.

        Args:
            x: Input counts (already log1p-transformed by caller).

        Returns:
            Hidden representation of shape (batch, hidden_dim).
        """
        h = x
        for layer in self.layers:
            h = nnx.relu(layer(h))
        return h


class _ModalityDecoder(nnx.Module):
    """Two-layer MLP decoder for a single modality.

    Maps the shared latent z back to reconstructed count-level outputs.

    Attributes:
        layers: List of linear layers.
        output_layer: Final projection to modality dimension.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise the decoder.

        Args:
            latent_dim: Latent space dimension (input).
            hidden_dim: Width of hidden layers.
            out_features: Output dimension (number of features for this modality).
            rngs: Flax NNX random number generators.
        """
        super().__init__()
        self.layers = nnx.List(
            [
                nnx.Linear(latent_dim, hidden_dim, rngs=rngs),
                nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            ]
        )
        self.output_layer = nnx.Linear(hidden_dim, out_features, rngs=rngs)

    def __call__(self, z: Float[Array, "batch latent"]) -> Float[Array, "batch features"]:
        """Decode latent z to reconstructed log-rates.

        Args:
            z: Latent representation.

        Returns:
            Reconstructed log-rates of shape (batch, out_features).
        """
        h = z
        for layer in self.layers:
            h = nnx.relu(layer(h))
        return self.output_layer(h)


# -------------------------------------------------------------------
# Main operator
# -------------------------------------------------------------------


class DifferentiableMultiOmicsVAE(EncoderDecoderOperator):
    """Multi-omics VAE with Product-of-Experts latent fusion.

    For each modality a dedicated encoder produces (mu_m, logvar_m).
    These are combined via PoE into a joint posterior from which z is
    sampled.  Per-modality decoders then reconstruct counts from z.

    The ELBO objective uses MSE reconstruction loss per modality,
    optionally weighted by learnable per-modality weights, plus
    a KL divergence term against a standard-normal prior.

    Data keys follow the convention ``<name>_counts`` for input and
    ``<name>_reconstructed`` for output.  When exactly two modalities
    are used the canonical names ``rna`` and ``atac`` are applied;
    otherwise ``modality_<i>`` is used.

    Attributes:
        encoders: Per-modality encoder modules.
        decoders: Per-modality decoder modules.
        mu_heads: Per-modality linear projection for latent mean.
        logvar_heads: Per-modality linear projection for latent logvar.
        log_modality_weights: Learnable log-weights (only in 'learnable' mode).
    """

    def __init__(
        self,
        config: MultiOmicsVAEConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialise the multi-omics VAE.

        Args:
            config: Operator configuration.
            rngs: Flax NNX random number generators.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        rngs = ensure_rngs(rngs)
        n_modalities = len(config.modality_dims)

        # Per-modality encoders ------------------------------------------
        encoders: list[_ModalityEncoder] = []
        mu_heads: list[nnx.Linear] = []
        logvar_heads: list[nnx.Linear] = []

        for dim in config.modality_dims:
            encoders.append(_ModalityEncoder(dim, config.hidden_dim, rngs=rngs))
            mu_heads.append(nnx.Linear(config.hidden_dim, config.latent_dim, rngs=rngs))
            logvar_heads.append(nnx.Linear(config.hidden_dim, config.latent_dim, rngs=rngs))

        self.encoders = nnx.List(encoders)
        self.mu_heads = nnx.List(mu_heads)
        self.logvar_heads = nnx.List(logvar_heads)

        # Per-modality decoders ------------------------------------------
        decoders: list[_ModalityDecoder] = []
        for dim in config.modality_dims:
            decoders.append(_ModalityDecoder(config.latent_dim, config.hidden_dim, dim, rngs=rngs))
        self.decoders = nnx.List(decoders)

        # Modality weight mode -------------------------------------------
        self._weight_mode = nnx.static(config.modality_weight_mode)
        if config.modality_weight_mode == "learnable":
            self.log_modality_weights = nnx.Param(jnp.zeros(n_modalities))

        # Assign canonical key names -------------------------------------
        if n_modalities == 2:
            self._modality_keys: tuple[str, ...] = _DEFAULT_MODALITY_KEYS
        else:
            self._modality_keys = tuple(f"modality_{i}" for i in range(n_modalities))

    # ------------------------------------------------------------------
    # PoE fusion
    # ------------------------------------------------------------------

    def product_of_experts(
        self,
        mu_list: list[Float[Array, "batch latent"]],
        logvar_list: list[Float[Array, "batch latent"]],
    ) -> tuple[Float[Array, "batch latent"], Float[Array, "batch latent"]]:
        """Fuse per-modality posteriors via Product-of-Experts.

        For M modalities the PoE joint posterior is Gaussian with:
            precision_joint = sum_m precision_m
            mu_joint = (sum_m mu_m * precision_m) / precision_joint
        where precision_m = 1 / sigma_m^2 = exp(-logvar_m).

        Args:
            mu_list: Per-modality means.
            logvar_list: Per-modality log-variances.

        Returns:
            (mu_joint, logvar_joint) tuple for the fused posterior.
        """
        # Stack precisions and weighted means across modalities
        precision_sum = jnp.zeros_like(mu_list[0])
        weighted_mu_sum = jnp.zeros_like(mu_list[0])

        for mu_m, logvar_m in zip(mu_list, logvar_list):
            precision_m = jnp.exp(-logvar_m)
            precision_sum = precision_sum + precision_m
            weighted_mu_sum = weighted_mu_sum + mu_m * precision_m

        mu_joint = weighted_mu_sum / (precision_sum + EPSILON)
        logvar_joint = -jnp.log(precision_sum + EPSILON)

        return mu_joint, logvar_joint

    # ------------------------------------------------------------------
    # Modality weights
    # ------------------------------------------------------------------

    def _get_modality_weights(self) -> Float[Array, "n_modalities"]:
        """Return normalised modality weights.

        Returns:
            Weight vector summing to 1 of length n_modalities.
        """
        n_modalities = len(self.config.modality_dims)
        if self._weight_mode == "learnable":
            return jax.nn.softmax(self.log_modality_weights[...])
        return jnp.ones(n_modalities) / n_modalities

    # ------------------------------------------------------------------
    # Multi-task loss balancing
    # ------------------------------------------------------------------

    def compute_balanced_loss(
        self,
        losses: dict[str, Float[Array, ""]],
    ) -> Float[Array, ""]:
        """Combine multiple loss terms, optionally using GradNormBalancer.

        When ``use_gradnorm`` is enabled in the config, uses
        ``opifex.core.physics.gradnorm.GradNormBalancer`` to dynamically
        weight the loss terms so that gradient magnitudes are balanced.
        Otherwise, returns a simple sum of the losses.

        Args:
            losses: Mapping of loss name to scalar loss value, e.g.
                ``{"recon_rna": ..., "recon_atac": ..., "kl": ...}``.

        Returns:
            Combined scalar loss.
        """
        loss_values = list(losses.values())
        if self.config.use_gradnorm:
            balancer = GradNormBalancer(num_losses=len(loss_values), rngs=nnx.Rngs(0))
            return balancer(loss_values)
        return sum(loss_values[1:], start=loss_values[0])

    # ------------------------------------------------------------------
    # Data key helpers
    # ------------------------------------------------------------------

    def _input_key(self, idx: int) -> str:
        """Return the data-dict key for modality *idx* input counts.

        Args:
            idx: Modality index.

        Returns:
            String key such as ``rna_counts`` or ``modality_0_counts``.
        """
        return f"{self._modality_keys[idx]}_counts"

    def _output_key(self, idx: int) -> str:
        """Return the data-dict key for modality *idx* reconstruction.

        Args:
            idx: Modality index.

        Returns:
            String key such as ``rna_reconstructed``.
        """
        return f"{self._modality_keys[idx]}_reconstructed"

    # ------------------------------------------------------------------
    # apply
    # ------------------------------------------------------------------

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Run the multi-omics VAE forward pass.

        Steps:
            1. Encode each modality to (mu_m, logvar_m).
            2. PoE fusion -> (mu_joint, logvar_joint).
            3. Reparameterise -> z.
            4. Decode each modality from z.
            5. Compute ELBO = weighted recon + KL.

        Args:
            data: Dictionary with ``<modality>_counts`` keys, each of shape
                (n_cells, modality_dim).
            state: Operator state (passed through unchanged).
            metadata: Operator metadata (passed through unchanged).
            random_params: Not used.
            stats: Not used.

        Returns:
            Tuple of (result_data, state, metadata) where result_data
            contains the original inputs plus ``joint_latent``,
            ``<modality>_reconstructed``, and ``elbo_loss``.
        """
        n_modalities = len(self.config.modality_dims)

        # 1. Encode each modality ----------------------------------------
        mu_list: list[jax.Array] = []
        logvar_list: list[jax.Array] = []

        for i in range(n_modalities):
            counts = data[self._input_key(i)]
            h = self.encoders[i](jnp.log1p(counts))
            mu = self.mu_heads[i](h)
            logvar = jnp.clip(self.logvar_heads[i](h), -10.0, 10.0)
            mu_list.append(mu)
            logvar_list.append(logvar)

        # 2. PoE fusion --------------------------------------------------
        mu_joint, logvar_joint = self.product_of_experts(mu_list, logvar_list)

        # 3. Sample z via reparameterisation (inherited) -----------------
        z = self.reparameterize(mu_joint, logvar_joint)

        # 4. Decode each modality ----------------------------------------
        reconstructions: list[jax.Array] = []
        for i in range(n_modalities):
            reconstructions.append(self.decoders[i](z))

        # 5. Compute ELBO ------------------------------------------------
        weights = self._get_modality_weights()

        # Weighted reconstruction loss (MSE per modality, summed over features)
        total_recon = jnp.array(0.0)
        for i in range(n_modalities):
            counts = data[self._input_key(i)]
            per_sample = jnp.sum((counts - reconstructions[i]) ** 2, axis=-1)
            mean_recon = reduce_loss(per_sample, reduction="mean")
            total_recon = total_recon + weights[i] * mean_recon

        # KL divergence (batch_sum: sum over latent, mean over batch)
        kl = gaussian_kl_divergence(mu_joint, logvar_joint, reduction="batch_sum")

        elbo_loss = total_recon + kl

        # Build output dict -----------------------------------------------
        result: dict[str, Any] = dict(data)
        result["joint_latent"] = z
        result["joint_mu"] = mu_joint
        result["joint_logvar"] = logvar_joint
        result["elbo_loss"] = elbo_loss

        for i in range(n_modalities):
            result[self._output_key(i)] = reconstructions[i]

        return result, state, metadata
