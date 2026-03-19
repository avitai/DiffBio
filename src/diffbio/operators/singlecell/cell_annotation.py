"""Cell type annotation operator for single-cell analysis.

This module provides a differentiable cell type annotator supporting three
annotation strategies inspired by popular tools:

- **celltypist**: Logistic-regression classifier on a VAE latent space.
- **cellassign**: Marker-gene likelihood model with learnable rate parameters.
- **scanvi**: Semi-supervised VAE that combines reconstruction, KL, and
  cross-entropy on labelled cells.

All three modes are end-to-end differentiable and JIT-compatible, enabling
gradient-based optimisation of annotation models within a Datarax pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp
from artifex.generative_models.core.losses.divergence import gaussian_kl_divergence
from datarax.core.config import OperatorConfig
from flax import nnx
from jaxtyping import Array, Float, Int, PyTree

from diffbio.constants import EPSILON
from diffbio.core.base_operators import EncoderDecoderOperator
from diffbio.utils.nn_utils import ensure_rngs


@dataclass
class CellAnnotatorConfig(OperatorConfig):
    """Configuration for cell type annotation.

    Attributes:
        annotation_mode: Annotation strategy to use.
        n_cell_types: Number of cell types to classify.
        n_genes: Number of input genes.
        latent_dim: Latent-space dimensionality for VAE encoder.
        hidden_dims: Hidden layer sizes for encoder and decoder.
        marker_matrix_shape: Shape (n_types, n_genes) for cellassign mode.
        stochastic: Whether the operator uses randomness.
        stream_name: RNG stream name for sampling.
    """

    annotation_mode: Literal["scanvi", "cellassign", "celltypist"] = "celltypist"
    n_cell_types: int = 10
    n_genes: int = 2000
    latent_dim: int = 10
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    marker_matrix_shape: tuple[int, int] | None = None
    stochastic: bool = True
    stream_name: str = "sample"


class DifferentiableCellAnnotator(EncoderDecoderOperator):
    """Differentiable cell type annotator with three annotation modes.

    Modes
    -----
    **celltypist** (logistic regression on latent):
        Encode counts to a VAE latent, apply a linear classifier head, softmax.

    **cellassign** (marker-gene likelihood):
        Given a binary marker matrix *M*, compute per-type Poisson
        log-likelihoods with learnable rate parameters, then softmax.

    **scanvi** (semi-supervised VAE):
        VAE encoder + classifier head.  For labelled cells the known labels
        contribute a cross-entropy term; for unlabelled cells the type
        probabilities are predicted from the latent.

    All modes additionally produce a latent representation via a shared
    VAE encoder.

    Inherits from EncoderDecoderOperator to get:

    - reparameterize() for the VAE sampling step
    - kl_divergence() for KL from standard normal
    - elbo_loss() for combining reconstruction and KL losses

    Args:
        config: CellAnnotatorConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        ```python
        config = CellAnnotatorConfig(
            annotation_mode="celltypist",
            n_cell_types=10,
            n_genes=2000,
            stochastic=True,
            stream_name="sample",
        )
        annotator = DifferentiableCellAnnotator(config, rngs=nnx.Rngs(42))
        data = {"counts": counts}
        result, state, meta = annotator.apply(data, {}, None)
        ```
    """

    def __init__(
        self,
        config: CellAnnotatorConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialise the cell type annotator.

        Args:
            config: Annotator configuration.
            rngs: Random number generators for initialisation and sampling.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        rngs = ensure_rngs(rngs)

        self.n_genes = config.n_genes
        self.n_cell_types = config.n_cell_types
        self.annotation_mode = nnx.static(config.annotation_mode)

        # --- shared VAE encoder (celltypist & scanvi) ---
        self._build_encoder(config, rngs)
        self._build_decoder(config, rngs)

        # --- mode-specific heads ---
        if config.annotation_mode in ("celltypist", "scanvi"):
            self.classifier_head = nnx.Linear(
                in_features=config.latent_dim,
                out_features=config.n_cell_types,
                rngs=rngs,
            )

        if config.annotation_mode == "cellassign":
            # Learnable log-rate parameters: mu_type_g (in log space).
            # Initialise to log(5) so Poisson rates start at ~5;
            # the x*log(mu) term is then sensitive to count magnitude,
            # letting the marker matrix drive type discrimination.
            self.log_mu = nnx.Param(
                jnp.full(
                    (config.n_cell_types, config.n_genes),
                    jnp.log(5.0),
                )
            )

    # ------------------------------------------------------------------
    # Network builders (DRY: shared between modes)
    # ------------------------------------------------------------------

    def _build_encoder(self, config: CellAnnotatorConfig, rngs: nnx.Rngs) -> None:
        """Build the VAE encoder layers.

        Args:
            config: Annotator configuration.
            rngs: Random number generators.
        """
        encoder_layers: list[nnx.Linear] = []
        prev_dim = config.n_genes
        for hidden_dim in config.hidden_dims:
            encoder_layers.append(
                nnx.Linear(in_features=prev_dim, out_features=hidden_dim, rngs=rngs)
            )
            prev_dim = hidden_dim
        self.encoder_layers = nnx.List(encoder_layers)
        self.fc_mean = nnx.Linear(in_features=prev_dim, out_features=config.latent_dim, rngs=rngs)
        self.fc_logvar = nnx.Linear(
            in_features=prev_dim, out_features=config.latent_dim, rngs=rngs
        )

    def _build_decoder(self, config: CellAnnotatorConfig, rngs: nnx.Rngs) -> None:
        """Build the VAE decoder layers.

        Args:
            config: Annotator configuration.
            rngs: Random number generators.
        """
        decoder_layers: list[nnx.Linear] = []
        prev_dim = config.latent_dim
        for hidden_dim in reversed(config.hidden_dims):
            decoder_layers.append(
                nnx.Linear(in_features=prev_dim, out_features=hidden_dim, rngs=rngs)
            )
            prev_dim = hidden_dim
        self.decoder_layers = nnx.List(decoder_layers)
        self.fc_output = nnx.Linear(in_features=prev_dim, out_features=config.n_genes, rngs=rngs)

    # ------------------------------------------------------------------
    # Encoder / decoder forward passes
    # ------------------------------------------------------------------

    def encode(
        self,
        counts: Float[Array, "batch n_genes"],
    ) -> tuple[Float[Array, "batch latent_dim"], Float[Array, "batch latent_dim"]]:
        """Encode count vectors to latent distribution parameters.

        Args:
            counts: Gene expression counts, shape ``(n, n_genes)``.

        Returns:
            Tuple of (mean, logvar) for the latent Gaussian.
        """
        x = jnp.log1p(counts)
        for layer in self.encoder_layers:
            x = nnx.relu(layer(x))
        mean = self.fc_mean(x)
        logvar = jnp.clip(self.fc_logvar(x), -10.0, 10.0)
        return mean, logvar

    def decode(
        self,
        z: Float[Array, "batch latent_dim"],
    ) -> Float[Array, "batch n_genes"]:
        """Decode latent vectors to log-rate gene expression.

        Args:
            z: Latent representations, shape ``(n, latent_dim)``.

        Returns:
            Reconstructed log-rates, shape ``(n, n_genes)``.
        """
        x = z
        for layer in self.decoder_layers:
            x = nnx.relu(layer(x))
        return self.fc_output(x)

    # ------------------------------------------------------------------
    # Per-mode annotation logic
    # ------------------------------------------------------------------

    def _annotate_celltypist(
        self,
        z: Float[Array, "batch latent_dim"],
    ) -> Float[Array, "batch n_cell_types"]:
        """Celltypist: logistic classifier on latent.

        Args:
            z: Latent representations.

        Returns:
            Cell type probabilities, shape ``(n, n_cell_types)``.
        """
        logits = self.classifier_head(z)
        return jax.nn.softmax(logits, axis=-1)

    def _annotate_cellassign(
        self,
        counts: Float[Array, "batch n_genes"],
        marker_matrix: Float[Array, "n_types n_genes"],
    ) -> Float[Array, "batch n_cell_types"]:
        """Cellassign: marker-gene Poisson likelihood.

        For each cell type, compute masked Poisson log-likelihood using only
        the marker genes for that type.

        Args:
            counts: Gene expression counts, shape ``(n, n_genes)``.
            marker_matrix: Binary marker matrix, shape ``(n_types, n_genes)``.

        Returns:
            Cell type probabilities, shape ``(n, n_cell_types)``.
        """
        # Learnable rates in positive space
        mu = jnp.exp(self.log_mu[...])  # (n_types, n_genes)

        # Poisson log-likelihood per gene per type:
        # log P(x_g | type) = x_g * log(mu_type_g) - mu_type_g - log(x_g!)
        # We mask by M so only marker genes contribute.
        # shape: (1, n_genes) vs (n_types, n_genes)
        log_mu = jnp.log(mu + EPSILON)  # (n_types, n_genes)

        # counts: (n, g), log_mu: (t, g), marker: (t, g)
        # Expand counts: (n, 1, g)
        counts_expanded = counts[:, None, :]  # (n, 1, g)

        # Per-gene log-likelihood (ignoring log(x!) which cancels in softmax)
        # ll_g = x_g * log(mu) - mu, masked by marker
        log_lik_per_gene = counts_expanded * log_mu[None, :, :] - mu[None, :, :]
        # Mask by marker matrix
        masked_log_lik = log_lik_per_gene * marker_matrix[None, :, :]  # (n, t, g)
        # Sum over genes to get per-type log-likelihood
        log_lik = jnp.sum(masked_log_lik, axis=-1)  # (n, t)

        return jax.nn.softmax(log_lik, axis=-1)

    def _annotate_scanvi(
        self,
        z: Float[Array, "batch latent_dim"],
        mean: Float[Array, "batch latent_dim"],
        logvar: Float[Array, "batch latent_dim"],
        counts: Float[Array, "batch n_genes"],
        known_labels: Int[Array, "n_labeled"] | None,
        label_indices: Int[Array, "n_labeled"] | None,
    ) -> Float[Array, "batch n_cell_types"]:
        """Scanvi: semi-supervised VAE annotation.

        The classifier head produces type probabilities from latent z.
        When ``known_labels`` and ``label_indices`` are provided, the
        probabilities of labelled cells are blended toward their known
        types via a soft cross-entropy weighting.

        Args:
            z: Latent representations.
            mean: Encoder mean.
            logvar: Encoder log-variance.
            counts: Original counts (for reconstruction context).
            known_labels: Known integer labels for a subset of cells.
            label_indices: Indices into the batch for the labelled cells.

        Returns:
            Cell type probabilities, shape ``(n, n_cell_types)``.
        """
        logits = self.classifier_head(z)  # (n, n_types)
        probs = jax.nn.softmax(logits, axis=-1)

        if known_labels is None or label_indices is None:
            return probs

        # Blend known labels into the probability for labelled cells.
        # Create one-hot targets for labelled cells.
        one_hot_targets = jax.nn.one_hot(known_labels, self.n_cell_types)  # (n_l, t)

        # Blend: 0.8 * one_hot + 0.2 * predicted (keeps differentiability)
        blended = 0.8 * one_hot_targets + 0.2 * probs[label_indices]

        # Scatter blended back
        probs = probs.at[label_indices].set(blended)

        return probs

    # ------------------------------------------------------------------
    # Training loss (scanvi ELBO)
    # ------------------------------------------------------------------

    def compute_elbo_loss(
        self,
        counts: Float[Array, "batch n_genes"],
        known_labels: Int[Array, "n_labeled"] | None = None,
        label_indices: Int[Array, "n_labeled"] | None = None,
        beta: float = 1.0,
    ) -> Float[Array, ""]:
        """Compute the negative ELBO for training (scanvi / celltypist).

        ELBO = reconstruction_loss + beta * KL + cross_entropy_on_labeled.

        Uses ``gaussian_kl_divergence`` from artifex for the KL term (DRY).

        Args:
            counts: Gene expression counts ``(n, n_genes)``.
            known_labels: Integer labels for labelled subset (scanvi).
            label_indices: Batch indices of labelled cells (scanvi).
            beta: KL weight (default 1.0, >1 for beta-VAE).

        Returns:
            Scalar negative ELBO loss.
        """
        mean, logvar = self.encode(counts)
        z = self.reparameterize(mean, logvar)

        # Reconstruction: Poisson NLL
        log_rate = self.decode(z)
        rate = jnp.exp(log_rate)
        recon_loss = jnp.sum(rate - counts * log_rate)

        # KL divergence via artifex (handles batched inputs)
        kl = gaussian_kl_divergence(mean, logvar, reduction="sum")

        loss = recon_loss + beta * kl

        # Cross-entropy on labelled cells (scanvi mode)
        if (
            self.annotation_mode in ("scanvi", "celltypist")
            and known_labels is not None
            and label_indices is not None
        ):
            logits = self.classifier_head(z)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            labeled_log_probs = log_probs[label_indices]
            one_hot = jax.nn.one_hot(known_labels, self.n_cell_types)
            ce = -jnp.sum(one_hot * labeled_log_probs)
            loss = loss + ce

        return loss

    # ------------------------------------------------------------------
    # apply()
    # ------------------------------------------------------------------

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Annotate cells with type probabilities.

        Args:
            data: Dictionary containing:
                - ``"counts"``: Gene expression counts ``(n, n_genes)``
                - (cellassign) ``"marker_matrix"``: Binary ``(n_types, n_genes)``
                - (scanvi) ``"known_labels"``: Integer labels ``(n_labeled,)``
                - (scanvi) ``"label_indices"``: Batch indices ``(n_labeled,)``
            state: Element state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: Not used.
            stats: Not used.

        Returns:
            Tuple of (transformed_data, state, metadata) where
            transformed_data adds:
                - ``"cell_type_probabilities"``: ``(n, n_cell_types)``
                - ``"cell_type_labels"``: ``(n,)`` argmax labels
                - ``"latent"``: ``(n, latent_dim)``
        """
        counts = data["counts"]

        # Shared VAE encoding
        mean, logvar = self.encode(counts)
        z = self.reparameterize(mean, logvar)

        # Mode dispatch
        mode = self.annotation_mode
        if mode == "celltypist":
            probs = self._annotate_celltypist(z)
        elif mode == "cellassign":
            marker_matrix = data["marker_matrix"]
            probs = self._annotate_cellassign(counts, marker_matrix)
        elif mode == "scanvi":
            known_labels = data.get("known_labels")
            label_indices = data.get("label_indices")
            probs = self._annotate_scanvi(
                z, mean, logvar, counts, known_labels, label_indices
            )
        else:
            raise ValueError(f"Unknown annotation_mode: {mode}")

        labels = jnp.argmax(probs, axis=-1)

        transformed_data = {
            **data,
            "cell_type_probabilities": probs,
            "cell_type_labels": labels,
            "latent": z,
        }

        return transformed_data, state, metadata
