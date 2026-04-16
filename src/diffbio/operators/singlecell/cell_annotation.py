"""Cell type annotation operator for single-cell analysis.

This module provides a differentiable cell type annotator supporting three
annotation strategies inspired by popular tools:

- **celltypist**: Logistic-regression classifier on a VAE latent space.
- **cellassign**: Marker-gene likelihood model with learnable rate parameters.
- **scanvi**: Semi-supervised VAE with type-conditioned latent prior.
  For each cell type y, learns prior parameters mu_y and logvar_y so that
  ``KL(q(z|x) || p(z|y))`` encourages different types to occupy distinct
  latent regions.  For unlabelled cells the KL is marginalised over
  predicted type probabilities.

All three modes are end-to-end differentiable and JIT-compatible, enabling
gradient-based optimisation of annotation models within a Datarax pipeline.
"""

import logging
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
from diffbio.operators._count_vae import CountReconstructionMixin, CountVAEBackboneMixin
from diffbio.utils.nn_utils import get_rng_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CellAnnotatorConfig(OperatorConfig):
    """Configuration for cell type annotation.

    Attributes:
        annotation_mode: Annotation strategy to use.
        n_cell_types: Number of cell types to classify.
        n_genes: Number of input genes.
        latent_dim: Latent-space dimensionality for VAE encoder.
        hidden_dims: Hidden layer sizes for encoder and decoder.
        marker_matrix_shape: Shape (n_types, n_genes) for cellassign mode.
        gene_likelihood: Reconstruction likelihood for scanvi mode.
            ``"poisson"`` for standard Poisson NLL (default),
            ``"zinb"`` for Zero-Inflated Negative Binomial.
    """

    annotation_mode: Literal["scanvi", "cellassign", "celltypist"] = "celltypist"
    n_cell_types: int = 10
    n_genes: int = 2000
    latent_dim: int = 10
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    marker_matrix_shape: tuple[int, int] | None = None
    gene_likelihood: Literal["poisson", "zinb"] = "poisson"

    def __post_init__(self) -> None:
        """Set stochastic defaults and validate."""
        object.__setattr__(self, "stochastic", True)
        if self.stream_name is None:
            object.__setattr__(self, "stream_name", "sample")
        super().__post_init__()

        if self.n_cell_types <= 0:
            raise ValueError(f"n_cell_types must be positive, got {self.n_cell_types}")
        if self.n_genes <= 0:
            raise ValueError(f"n_genes must be positive, got {self.n_genes}")
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError(
                f"hidden_dims must contain only positive values, got {self.hidden_dims}"
            )

        expected_marker_shape = (self.n_cell_types, self.n_genes)
        if self.annotation_mode == "cellassign":
            if self.marker_matrix_shape is None:
                raise ValueError(
                    "marker_matrix_shape must be provided for annotation_mode='cellassign'"
                )
            if self.marker_matrix_shape != expected_marker_shape:
                raise ValueError(
                    "marker_matrix_shape must match "
                    f"(n_cell_types, n_genes)={expected_marker_shape}, "
                    f"got {self.marker_matrix_shape}"
                )
        elif self.marker_matrix_shape is not None:
            raise ValueError(
                "marker_matrix_shape is only supported for annotation_mode='cellassign'"
            )

        if self.annotation_mode != "scanvi" and self.gene_likelihood != "poisson":
            raise ValueError("gene_likelihood is only configurable for annotation_mode='scanvi'")


class DifferentiableCellAnnotator(
    CountReconstructionMixin,
    CountVAEBackboneMixin,
    EncoderDecoderOperator,
):
    """Differentiable cell type annotator with three annotation modes.

    Modes
    -----
    **celltypist** (logistic regression on latent):
        Encode counts to a VAE latent, apply a linear classifier head, softmax.

    **cellassign** (marker-gene likelihood):
        Given a binary marker matrix *M*, compute per-type Poisson
        log-likelihoods with learnable rate parameters, then softmax.

    **scanvi** (semi-supervised VAE with type-conditioned prior):
        VAE encoder + classifier head with learnable per-type Gaussian priors
        in latent space.  The KL divergence uses ``p(z|y) = N(mu_y, sigma_y)``
        instead of the standard ``N(0, I)``, and is marginalised over predicted
        type probabilities for unlabelled cells.

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

        rngs = self._init_count_vae_operator(config=config, rngs=rngs)

        # --- mode-specific heads ---
        if config.annotation_mode in ("celltypist", "scanvi"):
            self.classifier_head = nnx.Linear(
                in_features=config.latent_dim,
                out_features=config.n_cell_types,
                rngs=rngs,
            )

        if config.annotation_mode == "scanvi":
            # Type-conditioned prior parameters: each cell type y has its own
            # Gaussian prior N(mu_y, diag(exp(logvar_y))) in latent space.
            params_key = get_rng_key(rngs, "params", fallback_seed=7)
            self.prior_means = nnx.Param(
                jax.random.normal(params_key, (config.n_cell_types, config.latent_dim)) * 0.01
            )
            self.prior_logvars = nnx.Param(jnp.zeros((config.n_cell_types, config.latent_dim)))

        if config.annotation_mode == "scanvi" and config.gene_likelihood == "zinb":
            # ZINB decoder heads: log-dispersion and dropout logit
            # Decoder reverses hidden_dims, so the final hidden dim is the first
            last_hidden = config.hidden_dims[0] if config.hidden_dims else config.latent_dim
            self.fc_log_theta = nnx.Linear(
                in_features=last_hidden,
                out_features=config.n_genes,
                rngs=rngs,
            )
            self.fc_pi_logit = nnx.Linear(
                in_features=last_hidden,
                out_features=config.n_genes,
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

    def decode(
        self,
        z: Float[Array, "batch latent_dim"],
    ) -> dict[str, Float[Array, "batch n_genes"]]:
        """Decode latent vectors to gene expression parameters.

        Args:
            z: Latent representations, shape ``(n, latent_dim)``.

        Returns:
            Dictionary with ``"log_rate"`` (always present) and optionally
            ``"log_theta"`` and ``"pi_logit"`` when ZINB likelihood is active.
        """
        x = self.decode_hidden(z)

        result: dict[str, Float[Array, "batch n_genes"]] = {
            "log_rate": self.fc_output(x),
        }

        if self.config.gene_likelihood == "zinb":
            result["log_theta"] = self.fc_log_theta(x)
            result["pi_logit"] = self.fc_pi_logit(x)

        return result

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

    def _type_conditioned_kl(
        self,
        mean: Float[Array, "batch latent_dim"],
        logvar: Float[Array, "batch latent_dim"],
        type_probs: Float[Array, "batch n_cell_types"],
    ) -> Float[Array, ""]:
        """KL divergence with type-conditioned prior, marginalised over types.

        For each cell type y with prior ``N(mu_y, diag(exp(logvar_y)))``,
        compute the analytic KL from ``q(z|x) = N(mean, diag(exp(logvar)))``
        then marginalise:

            KL = sum_y  p(y) * KL( q(z|x) || p(z|y) )

        Args:
            mean: Encoder mean, shape ``(n, latent_dim)``.
            logvar: Encoder log-variance, shape ``(n, latent_dim)``.
            type_probs: Cell-type probabilities, shape ``(n, n_cell_types)``.

        Returns:
            Scalar marginalised KL divergence (summed over batch).
        """
        prior_mu = self.prior_means[...]  # (n_types, latent_dim)
        prior_lv = self.prior_logvars[...]  # (n_types, latent_dim)

        # Expand for broadcasting:
        #   mean/logvar: (n, 1, d), prior: (1, t, d)
        mean_e = mean[:, None, :]  # (n, 1, d)
        logvar_e = logvar[:, None, :]  # (n, 1, d)
        prior_mu_e = prior_mu[None, :, :]  # (1, t, d)
        prior_lv_e = prior_lv[None, :, :]  # (1, t, d)

        # Analytic KL between two diagonal Gaussians per dimension:
        # KL(N(m1,s1)||N(m2,s2))
        #   = 0.5*(lv2-lv1 + (exp(lv1)+(m1-m2)^2)/exp(lv2) - 1)
        kl_per_dim = 0.5 * (
            prior_lv_e
            - logvar_e
            + (jnp.exp(logvar_e) + (mean_e - prior_mu_e) ** 2) / (jnp.exp(prior_lv_e) + EPSILON)
            - 1.0
        )  # (n, t, d)

        kl_per_type = jnp.sum(kl_per_dim, axis=-1)  # (n, t)

        # Marginalise over types: sum_y p(y) * KL_y
        kl_marginal = jnp.sum(type_probs * kl_per_type, axis=-1)  # (n,)

        return jnp.sum(kl_marginal)

    def _annotate_scanvi(
        self,
        z: Float[Array, "batch latent_dim"],
        mean: Float[Array, "batch latent_dim"],
        logvar: Float[Array, "batch latent_dim"],
        counts: Float[Array, "batch n_genes"],
        known_labels: Int[Array, "n_labeled"] | None,
        label_indices: Int[Array, "n_labeled"] | None,
    ) -> Float[Array, "batch n_cell_types"]:
        """Scanvi: semi-supervised VAE with type-conditioned prior.

        The classifier head produces type probabilities from latent z.
        For labelled cells the known labels are used directly as one-hot
        type probabilities.  For unlabelled cells the predicted
        probabilities are returned as-is.

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

        # For labelled cells, set probabilities to one-hot of the known type.
        one_hot_targets = jax.nn.one_hot(known_labels, self.config.n_cell_types)
        probs = probs.at[label_indices].set(one_hot_targets)

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
        """Compute the negative ELBO for training.

        For **scanvi** mode the KL term uses a type-conditioned prior:
        each cell type y has its own Gaussian prior ``N(mu_y, sigma_y)``
        and the KL is marginalised over predicted/known type probabilities.

        For other modes (celltypist, cellassign) the standard
        ``KL(q(z|x) || N(0,I))`` is used via artifex.

        When ``gene_likelihood="zinb"`` (scanvi only), the reconstruction
        loss uses the ZINB negative log-likelihood instead of Poisson NLL.

        Loss = reconstruction + beta * KL + cross_entropy_on_labeled.

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

        # Reconstruction loss (Poisson or ZINB depending on config)
        decode_output = self.decode(z)
        recon_loss = self.reconstruction_loss(counts, decode_output)

        # KL divergence -- type-conditioned for scanvi, standard otherwise
        if self.config.annotation_mode == "scanvi":
            logits = self.classifier_head(z)
            type_probs = jax.nn.softmax(logits, axis=-1)

            # For labelled cells, override predicted probs with known labels
            if known_labels is not None and label_indices is not None:
                one_hot_known = jax.nn.one_hot(known_labels, self.config.n_cell_types)
                type_probs = type_probs.at[label_indices].set(one_hot_known)

            kl = self._type_conditioned_kl(mean, logvar, type_probs)
        else:
            kl = gaussian_kl_divergence(mean, logvar, reduction="sum")

        loss = recon_loss + beta * kl

        # Cross-entropy on labelled cells (scanvi / celltypist)
        if (
            self.config.annotation_mode in ("scanvi", "celltypist")
            and known_labels is not None
            and label_indices is not None
        ):
            logits = self.classifier_head(z)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            labeled_log_probs = log_probs[label_indices]
            one_hot = jax.nn.one_hot(known_labels, self.config.n_cell_types)
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
        mode = self.config.annotation_mode
        if mode == "celltypist":
            probs = self._annotate_celltypist(z)
        elif mode == "cellassign":
            marker_matrix = data["marker_matrix"]
            probs = self._annotate_cellassign(counts, marker_matrix)
        elif mode == "scanvi":
            known_labels = data.get("known_labels")
            label_indices = data.get("label_indices")
            probs = self._annotate_scanvi(z, mean, logvar, counts, known_labels, label_indices)
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
