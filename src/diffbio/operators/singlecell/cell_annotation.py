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
from diffbio.utils.nn_utils import ensure_rngs, get_rng_key


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
        gene_likelihood: Reconstruction likelihood for scanvi mode.
            ``"poisson"`` for standard Poisson NLL (default),
            ``"zinb"`` for Zero-Inflated Negative Binomial.
        stochastic: Whether the operator uses randomness.
        stream_name: RNG stream name for sampling.
    """

    annotation_mode: Literal["scanvi", "cellassign", "celltypist"] = "celltypist"
    n_cell_types: int = 10
    n_genes: int = 2000
    latent_dim: int = 10
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    marker_matrix_shape: tuple[int, int] | None = None
    gene_likelihood: Literal["poisson", "zinb"] = "poisson"
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
        self.fc_logvar = nnx.Linear(in_features=prev_dim, out_features=config.latent_dim, rngs=rngs)

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
    ) -> dict[str, Float[Array, "batch n_genes"]]:
        """Decode latent vectors to gene expression parameters.

        Args:
            z: Latent representations, shape ``(n, latent_dim)``.

        Returns:
            Dictionary with ``"log_rate"`` (always present) and optionally
            ``"log_theta"`` and ``"pi_logit"`` when ZINB likelihood is active.
        """
        x = z
        for layer in self.decoder_layers:
            x = nnx.relu(layer(x))

        result: dict[str, Float[Array, "batch n_genes"]] = {
            "log_rate": self.fc_output(x),
        }

        if hasattr(self, "fc_log_theta"):
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
        one_hot_targets = jax.nn.one_hot(known_labels, self.n_cell_types)
        probs = probs.at[label_indices].set(one_hot_targets)

        return probs

    # ------------------------------------------------------------------
    # Reconstruction loss helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _poisson_nll(
        counts: Float[Array, "batch n_genes"],
        log_rate: Float[Array, "batch n_genes"],
    ) -> Float[Array, ""]:
        """Compute Poisson negative log-likelihood.

        Args:
            counts: Original counts.
            log_rate: Log rates from decoder.

        Returns:
            Negative log-likelihood (scalar, summed over all elements).
        """
        rate = jnp.exp(log_rate)
        return jnp.sum(rate - counts * log_rate)

    @staticmethod
    def _zinb_nll(
        counts: Float[Array, "batch n_genes"],
        log_rate: Float[Array, "batch n_genes"],
        log_theta: Float[Array, "batch n_genes"],
        pi_logit: Float[Array, "batch n_genes"],
    ) -> Float[Array, ""]:
        """Compute Zero-Inflated Negative Binomial negative log-likelihood.

        Uses the scVI-style logit-space formulation for numerical stability.
        Key identities used:
            ``log(sigmoid(pi))   = -softplus(-pi)``
            ``log(1-sigmoid(pi)) = -softplus(pi)``

        Args:
            counts: Original counts.
            log_rate: Log mean parameter from decoder.
            log_theta: Log dispersion parameter.
            pi_logit: Logit of zero-inflation probability.

        Returns:
            Negative log-likelihood (scalar, summed over all elements).
        """
        mu = jnp.exp(log_rate)
        theta = jnp.exp(jnp.clip(log_theta, -10.0, 10.0))
        eps = EPSILON

        # Log-space sigmoid computations (numerically stable)
        softplus_pi = jax.nn.softplus(-pi_logit)  # = -log(sigmoid(pi_logit))
        log_theta_mu = jnp.log(theta + mu + eps)

        # NB(0) in log-space combined with dropout logit
        pi_theta_log = -pi_logit + theta * (jnp.log(theta + eps) - log_theta_mu)

        # Case x == 0: log[sigmoid(pi) + (1 - sigmoid(pi)) * NB(0)]
        case_zero = jax.nn.softplus(pi_theta_log) - softplus_pi

        # Case x > 0: log[(1 - sigmoid(pi)) * NB(x)]
        case_nonzero = (
            -softplus_pi
            + pi_theta_log
            + counts * (jnp.log(mu + eps) - log_theta_mu)
            + jax.scipy.special.gammaln(counts + theta)
            - jax.scipy.special.gammaln(theta)
            - jax.scipy.special.gammaln(counts + 1.0)
        )

        is_zero = (counts < eps).astype(jnp.float32)
        log_prob = is_zero * case_zero + (1.0 - is_zero) * case_nonzero

        return -jnp.sum(log_prob)

    def _reconstruction_loss(
        self,
        counts: Float[Array, "batch n_genes"],
        decode_output: dict[str, Float[Array, "batch n_genes"]],
    ) -> Float[Array, ""]:
        """Compute reconstruction loss based on configured likelihood.

        Args:
            counts: Original counts.
            decode_output: Dictionary from ``decode()`` containing
                ``"log_rate"`` and optionally ``"log_theta"``, ``"pi_logit"``.

        Returns:
            Negative log-likelihood (scalar).
        """
        log_rate = decode_output["log_rate"]
        if "log_theta" in decode_output:
            return self._zinb_nll(
                counts,
                log_rate,
                decode_output["log_theta"],
                decode_output["pi_logit"],
            )
        return self._poisson_nll(counts, log_rate)

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
        recon_loss = self._reconstruction_loss(counts, decode_output)

        # KL divergence -- type-conditioned for scanvi, standard otherwise
        if self.annotation_mode == "scanvi":
            logits = self.classifier_head(z)
            type_probs = jax.nn.softmax(logits, axis=-1)

            # For labelled cells, override predicted probs with known labels
            if known_labels is not None and label_indices is not None:
                one_hot_known = jax.nn.one_hot(known_labels, self.n_cell_types)
                type_probs = type_probs.at[label_indices].set(one_hot_known)

            kl = self._type_conditioned_kl(mean, logvar, type_probs)
        else:
            kl = gaussian_kl_divergence(mean, logvar, reduction="sum")

        loss = recon_loss + beta * kl

        # Cross-entropy on labelled cells (scanvi / celltypist)
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
