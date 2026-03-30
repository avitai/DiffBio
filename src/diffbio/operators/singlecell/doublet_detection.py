"""Differentiable doublet detection for single-cell data.

This module provides two complementary doublet detection strategies:

1. **Scrublet-style (DifferentiableDoubletScorer)**: Generates synthetic
   doublets by summing random cell pairs, then scores real cells via a
   Bayesian k-NN likelihood ratio in PCA space.

2. **Solo-style VAE (DifferentiableSoloDetector)**: Encodes cells through a
   VAE into a latent space, generates synthetic doublets, and trains a binary
   classifier on latent representations to distinguish singlets from doublets.
   Based on Bernstein et al., Cell Systems 2020.

Key techniques:
- Soft k-NN neighbor counting with differentiable thresholding (Scrublet)
- VAE reparameterization + latent-space classifier (Solo)

Applications: Identifying doublet artifacts in scRNA-seq data as a
preprocessing step before downstream analysis (clustering, trajectory, DE).
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from artifex.generative_models.core.losses.divergence import gaussian_kl_divergence
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, PyTree

from diffbio.constants import DISTANCE_MASK_SENTINEL
from diffbio.core import soft_ops
from diffbio.core.base_operators import EncoderDecoderOperator
from diffbio.core.graph_utils import compute_pairwise_distances
from diffbio.utils.nn_utils import build_mlp_decoder, build_mlp_encoder, ensure_rngs, forward_mlp

logger = logging.getLogger(__name__)


def generate_synthetic_doublets(
    counts: Float[Array, "n_cells n_genes"],
    rng: jax.Array,
    sim_doublet_ratio: float,
) -> Float[Array, "n_synthetic n_genes"]:
    """Generate synthetic doublets by summing random cell pairs.

    The number of synthetic doublets is ``n_cells * sim_doublet_ratio``,
    matching Scrublet's default 2:1 synthetic-to-real ratio.

    Args:
        counts: Real count matrix of shape ``(n_cells, n_genes)``.
        rng: JAX random key for pair selection.
        sim_doublet_ratio: Ratio of synthetic doublets to real cells.

    Returns:
        Synthetic doublet profiles of shape
        ``(n_cells * sim_doublet_ratio, n_genes)``.
    """
    n_cells = counts.shape[0]
    n_synthetic = int(n_cells * sim_doublet_ratio)
    k1, k2 = jax.random.split(rng)
    idx_a = jax.random.randint(k1, (n_synthetic,), 0, n_cells)
    idx_b = jax.random.randint(k2, (n_synthetic,), 0, n_cells)
    return counts[idx_a] + counts[idx_b]


@dataclass(frozen=True)
class DoubletScorerConfig(OperatorConfig):
    """Configuration for Scrublet-style doublet detection.

    Attributes:
        n_neighbors: Base number of nearest neighbors for scoring (adjusted
            upward to account for synthetic pool size).
        expected_doublet_rate: Prior expected fraction of doublets (rho in
            the Bayesian likelihood ratio).
        sim_doublet_ratio: Ratio of synthetic doublets to real cells. Scrublet
            default is 2.0, meaning 2x as many synthetics as real cells.
        n_pca_components: Number of PCA components for embedding.
        n_genes: Number of genes in expression profiles.
        threshold_temperature: Temperature for sigmoid doublet thresholding.
    """

    n_neighbors: int = 30
    expected_doublet_rate: float = 0.06
    sim_doublet_ratio: float = 2.0
    n_pca_components: int = 30
    n_genes: int = 2000
    threshold_temperature: float = 10.0

    def __post_init__(self) -> None:
        """Set stochastic defaults and validate."""
        object.__setattr__(self, "stochastic", True)
        if self.stream_name is None:
            object.__setattr__(self, "stream_name", "sample")
        super().__post_init__()


class DifferentiableDoubletScorer(OperatorModule):
    """Differentiable Scrublet-style doublet detection operator.

    Detects doublets by generating synthetic doublet profiles from random
    cell pairs, embedding real and synthetic cells into PCA space, and
    scoring each real cell via the Bayesian k-NN likelihood ratio from
    Scrublet (Wolock et al., 2019).

    Algorithm:
        1. Generate ``n_cells * sim_doublet_ratio`` synthetic doublets
        2. Concatenate real and synthetic cells
        3. PCA-embed via truncated SVD
        4. Compute pairwise distances in PCA space
        5. Adjust k upward: ``k_adj = round(k * (1 + n_syn / n_cells))``
        6. Count soft synthetic neighbors in each real cell's k-NN
        7. Compute Laplace-smoothed fraction ``q`` of synthetic neighbors
        8. Bayesian likelihood ratio: ``Ld = q * rho / r / denom``
        9. Apply sigmoid threshold for predicted doublet calls

    Args:
        config: DoubletScorerConfig with operator parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = DoubletScorerConfig(n_neighbors=30, n_pca_components=30,
        ...                              n_genes=2000)
        >>> scorer = DifferentiableDoubletScorer(config, rngs=nnx.Rngs(0))
        >>> rng = jax.random.key(0)
        >>> rp = scorer.generate_random_params(rng, {"counts": (500, 2000)})
        >>> result, state, meta = scorer.apply({"counts": counts}, {}, None,
        ...                                    random_params=rp)
        >>> result["doublet_scores"].shape
        (500,)
    """

    def __init__(
        self,
        config: DoubletScorerConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the doublet scorer.

        Args:
            config: Doublet scorer configuration.
            rngs: Random number generators for stochastic operations.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

    def generate_random_params(
        self,
        rng: jax.Array,
        data_shapes: PyTree,
    ) -> jax.Array:
        """Generate random parameters for doublet pair selection.

        Produces two arrays of random cell indices used to form synthetic
        doublets by pairwise summation.

        Args:
            rng: JAX random key.
            data_shapes: PyTree with shapes, must contain ``"counts"`` key
                whose first dimension is the number of cells.

        Returns:
            A JAX random key for reproducible pair generation inside apply.
        """
        return rng

    def _pca_embed(
        self,
        data: Float[Array, "n_total n_genes"],
        n_components: int,
    ) -> Float[Array, "n_total n_components"]:
        """Embed data into PCA space via truncated SVD.

        Centers the data, computes SVD, and projects onto the top
        ``n_components`` principal components.

        Args:
            data: Input matrix of shape ``(n_total, n_genes)``.
            n_components: Number of PCA dimensions to keep.

        Returns:
            PCA embeddings of shape ``(n_total, n_components)``.
        """
        centered = data - jnp.mean(data, axis=0, keepdims=True)
        # Truncated SVD: project onto top-n_components right singular vectors
        _, _, vt = jnp.linalg.svd(centered, full_matrices=False)
        # Clamp n_components to available dimensions
        n_components_eff = min(n_components, vt.shape[0])
        projection = vt[:n_components_eff]  # (n_components_eff, n_genes)
        return centered @ projection.T

    def _compute_soft_knn_synthetic_count(
        self,
        distances: Float[Array, "n_real n_total"],
        n_real: int,
        n_total: int,
        k_adj: int,
    ) -> Float[Array, "n_real"]:
        """Compute soft count of synthetic neighbors in adjusted k-NN.

        Uses a softmax-weighted membership approach: for each real cell,
        the ``k_adj`` smallest distances are selected, and the weighted
        count of synthetic neighbors is returned for the Bayesian
        likelihood-ratio formula.

        Args:
            distances: Distance matrix from real cells to all cells,
                shape ``(n_real, n_total)``.
            n_real: Number of real cells.
            n_total: Total number of cells (real + synthetic).
            k_adj: Adjusted number of nearest neighbors (accounts for
                synthetic pool size).

        Returns:
            Soft synthetic neighbor count per real cell, shape ``(n_real,)``.
        """
        k_eff = min(k_adj, n_total - 1)

        # Create label vector: 0 for real, 1 for synthetic
        is_synthetic = jnp.concatenate(
            [
                jnp.zeros(n_real),
                jnp.ones(n_total - n_real),
            ]
        )

        # Mask self-distances for real cells (first n_real columns correspond to real)
        self_mask = jnp.eye(n_real, n_total) * DISTANCE_MASK_SENTINEL
        masked_distances = distances + self_mask

        # Soft k-NN: convert distances to weights via negative exponential
        # Use temperature scaling for smoother gradients
        sigma = jnp.sort(masked_distances, axis=-1)[:, k_eff - 1 : k_eff]
        sigma = jnp.maximum(sigma, 1e-8)
        weights = jnp.exp(-masked_distances / sigma)

        # Zero out self-connections
        weights = weights * (1.0 - jnp.eye(n_real, n_total))

        # Get top-k mask via soft approximation: use sorted threshold
        sorted_dists = jnp.sort(masked_distances, axis=-1)
        kth_dist = sorted_dists[:, k_eff - 1 : k_eff]  # (n_real, 1)
        # Soft indicator for being within k-NN (sigmoid approximation)
        temperature = 10.0
        knn_mask = soft_ops.greater(kth_dist, masked_distances, softness=1.0 / temperature)

        # Weighted synthetic count (not fraction -- the Bayesian formula needs count)
        masked_weights = weights * knn_mask
        synthetic_weight = masked_weights @ is_synthetic

        return synthetic_weight

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply doublet detection to single-cell count data.

        Implements Scrublet's Bayesian k-NN likelihood-ratio scoring:

        1. Generate ``n_cells * sim_doublet_ratio`` synthetic doublets
        2. Adjust k upward: ``k_adj = round(k * (1 + n_syn / n_cells))``
        3. For each real cell, count synthetic neighbors in its k-NN
        4. Compute Laplace-smoothed fraction ``q = (syn_count + 1) / (k_adj + 2)``
        5. Bayesian likelihood ratio:
           ``Ld = q * rho / r / (1 - rho - q*(1 - rho - rho/r))``

        Args:
            data: Dictionary containing:
                - ``"counts"``: Gene expression matrix ``(n_cells, n_genes)``
            state: Element state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: JAX random key for synthetic doublet generation.
            stats: Not used.

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - ``"counts"``: Original counts
                    - ``"doublet_scores"``: Bayesian likelihood ratio per cell
                    - ``"predicted_doublets"``: Soft doublet predictions in [0, 1]
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        counts = data["counts"]
        n_cells = counts.shape[0]
        config = self.config

        # Use provided random key or fallback
        rng = random_params if random_params is not None else jax.random.key(0)

        # Step 1: Generate synthetic doublets (n_cells * sim_doublet_ratio)
        synthetic = generate_synthetic_doublets(counts, rng, config.sim_doublet_ratio)
        n_synthetic = synthetic.shape[0]

        # Step 2: Combine real + synthetic
        combined = jnp.concatenate([counts, synthetic], axis=0)
        n_total = combined.shape[0]

        # Step 3: PCA embed
        pca = self._pca_embed(combined, config.n_pca_components)

        # Step 4: Pairwise distances in PCA space, then extract real-to-all block
        distances = compute_pairwise_distances(pca)
        real_to_all = distances[:n_cells]

        # Step 5: Adjust k for the enlarged pool (Scrublet convention)
        k_adj = round(config.n_neighbors * (1 + n_synthetic / n_cells))

        # Step 6: Soft k-NN synthetic neighbor count
        syn_neighbor_count = self._compute_soft_knn_synthetic_count(
            real_to_all, n_cells, n_total, k_adj
        )

        # Step 7: Bayesian likelihood-ratio scoring (Scrublet formula)
        # Clip soft synthetic count so it cannot exceed k_adj (soft weights
        # may overshoot the hard budget, making the denominator negative).
        syn_neighbor_count = jnp.clip(syn_neighbor_count, 0.0, float(k_adj))

        # q = Laplace-smoothed fraction of synthetic neighbors
        q = (syn_neighbor_count + 1) / (k_adj + 2)
        rho = config.expected_doublet_rate
        r = n_synthetic / n_cells  # synthetic-to-real ratio

        # Denominator with numerical guard to avoid division by zero
        denominator = jnp.maximum(1 - rho - q * (1 - rho - rho / r), 1e-8)
        doublet_scores = q * rho / r / denominator

        # Step 8: Soft threshold for predicted doublets (sigmoid on score)
        predicted_doublets = soft_ops.greater(
            doublet_scores, 0.5, softness=1.0 / config.threshold_temperature
        )

        transformed_data = {
            **data,
            "doublet_scores": doublet_scores,
            "predicted_doublets": predicted_doublets,
        }

        return transformed_data, state, metadata


# =============================================================================
# Solo-style VAE Doublet Detector
# =============================================================================


@dataclass(frozen=True)
class SoloDetectorConfig(OperatorConfig):
    """Configuration for Solo-style VAE doublet detection.

    Solo (Bernstein et al., Cell Systems 2020) trains a VAE on real cells,
    generates synthetic doublets, encodes both into latent space, and trains
    a binary classifier to distinguish singlets from doublets.

    Attributes:
        n_genes: Number of genes in expression profiles.
        latent_dim: Dimension of the VAE latent space.
        hidden_dims: Hidden layer dimensions for encoder/decoder.
        classifier_hidden_dim: Hidden dimension for the latent-space classifier.
        sim_doublet_ratio: Ratio of synthetic doublets to real cells.
    """

    n_genes: int = 2000
    latent_dim: int = 10
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    classifier_hidden_dim: int = 64
    sim_doublet_ratio: float = 2.0

    def __post_init__(self) -> None:
        """Set stochastic defaults and validate."""
        object.__setattr__(self, "stochastic", True)
        if self.stream_name is None:
            object.__setattr__(self, "stream_name", "sample")
        super().__post_init__()


class DifferentiableSoloDetector(EncoderDecoderOperator):
    """Solo-style VAE doublet detector.

    Detects doublets by encoding cells through a VAE, generating synthetic
    doublets in count space, then classifying real vs synthetic cells in the
    VAE latent space.

    Algorithm:
        1. Generate synthetic doublets by summing random cell pairs
        2. Concatenate real and synthetic counts
        3. Encode all cells through the VAE encoder to obtain (mean, logvar)
        4. Sample latent z via the reparameterization trick
        5. Run a binary classifier on real-cell latents
        6. Return doublet probabilities, labels, and latent representations

    Architecture:
        - Encoder: counts -> log1p -> hidden layers (ReLU) -> (mean, logvar)
        - Decoder: z -> hidden layers (ReLU) -> log_rate
        - Classifier: z -> Linear -> ReLU -> Linear -> sigmoid

    Args:
        config: SoloDetectorConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = SoloDetectorConfig(n_genes=2000, latent_dim=10)
        >>> detector = DifferentiableSoloDetector(config, rngs=nnx.Rngs(42))
        >>> rng = jax.random.key(0)
        >>> rp = detector.generate_random_params(rng, {"counts": (500, 2000)})
        >>> result, _, _ = detector.apply({"counts": counts}, {}, None, random_params=rp)
        >>> result["doublet_probabilities"].shape
        (500,)
    """

    def __init__(
        self,
        config: SoloDetectorConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the Solo VAE doublet detector.

        Args:
            config: Solo detector configuration.
            rngs: Random number generators for initialization and sampling.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        safe_rngs = ensure_rngs(rngs)

        self.n_genes = config.n_genes
        self.stream_name = nnx.static(config.stream_name)

        # --- VAE Encoder ---
        self.encoder_layers = build_mlp_encoder(config.n_genes, config.hidden_dims, rngs=safe_rngs)
        encoder_out_dim = config.hidden_dims[-1] if config.hidden_dims else config.n_genes

        # Latent space projection
        self.fc_mean = nnx.Linear(
            in_features=encoder_out_dim, out_features=config.latent_dim, rngs=safe_rngs
        )
        self.fc_logvar = nnx.Linear(
            in_features=encoder_out_dim, out_features=config.latent_dim, rngs=safe_rngs
        )

        # --- VAE Decoder ---
        self.decoder_layers = build_mlp_decoder(
            config.latent_dim, config.hidden_dims, rngs=safe_rngs
        )
        decoder_out_dim = config.hidden_dims[0] if config.hidden_dims else config.latent_dim

        self.fc_output = nnx.Linear(
            in_features=decoder_out_dim, out_features=config.n_genes, rngs=safe_rngs
        )

        # --- Classifier (operates on latent z) ---
        self.classifier_hidden = nnx.Linear(
            in_features=config.latent_dim,
            out_features=config.classifier_hidden_dim,
            rngs=safe_rngs,
        )
        self.classifier_output = nnx.Linear(
            in_features=config.classifier_hidden_dim,
            out_features=1,
            rngs=safe_rngs,
        )

    def generate_random_params(
        self,
        rng: jax.Array,
        data_shapes: PyTree,
    ) -> jax.Array:
        """Generate random parameters for synthetic doublet pair selection.

        Args:
            rng: JAX random key.
            data_shapes: PyTree with shapes (must contain ``"counts"`` key).

        Returns:
            A JAX random key for reproducible pair generation inside apply.
        """
        return rng

    def encode(
        self,
        counts: Float[Array, "batch n_genes"],
    ) -> tuple[Float[Array, "batch latent_dim"], Float[Array, "batch latent_dim"]]:
        """Encode counts to latent distribution parameters.

        Args:
            counts: Gene expression counts, shape ``(batch, n_genes)``.

        Returns:
            Tuple of (mean, logvar) for latent distribution.
        """
        x = jnp.log1p(counts)
        x = forward_mlp(self.encoder_layers, x)

        mean = self.fc_mean(x)
        logvar = jnp.clip(self.fc_logvar(x), -10.0, 10.0)

        return mean, logvar

    def decode(
        self,
        z: Float[Array, "batch latent_dim"],
    ) -> Float[Array, "batch n_genes"]:
        """Decode latent representation to gene expression log-rates.

        Args:
            z: Latent representation, shape ``(batch, latent_dim)``.

        Returns:
            Log rates for each gene, shape ``(batch, n_genes)``.
        """
        x = forward_mlp(self.decoder_layers, z)
        return self.fc_output(x)

    def classify(
        self,
        z: Float[Array, "batch latent_dim"],
    ) -> Float[Array, "batch"]:
        """Classify latent representations as singlet vs doublet.

        Args:
            z: Latent representations, shape ``(batch, latent_dim)``.

        Returns:
            Doublet probabilities in [0, 1], shape ``(batch,)``.
        """
        h = nnx.relu(self.classifier_hidden(z))
        logits = self.classifier_output(h)
        return jax.nn.sigmoid(logits).squeeze(-1)

    def compute_elbo_loss(
        self,
        counts: Float[Array, "batch n_genes"],
    ) -> Float[Array, ""]:
        """Compute negative ELBO loss for the VAE.

        Uses gaussian_kl_divergence from artifex for the KL term and
        Poisson NLL for reconstruction.

        Args:
            counts: Gene expression counts, shape ``(batch, n_genes)``.

        Returns:
            Negative ELBO (reconstruction loss + KL divergence).
        """
        mean, logvar = self.encode(counts)
        z = self.reparameterize(mean, logvar)
        log_rate = self.decode(z)

        # Poisson NLL reconstruction loss
        rate = jnp.exp(log_rate)
        recon_loss = jnp.sum(rate - counts * log_rate)

        # KL divergence via artifex
        kl = gaussian_kl_divergence(mean, logvar, reduction="sum")

        return recon_loss + kl

    def compute_solo_loss(
        self,
        counts: Float[Array, "batch n_genes"],
        random_params: jax.Array,
        classifier_weight: float = 1.0,
    ) -> dict[str, Float[Array, ""]]:
        """Full Solo training loss: VAE ELBO + classifier binary cross-entropy.

        Generates synthetic doublets, encodes all cells (real + synthetic)
        through the VAE, computes the ELBO on the combined set, and adds
        a binary cross-entropy term from the classifier distinguishing
        singlets from doublets in latent space.

        Args:
            counts: Real gene expression counts, shape ``(n_real, n_genes)``.
            random_params: JAX random key for synthetic doublet generation.
            classifier_weight: Weight for the classifier BCE term
                (default 1.0).

        Returns:
            Dictionary with ``"total_loss"``, ``"elbo"``, and
            ``"classifier_loss"`` scalar entries.
        """
        n_real = counts.shape[0]

        # 1. Generate synthetic doublets
        synthetic = generate_synthetic_doublets(
            counts, random_params, self.config.sim_doublet_ratio
        )
        n_synthetic = synthetic.shape[0]

        # 2. Combine real + synthetic
        combined = jnp.concatenate([counts, synthetic], axis=0)

        # 3. Encode all to latent space
        mean, logvar = self.encode(combined)
        z = self.reparameterize(mean, logvar)

        # 4. VAE ELBO on all cells
        log_rate = self.decode(z)
        rate = jnp.exp(log_rate)
        recon_loss = jnp.sum(rate - combined * log_rate)
        kl = gaussian_kl_divergence(mean, logvar, reduction="sum")
        elbo = recon_loss + kl

        # 5. Classifier BCE on all cells
        # Labels: 0 for real (first n_real), 1 for synthetic
        labels = jnp.concatenate([jnp.zeros(n_real), jnp.ones(n_synthetic)])
        probs = self.classify(z)  # returns sigmoid probabilities
        # classify returns sigmoid(logits), so we use log-sigmoid formulation
        bce = -jnp.mean(
            labels * jnp.log(probs + 1e-8) + (1.0 - labels) * jnp.log(1.0 - probs + 1e-8)
        )

        # 6. Total = ELBO + classifier_weight * BCE
        total = elbo + classifier_weight * bce

        return {"total_loss": total, "elbo": elbo, "classifier_loss": bce}

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply Solo-style VAE doublet detection.

        Steps:
            1. Generate synthetic doublets from random cell pairs
            2. Concatenate real and synthetic counts
            3. Encode all cells to latent space (mean, logvar)
            4. Sample z via reparameterization trick
            5. Run classifier on real-cell latents
            6. Return probabilities, labels, and latent for real cells only

        Args:
            data: Dictionary containing:
                - ``"counts"``: Gene expression matrix ``(n_cells, n_genes)``
            state: Element state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: JAX random key for synthetic doublet generation.
            stats: Not used.

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - ``"counts"``: Original counts
                    - ``"doublet_probabilities"``: Per-cell doublet probability
                    - ``"doublet_labels"``: Soft binary labels (sigmoid-thresholded)
                    - ``"latent"``: Latent representations for real cells
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        counts = data["counts"]
        n_cells = counts.shape[0]
        config = self.config

        # Use provided random key or fallback
        rng = random_params if random_params is not None else jax.random.key(0)

        # Step 1: Generate synthetic doublets
        synthetic = generate_synthetic_doublets(counts, rng, config.sim_doublet_ratio)

        # Step 2: Combine real + synthetic counts
        combined = jnp.concatenate([counts, synthetic], axis=0)

        # Step 3: Encode all to latent space
        mean, logvar = self.encode(combined)

        # Step 4: Sample z via reparameterization trick
        z = self.reparameterize(mean, logvar)

        # Step 5: Extract real-cell latents and classify
        z_real = z[:n_cells]
        doublet_probabilities = self.classify(z_real)

        # Step 6: Soft binary labels via thresholding at 0.5
        doublet_labels = jnp.round(doublet_probabilities)

        transformed_data = {
            **data,
            "doublet_probabilities": doublet_probabilities,
            "doublet_labels": doublet_labels,
            "latent": z_real,
        }

        return transformed_data, state, metadata
