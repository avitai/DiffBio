"""End-to-end differentiable single-cell analysis pipeline.

This module provides a complete single-cell RNA-seq analysis pipeline that composes:
1. Ambient RNA removal - CellBender-style decontamination
2. VAE normalization - scVI-style count normalization
3. Batch correction - Harmony-style integration
4. Dimensionality reduction - Parametric UMAP
5. Clustering - Soft k-means clustering

The pipeline is fully differentiable, enabling gradient-based optimization
of all analysis components jointly.
"""

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float

from diffbio.operators.normalization import (
    DifferentiableUMAP,
    UMAPConfig,
    VAENormalizer,
    VAENormalizerConfig,
)
from diffbio.operators.singlecell import (
    AmbientRemovalConfig,
    BatchCorrectionConfig,
    DifferentiableAmbientRemoval,
    DifferentiableHarmony,
    SoftClusteringConfig,
    SoftKMeansClustering,
)


@dataclass
class SingleCellPipelineConfig(OperatorConfig):
    # pylint: disable=too-many-instance-attributes
    """Configuration for the single-cell analysis pipeline.

    Attributes:
        n_genes: Number of genes in the expression matrix.
        n_clusters: Number of clusters for soft k-means.
        latent_dim: Dimension of the VAE latent space.
        hidden_dims: Hidden layer dimensions for VAE.
        umap_n_components: Number of UMAP output dimensions.
        batch_correction_clusters: Number of clusters for Harmony.
        batch_correction_iterations: Number of Harmony iterations.
        clustering_temperature: Temperature for soft clustering.
        enable_ambient_removal: Whether to enable ambient RNA removal.
        enable_batch_correction: Whether to enable batch correction.
        enable_dim_reduction: Whether to enable UMAP dimensionality reduction.
        enable_clustering: Whether to enable soft clustering.
        stochastic: Whether the pipeline uses stochastic operations.
        stream_name: RNG stream name for stochastic operations.
    """

    n_genes: int = 2000
    n_clusters: int = 10
    latent_dim: int = 64
    hidden_dims: tuple[int, ...] = (128, 64)
    umap_n_components: int = 2
    batch_correction_clusters: int = 100
    batch_correction_iterations: int = 10
    clustering_temperature: float = 1.0
    enable_ambient_removal: bool = True
    enable_batch_correction: bool = True
    enable_dim_reduction: bool = True
    enable_clustering: bool = True
    stochastic: bool = field(default=True, repr=False)
    stream_name: str | None = field(default="sample", repr=False)


class SingleCellPipeline(OperatorModule):
    """End-to-end differentiable single-cell analysis pipeline.

    This pipeline processes single-cell RNA-seq data through multiple analysis steps:

    Input data structure:
        - counts: Float[Array, "n_cells n_genes"] - Raw count matrix
        - ambient_profile: Float[Array, "n_genes"] - Ambient expression profile
        - batch_labels: Int[Array, "n_cells"] - Batch assignments

    Output data structure (adds):
        - decontaminated_counts: Ambient-removed counts (if enabled)
        - normalized: VAE-normalized expression
        - latent: Latent space representation
        - corrected_embeddings: Batch-corrected embeddings (if enabled)
        - embeddings_2d: 2D UMAP embeddings (if enabled)
        - cluster_assignments: Soft cluster assignments

    The pipeline is fully differentiable, supporting gradient-based training
    to optimize all components jointly for tasks like:
    - Supervised cell type classification
    - Semi-supervised clustering
    - Multi-task learning across batches

    Example:
        ```python
        config = SingleCellPipelineConfig(n_genes=2000, n_clusters=10)
        pipeline = SingleCellPipeline(config, rngs=nnx.Rngs(42))
        result, state, meta = pipeline.apply(data, {}, None)
        clusters = result["cluster_assignments"]
        ```
    """

    def __init__(
        self,
        config: SingleCellPipelineConfig,
        *,
        rngs: nnx.Rngs,
        name: str | None = None,
    ):
        """Initialize the single-cell analysis pipeline.

        Args:
            config: Pipeline configuration.
            rngs: Random number generators for parameter initialization.
            name: Optional name for the pipeline.
        """
        super().__init__(config, rngs=rngs, name=name)

        # 1. Ambient RNA removal (optional)
        self.ambient_removal = (
            DifferentiableAmbientRemoval(
                AmbientRemovalConfig(
                    n_genes=config.n_genes,
                    latent_dim=config.latent_dim,
                    hidden_dims=list(config.hidden_dims),
                ),
                rngs=rngs,
            )
            if config.enable_ambient_removal
            else None
        )

        # 2. VAE normalization (always enabled - core component)
        self.vae_normalizer = VAENormalizer(
            VAENormalizerConfig(
                n_genes=config.n_genes,
                latent_dim=config.latent_dim,
                hidden_dims=list(config.hidden_dims),
            ),
            rngs=rngs,
        )

        # 3. Batch correction (optional)
        self.batch_correction = (
            DifferentiableHarmony(
                BatchCorrectionConfig(
                    n_clusters=config.batch_correction_clusters,
                    n_features=config.latent_dim,  # Must match latent dimension
                    n_iterations=config.batch_correction_iterations,
                ),
                rngs=rngs,
            )
            if config.enable_batch_correction
            else None
        )

        # 4. Dimensionality reduction (optional)
        self.dim_reduction = (
            DifferentiableUMAP(
                UMAPConfig(
                    input_features=config.latent_dim,
                    n_components=config.umap_n_components,
                ),
                rngs=rngs,
            )
            if config.enable_dim_reduction
            else None
        )

        # 5. Clustering (optional but typically used)
        self.clustering = (
            SoftKMeansClustering(
                SoftClusteringConfig(
                    n_clusters=config.n_clusters,
                    n_features=config.latent_dim,
                    temperature=config.clustering_temperature,
                ),
                rngs=rngs,
            )
            if config.enable_clustering
            else None
        )

    def apply(
        self,
        data: dict[str, Array],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,  # noqa: ARG002
        stats: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[dict[str, Array], dict[str, Any], dict[str, Any] | None]:
        """Apply the full single-cell analysis pipeline.

        Args:
            data: Input data containing:
                - counts: Float[Array, "n_cells n_genes"]
                - ambient_profile: Float[Array, "n_genes"]
                - batch_labels: Int[Array, "n_cells"]
            state: Element state (passed through).
            metadata: Element metadata (passed through).
            random_params: Random parameters for stochastic operations.
            stats: Optional statistics dict.

        Returns:
            Tuple of (output_data, state, metadata) where output_data contains
            all input keys plus analysis outputs.
        """
        counts = data["counts"]
        n_cells = counts.shape[0]

        # Step 1: Ambient RNA removal (optional)
        if self.ambient_removal is not None:
            ambient_data = {
                "counts": counts,
                "ambient_profile": data["ambient_profile"],
            }
            ambient_result, _, _ = self.ambient_removal.apply(ambient_data, {}, None)
            decontaminated = ambient_result["decontaminated_counts"]
        else:
            decontaminated = counts

        # Step 2: VAE normalization (per-cell using vmap for efficiency)
        # The VAE normalizer expects single-cell input with library_size
        def normalize_cell(cell_counts: Float[Array, "n_genes"]) -> dict[str, Array]:
            # Compute library size (total counts per cell)
            library_size = cell_counts.sum()
            vae_data = {"counts": cell_counts, "library_size": library_size}
            result, _, _ = self.vae_normalizer.apply(vae_data, {}, None)
            return result

        # Use vmap for batch processing
        vmap_normalize = jax.vmap(normalize_cell)
        normalized_results = vmap_normalize(decontaminated)

        normalized = normalized_results["normalized"]
        latent = normalized_results["latent_z"]  # VAENormalizer outputs latent_z

        # Step 3: Batch correction (optional)
        if self.batch_correction is not None:
            batch_data = {
                "embeddings": latent,
                "batch_labels": data["batch_labels"],
            }
            batch_result, _, _ = self.batch_correction.apply(batch_data, {}, None)
            corrected_embeddings = batch_result["corrected_embeddings"]
        else:
            corrected_embeddings = latent

        # Step 4: Dimensionality reduction (optional)
        if self.dim_reduction is not None:
            umap_data = {"features": corrected_embeddings}
            umap_result, _, _ = self.dim_reduction.apply(umap_data, {}, None)
            embeddings_2d = umap_result["embedding"]  # UMAP outputs singular "embedding"
        else:
            # Use first 2 dimensions of latent if no UMAP
            embeddings_2d = corrected_embeddings[:, : self.config.umap_n_components]

        # Step 5: Clustering (optional)
        if self.clustering is not None:
            cluster_data = {"embeddings": corrected_embeddings}
            cluster_result, _, _ = self.clustering.apply(cluster_data, {}, None)
            cluster_assignments = cluster_result["cluster_assignments"]
        else:
            # Return uniform assignments if no clustering
            n_clusters = self.config.n_clusters
            cluster_assignments = jnp.ones((n_cells, n_clusters)) / n_clusters

        # Build output preserving input keys
        output_data = {
            **data,
            "normalized": normalized,
            "latent": latent,
            "corrected_embeddings": corrected_embeddings,
            "embeddings_2d": embeddings_2d,
            "cluster_assignments": cluster_assignments,
        }

        # Add optional outputs
        if self.ambient_removal is not None:
            output_data["decontaminated_counts"] = decontaminated

        return output_data, state, metadata


def create_single_cell_pipeline(
    n_genes: int = 2000,
    n_clusters: int = 10,
    latent_dim: int = 64,
    umap_n_components: int = 2,
    enable_ambient_removal: bool = True,
    enable_batch_correction: bool = True,
    enable_dim_reduction: bool = True,
    enable_clustering: bool = True,
    seed: int = 42,
) -> SingleCellPipeline:
    """Factory function to create a single-cell analysis pipeline.

    Args:
        n_genes: Number of genes in the expression matrix.
        n_clusters: Number of clusters for soft k-means.
        latent_dim: Dimension of the VAE latent space.
        umap_n_components: Number of UMAP output dimensions.
        enable_ambient_removal: Whether to enable ambient RNA removal.
        enable_batch_correction: Whether to enable batch correction.
        enable_dim_reduction: Whether to enable UMAP.
        enable_clustering: Whether to enable soft clustering.
        seed: Random seed.

    Returns:
        Configured SingleCellPipeline instance.
    """
    config = SingleCellPipelineConfig(
        n_genes=n_genes,
        n_clusters=n_clusters,
        latent_dim=latent_dim,
        umap_n_components=umap_n_components,
        enable_ambient_removal=enable_ambient_removal,
        enable_batch_correction=enable_batch_correction,
        enable_dim_reduction=enable_dim_reduction,
        enable_clustering=enable_clustering,
    )
    rngs = nnx.Rngs(seed)
    return SingleCellPipeline(config, rngs=rngs)
