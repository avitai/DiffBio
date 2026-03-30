#!/usr/bin/env python3
"""scVI-style VAE Benchmark for DiffBio.

This benchmark evaluates DiffBio's VAENormalizer with ZINB likelihood
against scVI-like metrics on synthetic PBMC-like data.

Metrics (all from calibrax):
- ELBO: from VAENormalizer.compute_elbo_loss()
- Reconstruction MSE: mean squared error between counts and reconstructed
- Latent silhouette: biological conservation via silhouette_score
- Batch ASW: batch removal quality via 1 - abs(silhouette_score)
- ARI: adjusted Rand index of k-means clusters vs true labels
- NMI: normalized mutual information of k-means clusters vs true labels

Reference architecture: scVI (Lopez et al., 2018)
- n_hidden=128, n_latent=10, n_layers=1, gene_likelihood='zinb'

Usage:
    python benchmarks/singlecell/scvi_benchmark.py
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime

import jax
import jax.numpy as jnp
import optax
from calibrax.metrics.functional.clustering import (
    adjusted_rand_index,
    normalized_mutual_information_clustering,
    silhouette_score,
)
from flax import nnx

from benchmarks._common import check_gradient_flow, save_benchmark_result
from diffbio.operators.normalization.vae_normalizer import (
    VAENormalizer,
    VAENormalizerConfig,
)
from diffbio.operators.singlecell.soft_clustering import (
    SoftClusteringConfig,
    SoftKMeansClustering,
)


@dataclass(frozen=True, kw_only=True)
class ScVIBenchmarkResult:
    """Results from scVI-style benchmark.

    Attributes:
        timestamp: ISO-format timestamp of the benchmark run.
        n_cells: Number of cells in synthetic dataset.
        n_genes: Number of genes in synthetic dataset.
        n_batches: Number of experimental batches.
        n_types: Number of cell types.
        n_epochs: Number of training epochs.
        elbo: Final ELBO training loss.
        reconstruction_mse: Mean squared reconstruction error.
        silhouette: Silhouette score for cell-type separation.
        batch_asw: Batch ASW (1 - |batch silhouette|).
        ari: Adjusted Rand index of clusters vs true labels.
        nmi: Normalized mutual information of clusters.
        gradient_norm: L2 norm of gradients through the model.
        gradient_nonzero: Whether gradient norm exceeds threshold.
        training_time_ms: Wall-clock training time in milliseconds.
    """

    timestamp: str
    n_cells: int
    n_genes: int
    n_batches: int
    n_types: int
    n_epochs: int
    elbo: float
    reconstruction_mse: float
    silhouette: float
    batch_asw: float
    ari: float
    nmi: float
    gradient_norm: float
    gradient_nonzero: bool
    training_time_ms: float


def generate_synthetic_pbmc_data(
    n_cells: int = 500,
    n_genes: int = 200,
    n_batches: int = 2,
    n_types: int = 3,
    seed: int = 42,
) -> dict[str, jax.Array | int]:
    """Generate synthetic scRNA-seq data with known structure.

    Simulates PBMC-like data with:
    - Per-type mean expression profiles (different for each cell type)
    - Per-batch shift (additive on log scale)
    - Negative binomial sampling for realistic count distributions

    Args:
        n_cells: Total number of cells to generate.
        n_genes: Number of genes.
        n_batches: Number of experimental batches.
        n_types: Number of cell types.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys: counts, library_size, batch_labels,
        cell_type_labels, n_cells, n_genes, n_batches, n_types.
    """
    key = jax.random.key(seed)
    keys = jax.random.split(key, 6)

    # Per-type mean expression profiles on log scale
    type_log_means = (
        jax.random.normal(keys[0], (n_types, n_genes)) * 1.5 + 2.0
    )

    # Per-batch additive shift on log scale (batch effects)
    batch_shifts = (
        jax.random.normal(keys[1], (n_batches, n_genes)) * 0.5
    )

    # Assign cells to types (roughly equal)
    cells_per_type = n_cells // n_types
    type_labels_list: list[int] = []
    for t in range(n_types):
        count = (
            cells_per_type
            if t < n_types - 1
            else n_cells - len(type_labels_list)
        )
        type_labels_list.extend([t] * count)
    cell_type_labels = jnp.array(type_labels_list)

    # Assign cells to batches (roughly equal)
    cells_per_batch = n_cells // n_batches
    batch_labels_list: list[int] = []
    for b in range(n_batches):
        count = (
            cells_per_batch
            if b < n_batches - 1
            else n_cells - len(batch_labels_list)
        )
        batch_labels_list.extend([b] * count)
    batch_labels = jnp.array(batch_labels_list)

    # Build per-cell log-mean expression: type mean + batch shift + noise
    cell_log_means = (
        type_log_means[cell_type_labels] + batch_shifts[batch_labels]
    )
    cell_noise = jax.random.normal(keys[2], (n_cells, n_genes)) * 0.3
    cell_log_means = cell_log_means + cell_noise

    # Convert to rates (ensure positive)
    rates = jnp.exp(cell_log_means)

    # Negative binomial sampling: use Gamma-Poisson mixture
    dispersion = 5.0
    gamma_samples = jax.random.gamma(
        keys[3], dispersion, (n_cells, n_genes)
    )
    scaled_rates = rates * gamma_samples / dispersion
    counts = jax.random.poisson(keys[4], scaled_rates).astype(jnp.float32)

    # Library size per cell
    library_size = jnp.sum(counts, axis=-1)

    return {
        "counts": counts,
        "library_size": library_size,
        "batch_labels": batch_labels,
        "cell_type_labels": cell_type_labels,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "n_batches": n_batches,
        "n_types": n_types,
    }


def create_jit_train_step(
    model: VAENormalizer,
    nnx_optimizer: nnx.Optimizer,
) -> Callable[..., jax.Array]:
    """Create and return a JIT-compiled training step function.

    Uses nnx.Optimizer + @nnx.jit for correct state management,
    following the established pattern in the codebase.

    Args:
        model: VAENormalizer model (mutated in-place by optimizer).
        nnx_optimizer: NNX Optimizer wrapping the model.

    Returns:
        JIT-compiled train_step function.
    """

    @nnx.jit
    def train_step(
        m: VAENormalizer,
        opt: nnx.Optimizer,
        counts_batch: jax.Array,
        library_size_batch: jax.Array,
    ) -> jax.Array:
        """Single JIT-compiled training step.

        Args:
            m: VAENormalizer model.
            opt: NNX optimizer.
            counts_batch: Batch of count vectors (n_cells, n_genes).
            library_size_batch: Library sizes (n_cells,).

        Returns:
            Scalar loss value.
        """

        def loss_fn(
            model_inner: VAENormalizer,
        ) -> jax.Array:
            """Compute mean ELBO loss over batch."""

            def per_cell_loss(
                counts_i: jax.Array,
                lib_i: jax.Array,
            ) -> jax.Array:
                return model_inner.compute_elbo_loss(
                    counts_i, lib_i
                )

            losses = jax.vmap(per_cell_loss)(
                counts_batch, library_size_batch
            )
            return jnp.mean(losses)

        loss, grads = nnx.value_and_grad(
            loss_fn, argnums=nnx.DiffState(0, nnx.Param)
        )(m)
        opt.update(m, grads)
        return loss

    return train_step


def _compute_latent_representations(
    model: VAENormalizer,
    counts: jax.Array,
    library_size: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute latent means and reconstructions for all cells.

    Args:
        model: Trained VAENormalizer.
        counts: Count matrix (n_cells, n_genes).
        library_size: Library sizes (n_cells,).

    Returns:
        Tuple of (latent_means, reconstructed_rates).
    """

    def encode_cell(
        counts_i: jax.Array,
        lib_i: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Encode a single cell and decode for reconstruction."""
        mean, _ = model.encode(counts_i)
        decode_out = model.decode(mean, lib_i)
        reconstructed = jnp.exp(decode_out["log_rate"])
        return mean, reconstructed

    latent_means, reconstructed = jax.vmap(encode_cell)(
        counts, library_size
    )
    return latent_means, reconstructed


def _cluster_latent(
    latent: jax.Array,
    n_clusters: int,
    seed: int = 0,
) -> jax.Array:
    """Cluster latent representations using SoftKMeansClustering.

    Args:
        latent: Latent representations (n_cells, latent_dim).
        n_clusters: Number of clusters.
        seed: Random seed.

    Returns:
        Hard cluster labels (n_cells,).
    """
    config = SoftClusteringConfig(
        n_clusters=n_clusters,
        n_features=latent.shape[1],
        temperature=0.1,
    )
    clusterer = SoftKMeansClustering(config, rngs=nnx.Rngs(seed))
    result, _, _ = clusterer.apply({"embeddings": latent}, {}, None)
    return result["cluster_labels"]


def _run_scvi_benchmark_inner(
    *,
    n_cells: int,
    n_genes: int,
    n_epochs: int,
    seed: int,
) -> dict[str, float]:
    """Train VAENormalizer and evaluate against scVI-like metrics.

    This is the internal implementation that performs the full
    training and evaluation pipeline.

    Pipeline:
    1. Generate synthetic PBMC-like data
    2. Create VAENormalizer with ZINB likelihood
    3. JIT-compiled training loop
    4. Compute evaluation metrics via calibrax
    5. Check gradient flow via benchmarks._common

    Args:
        n_cells: Number of cells in synthetic data.
        n_genes: Number of genes in synthetic data.
        n_epochs: Number of training epochs.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with metric names as keys and float values.
    """
    # 1. Generate data
    data = generate_synthetic_pbmc_data(
        n_cells=n_cells, n_genes=n_genes, seed=seed
    )
    counts = jnp.asarray(data["counts"])
    library_size = jnp.asarray(data["library_size"])
    cell_type_labels = jnp.asarray(data["cell_type_labels"])
    batch_labels = jnp.asarray(data["batch_labels"])
    n_types = int(data["n_types"])

    # 2. Create VAENormalizer with ZINB (scVI-like config)
    config = VAENormalizerConfig(
        n_genes=n_genes,
        latent_dim=10,
        hidden_dims=[128],
        likelihood="zinb",
    )
    model = VAENormalizer(config, rngs=nnx.Rngs(seed))

    # 3. JIT-compiled training loop using nnx.Optimizer
    nnx_optimizer = nnx.Optimizer(
        model, optax.adam(1e-3), wrt=nnx.Param
    )
    train_step = create_jit_train_step(model, nnx_optimizer)

    start_time = time.time()
    for _epoch in range(n_epochs):
        loss = train_step(model, nnx_optimizer, counts, library_size)
    training_time_ms = (time.time() - start_time) * 1000

    # model is updated in-place by nnx.Optimizer
    trained_model = model

    # 4. Compute metrics
    elbo_value = float(loss)

    # Reconstruction MSE
    latent_means, reconstructed = _compute_latent_representations(
        trained_model, counts, library_size
    )
    reconstruction_mse = float(
        jnp.mean((counts - reconstructed) ** 2)
    )

    # Latent silhouette (bio conservation)
    sil = float(silhouette_score(latent_means, cell_type_labels))

    # Batch ASW (batch removal): ideal is 0 silhouette on batch labels
    batch_sil = float(silhouette_score(latent_means, batch_labels))
    batch_asw = 1.0 - abs(batch_sil)

    # Cluster with SoftKMeans and evaluate
    cluster_labels = _cluster_latent(
        latent_means, n_clusters=n_types, seed=seed
    )
    ari = float(adjusted_rand_index(cell_type_labels, cluster_labels))
    nmi = float(
        normalized_mutual_information_clustering(
            cell_type_labels, cluster_labels
        )
    )

    # 5. Gradient flow check via shared utility
    def grad_loss_fn(
        m: VAENormalizer,
        c: jax.Array,
        ls: jax.Array,
    ) -> jax.Array:
        """Scalar loss for gradient flow verification."""

        def per_cell(ci: jax.Array, li: jax.Array) -> jax.Array:
            return m.compute_elbo_loss(ci, li)

        return jnp.mean(jax.vmap(per_cell)(c, ls))

    grad_result = check_gradient_flow(
        grad_loss_fn, trained_model, counts, library_size
    )

    return {
        "elbo": elbo_value,
        "reconstruction_mse": reconstruction_mse,
        "silhouette": sil,
        "batch_asw": batch_asw,
        "ari": ari,
        "nmi": nmi,
        "gradient_norm": grad_result.gradient_norm,
        "gradient_nonzero": grad_result.gradient_nonzero,
        "training_time_ms": training_time_ms,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "n_epochs": n_epochs,
    }


def run_benchmark(*, quick: bool = False) -> ScVIBenchmarkResult:
    """Run the complete scVI-style VAE benchmark.

    Args:
        quick: If True, use reduced dataset sizes and fewer epochs
            for faster CI runs (n_cells=50, n_genes=20, n_epochs=5).

    Returns:
        Benchmark results dataclass.
    """
    n_cells = 50 if quick else 500
    n_genes = 20 if quick else 200
    n_epochs = 5 if quick else 50
    seed = 42

    print("=" * 60)
    print("DiffBio scVI-style VAE Benchmark")
    if quick:
        print("  (quick mode)")
    print("=" * 60)

    print("\nGenerating synthetic PBMC-like data...")
    data = generate_synthetic_pbmc_data(
        n_cells=n_cells, n_genes=n_genes, seed=seed
    )
    print(f"  Cells: {data['n_cells']}")
    print(f"  Genes: {data['n_genes']}")
    print(f"  Batches: {data['n_batches']}")
    print(f"  Cell types: {data['n_types']}")

    print("\nTraining VAENormalizer (ZINB likelihood)...")
    results = _run_scvi_benchmark_inner(
        n_cells=n_cells,
        n_genes=n_genes,
        n_epochs=n_epochs,
        seed=seed,
    )

    print("\nResults:")
    print(f"  ELBO:               {results['elbo']:.2f}")
    print(f"  Reconstruction MSE: {results['reconstruction_mse']:.4f}")
    print(f"  Silhouette (bio):   {results['silhouette']:.4f}")
    print(f"  Batch ASW:          {results['batch_asw']:.4f}")
    print(f"  ARI:                {results['ari']:.4f}")
    print(f"  NMI:                {results['nmi']:.4f}")
    print(f"  Gradient norm:      {results['gradient_norm']:.6f}")
    print(f"  Gradient non-zero:  {results['gradient_nonzero']}")
    print(
        f"  Training time:      {results['training_time_ms']:.0f}ms"
    )

    benchmark_result = ScVIBenchmarkResult(
        timestamp=datetime.now().isoformat(),
        n_cells=n_cells,
        n_genes=n_genes,
        n_batches=int(data["n_batches"]),
        n_types=int(data["n_types"]),
        n_epochs=n_epochs,
        elbo=results["elbo"],
        reconstruction_mse=results["reconstruction_mse"],
        silhouette=results["silhouette"],
        batch_asw=results["batch_asw"],
        ari=results["ari"],
        nmi=results["nmi"],
        gradient_norm=results["gradient_norm"],
        gradient_nonzero=results["gradient_nonzero"],
        training_time_ms=results["training_time_ms"],
    )

    print("\n" + "=" * 60)
    print("Benchmark complete.")
    print("=" * 60)

    return benchmark_result


def main() -> None:
    """Run benchmark and save results."""
    result = run_benchmark()
    result_dict = asdict(result)
    output_path = save_benchmark_result(
        result_dict,
        domain="singlecell",
        benchmark_name="scvi",
    )
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
