#!/usr/bin/env python3
"""Multi-omics integration benchmark for DiffBio.

Evaluates DiffBio's multi-omics operators for correctness,
differentiability, and throughput:

- DifferentiableMultiOmicsVAE  (Product-of-Experts fusion)
- SpatialDeconvolution         (cell-type deconvolution)
- HiCContactAnalysis           (chromatin contact analysis)
- DifferentiableSpatialGeneDetector  (spatially variable genes)

Usage:
    python benchmarks/multiomics/multiomics_benchmark.py
    python benchmarks/multiomics/multiomics_benchmark.py --quick
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from benchmarks._gradient import check_gradient_flow
from diffbio.operators.multiomics.hic_contact import (
    HiCContactAnalysis,
    HiCContactAnalysisConfig,
)
from diffbio.operators.multiomics.multiomics_vae import (
    DifferentiableMultiOmicsVAE,
    MultiOmicsVAEConfig,
)
from diffbio.operators.multiomics.spatial_deconvolution import (
    SpatialDeconvolution,
    SpatialDeconvolutionConfig,
)
from diffbio.operators.multiomics.spatial_gene_detection import (
    DifferentiableSpatialGeneDetector,
    SpatialGeneDetectorConfig,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class MultiOmicsBenchmarkResult:
    """Aggregated results from multi-omics benchmark.

    Attributes:
        timestamp: ISO-format timestamp of the run.
        quick: Whether this was a quick (reduced-size) run.
        vae_results: Per-metric dict for the multi-omics VAE.
        deconv_results: Per-metric dict for spatial deconvolution.
        hic_results: Per-metric dict for Hi-C contact analysis.
        spatial_gene_results: Per-metric dict for spatial gene detection.
        all_shapes_correct: Whether all output shapes matched expectations.
        all_gradients_flow: Whether all operators produced non-zero grads.
    """

    timestamp: str
    quick: bool
    vae_results: dict[str, Any] = field(default_factory=dict)
    deconv_results: dict[str, Any] = field(default_factory=dict)
    hic_results: dict[str, Any] = field(default_factory=dict)
    spatial_gene_results: dict[str, Any] = field(default_factory=dict)
    all_shapes_correct: bool = False
    all_gradients_flow: bool = False


# ------------------------------------------------------------------
# Synthetic data generators
# ------------------------------------------------------------------


def _generate_vae_data(
    n_cells: int,
    n_rna_genes: int,
    n_atac_peaks: int,
    seed: int = 0,
) -> dict[str, jax.Array]:
    """Generate synthetic multi-omics count matrices.

    Args:
        n_cells: Number of cells.
        n_rna_genes: Number of RNA genes (first modality).
        n_atac_peaks: Number of ATAC peaks (second modality).
        seed: Random seed.

    Returns:
        Dictionary with ``rna_counts`` and ``atac_counts`` arrays.
    """
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)
    rna = jnp.abs(jax.random.normal(k1, (n_cells, n_rna_genes))) * 5.0
    atac = jnp.abs(jax.random.normal(k2, (n_cells, n_atac_peaks))) * 3.0
    return {"rna_counts": rna, "atac_counts": atac}


def _generate_deconv_data(
    n_spots: int,
    n_genes: int,
    n_cell_types: int,
    seed: int = 1,
) -> dict[str, jax.Array]:
    """Generate synthetic spatial deconvolution inputs.

    Creates spot expression, reference profiles, and 2-D coordinates.

    Args:
        n_spots: Number of spatial spots.
        n_genes: Number of genes.
        n_cell_types: Number of reference cell types.
        seed: Random seed.

    Returns:
        Dictionary with ``spot_expression``, ``reference_profiles``,
        and ``coordinates`` arrays.
    """
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    spot_expr = jnp.abs(
        jax.random.normal(k1, (n_spots, n_genes))
    ) * 4.0
    ref_profiles = jnp.abs(
        jax.random.normal(k2, (n_cell_types, n_genes))
    ) * 3.0
    coords = jax.random.uniform(k3, (n_spots, 2)) * 10.0
    return {
        "spot_expression": spot_expr,
        "reference_profiles": ref_profiles,
        "coordinates": coords,
    }


def _generate_hic_data(
    n_bins: int,
    n_tad_blocks: int = 5,
    bin_features_dim: int = 16,
    seed: int = 2,
) -> dict[str, jax.Array]:
    """Generate synthetic Hi-C contact matrix with block-diagonal TADs.

    Builds a symmetric contact matrix where blocks along the diagonal
    have elevated counts to simulate topologically associating domains.

    Args:
        n_bins: Number of genomic bins.
        n_tad_blocks: Approximate number of TAD blocks.
        bin_features_dim: Dimension of per-bin feature vectors.
        seed: Random seed.

    Returns:
        Dictionary with ``contact_matrix`` and ``bin_features`` arrays.
    """
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    # Background contacts decay with distance
    positions = jnp.arange(n_bins, dtype=jnp.float32)
    dist = jnp.abs(positions[:, None] - positions[None, :])
    background = jnp.exp(-dist / (n_bins / 4.0))

    # Block-diagonal TAD structure
    block_size = max(n_bins // n_tad_blocks, 1)
    block_ids = positions // block_size
    same_block = (block_ids[:, None] == block_ids[None, :]).astype(
        jnp.float32
    )

    noise = jnp.abs(jax.random.normal(k1, (n_bins, n_bins))) * 0.1
    contact_matrix = background + same_block * 2.0 + noise
    # Symmetrize
    contact_matrix = (contact_matrix + contact_matrix.T) / 2.0

    bin_features = jax.random.normal(k2, (n_bins, bin_features_dim))
    return {
        "contact_matrix": contact_matrix,
        "bin_features": bin_features,
    }


def _generate_spatial_gene_data(
    n_spots: int,
    n_genes: int,
    n_spatial_genes: int = 5,
    seed: int = 3,
) -> dict[str, jax.Array]:
    """Generate synthetic spatial gene expression data.

    A subset of genes are given spatially correlated expression
    patterns (sinusoidal over coordinates) while the rest are
    random noise.

    Args:
        n_spots: Number of spatial spots.
        n_genes: Number of genes.
        n_spatial_genes: Number of genes with spatial patterns.
        seed: Random seed.

    Returns:
        Dictionary with ``spatial_coords``, ``expression``, and
        ``total_counts`` arrays.
    """
    key = jax.random.key(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    coords = jax.random.uniform(k1, (n_spots, 2)) * 10.0

    # Non-spatial (noise) genes
    expression = jnp.abs(
        jax.random.normal(k2, (n_spots, n_genes))
    ) * 2.0

    # Overwrite first n_spatial_genes with a spatial pattern
    freqs = jax.random.uniform(
        k3, (n_spatial_genes,), minval=0.5, maxval=2.0
    )
    phases = jax.random.uniform(k4, (n_spatial_genes,)) * 2.0 * jnp.pi
    # Sinusoidal pattern driven by x-coordinate
    spatial_signal = jnp.sin(
        coords[:, 0:1] * freqs[None, :] + phases[None, :]
    )
    spatial_expr = jnp.abs(spatial_signal) * 5.0 + 1.0
    expression = expression.at[:, :n_spatial_genes].set(spatial_expr)

    total_counts = jnp.sum(expression, axis=-1)
    return {
        "spatial_coords": coords,
        "expression": expression,
        "total_counts": total_counts,
    }


# ------------------------------------------------------------------
# Individual operator benchmarks
# ------------------------------------------------------------------


def _benchmark_vae(
    quick: bool,
) -> dict[str, Any]:
    """Benchmark DifferentiableMultiOmicsVAE.

    Args:
        quick: Use reduced data sizes when True.

    Returns:
        Dictionary of VAE benchmark metrics.
    """
    n_cells = 30 if quick else 100
    n_rna = 50
    n_atac = 30
    n_types = 3  # noqa: F841 (documents intent)
    latent_dim = 8

    print(f"  cells={n_cells}, rna={n_rna}, atac={n_atac}")

    config = MultiOmicsVAEConfig(
        modality_dims=[n_rna, n_atac],
        latent_dim=latent_dim,
        hidden_dim=32,
        modality_weight_mode="equal",
    )
    rngs = nnx.Rngs(42)
    model = DifferentiableMultiOmicsVAE(config, rngs=rngs)

    data = _generate_vae_data(n_cells, n_rna, n_atac)
    result, _, _ = model.apply(data, {}, None)

    # --- shape checks ---
    latent = result["joint_latent"]
    rna_recon = result["rna_reconstructed"]
    atac_recon = result["atac_reconstructed"]
    elbo = result["elbo_loss"]

    shapes_ok = (
        latent.shape == (n_cells, latent_dim)
        and rna_recon.shape == (n_cells, n_rna)
        and atac_recon.shape == (n_cells, n_atac)
        and elbo.ndim == 0
    )

    # --- latent quality ---
    latent_mean = float(jnp.mean(latent))
    latent_std = float(jnp.std(latent))
    recon_finite = bool(
        jnp.all(jnp.isfinite(rna_recon))
        and jnp.all(jnp.isfinite(atac_recon))
    )

    # --- gradient flow ---
    def _loss(mdl: DifferentiableMultiOmicsVAE, d: dict) -> jax.Array:
        r, _, _ = mdl.apply(d, {}, None)
        return r["elbo_loss"]

    grad_info = check_gradient_flow(_loss, model, data)

    # --- throughput ---
    n_iter = 20 if quick else 50
    tp = measure_throughput(
        lambda d: model.apply(d, {}, None),
        (data,),
        n_iterations=n_iter,
        warmup=3,
    )

    metrics: dict[str, Any] = {
        "shapes_correct": shapes_ok,
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "reconstruction_finite": recon_finite,
        "elbo_loss": float(elbo),
        **asdict(grad_info),
        "throughput_items_per_sec": tp["items_per_sec"],
        "per_item_ms": tp["per_item_ms"],
    }

    print(f"    shapes_correct={shapes_ok}")
    print(f"    latent_std={latent_std:.4f}")
    print(f"    recon_finite={recon_finite}")
    print(f"    gradient_nonzero={grad_info.gradient_nonzero}")
    print(f"    throughput={tp['items_per_sec']:.1f} items/s")
    return metrics


def _benchmark_deconv(
    quick: bool,
) -> dict[str, Any]:
    """Benchmark SpatialDeconvolution.

    Args:
        quick: Use reduced data sizes when True.

    Returns:
        Dictionary of deconvolution benchmark metrics.
    """
    n_spots = 20 if quick else 50
    n_genes = 40
    n_cell_types = 5

    print(f"  spots={n_spots}, genes={n_genes}, types={n_cell_types}")

    config = SpatialDeconvolutionConfig(
        n_genes=n_genes,
        n_cell_types=n_cell_types,
        hidden_dim=32,
        num_layers=2,
        spatial_hidden=16,
        temperature=1.0,
    )
    rngs = nnx.Rngs(42)
    model = SpatialDeconvolution(config, rngs=rngs)

    data = _generate_deconv_data(n_spots, n_genes, n_cell_types)
    result, _, _ = model.apply(data, {}, None)

    # --- shape checks ---
    proportions = result["cell_proportions"]
    recon_expr = result["reconstructed_expression"]
    spatial_emb = result["spatial_embeddings"]

    shapes_ok = (
        proportions.shape == (n_spots, n_cell_types)
        and recon_expr.shape == (n_spots, n_genes)
        and spatial_emb.shape[0] == n_spots
    )

    # --- proportions sum to ~1 ---
    row_sums = jnp.sum(proportions, axis=-1)
    sum_close_to_one = bool(
        jnp.allclose(row_sums, 1.0, atol=1e-4)
    )
    max_sum_deviation = float(jnp.max(jnp.abs(row_sums - 1.0)))

    # --- gradient flow ---
    def _loss(mdl: SpatialDeconvolution, d: dict) -> jax.Array:
        r, _, _ = mdl.apply(d, {}, None)
        return jnp.sum(r["cell_proportions"])

    grad_info = check_gradient_flow(_loss, model, data)

    # --- throughput ---
    n_iter = 20 if quick else 50
    tp = measure_throughput(
        lambda d: model.apply(d, {}, None),
        (data,),
        n_iterations=n_iter,
        warmup=3,
    )

    metrics: dict[str, Any] = {
        "shapes_correct": shapes_ok,
        "proportions_sum_to_one": sum_close_to_one,
        "max_sum_deviation": max_sum_deviation,
        **asdict(grad_info),
        "throughput_items_per_sec": tp["items_per_sec"],
        "per_item_ms": tp["per_item_ms"],
    }

    print(f"    shapes_correct={shapes_ok}")
    print(f"    proportions_sum_to_one={sum_close_to_one}")
    print(f"    max_sum_deviation={max_sum_deviation:.6f}")
    print(f"    gradient_nonzero={grad_info.gradient_nonzero}")
    print(f"    throughput={tp['items_per_sec']:.1f} items/s")
    return metrics


def _benchmark_hic(
    quick: bool,
) -> dict[str, Any]:
    """Benchmark HiCContactAnalysis.

    Args:
        quick: Use reduced data sizes when True.

    Returns:
        Dictionary of Hi-C benchmark metrics.
    """
    n_bins = 50 if quick else 100
    bin_features_dim = 16

    print(f"  n_bins={n_bins}, bin_features={bin_features_dim}")

    config = HiCContactAnalysisConfig(
        n_bins=n_bins,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        bin_features=bin_features_dim,
        temperature=1.0,
    )
    rngs = nnx.Rngs(42)
    model = HiCContactAnalysis(config, rngs=rngs)

    data = _generate_hic_data(n_bins, bin_features_dim=bin_features_dim)
    result, _, _ = model.apply(data, {}, None)

    # --- shape checks ---
    compartment = result["compartment_scores"]
    boundary = result["tad_boundary_scores"]
    pred_contacts = result["predicted_contacts"]
    bin_emb = result["bin_embeddings"]

    shapes_ok = (
        compartment.shape == (n_bins,)
        and boundary.shape == (n_bins,)
        and pred_contacts.shape == (n_bins, n_bins)
        and bin_emb.shape[0] == n_bins
    )

    # --- outputs finite ---
    outputs_finite = bool(
        jnp.all(jnp.isfinite(compartment))
        and jnp.all(jnp.isfinite(boundary))
        and jnp.all(jnp.isfinite(pred_contacts))
    )

    # --- gradient flow ---
    def _loss(mdl: HiCContactAnalysis, d: dict) -> jax.Array:
        r, _, _ = mdl.apply(d, {}, None)
        return jnp.sum(r["compartment_scores"])

    grad_info = check_gradient_flow(_loss, model, data)

    # --- throughput ---
    n_iter = 20 if quick else 50
    tp = measure_throughput(
        lambda d: model.apply(d, {}, None),
        (data,),
        n_iterations=n_iter,
        warmup=3,
    )

    metrics: dict[str, Any] = {
        "shapes_correct": shapes_ok,
        "outputs_finite": outputs_finite,
        **asdict(grad_info),
        "throughput_items_per_sec": tp["items_per_sec"],
        "per_item_ms": tp["per_item_ms"],
    }

    print(f"    shapes_correct={shapes_ok}")
    print(f"    outputs_finite={outputs_finite}")
    print(f"    gradient_nonzero={grad_info.gradient_nonzero}")
    print(f"    throughput={tp['items_per_sec']:.1f} items/s")
    return metrics


def _benchmark_spatial_gene(
    quick: bool,
) -> dict[str, Any]:
    """Benchmark DifferentiableSpatialGeneDetector.

    Args:
        quick: Use reduced data sizes when True.

    Returns:
        Dictionary of spatial gene detection benchmark metrics.
    """
    n_spots = 15 if quick else 30
    n_genes = 50
    n_spatial_genes = 5

    print(f"  spots={n_spots}, genes={n_genes}, spatial={n_spatial_genes}")

    config = SpatialGeneDetectorConfig(
        n_genes=n_genes,
        hidden_dims=[32, 16],
        temperature=1.0,
        pvalue_threshold=0.05,
        learnable_kernel=True,
    )
    rngs = nnx.Rngs(42)
    model = DifferentiableSpatialGeneDetector(config, rngs=rngs)

    data = _generate_spatial_gene_data(
        n_spots, n_genes, n_spatial_genes
    )
    result, _, _ = model.apply(data, {}, None)

    # --- shape checks ---
    spatial_var = result["spatial_variance"]
    fsv = result["fsv"]
    pvalues = result["spatial_pvalues"]
    is_spatial = result["is_spatial"]
    smoothed = result["smoothed_expression"]

    shapes_ok = (
        spatial_var.shape == (n_genes,)
        and fsv.shape == (n_genes,)
        and pvalues.shape == (n_genes,)
        and is_spatial.shape == (n_genes,)
        and smoothed.shape == (n_spots, n_genes)
    )

    # --- outputs finite ---
    outputs_finite = bool(
        jnp.all(jnp.isfinite(spatial_var))
        and jnp.all(jnp.isfinite(fsv))
        and jnp.all(jnp.isfinite(pvalues))
        and jnp.all(jnp.isfinite(is_spatial))
    )

    # --- gradient flow ---
    def _loss(
        mdl: DifferentiableSpatialGeneDetector,
        d: dict,
    ) -> jax.Array:
        r, _, _ = mdl.apply(d, {}, None)
        return jnp.sum(r["is_spatial"])

    grad_info = check_gradient_flow(_loss, model, data)

    # --- throughput ---
    n_iter = 20 if quick else 50
    tp = measure_throughput(
        lambda d: model.apply(d, {}, None),
        (data,),
        n_iterations=n_iter,
        warmup=3,
    )

    metrics: dict[str, Any] = {
        "shapes_correct": shapes_ok,
        "outputs_finite": outputs_finite,
        "n_detected_spatial": int(jnp.sum(is_spatial > 0.5)),
        "mean_fsv": float(jnp.mean(fsv)),
        **asdict(grad_info),
        "throughput_items_per_sec": tp["items_per_sec"],
        "per_item_ms": tp["per_item_ms"],
    }

    print(f"    shapes_correct={shapes_ok}")
    print(f"    outputs_finite={outputs_finite}")
    print(
        f"    detected_spatial={metrics['n_detected_spatial']}"
        f"/{n_genes}"
    )
    print(f"    mean_fsv={metrics['mean_fsv']:.4f}")
    print(f"    gradient_nonzero={grad_info.gradient_nonzero}")
    print(f"    throughput={tp['items_per_sec']:.1f} items/s")
    return metrics


# ------------------------------------------------------------------
# Top-level runner
# ------------------------------------------------------------------


def run_benchmark(
    quick: bool = False,
) -> MultiOmicsBenchmarkResult:
    """Run the complete multi-omics integration benchmark.

    Args:
        quick: When True, use smaller data sizes for faster execution.

    Returns:
        Aggregated benchmark results.
    """
    mode = "QUICK" if quick else "FULL"
    print("=" * 60)
    print(f"DiffBio Multi-Omics Benchmark ({mode})")
    print("=" * 60)

    # 1. Multi-omics VAE
    print("\n[1/4] DifferentiableMultiOmicsVAE")
    vae_res = _benchmark_vae(quick)

    # 2. Spatial deconvolution
    print("\n[2/4] SpatialDeconvolution")
    deconv_res = _benchmark_deconv(quick)

    # 3. Hi-C contact analysis
    print("\n[3/4] HiCContactAnalysis")
    hic_res = _benchmark_hic(quick)

    # 4. Spatial gene detection
    print("\n[4/4] DifferentiableSpatialGeneDetector")
    sgene_res = _benchmark_spatial_gene(quick)

    # Aggregate summary flags
    all_shapes = (
        vae_res["shapes_correct"]
        and deconv_res["shapes_correct"]
        and hic_res["shapes_correct"]
        and sgene_res["shapes_correct"]
    )
    all_grads = (
        vae_res["gradient_nonzero"]
        and deconv_res["gradient_nonzero"]
        and hic_res["gradient_nonzero"]
        and sgene_res["gradient_nonzero"]
    )

    result = MultiOmicsBenchmarkResult(
        timestamp=datetime.now().isoformat(),
        quick=quick,
        vae_results=vae_res,
        deconv_results=deconv_res,
        hic_results=hic_res,
        spatial_gene_results=sgene_res,
        all_shapes_correct=all_shapes,
        all_gradients_flow=all_grads,
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  All shapes correct : {all_shapes}")
    print(f"  All gradients flow : {all_grads}")
    print("=" * 60)

    return result


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def main() -> None:
    """CLI entry point for multi-omics benchmark."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="DiffBio multi-omics benchmark",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller data sizes for faster execution.",
    )
    args = parser.parse_args()

    result = run_benchmark(quick=args.quick)

    path = save_benchmark_result(
        asdict(result),
        domain="multiomics",
        benchmark_name="multiomics_benchmark",
    )
    print(f"Results saved to: {path}")


if __name__ == "__main__":
    main()
