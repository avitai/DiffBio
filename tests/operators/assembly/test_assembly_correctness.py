#!/usr/bin/env python3
"""Assembly Benchmark for DiffBio.

This benchmark evaluates DiffBio's genome assembly operators:
- GNNAssemblyNavigator (graph attention-based assembly traversal)
- DifferentiableMetagenomicBinner (VAE-based metagenomic binning)

Metrics:
- Output shape correctness and value finiteness
- Cluster assignment and latent space shape validation
- Gradient flow for both operators
- Throughput (graphs/second, contigs/second)

Usage:
    python benchmarks/assembly/assembly_benchmark.py
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime

import jax
import jax.numpy as jnp
from flax import nnx

from benchmarks._gradient import check_gradient_flow
from diffbio.operators.assembly.gnn_assembly import (
    GNNAssemblyNavigator,
    GNNAssemblyNavigatorConfig,
)
from diffbio.operators.assembly.metagenomic_binning import (
    DifferentiableMetagenomicBinner,
    MetagenomicBinnerConfig,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class AssemblyBenchmarkResult:
    """Results from the assembly benchmark suite.

    Attributes:
        timestamp: ISO-formatted timestamp of the run.
        n_nodes: Number of nodes in the GNN assembly graph.
        n_edges: Number of edges in the GNN assembly graph.
        n_contigs: Number of contigs for metagenomic binning.
        gnn_edge_scores_shape_ok: Whether GNN edge scores have
            the expected shape.
        gnn_edge_scores_finite: Whether all GNN edge scores are
            finite (no NaN or inf).
        gnn_traversal_probs_shape_ok: Whether traversal
            probabilities have the expected shape.
        gnn_node_embeddings_shape_ok: Whether node embeddings
            have the expected shape.
        binner_cluster_shape_ok: Whether cluster assignments
            have the expected shape.
        binner_latent_shape_ok: Whether latent representations
            have the expected shape.
        binner_reconstruction_shape_ok: Whether reconstructed
            outputs have the expected shapes.
        gnn_gradient: Gradient flow results for the GNN
            navigator.
        binner_gradient: Gradient flow results for the
            metagenomic binner.
        gnn_throughput: Throughput metrics for the GNN navigator.
        binner_throughput: Throughput metrics for the metagenomic
            binner.
    """

    timestamp: str
    n_nodes: int
    n_edges: int
    n_contigs: int
    # GNN shape / value correctness
    gnn_edge_scores_shape_ok: bool
    gnn_edge_scores_finite: bool
    gnn_traversal_probs_shape_ok: bool
    gnn_node_embeddings_shape_ok: bool
    # Binner shape correctness
    binner_cluster_shape_ok: bool
    binner_latent_shape_ok: bool
    binner_reconstruction_shape_ok: bool
    # Gradient flow
    gnn_gradient: dict[str, float | bool] = field(
        default_factory=dict,
    )
    binner_gradient: dict[str, float | bool] = field(
        default_factory=dict,
    )
    # Throughput
    gnn_throughput: dict[str, float] = field(
        default_factory=dict,
    )
    binner_throughput: dict[str, float] = field(
        default_factory=dict,
    )


# ------------------------------------------------------------------
# Synthetic data generators
# ------------------------------------------------------------------


def _generate_assembly_graph(
    n_nodes: int,
    n_edges: int,
    node_feature_dim: int = 64,
    edge_feature_dim: int = 8,
    seed: int = 42,
) -> dict[str, jnp.ndarray]:
    """Generate a synthetic assembly graph.

    Creates random node features and a random edge structure
    with overlap features, mimicking an assembly overlap graph.

    Args:
        n_nodes: Number of contig/read nodes.
        n_edges: Number of overlap edges.
        node_feature_dim: Dimension of node feature vectors.
        edge_feature_dim: Dimension of edge feature vectors.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with node_features, edge_index, and
        edge_features arrays.
    """
    key = jax.random.key(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    node_features = jax.random.normal(
        k1, (n_nodes, node_feature_dim),
    )
    # Random edge indices (source, target) in [0, n_nodes)
    sources = jax.random.randint(k2, (n_edges,), 0, n_nodes)
    targets = jax.random.randint(k3, (n_edges,), 0, n_nodes)
    edge_index = jnp.stack([sources, targets], axis=0)

    edge_features = jax.random.normal(
        k4, (n_edges, edge_feature_dim),
    )

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
    }


def _generate_binning_data(
    n_contigs: int,
    n_tnf_features: int = 136,
    n_abundance_features: int = 3,
    seed: int = 42,
) -> dict[str, jnp.ndarray]:
    """Generate synthetic metagenomic binning data.

    Creates tetranucleotide frequency (TNF) profiles and
    abundance vectors for a set of contigs.

    Args:
        n_contigs: Number of contigs.
        n_tnf_features: Dimension of TNF feature vectors.
        n_abundance_features: Number of abundance samples.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with tnf and abundance arrays.
    """
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key, 2)

    # TNF as softmax-normalised frequencies
    tnf_logits = jax.random.normal(k1, (n_contigs, n_tnf_features))
    tnf = jax.nn.softmax(tnf_logits, axis=-1)

    # Abundance as positive values
    abundance = jax.nn.softplus(
        jax.random.normal(k2, (n_contigs, n_abundance_features)),
    )

    return {"tnf": tnf, "abundance": abundance}


# ------------------------------------------------------------------
# Operator tests
# ------------------------------------------------------------------


def _test_gnn_navigator(
    data: dict[str, jnp.ndarray],
    n_nodes: int,
    n_edges: int,
    hidden_dim: int = 128,
) -> tuple[dict[str, bool], GNNAssemblyNavigator]:
    """Test GNNAssemblyNavigator on synthetic graph data.

    Args:
        data: Assembly graph data dictionary.
        n_nodes: Expected number of nodes.
        n_edges: Expected number of edges.
        hidden_dim: Hidden dimension for the GNN.

    Returns:
        Tuple of (metrics dict, operator instance).
    """
    config = GNNAssemblyNavigatorConfig(
        node_features=data["node_features"].shape[1],
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
        edge_features=data["edge_features"].shape[1],
        dropout_rate=0.0,
        temperature=1.0,
    )
    navigator = GNNAssemblyNavigator(
        config, rngs=nnx.Rngs(42),
    )

    result, _, _ = navigator.apply(data, {}, None)

    edge_scores = result["edge_scores"]
    traversal_probs = result["traversal_probs"]
    node_embeddings = result["node_embeddings"]

    edge_scores_shape_ok = edge_scores.shape == (n_edges,)
    edge_scores_finite = bool(jnp.all(jnp.isfinite(edge_scores)))
    traversal_probs_shape_ok = (
        traversal_probs.shape == (n_edges,)
    )
    node_embeddings_shape_ok = (
        node_embeddings.shape == (n_nodes, hidden_dim)
    )

    return {
        "edge_scores_shape_ok": edge_scores_shape_ok,
        "edge_scores_finite": edge_scores_finite,
        "traversal_probs_shape_ok": traversal_probs_shape_ok,
        "node_embeddings_shape_ok": node_embeddings_shape_ok,
    }, navigator


def _test_metagenomic_binner(
    data: dict[str, jnp.ndarray],
    n_contigs: int,
    n_abundance_features: int = 3,
    n_clusters: int = 20,
    latent_dim: int = 32,
) -> tuple[dict[str, bool], DifferentiableMetagenomicBinner]:
    """Test DifferentiableMetagenomicBinner on synthetic data.

    Args:
        data: Binning data dictionary with tnf and abundance.
        n_contigs: Expected number of contigs.
        n_abundance_features: Number of abundance samples.
        n_clusters: Number of target clusters.
        latent_dim: Latent space dimension.

    Returns:
        Tuple of (metrics dict, operator instance).
    """
    config = MetagenomicBinnerConfig(
        n_tnf_features=data["tnf"].shape[1],
        n_abundance_features=n_abundance_features,
        latent_dim=latent_dim,
        hidden_dims=(256, 128),
        dropout_rate=0.0,
        beta=1.0,
        temperature=1.0,
        n_clusters=n_clusters,
    )
    binner = DifferentiableMetagenomicBinner(
        config, rngs=nnx.Rngs(42),
    )

    result, _, _ = binner.apply(data, {}, None)

    cluster_assignments = result["cluster_assignments"]
    latent_z = result["latent_z"]
    recon_tnf = result["reconstructed_tnf"]
    recon_abundance = result["reconstructed_abundance"]

    cluster_shape_ok = (
        cluster_assignments.shape == (n_contigs, n_clusters)
    )
    latent_shape_ok = (
        latent_z.shape == (n_contigs, latent_dim)
    )
    reconstruction_shape_ok = (
        recon_tnf.shape == data["tnf"].shape
        and recon_abundance.shape == data["abundance"].shape
    )

    return {
        "cluster_shape_ok": cluster_shape_ok,
        "latent_shape_ok": latent_shape_ok,
        "reconstruction_shape_ok": reconstruction_shape_ok,
    }, binner


# ------------------------------------------------------------------
# Main benchmark
# ------------------------------------------------------------------


def run_benchmark(
    *, quick: bool = False,
) -> AssemblyBenchmarkResult:
    """Run the complete assembly benchmark.

    Args:
        quick: If True, use smaller data for faster execution.

    Returns:
        Benchmark results dataclass.
    """
    n_nodes = 10 if quick else 20
    n_edges = 20 if quick else 40
    n_contigs = 30 if quick else 100
    n_abundance_features = 3
    n_clusters = 20
    latent_dim = 32
    hidden_dim = 128
    n_throughput_iters = 20 if quick else 100

    print("=" * 60)
    print("DiffBio Assembly Benchmark")
    print("=" * 60)
    print(f"  Nodes (GNN)     : {n_nodes}")
    print(f"  Edges (GNN)     : {n_edges}")
    print(f"  Contigs (Binner): {n_contigs}")
    print(f"  Quick mode      : {quick}")

    # ----- Synthetic data -----
    print("\nGenerating synthetic data...")
    gnn_data = _generate_assembly_graph(
        n_nodes=n_nodes,
        n_edges=n_edges,
        node_feature_dim=64,
        edge_feature_dim=8,
    )
    binner_data = _generate_binning_data(
        n_contigs=n_contigs,
        n_tnf_features=136,
        n_abundance_features=n_abundance_features,
    )

    # ----- GNN Assembly Navigator -----
    print("\nTesting GNNAssemblyNavigator...")
    gnn_metrics, navigator = _test_gnn_navigator(
        gnn_data, n_nodes, n_edges, hidden_dim=hidden_dim,
    )
    print(
        f"  Edge scores shape OK     :"
        f" {gnn_metrics['edge_scores_shape_ok']}"
    )
    print(
        f"  Edge scores finite       :"
        f" {gnn_metrics['edge_scores_finite']}"
    )
    print(
        f"  Traversal probs shape OK :"
        f" {gnn_metrics['traversal_probs_shape_ok']}"
    )
    print(
        f"  Node embeddings shape OK :"
        f" {gnn_metrics['node_embeddings_shape_ok']}"
    )

    # ----- Metagenomic Binner -----
    print("\nTesting DifferentiableMetagenomicBinner...")
    binner_metrics, binner = _test_metagenomic_binner(
        binner_data,
        n_contigs,
        n_abundance_features=n_abundance_features,
        n_clusters=n_clusters,
        latent_dim=latent_dim,
    )
    print(
        f"  Cluster shape OK         :"
        f" {binner_metrics['cluster_shape_ok']}"
    )
    print(
        f"  Latent shape OK          :"
        f" {binner_metrics['latent_shape_ok']}"
    )
    print(
        f"  Reconstruction shape OK  :"
        f" {binner_metrics['reconstruction_shape_ok']}"
    )

    # ----- Gradient flow -----
    print("\nChecking gradient flow...")

    def _gnn_loss(model: GNNAssemblyNavigator) -> jax.Array:
        """Loss for GNN navigator gradient check."""
        out, _, _ = model.apply(gnn_data, {}, None)
        return jnp.sum(out["edge_scores"])

    gnn_grad = check_gradient_flow(_gnn_loss, navigator)
    print(
        f"  GNNNavigator   : norm={gnn_grad.gradient_norm:.6f}"
        f"  nonzero={gnn_grad.gradient_nonzero}"
    )

    def _binner_loss(
        model: DifferentiableMetagenomicBinner,
    ) -> jax.Array:
        """Loss for metagenomic binner gradient check."""
        out, _, _ = model.apply(binner_data, {}, None)
        return jnp.sum(out["latent_mu"])

    binner_grad = check_gradient_flow(_binner_loss, binner)
    print(
        f"  MetaBinner     : norm={binner_grad.gradient_norm:.6f}"
        f"  nonzero={binner_grad.gradient_nonzero}"
    )

    # ----- Throughput -----
    print("\nMeasuring throughput...")

    gnn_tp = measure_throughput(
        lambda: navigator.apply(gnn_data, {}, None),
        args=(),
        n_iterations=n_throughput_iters,
        warmup=3,
    )
    gnn_graphs_per_sec = gnn_tp["items_per_sec"]
    print(
        f"  GNNNavigator   : {gnn_graphs_per_sec:.0f} graphs/s"
        f"  ({gnn_tp['per_item_ms']:.2f} ms/call)"
    )

    binner_tp = measure_throughput(
        lambda: binner.apply(binner_data, {}, None),
        args=(),
        n_iterations=n_throughput_iters,
        warmup=3,
    )
    binner_contigs_per_sec = (
        n_contigs * binner_tp["items_per_sec"]
    )
    print(
        f"  MetaBinner     :"
        f" {binner_contigs_per_sec:.0f} contigs/s"
        f"  ({binner_tp['per_item_ms']:.2f} ms/call)"
    )

    # ----- Compile result -----
    result = AssemblyBenchmarkResult(
        timestamp=datetime.now().isoformat(),
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_contigs=n_contigs,
        gnn_edge_scores_shape_ok=(
            gnn_metrics["edge_scores_shape_ok"]
        ),
        gnn_edge_scores_finite=gnn_metrics["edge_scores_finite"],
        gnn_traversal_probs_shape_ok=(
            gnn_metrics["traversal_probs_shape_ok"]
        ),
        gnn_node_embeddings_shape_ok=(
            gnn_metrics["node_embeddings_shape_ok"]
        ),
        binner_cluster_shape_ok=(
            binner_metrics["cluster_shape_ok"]
        ),
        binner_latent_shape_ok=(
            binner_metrics["latent_shape_ok"]
        ),
        binner_reconstruction_shape_ok=(
            binner_metrics["reconstruction_shape_ok"]
        ),
        gnn_gradient=gnn_grad,
        binner_gradient=binner_grad,
        gnn_throughput={
            **gnn_tp,
            "graphs_per_sec": gnn_graphs_per_sec,
        },
        binner_throughput={
            **binner_tp,
            "contigs_per_sec": binner_contigs_per_sec,
        },
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    gnn_shapes_ok = (
        gnn_metrics["edge_scores_shape_ok"]
        and gnn_metrics["traversal_probs_shape_ok"]
        and gnn_metrics["node_embeddings_shape_ok"]
    )
    binner_shapes_ok = (
        binner_metrics["cluster_shape_ok"]
        and binner_metrics["latent_shape_ok"]
        and binner_metrics["reconstruction_shape_ok"]
    )
    all_grads_nonzero = gnn_grad.gradient_nonzero and binner_grad.gradient_nonzero
    print(f"  GNN shapes OK           : {gnn_shapes_ok}")
    print(f"  GNN values finite       : {gnn_metrics['edge_scores_finite']}")
    print(f"  Binner shapes OK        : {binner_shapes_ok}")
    print(f"  All gradients nonzero   : {all_grads_nonzero}")
    print("=" * 60)

    return result


def main() -> None:
    """Entry point for the assembly benchmark."""
    result = run_benchmark()
    output_path = save_benchmark_result(
        asdict(result),
        domain="assembly",
        benchmark_name="assembly_benchmark",
    )
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
