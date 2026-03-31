#!/usr/bin/env python3
"""Assembly correctness tests for DiffBio.

Validates DiffBio's genome assembly operators for output shape
correctness, value finiteness, and gradient flow:
- GNNAssemblyNavigator (graph attention-based assembly traversal)
- DifferentiableMetagenomicBinner (VAE-based metagenomic binning)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from diffbio.operators.assembly.gnn_assembly import (
    GNNAssemblyNavigator,
    GNNAssemblyNavigatorConfig,
)
from diffbio.operators.assembly.metagenomic_binning import (
    DifferentiableMetagenomicBinner,
    MetagenomicBinnerConfig,
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
        k1,
        (n_nodes, node_feature_dim),
    )
    # Random edge indices (source, target) in [0, n_nodes)
    sources = jax.random.randint(k2, (n_edges,), 0, n_nodes)
    targets = jax.random.randint(k3, (n_edges,), 0, n_nodes)
    edge_index = jnp.stack([sources, targets], axis=0)

    edge_features = jax.random.normal(
        k4,
        (n_edges, edge_feature_dim),
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
        config,
        rngs=nnx.Rngs(42),
    )

    result, _, _ = navigator.apply(data, {}, None)

    edge_scores = result["edge_scores"]
    traversal_probs = result["traversal_probs"]
    node_embeddings = result["node_embeddings"]

    edge_scores_shape_ok = edge_scores.shape == (n_edges,)
    edge_scores_finite = bool(jnp.all(jnp.isfinite(edge_scores)))
    traversal_probs_shape_ok = traversal_probs.shape == (n_edges,)
    node_embeddings_shape_ok = node_embeddings.shape == (n_nodes, hidden_dim)

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
        config,
        rngs=nnx.Rngs(42),
    )

    result, _, _ = binner.apply(data, {}, None)

    cluster_assignments = result["cluster_assignments"]
    latent_z = result["latent_z"]
    recon_tnf = result["reconstructed_tnf"]
    recon_abundance = result["reconstructed_abundance"]

    cluster_shape_ok = cluster_assignments.shape == (n_contigs, n_clusters)
    latent_shape_ok = latent_z.shape == (n_contigs, latent_dim)
    reconstruction_shape_ok = (
        recon_tnf.shape == data["tnf"].shape and recon_abundance.shape == data["abundance"].shape
    )

    return {
        "cluster_shape_ok": cluster_shape_ok,
        "latent_shape_ok": latent_shape_ok,
        "reconstruction_shape_ok": reconstruction_shape_ok,
    }, binner
