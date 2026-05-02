# Assembly Operators API

Differentiable operators for genome assembly using graph neural networks and VAE-based binning.

## GNNAssemblyNavigator

::: diffbio.operators.assembly.gnn_assembly.GNNAssemblyNavigator
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## GNNAssemblyNavigatorConfig

::: diffbio.operators.assembly.gnn_assembly.GNNAssemblyNavigatorConfig
    options:
      show_root_heading: true
      members: []

## DifferentiableMetagenomicBinner

::: diffbio.operators.assembly.metagenomic_binning.DifferentiableMetagenomicBinner
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply
        - encode
        - decode
        - soft_cluster

## MetagenomicBinnerConfig

::: diffbio.operators.assembly.metagenomic_binning.MetagenomicBinnerConfig
    options:
      show_root_heading: true
      members: []

## create_metagenomic_binner

::: diffbio.operators.assembly.metagenomic_binning.create_metagenomic_binner
    options:
      show_root_heading: true
      show_source: false

## Usage Examples

### Assembly Graph Navigation

```python
from flax import nnx
from diffbio.operators.assembly import GNNAssemblyNavigator, GNNAssemblyNavigatorConfig

config = GNNAssemblyNavigatorConfig(
    node_features=64,
    edge_features=32,
    hidden_dim=128,
    num_layers=3,
)
navigator = GNNAssemblyNavigator(config, rngs=nnx.Rngs(42))

data = {
    "node_features": node_feats,   # (n_nodes, node_dim)
    "edge_index": edges,           # (2, n_edges)
    "edge_features": edge_feats,   # (n_edges, edge_dim)
}
result, _, _ = navigator.apply(data, {}, None)
next_node_probs = result["next_node_probs"]
```

### De Bruijn Graph Construction

```python
import jax.numpy as jnp

# Create k-mer node features
k = 31
kmers = extract_kmers(sequences, k)
kmer_embeddings = embed_kmers(kmers)
coverage = compute_kmer_coverage(kmers)

node_features = jnp.concatenate([
    kmer_embeddings,
    coverage[:, None],
], axis=-1)

# Create edges for k-1 overlaps
edge_index = find_overlapping_kmers(kmers, k-1)
```

### Metagenomic Binning

```python
from flax import nnx
from diffbio.operators.assembly import (
    DifferentiableMetagenomicBinner,
    MetagenomicBinnerConfig,
    create_metagenomic_binner,
)

# Using config
config = MetagenomicBinnerConfig(
    n_tnf_features=136,
    n_abundance_features=10,
    latent_dim=32,
    n_clusters=100,
)
binner = DifferentiableMetagenomicBinner(config, rngs=nnx.Rngs(42))

# Or using factory function
binner = create_metagenomic_binner(
    n_abundance_features=10,
    n_clusters=100,
    latent_dim=32,
)

# Apply binning
data = {
    "tnf": tnf_features,      # (n_contigs, 136)
    "abundance": abundances,   # (n_contigs, n_samples)
}
result, _, _ = binner.apply(data, {}, None)

# Get cluster assignments
bins = result["cluster_assignments"].argmax(axis=-1)
latent = result["latent_z"]
```
