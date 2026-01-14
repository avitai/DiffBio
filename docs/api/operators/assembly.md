# Assembly Operators API

Differentiable operators for genome assembly using graph neural networks.

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

## Usage Examples

### Assembly Graph Navigation

```python
from flax import nnx
from diffbio.operators.assembly import GNNAssemblyNavigator, GNNAssemblyConfig

config = GNNAssemblyConfig(
    node_dim=64,
    edge_dim=32,
    hidden_dim=128,
    n_layers=3,
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
