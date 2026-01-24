# Drug Discovery Operators API

Differentiable operators for molecular property prediction, fingerprint computation, and similarity scoring.

## MolecularPropertyPredictor

::: diffbio.operators.drug_discovery.property_predictor.MolecularPropertyPredictor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## MolecularPropertyConfig

::: diffbio.operators.drug_discovery.property_predictor.MolecularPropertyConfig
    options:
      show_root_heading: true
      members: []

## DifferentiableMolecularFingerprint

::: diffbio.operators.drug_discovery.fingerprint.DifferentiableMolecularFingerprint
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## MolecularFingerprintConfig

::: diffbio.operators.drug_discovery.fingerprint.MolecularFingerprintConfig
    options:
      show_root_heading: true
      members: []

## MolecularSimilarityOperator

::: diffbio.operators.drug_discovery.similarity.MolecularSimilarityOperator
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## MolecularSimilarityConfig

::: diffbio.operators.drug_discovery.similarity.MolecularSimilarityConfig
    options:
      show_root_heading: true
      members: []

## Message Passing Layers

### MessagePassingLayer

::: diffbio.operators.drug_discovery.message_passing.MessagePassingLayer
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - __call__

### StackedMessagePassing

::: diffbio.operators.drug_discovery.message_passing.StackedMessagePassing
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - __call__

## Factory Functions

### create_property_predictor

::: diffbio.operators.drug_discovery.property_predictor.create_property_predictor
    options:
      show_root_heading: true

### create_fingerprint_operator

::: diffbio.operators.drug_discovery.fingerprint.create_fingerprint_operator
    options:
      show_root_heading: true

### create_similarity_operator

::: diffbio.operators.drug_discovery.similarity.create_similarity_operator
    options:
      show_root_heading: true

## Similarity Functions

### tanimoto_similarity

::: diffbio.operators.drug_discovery.similarity.tanimoto_similarity
    options:
      show_root_heading: true

### cosine_similarity

::: diffbio.operators.drug_discovery.similarity.cosine_similarity
    options:
      show_root_heading: true

### dice_similarity

::: diffbio.operators.drug_discovery.similarity.dice_similarity
    options:
      show_root_heading: true

## Graph Conversion Utilities

### smiles_to_graph

::: diffbio.operators.drug_discovery.primitives.smiles_to_graph
    options:
      show_root_heading: true

### batch_smiles_to_graphs

::: diffbio.operators.drug_discovery.primitives.batch_smiles_to_graphs
    options:
      show_root_heading: true

### AtomFeatureConfig

::: diffbio.operators.drug_discovery.primitives.AtomFeatureConfig
    options:
      show_root_heading: true
      members: []

## Constants

### DEFAULT_ATOM_FEATURES

::: diffbio.operators.drug_discovery.primitives.DEFAULT_ATOM_FEATURES
    options:
      show_root_heading: true

### DEFAULT_ATOM_CONFIG

::: diffbio.operators.drug_discovery.primitives.DEFAULT_ATOM_CONFIG
    options:
      show_root_heading: true

## Usage Examples

### Property Prediction

```python
from diffbio.operators.drug_discovery import (
    MolecularPropertyPredictor,
    MolecularPropertyConfig,
    smiles_to_graph,
    DEFAULT_ATOM_FEATURES,
)
from flax import nnx

# Create predictor
config = MolecularPropertyConfig(
    hidden_dim=64,
    num_message_passing_steps=3,
    num_output_tasks=1,
    in_features=DEFAULT_ATOM_FEATURES,
)
predictor = MolecularPropertyPredictor(config, rngs=nnx.Rngs(42))

# Convert SMILES to graph
node_features, adjacency, edge_features = smiles_to_graph("CCO")

# Predict properties
data = {
    "node_features": node_features,
    "adjacency": adjacency,
    "edge_features": edge_features,
}
result, _, _ = predictor.apply(data, {}, None)
predictions = result["predictions"]  # (1,)
```

### Fingerprint Computation

```python
from diffbio.operators.drug_discovery import (
    DifferentiableMolecularFingerprint,
    MolecularFingerprintConfig,
    smiles_to_graph,
    DEFAULT_ATOM_FEATURES,
)
from flax import nnx

# Create fingerprint operator
config = MolecularFingerprintConfig(
    fingerprint_dim=256,
    hidden_dim=128,
    num_layers=3,
    in_features=DEFAULT_ATOM_FEATURES,
    normalize=True,
)
fp_op = DifferentiableMolecularFingerprint(config, rngs=nnx.Rngs(42))

# Compute fingerprint
node_features, adjacency, _ = smiles_to_graph("c1ccccc1")
data = {"node_features": node_features, "adjacency": adjacency}
result, _, _ = fp_op.apply(data, {}, None)
fingerprint = result["fingerprint"]  # (256,)
```

### Similarity Computation

```python
from diffbio.operators.drug_discovery import (
    MolecularSimilarityOperator,
    MolecularSimilarityConfig,
    tanimoto_similarity,
)
from flax import nnx
import jax.numpy as jnp

# Using operator
config = MolecularSimilarityConfig(similarity_type="tanimoto")
sim_op = MolecularSimilarityOperator(config, rngs=nnx.Rngs(42))

fp1 = jnp.array([1.0, 0.5, 0.0, 0.8])
fp2 = jnp.array([0.9, 0.6, 0.1, 0.7])

data = {"fingerprint_a": fp1, "fingerprint_b": fp2}
result, _, _ = sim_op.apply(data, {}, None)
similarity = result["similarity"]

# Using standalone function
sim = tanimoto_similarity(fp1, fp2)
```

### Batch Processing

```python
from diffbio.operators.drug_discovery import batch_smiles_to_graphs

smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
node_features, adjacency, edge_features, masks = batch_smiles_to_graphs(
    smiles_list,
    max_atoms=20,
)
# node_features: (3, 20, 34)
# adjacency: (3, 20, 20)
# edge_features: (3, 20, 20, 4)
# masks: (3, 20)
```

### Gradient Computation

```python
import jax
from flax import nnx
from diffbio.operators.drug_discovery import (
    create_property_predictor,
    smiles_to_graph,
    DEFAULT_ATOM_FEATURES,
)

predictor = create_property_predictor(
    hidden_dim=32,
    num_layers=2,
    num_tasks=1,
)

node_features, adjacency, edge_features = smiles_to_graph("CCO")
data = {
    "node_features": node_features,
    "adjacency": adjacency,
    "edge_features": edge_features,
}

def loss_fn(model, data):
    result, _, _ = model.apply(data, {}, None)
    return result["predictions"].sum()

# Compute gradients with nnx.grad
grads = nnx.grad(loss_fn)(predictor, data)
```

## Input Specifications

### MolecularPropertyPredictor

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `node_features` | (n, in_features) | float32 | Atom feature vectors |
| `adjacency` | (n, n) | float32 | Adjacency matrix |
| `edge_features` | (n, n, num_edge_features) | float32 | Optional bond features |
| `node_mask` | (n,) | float32 | Optional mask for valid atoms |

### DifferentiableMolecularFingerprint

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `node_features` | (n, in_features) | float32 | Atom feature vectors |
| `adjacency` | (n, n) | float32 | Adjacency matrix |
| `edge_features` | (n, n, num_edge_features) | float32 | Optional bond features |
| `node_mask` | (n,) | float32 | Optional mask for valid atoms |

### MolecularSimilarityOperator

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `fingerprint_a` | (dim,) | float32 | First fingerprint vector |
| `fingerprint_b` | (dim,) | float32 | Second fingerprint vector |

## Output Specifications

### MolecularPropertyPredictor

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `predictions` | (num_tasks,) | float32 | Property predictions |
| `graph_representation` | (hidden_dim,) | float32 | Graph-level embedding |

### DifferentiableMolecularFingerprint

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `fingerprint` | (fingerprint_dim,) | float32 | Molecular fingerprint |

### MolecularSimilarityOperator

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `similarity` | () | float32 | Similarity score |

## Atom Feature Dimensions

| Feature | Dimensions | Description |
|---------|------------|-------------|
| Atom type | 10 | C, N, O, S, F, Cl, Br, I, P, other |
| Degree | 6 | 0-5+ neighbors |
| Formal charge | 5 | -2 to +2 |
| Hybridization | 5 | SP, SP2, SP3, SP3D, SP3D2 |
| Aromaticity | 1 | Binary |
| Hydrogens | 5 | 0-4+ |
| In ring | 1 | Binary |
| Chiral | 1 | Binary |
| **Total** | **34** | `DEFAULT_ATOM_FEATURES` |
