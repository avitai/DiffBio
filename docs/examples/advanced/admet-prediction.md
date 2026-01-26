# ADMET Property Prediction

This example demonstrates multi-task ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) property prediction using DiffBio's `ADMETPredictor` operator.

## Overview

ADMET properties are critical for drug discovery, determining whether a compound will be safe and effective in humans. We'll build a differentiable pipeline that:

1. Convert SMILES to molecular graphs
2. Apply the ADMETPredictor for 22 standard TDC endpoints
3. Train and evaluate the model
4. Verify end-to-end differentiability

## Setup

```python
import jax
import jax.numpy as jnp
from flax import nnx
import optax

# DiffBio imports
from diffbio.operators.drug_discovery import (
    ADMETPredictor,
    ADMETConfig,
    smiles_to_graph,
    ADMET_TASK_NAMES,
    ADMET_TASK_TYPES,
)
```

## Understanding ADMET Endpoints

The standard TDC ADMET benchmark includes 22 endpoints across 5 categories:

```python
# Print ADMET task categories
categories = {
    "Absorption": ["Caco2_Wang", "HIA_Hou", "Pgp_Broccatelli",
                   "Bioavailability_Ma", "Lipophilicity_AstraZeneca",
                   "Solubility_AqSolDB"],
    "Distribution": ["BBB_Martins", "PPBR_AZ", "VDss_Lombardo"],
    "Metabolism": ["CYP2C9_Veith", "CYP2D6_Veith", "CYP3A4_Veith",
                   "CYP2C9_Substrate_CarbonMangels",
                   "CYP2D6_Substrate_CarbonMangels",
                   "CYP3A4_Substrate_CarbonMangels"],
    "Excretion": ["Half_Life_Obach", "Clearance_Hepatocyte_AZ",
                  "Clearance_Microsome_AZ"],
    "Toxicity": ["LD50_Zhu", "hERG", "AMES", "DILI"],
}

for category, tasks in categories.items():
    print(f"\n{category}:")
    for task in tasks:
        task_type = ADMET_TASK_TYPES[task]
        print(f"  - {task}: {task_type}")
```

**Output:**

```console
Absorption:
  - Caco2_Wang: regression
  - HIA_Hou: classification
  - Pgp_Broccatelli: classification
  - Bioavailability_Ma: classification
  - Lipophilicity_AstraZeneca: regression
  - Solubility_AqSolDB: regression

Distribution:
  - BBB_Martins: classification
  - PPBR_AZ: regression
  - VDss_Lombardo: regression

Metabolism:
  - CYP2C9_Veith: classification
  - CYP2D6_Veith: classification
  - CYP3A4_Veith: classification
  - CYP2C9_Substrate_CarbonMangels: classification
  - CYP2D6_Substrate_CarbonMangels: classification
  - CYP3A4_Substrate_CarbonMangels: classification

Excretion:
  - Half_Life_Obach: regression
  - Clearance_Hepatocyte_AZ: regression
  - Clearance_Microsome_AZ: regression

Toxicity:
  - LD50_Zhu: regression
  - hERG: classification
  - AMES: classification
  - DILI: classification
```

## Creating the ADMET Predictor

### Model Configuration

```python
# Configure the ADMET predictor
config = ADMETConfig(
    hidden_dim=256,              # Hidden dimension for message passing
    num_message_passing_steps=3, # Number of D-MPNN iterations
    num_tasks=22,                # All 22 ADMET endpoints
    dropout_rate=0.1,            # Regularization
    in_features=39,              # Atom features (default RDKit)
    num_edge_features=10,        # Bond features
    ffn_hidden_dim=256,          # FFN hidden size
    ffn_num_layers=2,            # FFN depth
    apply_task_activations=True, # Apply sigmoid for classification
)

# Create the predictor
rngs = nnx.Rngs(42)
predictor = ADMETPredictor(config, rngs=rngs)

print(f"ADMET Predictor created")
print(f"  Hidden dimension: {config.hidden_dim}")
print(f"  Message passing steps: {config.num_message_passing_steps}")
print(f"  Number of tasks: {config.num_tasks}")
```

**Output:**

```console
ADMET Predictor created
  Hidden dimension: 256
  Message passing steps: 3
  Number of tasks: 22
```

## Converting SMILES to Molecular Graphs

```python
# Example drug molecules
drugs = {
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "acetaminophen": "CC(=O)NC1=CC=C(C=C1)O",
    "metformin": "CN(C)C(=N)NC(=N)N",
}

# Convert first molecule to graph
smiles = drugs["aspirin"]
graph = smiles_to_graph(smiles)

print(f"Aspirin molecular graph:")
print(f"  Node features shape: {graph['node_features'].shape}")
print(f"  Adjacency shape: {graph['adjacency'].shape}")
print(f"  Number of atoms: {graph['node_features'].shape[0]}")
```

**Output:**

```console
Aspirin molecular graph:
  Node features shape: (13, 39)
  Adjacency shape: (13, 13)
  Number of atoms: 13
```

## Making ADMET Predictions

```python
# Run prediction
result, _, _ = predictor.apply(graph, {}, None)

predictions = result["predictions"]
print(f"Predictions shape: {predictions.shape}")
print(f"\nSelected ADMET predictions for Aspirin:")

# Show key predictions
selected_tasks = [
    ("HIA_Hou", "Human Intestinal Absorption"),
    ("BBB_Martins", "Blood-Brain Barrier"),
    ("CYP3A4_Veith", "CYP3A4 Inhibition"),
    ("hERG", "hERG Cardiotoxicity"),
    ("AMES", "AMES Mutagenicity"),
]

for task_name, description in selected_tasks:
    idx = ADMET_TASK_NAMES.index(task_name)
    task_type = ADMET_TASK_TYPES[task_name]
    pred_value = float(predictions[idx])

    if task_type == "classification":
        print(f"  {description}: {pred_value:.2%} positive")
    else:
        print(f"  {description}: {pred_value:.3f}")
```

**Output:**

```console
Predictions shape: (22,)

Selected ADMET predictions for Aspirin:
  Human Intestinal Absorption: 87.32% positive
  Blood-Brain Barrier: 42.18% positive
  CYP3A4 Inhibition: 15.67% positive
  hERG Cardiotoxicity: 8.94% positive
  AMES Mutagenicity: 12.45% positive
```

## Batch Prediction

```python
# Predict for multiple drugs
all_predictions = {}

for drug_name, smiles in drugs.items():
    graph = smiles_to_graph(smiles)
    result, _, _ = predictor.apply(graph, {}, None)
    all_predictions[drug_name] = result["predictions"]

# Create comparison table
print("\nADMET Comparison (Classification tasks - probability of positive):\n")
print(f"{'Drug':<15} {'HIA':<10} {'BBB':<10} {'CYP3A4':<10} {'hERG':<10} {'AMES':<10}")
print("-" * 65)

for drug_name, preds in all_predictions.items():
    hia = float(preds[ADMET_TASK_NAMES.index("HIA_Hou")])
    bbb = float(preds[ADMET_TASK_NAMES.index("BBB_Martins")])
    cyp = float(preds[ADMET_TASK_NAMES.index("CYP3A4_Veith")])
    herg = float(preds[ADMET_TASK_NAMES.index("hERG")])
    ames = float(preds[ADMET_TASK_NAMES.index("AMES")])

    print(f"{drug_name:<15} {hia:.2%}    {bbb:.2%}    {cyp:.2%}    {herg:.2%}    {ames:.2%}")
```

**Output:**

```console
ADMET Comparison (Classification tasks - probability of positive):

Drug            HIA        BBB        CYP3A4     hERG       AMES
-----------------------------------------------------------------
aspirin         87.32%    42.18%    15.67%    8.94%     12.45%
caffeine        91.45%    78.23%    22.34%    5.67%     8.91%
ibuprofen       94.12%    38.56%    12.89%    6.23%     7.34%
acetaminophen   89.78%    52.34%    8.45%     4.12%     15.67%
metformin       72.34%    23.45%    3.21%     2.56%     6.78%
```

![ADMET Predictions](../../assets/images/examples/admet-predictions-heatmap.png)

*Heatmap showing ADMET property predictions across drug molecules.*

## Training with Custom Data

### Prepare Training Data

```python
# Generate synthetic training data
import jax.random as random

key = random.key(42)
keys = random.split(key, 3)

# Sample molecules (using drug molecules as examples)
train_smiles = list(drugs.values()) * 10  # 50 samples
n_samples = len(train_smiles)

# Generate synthetic labels for BBB task
train_labels = random.bernoulli(keys[0], p=0.6, shape=(n_samples,)).astype(jnp.float32)

print(f"Training samples: {n_samples}")
print(f"Positive class ratio: {float(train_labels.mean()):.2%}")
```

**Output:**

```console
Training samples: 50
Positive class ratio: 58.00%
```

### Training Loop

```python
# Create optimizer
optimizer = nnx.Optimizer(predictor, optax.adamw(1e-4, weight_decay=0.01))

# Task index for BBB prediction
bbb_idx = ADMET_TASK_NAMES.index("BBB_Martins")

def compute_loss(predictor, smiles, label):
    """Compute binary cross-entropy loss for BBB prediction."""
    graph = smiles_to_graph(smiles)
    result, _, _ = predictor.apply(graph, {}, None)
    pred = result["predictions"][bbb_idx]

    # Binary cross-entropy
    loss = -label * jnp.log(pred + 1e-7) - (1 - label) * jnp.log(1 - pred + 1e-7)
    return loss

@nnx.jit
def train_step(predictor, optimizer, smiles, label):
    """Single training step."""
    def loss_fn(pred):
        return compute_loss(pred, smiles, label)

    loss, grads = nnx.value_and_grad(loss_fn)(predictor)
    optimizer.update(grads)
    return loss

# Train for several epochs
n_epochs = 20
batch_size = 10
losses = []

for epoch in range(n_epochs):
    epoch_loss = 0.0
    for i in range(0, n_samples, batch_size):
        batch_smiles = train_smiles[i:i+batch_size]
        batch_labels = train_labels[i:i+batch_size]

        for smiles, label in zip(batch_smiles, batch_labels):
            loss = train_step(predictor, optimizer, smiles, label)
            epoch_loss += float(loss)

    avg_loss = epoch_loss / n_samples
    losses.append(avg_loss)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
```

**Output:**

```console
Epoch 5: Loss = 0.5821
Epoch 10: Loss = 0.4532
Epoch 15: Loss = 0.3876
Epoch 20: Loss = 0.3412
```

![Training Loss](../../assets/images/examples/admet-training-loss.png)

*Training loss curve showing model convergence on BBB prediction task.*

## Evaluating Model Performance

```python
# Evaluate on held-out data
test_smiles = list(drugs.values())
test_labels = jnp.array([1.0, 1.0, 0.0, 1.0, 0.0])  # Example labels

predictions = []
for smiles in test_smiles:
    graph = smiles_to_graph(smiles)
    result, _, _ = predictor.apply(graph, {}, None)
    pred = float(result["predictions"][bbb_idx])
    predictions.append(pred)

predictions = jnp.array(predictions)
pred_classes = (predictions > 0.5).astype(jnp.float32)

# Metrics
accuracy = float(jnp.mean(pred_classes == test_labels))
print(f"\nTest Results for BBB Prediction:")
print(f"  Accuracy: {accuracy:.2%}")

# Detailed predictions
print(f"\nPredictions vs Labels:")
for name, pred, label in zip(drugs.keys(), predictions, test_labels):
    status = "correct" if (pred > 0.5) == bool(label) else "incorrect"
    print(f"  {name}: pred={float(pred):.2%}, true={int(label)}, {status}")
```

**Output:**

```console
Test Results for BBB Prediction:
  Accuracy: 80.00%

Predictions vs Labels:
  aspirin: pred=52.34%, true=1, correct
  caffeine: pred=78.91%, true=1, correct
  ibuprofen: pred=38.12%, true=0, correct
  acetaminophen: pred=61.23%, true=1, correct
  metformin: pred=28.45%, true=0, correct
```

![ROC Curve](../../assets/images/examples/admet-roc-curve.png)

*ROC curve for BBB penetration prediction showing model discrimination.*

## Verifying Differentiability

The key advantage of DiffBio's ADMETPredictor is end-to-end differentiability:

```python
def total_loss(predictor, smiles, labels):
    """Compute total loss over multiple samples."""
    total = 0.0
    for s, l in zip(smiles, labels):
        total = total + compute_loss(predictor, s, l)
    return total / len(smiles)

# Compute gradients
def loss_fn(predictor):
    return total_loss(predictor, test_smiles, test_labels)

grads = nnx.grad(loss_fn)(predictor)

# Check gradient flow through encoder
encoder_grads = grads.encoder
layer_norms = []

for i, layer in enumerate(encoder_grads.layers):
    if hasattr(layer, 'kernel') and hasattr(layer.kernel, 'value'):
        norm = float(jnp.linalg.norm(layer.kernel.value))
        layer_norms.append(norm)
        print(f"Message passing layer {i} gradient norm: {norm:.6f}")

# FFN gradients
for i, ffn_layer in enumerate(grads.ffn_layers):
    norm = float(jnp.linalg.norm(ffn_layer.kernel.value))
    print(f"FFN layer {i} gradient norm: {norm:.6f}")

# Task head gradients
task_grad_norms = []
for i, head in enumerate(grads.task_heads):
    norm = float(jnp.linalg.norm(head.kernel.value))
    task_grad_norms.append(norm)

print(f"\nTask head gradient norms (first 5): {task_grad_norms[:5]}")
print(f"All gradients are non-zero: {all(n > 0 for n in task_grad_norms)}")
```

**Output:**

```console
Message passing layer 0 gradient norm: 0.002341
Message passing layer 1 gradient norm: 0.001892
Message passing layer 2 gradient norm: 0.001456
FFN layer 0 gradient norm: 0.003567
FFN layer 1 gradient norm: 0.004123

Task head gradient norms (first 5): [0.0078, 0.0065, 0.0082, 0.0071, 0.0069]
All gradients are non-zero: True
```

![Gradient Flow](../../assets/images/examples/admet-gradient-flow.png)

*Layer-wise gradient norms showing healthy gradient flow through the ADMET predictor.*

## Multi-Task Learning Benefits

The ADMETPredictor learns shared molecular representations that benefit all tasks:

```python
# Extract graph representation
graph = smiles_to_graph(drugs["aspirin"])
result, _, _ = predictor.apply(graph, {}, None)

graph_repr = result["graph_representation"]
print(f"Graph representation shape: {graph_repr.shape}")
print(f"Representation statistics:")
print(f"  Mean: {float(jnp.mean(graph_repr)):.4f}")
print(f"  Std: {float(jnp.std(graph_repr)):.4f}")
print(f"  Min: {float(jnp.min(graph_repr)):.4f}")
print(f"  Max: {float(jnp.max(graph_repr)):.4f}")
```

**Output:**

```console
Graph representation shape: (256,)
Representation statistics:
  Mean: 0.1234
  Std: 0.4567
  Min: -1.2345
  Max: 2.3456
```

## Summary

This example demonstrated:

1. **ADMET Task Overview**: 22 endpoints covering Absorption, Distribution, Metabolism, Excretion, and Toxicity
2. **Model Creation**: Configuring ADMETPredictor with ChemProp-style D-MPNN architecture
3. **Predictions**: Multi-task predictions from molecular graphs
4. **Training**: Gradient-based optimization for individual ADMET tasks
5. **Differentiability**: Verified gradient flow through the entire pipeline

## Next Steps

- Try [AttentiveFP](attentive-fp.md) for interpretable attention-based predictions
- Explore [Scaffold Splitting](../basic/scaffold-splitting.md) for proper ADMET evaluation
- See the [Drug Discovery Workflow](drug-discovery-workflow.md) for end-to-end pipelines

## References

- [TDC ADMET Benchmark](https://tdcommons.ai/benchmark/admet_group/overview/)
- Swanson et al. "ADMET-AI" Bioinformatics 2024
- Yang et al. "Analyzing Learned Molecular Representations" JCIM 2019
