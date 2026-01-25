# Population Genetics Operators

DiffBio provides differentiable operators for population genetics analysis including ancestry estimation, enabling gradient-based optimization of population structure models.

<span class="operator-population">Population</span> <span class="diff-high">Fully Differentiable</span>

## Overview

Population genetics operators enable end-to-end optimization of:

- **DifferentiableAncestryEstimator**: Neural ADMIXTURE-style ancestry estimation

## DifferentiableAncestryEstimator

Autoencoder-based ancestry estimation that learns to decompose individual genotypes into proportions from K ancestral populations.

### Quick Start

```python
from flax import nnx
from diffbio.operators.population import (
    DifferentiableAncestryEstimator,
    AncestryEstimatorConfig,
    create_ancestry_estimator,
)

# Configure estimator
config = AncestryEstimatorConfig(
    n_snps=10000,           # Number of SNP markers
    n_populations=5,        # Number of ancestral populations (K)
    hidden_dims=(128, 64),  # Encoder hidden layers
    temperature=1.0,        # Softmax temperature
    dropout_rate=0.1,       # Regularization
)

# Create operator
rngs = nnx.Rngs(42)
estimator = DifferentiableAncestryEstimator(config, rngs=rngs)

# Apply ancestry estimation
data = {
    "genotypes": genotype_matrix,  # (n_samples, n_snps), values 0/1/2
}
result, state, metadata = estimator.apply(data, {}, None)

# Get ancestry proportions
ancestry = result["ancestry_proportions"]  # (n_samples, K)
reconstructed = result["reconstructed"]     # Reconstructed genotypes
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_snps` | int | 10000 | Number of SNP markers in input |
| `n_populations` | int | 5 | Number of ancestral populations (K) |
| `hidden_dims` | tuple[int, ...] | (128, 64) | Encoder hidden layer dimensions |
| `temperature` | float | 1.0 | Temperature for softmax proportions |
| `dropout_rate` | float | 0.1 | Dropout rate for regularization |

### ADMIXTURE Model

The operator implements the classic ADMIXTURE generative model:

$$G_{ij} = \sum_k Q_{ik} \cdot P_{kj}$$

Where:

- $G$ = genotype matrix (individuals × SNPs)
- $Q$ = ancestry proportion matrix (individuals × K)
- $P$ = population allele frequency matrix (K × SNPs)

The neural network learns both Q (via encoder) and P (learnable parameters).

### Architecture

```
Genotypes ──→ Encoder ──→ Latent ──→ Softmax ──→ Ancestry (Q)
    (n, s)    (MLP)       (n, h)     (τ)         (n, K)
                                                    │
                                                    ↓
                                     Population Frequencies (P)
                                           (K, s)
                                                    │
                                                    ↓
                          Reconstructed ←── Q @ P ──┘
                             (n, s)
```

The encoder maps genotypes to a latent space, then computes ancestry proportions via temperature-controlled softmax. The decoder reconstructs genotypes using the ADMIXTURE model.

### Temperature Control

The temperature parameter controls the sharpness of ancestry assignments:

| Temperature | Effect |
|-------------|--------|
| High (5.0+) | Softer, more uniform proportions |
| Medium (1.0) | Balanced |
| Low (0.1) | Sharper, more confident assignments |

```python
# High temperature (exploration)
config_soft = AncestryEstimatorConfig(temperature=5.0)

# Low temperature (exploitation)
config_sharp = AncestryEstimatorConfig(temperature=0.1)
```

### Training

```python
import optax
from flax import nnx

estimator = create_ancestry_estimator(n_snps=10000, n_populations=5)
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(nnx.state(estimator, nnx.Param))

def loss_fn(model, genotypes):
    """Reconstruction loss for unsupervised training."""
    result, _, _ = model.apply({"genotypes": genotypes}, {}, None)

    # Reconstruction loss
    recon_loss = jnp.mean((result["reconstructed"] - genotypes) ** 2)

    # Optional: Entropy regularization for sparse ancestry
    proportions = result["ancestry_proportions"]
    entropy = -jnp.mean(jnp.sum(proportions * jnp.log(proportions + 1e-10), axis=-1))

    return recon_loss - 0.01 * entropy

@jax.jit
def train_step(model, opt_state, genotypes):
    loss, grads = jax.value_and_grad(loss_fn)(model, genotypes)
    params = nnx.state(model, nnx.Param)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    nnx.update(model, optax.apply_updates(params, updates))
    return loss, opt_state

# Training loop
estimator.train()
for epoch in range(100):
    loss, opt_state = train_step(estimator, opt_state, train_genotypes)

estimator.eval()
```

### Supervised Training

With known population labels:

```python
def supervised_loss(model, genotypes, true_labels):
    """Cross-entropy loss for supervised training."""
    result, _, _ = model.apply({"genotypes": genotypes}, {}, None)

    # One-hot encode true labels
    true_onehot = jax.nn.one_hot(true_labels, num_classes=K)

    # Cross-entropy
    log_probs = jnp.log(result["ancestry_proportions"] + 1e-10)
    ce_loss = -jnp.mean(jnp.sum(true_onehot * log_probs, axis=-1))

    return ce_loss
```

### Inference

```python
estimator.eval()

result, _, _ = estimator.apply({"genotypes": test_genotypes}, {}, None)

# Get ancestry proportions
ancestry = result["ancestry_proportions"]  # (n_samples, K)

# Find primary ancestry
primary_ancestry = jnp.argmax(ancestry, axis=-1)

# Visualize admixture
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 3))
plt.bar(range(len(ancestry)), ancestry[:, 0], label="Pop 1")
bottom = ancestry[:, 0]
for k in range(1, K):
    plt.bar(range(len(ancestry)), ancestry[:, k], bottom=bottom, label=f"Pop {k+1}")
    bottom += ancestry[:, k]
plt.ylabel("Ancestry Proportion")
plt.xlabel("Individual")
plt.legend()
plt.show()
```

### Accessing Learned Parameters

```python
# Population allele frequencies (P matrix)
pop_freqs = estimator.population_frequencies[...]  # (K, n_snps)

# Encoder weights
encoder_params = nnx.state(estimator, nnx.Param)

# Temperature
temperature = estimator.temperature[...]
```

## Use Cases

| Application | Operator | Description |
|-------------|----------|-------------|
| Ancestry inference | DifferentiableAncestryEstimator | Estimate ancestry proportions |
| Population structure | DifferentiableAncestryEstimator | Discover population structure |
| Admixture analysis | DifferentiableAncestryEstimator | Model genetic admixture |

## References

1. Alexander, D.H. et al. (2009). "Fast model-based estimation of ancestry in unrelated individuals." *Genome Research*.

2. Dias, A. et al. (2022). "Neural ADMIXTURE: Rapid population clustering with autoencoders." *Nature Computational Science*.

3. Pritchard, J.K. et al. (2000). "Inference of Population Structure Using Multilocus Genotype Data." *Genetics*.

## Next Steps

- See [Statistical Operators](statistical.md) for related statistical methods
- Explore [Single-Cell Operators](singlecell.md) for clustering approaches
