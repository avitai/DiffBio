# Loss Functions Overview

DiffBio provides specialized loss functions for training differentiable bioinformatics pipelines.

## Available Loss Functions

### Single-Cell Losses

| Loss | Description |
|------|-------------|
| [BatchMixingLoss](singlecell.md#batchmixingloss) | Maximizes batch mixing in latent space |
| [ClusteringCompactnessLoss](singlecell.md#clusteringcompactnessloss) | Encourages tight, well-separated clusters |
| [VelocityConsistencyLoss](singlecell.md#velocityconsistencyloss) | Ensures RNA velocity consistency |
| [ShannonDiversityLoss](singlecell.md#shannondiversityloss) | Shannon entropy of cluster assignments |
| [SimpsonDiversityLoss](singlecell.md#simpsondiversityloss) | Simpson concentration index of assignments |

### Metric Losses

| Loss | Description |
|------|-------------|
| [DifferentiableAUROC](metric.md#differentiableauroc) | Sigmoid-approximated AUROC for training |
| [ExactAUROC](metric.md#exactauroc) | Exact trapezoidal-rule AUROC for evaluation |

### Statistical Losses

| Loss | Description |
|------|-------------|
| [NegativeBinomialLoss](statistical.md#negativebinomialloss) | NB log-likelihood for count data |
| [VAELoss](statistical.md#vaeloss) | ELBO loss with KL regularization |
| [HMMLikelihoodLoss](statistical.md#hmmlikelihoodloss) | HMM forward algorithm loss |

### Alignment Losses

| Loss | Description |
|------|-------------|
| `AlignmentScoreLoss` | Alignment quality loss |
| `GapPenaltyLoss` | Learnable gap penalty regularization |

## Using Loss Functions

### Basic Usage

```python
from diffbio.losses import NegativeBinomialLoss

# Create loss function
nb_loss = NegativeBinomialLoss()

# Compute loss
loss = nb_loss(
    counts=observed_counts,
    predicted_mean=model_predictions,
    dispersion=dispersions,
)
```

### With Training

```python
from flax import nnx
from diffbio.losses import BatchMixingLoss

batch_loss = BatchMixingLoss(n_neighbors=15, temperature=1.0)

def train_step(model, data):
    def loss_fn(m):
        result, _, _ = m.apply(data, {}, None)
        return batch_loss(result["embeddings"], data["batch_ids"])

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    return loss, grads
```

### Combining Multiple Losses

```python
from diffbio.losses import (
    BatchMixingLoss,
    ClusteringCompactnessLoss,
)

batch_loss = BatchMixingLoss()
cluster_loss = ClusteringCompactnessLoss()

def combined_loss(model, data):
    result, _, _ = model.apply(data, {}, None)

    # Batch mixing (maximize, so negate)
    l_batch = -batch_loss(result["embeddings"], data["batch_ids"])

    # Clustering compactness (minimize)
    l_cluster = cluster_loss(result["embeddings"], result["assignments"])

    # Weighted combination
    return l_batch + 0.5 * l_cluster
```

## Loss Function Interface

All DiffBio losses follow a consistent interface:

```python
class Loss:
    def __init__(self, **config):
        """Initialize loss with configuration."""
        pass

    def __call__(self, **inputs) -> jax.Array:
        """Compute loss value.

        Returns:
            Scalar loss value.
        """
        pass
```

## Gradient Properties

All losses are designed for:

- **Numerical stability**: Using log-space computations where needed
- **Smooth gradients**: Temperature-controlled soft operations
- **JAX compatibility**: Full support for `jax.grad`, `jax.jit`

```python
# Gradient computation
loss_fn = lambda model: loss(model.apply(data, {}, None)[0])
grads = jax.grad(loss_fn)(model)

# JIT compilation
@jax.jit
def compute_loss(model, data):
    result, _, _ = model.apply(data, {}, None)
    return loss(result)
```

## Next Steps

- Learn about [Single-Cell Losses](singlecell.md)
- Explore [Statistical Losses](statistical.md)
- Check [Metric Losses](metric.md) for AUROC training surrogates
- See [Training Overview](../training/overview.md) for training workflows
