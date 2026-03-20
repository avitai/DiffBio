# Metric Losses

DiffBio provides differentiable metric-based loss functions that can serve as training objectives, plus exact evaluation metrics backed by calibrax.

## DifferentiableAUROC

Smooth training surrogate for the Area Under the ROC Curve. Replaces the hard indicator in the Wilcoxon-Mann-Whitney statistic with a sigmoid function, making AUROC fully differentiable and JIT-compatible.

For every (positive, negative) pair, the hard AUROC checks whether the positive score exceeds the negative score. This module replaces that indicator with $\sigma((s_+ - s_-) / T)$, yielding a smooth surrogate whose gradient can drive optimisation.

### Quick Start

```python
from diffbio.losses.metric_losses import DifferentiableAUROC

auroc_loss = DifferentiableAUROC(temperature=1.0)

predictions = jnp.array([0.9, 0.8, 0.1, 0.2])
labels = jnp.array([1.0, 1.0, 0.0, 0.0])

value = auroc_loss(predictions, labels)  # ~1.0 for well-separated predictions
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 1.0 | Sigmoid sharpness (lower = closer to hard indicator) |

### Usage as Training Loss

```python
from flax import nnx

def auroc_training_loss(model, data, labels):
    result, _, _ = model.apply(data, {}, None)
    predictions = result["predictions"]

    # Maximize AUROC -> minimize negative AUROC
    return -auroc_loss(predictions, labels)

grads = nnx.grad(auroc_training_loss)(model, data, labels)
```

### Algorithm

$$\text{AUROC}_{\text{soft}} = \frac{1}{|P| \cdot |N|} \sum_{i \in P} \sum_{j \in N} \sigma\left(\frac{s_i - s_j}{T}\right)$$

Where $P$ and $N$ are the sets of positive and negative samples.

## ExactAUROC

Exact AUROC metric using calibrax's trapezoidal-rule implementation. Delegates to `calibrax.metrics.functional.classification.roc_auc` for the threshold-sweep and trapezoidal-rule computation.

Use this for **evaluation only**. The sorting-based trapezoidal rule has zero gradients with respect to predictions because `argsort` is not differentiable.

### Quick Start

```python
from diffbio.losses.metric_losses import ExactAUROC

exact = ExactAUROC()

predictions = jnp.array([0.9, 0.8, 0.1, 0.2])
labels = jnp.array([1.0, 1.0, 0.0, 0.0])

value = exact(predictions, labels)  # 1.0 (exact)
```

### Parameters

ExactAUROC has no learnable parameters.

### When to Use Which

| Metric | Differentiable | Use Case |
|--------|---------------|----------|
| DifferentiableAUROC | Yes | Training objective for gradient-based optimizers |
| ExactAUROC | No | Evaluation, reporting, model selection |

## Training Pattern

A typical workflow uses the differentiable variant for training and the exact variant for evaluation:

```python
from diffbio.losses.metric_losses import DifferentiableAUROC, ExactAUROC

# Training
train_auroc = DifferentiableAUROC(temperature=1.0)

def train_loss(model, data, labels):
    result, _, _ = model.apply(data, {}, None)
    return -train_auroc(result["scores"], labels)

# Evaluation
eval_auroc = ExactAUROC()

def evaluate(model, data, labels):
    result, _, _ = model.apply(data, {}, None)
    return eval_auroc(result["scores"], labels)
```

## Next Steps

- See [Single-Cell Losses](singlecell.md) for single-cell training objectives
- Explore [Statistical Losses](statistical.md) for count-based losses
