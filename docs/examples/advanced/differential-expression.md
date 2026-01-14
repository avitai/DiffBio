# Differential Expression Analysis Example

This example demonstrates end-to-end differentiable differential expression analysis using DiffBio's DESeq2-style pipeline.

## Setup

```python
import jax
import jax.numpy as jnp
from flax import nnx
from diffbio.pipelines import (
    DifferentialExpressionPipeline,
    DEPipelineConfig,
)
```

## Generate Synthetic RNA-seq Data

```python
def generate_de_data(n_genes=2000, n_samples=100, n_de_genes=200, fold_change=2.0, seed=42):
    key = jax.random.key(seed)
    keys = jax.random.split(key, 5)

    # Condition assignments
    n_control = n_samples // 2
    condition = jnp.concatenate([
        jnp.zeros(n_control),
        jnp.ones(n_samples - n_control),
    ]).astype(jnp.int32)

    # Base expression levels
    base_mean = jnp.exp(jax.random.normal(keys[0], (n_genes,)) * 1.5 + 3)

    # DE genes
    de_mask = jnp.arange(n_genes) < n_de_genes
    log_fc = jnp.where(de_mask, jnp.log2(fold_change), 0.0)

    # Generate counts
    treatment_factor = condition[:, None] * log_fc[None, :]
    mean_expr = base_mean[None, :] * jnp.power(2, treatment_factor)
    counts = jax.random.poisson(keys[2], mean_expr).astype(jnp.float32)

    # Design matrix
    design = jnp.column_stack([jnp.ones(n_samples), condition.astype(jnp.float32)])

    return {"counts": counts, "design": design, "condition": condition, "true_de": de_mask}

data = generate_de_data()
print(f"Counts shape: {data['counts'].shape}")  # (100, 2000)
```

## Create and Run DE Pipeline

```python
config = DEPipelineConfig(
    n_genes=2000,
    n_conditions=2,
    alpha=0.05,
)

rngs = nnx.Rngs(42)
de_pipeline = DifferentialExpressionPipeline(config, rngs=rngs)

pipeline_data = {"counts": data["counts"], "design": data["design"]}
result, state, metadata = de_pipeline.apply(pipeline_data, {}, None)

log2fc = result["log_fold_change"]
pvalues = result["p_values"]
significant = result["significant"]

print(f"Detected significant genes: {significant.sum()}")
```

## Visualization

```python
import matplotlib.pyplot as plt

# Volcano plot
neg_log_p = -jnp.log10(pvalues + 1e-300)

# Convert soft significance to boolean for plotting
is_sig = significant > 0.5

plt.figure(figsize=(10, 8))
plt.scatter(log2fc[~is_sig], neg_log_p[~is_sig], c='gray', alpha=0.5, s=10)
plt.scatter(log2fc[is_sig], neg_log_p[is_sig], c='red', alpha=0.7, s=20)
plt.axhline(-jnp.log10(0.05), ls='--', c='black')
plt.xlabel('Log2 Fold Change')
plt.ylabel('-Log10 P-value')
plt.title('Volcano Plot')
plt.show()
```

## Next Steps

- See [Statistical Operators](../../user-guide/operators/statistical.md) for NB-GLM details
- Explore [Statistical Losses](../../user-guide/losses/statistical.md) for training objectives
