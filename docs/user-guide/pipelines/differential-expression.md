# Differential Expression Pipeline

The `DifferentialExpressionPipeline` provides DESeq2-style differential expression analysis with end-to-end differentiability.

<span class="pipeline-deseq">Differential Expression</span> <span class="diff-high">Fully Differentiable</span>

## Overview

The differential expression pipeline implements:

1. **Size Factor Estimation**: Median-of-ratios normalization
2. **Dispersion Estimation**: Gene-wise dispersion fitting
3. **NB-GLM Fitting**: Negative binomial generalized linear model
4. **Statistical Testing**: Wald test for significance
5. **Multiple Testing Correction**: Benjamini-Hochberg FDR

## Quick Start

```python
from flax import nnx
from diffbio.pipelines import (
    DifferentialExpressionPipeline,
    DEPipelineConfig,
)

# Configure pipeline
config = DEPipelineConfig(
    n_genes=2000,
    n_conditions=2,
    alpha=0.05,
)

# Create pipeline
rngs = nnx.Rngs(42)
de_pipeline = DifferentialExpressionPipeline(config, rngs=rngs)

# Run differential expression analysis
data = {
    "counts": count_matrix,      # (n_samples, n_genes)
    "design": design_matrix,     # (n_samples, n_conditions)
}
result, state, metadata = de_pipeline.apply(data, {}, None)

# Get results
log2fc = result["log_fold_change"]       # (n_genes,)
pvalues = result["p_values"]             # (n_genes,)
significant = result["significant"]      # Soft significance indicator
```

## Configuration

### DEPipelineConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_genes` | int | 1000 | Number of genes |
| `n_conditions` | int | 2 | Number of conditions/covariates |
| `alpha` | float | 0.05 | Significance threshold |
| `use_size_factors` | bool | True | Whether to compute size factors |

### Detailed Configuration

```python
config = DEPipelineConfig(
    # Data dimensions
    n_genes=2000,
    n_conditions=2,

    # Testing
    alpha=0.05,
    use_size_factors=True,
)
```

## Pipeline Stages

### Stage 1: Size Factor Estimation

Median-of-ratios method (DESeq2 style):

```python
# Geometric mean per gene
geo_means = jnp.exp(jnp.mean(jnp.log(counts + 1), axis=0))

# Size factors per sample
ratios = counts / geo_means
size_factors = jnp.median(ratios, axis=1)
```

### Stage 2: Dispersion Estimation

Gene-wise dispersion with shrinkage:

```python
# Initial dispersion estimate (method of moments)
mean_counts = counts.mean(axis=0)
var_counts = counts.var(axis=0)
alpha_init = (var_counts - mean_counts) / (mean_counts ** 2)

# Shrinkage towards trend
dispersion = shrink_dispersions(alpha_init, mean_counts)
```

### Stage 3: NB-GLM Fitting

Fit negative binomial GLM:

$$Y_{ij} \sim NB(\mu_{ij}, \alpha_j)$$
$$\log(\mu_{ij}) = \log(s_i) + X_i \cdot \beta_j$$

Where:
- $s_i$ = size factor for sample $i$
- $X_i$ = design matrix row
- $\beta_j$ = coefficients for gene $j$
- $\alpha_j$ = dispersion for gene $j$

### Stage 4: Wald Test

Test for significant coefficients:

```python
# Wald statistic
wald_stat = beta / se_beta

# Two-sided p-value
from jax.scipy.stats import norm
pvalue = 2 * (1 - norm.cdf(jnp.abs(wald_stat)))
```

### Stage 5: Multiple Testing Correction

Benjamini-Hochberg FDR:

```python
# Sort p-values
sorted_idx = jnp.argsort(pvalues)
sorted_pvals = pvalues[sorted_idx]

# BH correction
n = len(pvalues)
adjusted = sorted_pvals * n / (jnp.arange(n) + 1)
adjusted = jnp.minimum.accumulate(adjusted[::-1])[::-1]

# Unsort
padj = adjusted[jnp.argsort(sorted_idx)]
```

## Output Format

The pipeline returns a dictionary with:

| Key | Shape | Description |
|-----|-------|-------------|
| `log_fold_change` | (n_genes,) | Log2 fold change |
| `pvalues` | (n_genes,) | Raw p-values |
| `p_values` | (n_genes,) | FDR-adjusted p-values |
| `significant` | (n_genes,) | Boolean significance mask |
| `coefficients` | (n_genes, n_conditions) | GLM coefficients |
| `dispersions` | (n_genes,) | Gene dispersions |
| `size_factors` | (n_samples,) | Sample size factors |
| `normalized_counts` | (n_samples, n_genes) | Normalized counts |

## Training / Fine-tuning

### Loss Function for DE

```python
from diffbio.losses.statistical_losses import NegativeBinomialLoss

nb_loss = NegativeBinomialLoss()

def de_loss(pipeline, counts, design, known_de_genes):
    """Train DE pipeline with known DE genes."""
    data = {"counts": counts, "design": design, "condition": design[:, 1]}
    result, _, _ = pipeline.apply(data, {}, None)

    # Likelihood loss
    likelihood = nb_loss(
        counts=counts,
        predicted_mean=result["predicted_means"],
        dispersion=result["dispersions"],
    )

    # Optional: supervised loss if DE genes are known
    if known_de_genes is not None:
        sig_loss = binary_cross_entropy(
            result["significant"].astype(float),
            known_de_genes.astype(float),
        )
        return likelihood + 0.1 * sig_loss

    return likelihood
```

### End-to-End Optimization

```python
import optax
from flax import nnx

# Create optimizer
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(nnx.state(de_pipeline, nnx.Param))

@jax.jit
def train_step(pipeline, counts, design, opt_state):
    def loss_fn(pipe):
        data = {"counts": counts, "design": design, "condition": design[:, 1]}
        result, _, _ = pipe.apply(data, {}, None)
        return nb_loss(counts, result["predicted_means"], result["dispersions"])

    loss, grads = nnx.value_and_grad(loss_fn)(pipeline)
    params = nnx.state(pipeline, nnx.Param)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    nnx.update(pipeline, optax.apply_updates(params, updates))
    return loss, opt_state
```

## Visualization

### Volcano Plot

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))

log2fc = result["log_fold_change"]
neg_log_p = -jnp.log10(result["p_values"] + 1e-300)
significant = result["significant"]

# Non-significant
ax.scatter(
    log2fc[~significant],
    neg_log_p[~significant],
    c='gray', alpha=0.5, s=10
)

# Significant
ax.scatter(
    log2fc[significant],
    neg_log_p[significant],
    c='red', alpha=0.7, s=20
)

ax.axhline(-jnp.log10(0.05), ls='--', c='black')
ax.axvline(-1, ls='--', c='black')
ax.axvline(1, ls='--', c='black')

ax.set_xlabel('Log2 Fold Change')
ax.set_ylabel('-Log10 Adjusted P-value')
ax.set_title('Volcano Plot')
plt.show()
```

### MA Plot

```python
base_mean = result["normalized_counts"].mean(axis=0)
log2fc = result["log_fold_change"]

plt.figure(figsize=(10, 8))
plt.scatter(jnp.log10(base_mean + 1), log2fc, c='gray', alpha=0.5, s=10)
plt.scatter(
    jnp.log10(base_mean[significant] + 1),
    log2fc[significant],
    c='red', alpha=0.7, s=20
)
plt.axhline(0, ls='--', c='black')
plt.xlabel('Log10 Mean Expression')
plt.ylabel('Log2 Fold Change')
plt.title('MA Plot')
plt.show()
```

## Use Cases

| Application | Description |
|-------------|-------------|
| RNA-seq DE | Compare gene expression between conditions |
| Single-cell DE | Marker gene discovery |
| Time series | Identify temporally regulated genes |
| Drug response | Find genes responding to treatment |

## Next Steps

- See [Statistical Operators](../operators/statistical.md) for NB-GLM details
- Explore [Statistical Losses](../losses/statistical.md) for loss functions
- Check [Differential Expression Example](../../examples/advanced/differential-expression.md)
