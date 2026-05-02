# Differential Expression Pipeline API

DESeq2-style differential expression analysis pipeline.

## DifferentialExpressionPipeline

::: diffbio.pipelines.differential_expression.DifferentialExpressionPipeline
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## DEPipelineConfig

::: diffbio.pipelines.differential_expression.DEPipelineConfig
    options:
      show_root_heading: true
      members: []

## Usage Examples

### Basic Differential Expression

```python
from flax import nnx
from diffbio.pipelines import (
    DifferentialExpressionPipeline,
    DEPipelineConfig,
)

config = DEPipelineConfig(
    n_genes=2000,
    n_conditions=2,
    alpha=0.05,
)

pipeline = DifferentialExpressionPipeline(config, rngs=nnx.Rngs(42))

data = {
    "counts": count_matrix,      # (n_samples, n_genes)
    "design": design_matrix,     # (n_samples, n_conditions)
}
result, _, _ = pipeline.apply(data, {}, None)

log2fc = result["log_fold_change"]
pvalues = result["p_values"]
significant = result["significant"]
```

### Access Intermediate Results

```python
# Size factors
size_factors = result["size_factors"]

# Predicted mean expression
predicted_mean = result["predicted_mean"]

# Log fold change estimates
log_fold_change = result["log_fold_change"]

# Wald test statistics and standard errors
wald_statistic = result["wald_statistic"]
standard_error = result["standard_error"]

# P-values and significance indicators
p_values = result["p_values"]
significant = result["significant"]
```
