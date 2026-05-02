# Population Genetics Operators API

Differentiable operators for population genetics analysis including ancestry estimation.

## DifferentiableAncestryEstimator

::: diffbio.operators.population.ancestry_estimation.DifferentiableAncestryEstimator
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply
        - encode
        - compute_ancestry
        - decode

## AncestryEstimatorConfig

::: diffbio.operators.population.ancestry_estimation.AncestryEstimatorConfig
    options:
      show_root_heading: true
      members: []

## create_ancestry_estimator

::: diffbio.operators.population.ancestry_estimation.create_ancestry_estimator
    options:
      show_root_heading: true
      show_source: false

## Usage Examples

### Basic Ancestry Estimation

```python
from flax import nnx
from diffbio.operators.population import (
    DifferentiableAncestryEstimator,
    AncestryEstimatorConfig,
    create_ancestry_estimator,
)

# Using config
config = AncestryEstimatorConfig(
    n_snps=10000,
    n_populations=5,
    hidden_dims=(128, 64),
    temperature=1.0,
)
estimator = DifferentiableAncestryEstimator(config, rngs=nnx.Rngs(42))

# Or using factory function
estimator = create_ancestry_estimator(
    n_snps=10000,
    n_populations=5,
)

# Apply ancestry estimation
data = {"genotypes": genotype_matrix}  # (n_samples, n_snps)
result, _, _ = estimator.apply(data, {}, None)

# Get ancestry proportions
ancestry = result["ancestry_proportions"]  # (n_samples, K)
```

### Training Mode

```python
# Enable dropout during training
estimator.train()

for batch in train_dataloader:
    loss = train_step(estimator, batch)

# Disable dropout for inference
estimator.eval()
```

### Accessing Components

```python
# Population allele frequencies
pop_freqs = estimator.population_frequencies[...]  # (K, n_snps)

# Temperature parameter (read from config; the live Param is `_temperature`)
temperature = estimator.config.temperature

# Encoder layers
encoder = estimator.backbone
```

## Input Specifications

| Key | Shape | Description |
|-----|-------|-------------|
| `genotypes` | (n_samples, n_snps) | Genotype matrix with values 0, 1, or 2 |

## Output Specifications

| Key | Shape | Description |
|-----|-------|-------------|
| `genotypes` | (n_samples, n_snps) | Original genotype matrix |
| `ancestry_proportions` | (n_samples, n_populations) | Estimated ancestry proportions (sum to 1) |
| `reconstructed` | (n_samples, n_snps) | Reconstructed genotypes |
| `latent` | (n_samples, hidden_dims[-1]) | Latent representation |
