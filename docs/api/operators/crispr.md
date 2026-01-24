# CRISPR Operators API

Differentiable operators for CRISPR guide RNA design and on-target efficiency prediction.

## DifferentiableCRISPRScorer

::: diffbio.operators.crispr.guide_scoring.DifferentiableCRISPRScorer
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply
        - extract_features
        - predict_efficiency

## CRISPRScorerConfig

::: diffbio.operators.crispr.guide_scoring.CRISPRScorerConfig
    options:
      show_root_heading: true
      members: []

## create_crispr_scorer

::: diffbio.operators.crispr.guide_scoring.create_crispr_scorer
    options:
      show_root_heading: true
      show_source: false

## Usage Examples

### Basic Guide Scoring

```python
from flax import nnx
import jax
from diffbio.operators.crispr import (
    DifferentiableCRISPRScorer,
    CRISPRScorerConfig,
    create_crispr_scorer,
)

# Using config
config = CRISPRScorerConfig(
    guide_length=23,
    hidden_channels=(64, 128, 256),
    fc_dims=(256, 128),
)
scorer = DifferentiableCRISPRScorer(config, rngs=nnx.Rngs(42))

# Or using factory function
scorer = create_crispr_scorer(guide_length=23)

# Score guides
guide_indices = jax.random.randint(jax.random.PRNGKey(0), (100, 23), 0, 4)
guides = jax.nn.one_hot(guide_indices, 4)

result, _, _ = scorer.apply({"guides": guides}, {}, None)
scores = result["efficiency_scores"]  # (100,) in [0, 1]
```

### Training Mode

```python
# Enable dropout during training
scorer.train()

for batch in train_dataloader:
    loss = train_step(scorer, batch)

# Disable dropout for inference
scorer.eval()
```

### Accessing Components

```python
# Convolutional layers
conv_layers = scorer.conv_layers

# Fully connected layers
fc_layers = scorer.fc_layers

# Output head
output_head = scorer.output_head
```

## Input Specifications

| Key | Shape | Description |
|-----|-------|-------------|
| `guides` | (n_guides, guide_length, 4) | One-hot encoded guide sequences |

## Output Specifications

| Key | Shape | Description |
|-----|-------|-------------|
| `guides` | (n_guides, guide_length, 4) | Original guide sequences |
| `efficiency_scores` | (n_guides,) | Predicted efficiency scores in [0, 1] |
| `features` | (n_guides, feature_dim) | Extracted CNN features |
