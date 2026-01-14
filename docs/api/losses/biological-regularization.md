# Biological Regularization API

Regularization losses based on biological constraints and priors.

## BiologicalPlausibilityLoss

::: diffbio.losses.biological_regularization.BiologicalPlausibilityLoss
    options:
      show_root_heading: true
      show_source: false

## BiologicalRegularizationConfig

::: diffbio.losses.biological_regularization.BiologicalRegularizationConfig
    options:
      show_root_heading: true
      members: []

## GCContentRegularization

::: diffbio.losses.biological_regularization.GCContentRegularization
    options:
      show_root_heading: true
      show_source: false

## GapPatternRegularization

::: diffbio.losses.biological_regularization.GapPatternRegularization
    options:
      show_root_heading: true
      show_source: false

## SequenceComplexityLoss

::: diffbio.losses.biological_regularization.SequenceComplexityLoss
    options:
      show_root_heading: true
      show_source: false

## Usage Example

```python
from diffbio.losses import (
    BiologicalPlausibilityLoss,
    GCContentRegularization,
    SequenceComplexityLoss,
)

# GC content regularization
gc_reg = GCContentRegularization(target_gc=0.5, weight=1.0)
gc_loss = gc_reg(sequences=predicted_sequences)

# Sequence complexity loss
complexity = SequenceComplexityLoss()
comp_loss = complexity(sequences=predicted_sequences)
```
