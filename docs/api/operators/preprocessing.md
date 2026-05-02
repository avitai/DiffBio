# Preprocessing Operators API

Differentiable preprocessing operators for read quality control, adapter removal, and error correction.

## SoftAdapterRemoval

::: diffbio.operators.preprocessing.adapter_removal.SoftAdapterRemoval
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## AdapterRemovalConfig

::: diffbio.operators.preprocessing.adapter_removal.AdapterRemovalConfig
    options:
      show_root_heading: true
      members: []

## DifferentiableDuplicateWeighting

::: diffbio.operators.preprocessing.duplicate_filter.DifferentiableDuplicateWeighting
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## DuplicateWeightingConfig

::: diffbio.operators.preprocessing.duplicate_filter.DuplicateWeightingConfig
    options:
      show_root_heading: true
      members: []

## SoftErrorCorrection

::: diffbio.operators.preprocessing.error_correction.SoftErrorCorrection
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## ErrorCorrectionConfig

::: diffbio.operators.preprocessing.error_correction.ErrorCorrectionConfig
    options:
      show_root_heading: true
      members: []

## Usage Examples

### Adapter Removal

```python
from flax import nnx
from diffbio.operators.preprocessing import SoftAdapterRemoval, AdapterRemovalConfig

config = AdapterRemovalConfig(
    adapter_sequence="AGATCGGAAGAG",
    temperature=1.0,
    match_threshold=0.8,
)
adapter_removal = SoftAdapterRemoval(config, rngs=nnx.Rngs(42))

data = {"sequence": read, "quality_scores": quality}
result, _, _ = adapter_removal.apply(data, {}, None)
trimmed = result["sequence"]
```

### Duplicate Weighting

```python
from diffbio.operators.preprocessing import (
    DifferentiableDuplicateWeighting,
    DuplicateWeightingConfig,
)

config = DuplicateWeightingConfig(embedding_dim=32)
dup_weighting = DifferentiableDuplicateWeighting(config, rngs=nnx.Rngs(42))

data = {"sequences": reads, "quality_scores": quality}
result, _, _ = dup_weighting.apply(data, {}, None)
weights = result["weights"]
```

### Error Correction

```python
from diffbio.operators.preprocessing import SoftErrorCorrection, ErrorCorrectionConfig

config = ErrorCorrectionConfig(hidden_dim=64, window_size=5)
error_correction = SoftErrorCorrection(config, rngs=nnx.Rngs(42))

data = {"sequence": read, "quality_scores": quality}
result, _, _ = error_correction.apply(data, {}, None)
corrected = result["sequence"]
```
