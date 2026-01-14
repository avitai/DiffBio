# Epigenomics Operators API

Differentiable operators for epigenomic analysis including peak calling and chromatin state annotation.

## DifferentiablePeakCaller

::: diffbio.operators.epigenomics.peak_calling.DifferentiablePeakCaller
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## PeakCallerConfig

::: diffbio.operators.epigenomics.peak_calling.PeakCallerConfig
    options:
      show_root_heading: true
      members: []

## ChromatinStateAnnotator

::: diffbio.operators.epigenomics.chromatin_state.ChromatinStateAnnotator
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## ChromatinStateConfig

::: diffbio.operators.epigenomics.chromatin_state.ChromatinStateConfig
    options:
      show_root_heading: true
      members: []

## Usage Examples

### Peak Calling

```python
from flax import nnx
from diffbio.operators.epigenomics import DifferentiablePeakCaller, PeakCallerConfig

config = PeakCallerConfig(num_filters=32, kernel_sizes=[3, 5, 7])
peak_caller = DifferentiablePeakCaller(config, rngs=nnx.Rngs(42))

data = {"signal": signal_track}  # (length,)
result, _, _ = peak_caller.apply(data, {}, None)
peaks = result["peak_scores"]
```

### Chromatin State Annotation

```python
from diffbio.operators.epigenomics import ChromatinStateAnnotator, ChromatinStateConfig

config = ChromatinStateConfig(num_states=15, num_marks=6)
annotator = ChromatinStateAnnotator(config, rngs=nnx.Rngs(42))

data = {"histone_marks": marks}  # (length, num_marks)
result, _, _ = annotator.apply(data, {}, None)
states = result["state_probabilities"]
```
