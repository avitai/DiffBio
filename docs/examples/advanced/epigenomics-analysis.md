# Epigenomics Analysis Example

This example demonstrates differentiable epigenomics analysis using DiffBio's peak calling and chromatin state annotation operators.

## Setup

```python
import jax
import jax.numpy as jnp
from flax import nnx
from diffbio.operators.epigenomics import (
    DifferentiablePeakCaller,
    PeakCallerConfig,
    ChromatinStateAnnotator,
    ChromatinStateConfig,
)
```

## Peak Calling

### Generate Synthetic ChIP-seq Signal

```python
def generate_chipseq_signal(length=10000, n_peaks=50, seed=42):
    key = jax.random.key(seed)
    keys = jax.random.split(key, 4)

    background = jax.random.exponential(keys[0], (length,)) * 0.5
    peak_positions = jax.random.randint(keys[1], (n_peaks,), 100, length - 100)

    signal = background
    peak_labels = jnp.zeros(length)

    # Add Gaussian peaks
    for i in range(n_peaks):
        pos = peak_positions[i]
        x = jnp.arange(length)
        peak = 10.0 * jnp.exp(-0.5 * ((x - pos) / 25) ** 2)
        signal = signal + peak

    return signal, peak_labels

signal, peak_labels = generate_chipseq_signal()
```

### Create and Train Peak Caller

```python
config = PeakCallerConfig(num_filters=32, kernel_sizes=[3, 5, 7], threshold=0.5)
rngs = nnx.Rngs(42)
peak_caller = DifferentiablePeakCaller(config, rngs=rngs)

result, _, _ = peak_caller.apply({"coverage": signal}, {}, None)
peak_probs = result["peak_probabilities"]
print(f"Predicted peaks: {(peak_probs > 0.5).sum()}")
```

## Chromatin State Annotation

```python
# Generate histone marks
histone_marks = jax.random.bernoulli(jax.random.key(0), 0.3, (5000, 6)).astype(jnp.float32)

config = ChromatinStateConfig(num_states=5, num_marks=6)
annotator = ChromatinStateAnnotator(config, rngs=nnx.Rngs(42))

result, _, _ = annotator.apply({"histone_marks": histone_marks}, {}, None)
state_probs = result["state_posteriors"]
print(f"State probabilities shape: {state_probs.shape}")
```

## Next Steps

- See [Multi-omics Integration](multiomics-integration.md) for cross-assay analysis
- Explore [Epigenomics Operators](../../user-guide/operators/epigenomics.md) for details
