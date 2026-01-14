# Core Concepts

This guide covers the foundational concepts you need to understand DiffBio.

## Differentiable Bioinformatics

Traditional bioinformatics algorithms use discrete operations like `max`, `argmax`, and hard thresholds that block gradient flow. DiffBio makes these operations differentiable by using smooth approximations:

| Discrete Operation | Differentiable Approximation |
|-------------------|------------------------------|
| `max(a, b)` | `logsumexp(a, b) / temperature` |
| `argmax(x)` | `softmax(x / temperature)` |
| `threshold(x > t)` | `sigmoid(x - t)` |
| Hard counting | Soft weighted accumulation |

The **temperature** parameter controls the smoothness:

- **Low temperature** ($\tau \to 0$): Approaches the hard/discrete behavior
- **High temperature** ($\tau \to \infty$): More uniform/smooth distribution

## The Datarax Operator Pattern

DiffBio operators inherit from Datarax's `OperatorModule`, providing a consistent interface for composable data processing:

```python
class OperatorModule:
    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict | None,
        random_params: Any = None,
        stats: dict | None = None,
    ) -> tuple[PyTree, PyTree, dict | None]:
        """Transform data through the operator."""
        ...
```

### Key Concepts

**Data**: A PyTree (usually a dictionary) containing input tensors.

**State**: Per-element state that persists across operator applications.

**Metadata**: Optional metadata about the data element.

### Operator Configuration

Each operator has a corresponding configuration dataclass:

```python
from dataclasses import dataclass
from datarax.core.config import OperatorConfig

@dataclass
class SmithWatermanConfig(OperatorConfig):
    temperature: float = 1.0
    gap_open: float = -10.0
    gap_extend: float = -1.0
```

## Sequence Representation

DiffBio uses **one-hot encoding** for sequences to enable gradient flow:

```python
import jax.numpy as jnp

# DNA alphabet: A=0, C=1, G=2, T=3
# Sequence "ACGT" as indices
seq_indices = jnp.array([0, 1, 2, 3])

# One-hot encode
seq_onehot = jnp.eye(4)[seq_indices]
# Result: [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
```

One-hot encoding allows gradients to flow through sequence-dependent operations like scoring matrices.

## Learnable Parameters

DiffBio operators use Flax's `nnx.Param` to mark learnable parameters:

```python
from flax import nnx

class SmoothSmithWaterman(OperatorModule):
    def __init__(self, config, scoring_matrix, ...):
        # Learnable parameters
        self.temperature = nnx.Param(jnp.array(config.temperature))
        self.scoring_matrix = nnx.Param(scoring_matrix)
        self.gap_open = nnx.Param(jnp.array(config.gap_open))
```

These parameters can be optimized using gradient descent:

```python
import optax

# Get all learnable parameters
params = nnx.state(aligner, nnx.Param)

# Create optimizer
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

# Update step
def update(params, grads, opt_state):
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state
```

## Gradient Flow

The goal of DiffBio is to enable gradient flow through entire pipelines:

```
Input Sequences
      │
      ▼
┌─────────────────┐
│ Quality Filter  │ ← ∂L/∂threshold
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Smith-Waterman  │ ← ∂L/∂scoring_matrix, ∂L/∂gap_penalties
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Pileup          │ ← ∂L/∂temperature
└────────┬────────┘
         │
         ▼
     Loss Function
```

Gradients flow backward through all operators, allowing joint optimization of all parameters.

## Temperature as a Hyperparameter

The temperature parameter ($\tau$) appears throughout DiffBio and controls the trade-off between:

- **Accuracy** (low $\tau$): Closer to true discrete algorithm behavior
- **Trainability** (high $\tau$): Smoother gradients, easier optimization

### Temperature Scheduling

A common practice is to start with high temperature and anneal:

```python
def temperature_schedule(step, initial=10.0, final=0.1, decay_steps=10000):
    """Exponential temperature decay."""
    decay_rate = (final / initial) ** (1.0 / decay_steps)
    return initial * (decay_rate ** step)
```

This allows the model to first explore broadly, then focus on discrete solutions.

## JAX Transformations

DiffBio operators are designed to work with JAX transformations:

### JIT Compilation

```python
import jax

@jax.jit
def align_batch(aligner, seq1, seq2):
    return aligner.align(seq1, seq2)

# First call compiles, subsequent calls are fast
result = align_batch(aligner, seq1, seq2)
```

### Vectorization (vmap)

```python
# Align many sequence pairs in parallel
batch_align = jax.vmap(
    lambda s1, s2: aligner.align(s1, s2),
    in_axes=(0, 0)
)

results = batch_align(batch_seq1, batch_seq2)
```

### Parallelization (pmap)

```python
# Distribute across multiple devices
parallel_align = jax.pmap(
    lambda s1, s2: aligner.align(s1, s2),
    in_axes=(0, 0)
)
```

## Functional vs Object-Oriented

DiffBio uses Flax NNX which supports both styles:

### Object-Oriented (Recommended for Simple Cases)

```python
aligner = SmoothSmithWaterman(config, scoring_matrix)
result = aligner.align(seq1, seq2)
```

### Functional (For JAX Transformations)

```python
# Split into state and functions
graphdef, state = nnx.split(aligner)

def align_fn(state, seq1, seq2):
    model = nnx.merge(graphdef, state)
    return model.align(seq1, seq2)

# Now safe for jax.jit, jax.grad, etc.
result = jax.jit(align_fn)(state, seq1, seq2)
```

## Next Steps

- See the [User Guide](../user-guide/concepts/differentiable-bioinformatics.md) for detailed algorithm explanations
- Explore [Operators](../user-guide/operators/overview.md) to learn about specific operators
- Read about [Training](../user-guide/training/overview.md) for optimization techniques
