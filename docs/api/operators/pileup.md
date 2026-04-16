# Pileup API

Differentiable pileup generation for variant calling.

## DifferentiablePileup

::: diffbio.operators.variant.pileup.DifferentiablePileup
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - compute_pileup
        - apply

## PileupConfig

::: diffbio.operators.variant.pileup.PileupConfig
    options:
      show_root_heading: true
      members: []

## Usage Examples

### Basic Pileup Generation

```python
import jax
import jax.numpy as jnp
from diffbio.operators.variant import DifferentiablePileup, PileupConfig

# Configure
config = PileupConfig(
    reference_length=100,
    use_quality_weights=True,
)
pileup_op = DifferentiablePileup(config)

# Prepare data
num_reads = 20
read_length = 30

reads = jax.nn.softmax(
    jax.random.uniform(jax.random.PRNGKey(0), (num_reads, read_length, 4)),
    axis=-1
)
positions = jax.random.randint(jax.random.PRNGKey(1), (num_reads,), 0, 70)
quality = jax.random.uniform(jax.random.PRNGKey(2), (num_reads, read_length), 10, 40)

# Generate pileup
result = pileup_op.compute_pileup(reads, positions, quality, 100)
pileup = result["pileup"]
print(f"Pileup shape: {pileup.shape}")  # (100, 4)
```

### Datarax Interface

```python
data = {
    "reads": reads,
    "positions": positions,
    "quality": quality,
}

result_data, state, metadata = pileup_op.apply(data, {}, None)
pileup = result_data["pileup"]
```

### Gradient Computation

```python
import jax

def pileup_loss(pileup_op, reads, positions, quality, target_pileup):
    data = {"reads": reads, "positions": positions, "quality": quality}
    result, _, _ = pileup_op.apply(data, {}, None)
    return jnp.mean((result["pileup"] - target_pileup) ** 2)

grads = jax.grad(pileup_loss)(pileup_op, reads, positions, quality, target)
```

## Input Specifications

### reads

| Property | Value |
|----------|-------|
| Shape | `(num_reads, read_length, 4)` |
| Type | `Float[Array, ...]` |
| Description | One-hot encoded read sequences |

### positions

| Property | Value |
|----------|-------|
| Shape | `(num_reads,)` |
| Type | `Int[Array, ...]` |
| Description | Starting position of each read |

### quality

| Property | Value |
|----------|-------|
| Shape | `(num_reads, read_length)` |
| Type | `Float[Array, ...]` |
| Description | Phred quality scores |

## Output Specifications

### pileup

| Property | Value |
|----------|-------|
| Shape | `(reference_length, 4)` |
| Type | `Float[Array, ...]` |
| Description | Nucleotide distribution at each position |
