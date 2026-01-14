# Quick Start

This guide walks you through the basics of using DiffBio for differentiable bioinformatics.

## Your First Alignment

Let's compute a differentiable Smith-Waterman alignment:

```python
import jax.numpy as jnp
from diffbio.operators import SmoothSmithWaterman, SmithWatermanConfig
from diffbio.operators.alignment import create_dna_scoring_matrix

# Create a scoring matrix for DNA sequences
# A=0, C=1, G=2, T=3
scoring_matrix = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)

# Configure the aligner
config = SmithWatermanConfig(
    temperature=1.0,      # Smoothness parameter
    gap_open=-10.0,       # Gap opening penalty
    gap_extend=-1.0       # Gap extension penalty
)

# Create the differentiable aligner
aligner = SmoothSmithWaterman(config, scoring_matrix=scoring_matrix)

# One-hot encode sequences
# Sequence: ACGT -> [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
def one_hot_dna(sequence_indices):
    return jnp.eye(4)[sequence_indices]

seq1 = one_hot_dna(jnp.array([0, 1, 2, 3, 0, 1]))  # ACGTAC
seq2 = one_hot_dna(jnp.array([0, 1, 0, 3, 0, 1]))  # ACATAC

# Perform alignment
result = aligner.align(seq1, seq2)

print(f"Alignment score: {result.score:.4f}")
print(f"Alignment matrix shape: {result.alignment_matrix.shape}")
print(f"Soft alignment shape: {result.soft_alignment.shape}")
```

## Computing Gradients

The key feature of DiffBio is that all operations are differentiable:

```python
import jax

# Define a loss function based on alignment score
def alignment_loss(scoring_matrix, seq1, seq2):
    config = SmithWatermanConfig(temperature=1.0)
    aligner = SmoothSmithWaterman(config, scoring_matrix=scoring_matrix)
    result = aligner.align(seq1, seq2)
    return -result.score  # Negative because we want to maximize

# Compute gradients with respect to the scoring matrix
grad_fn = jax.grad(alignment_loss)
grads = grad_fn(scoring_matrix, seq1, seq2)

print(f"Gradient shape: {grads.shape}")
print(f"Gradient:\n{grads}")
```

## Using the Datarax Interface

DiffBio operators implement the Datarax `OperatorModule` interface for batch processing:

```python
from diffbio.operators import SmoothSmithWaterman, SmithWatermanConfig
from diffbio.operators.alignment import create_dna_scoring_matrix

# Setup
config = SmithWatermanConfig(temperature=1.0)
scoring = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)
aligner = SmoothSmithWaterman(config, scoring_matrix=scoring)

# Prepare data as dictionary (Datarax format)
data = {
    "seq1": seq1,
    "seq2": seq2,
}
state = {}
metadata = None

# Apply operator
result_data, state, metadata = aligner.apply(data, state, metadata)

print(f"Score: {result_data['score']:.4f}")
```

## Quality Filtering

Apply soft quality filtering to sequence data:

```python
from diffbio.operators import DifferentiableQualityFilter, QualityFilterConfig

# Create quality filter
config = QualityFilterConfig(initial_threshold=20.0)  # Phred 20
filter_op = DifferentiableQualityFilter(config)

# Prepare data
data = {
    "sequence": seq1,  # One-hot encoded sequence
    "quality_scores": jnp.array([30.0, 25.0, 10.0, 35.0, 20.0, 28.0]),
}

# Apply filtering
filtered_data, _, _ = filter_op.apply(data, {}, None)

# Low-quality positions are down-weighted
print(f"Original sequence sum: {seq1.sum():.2f}")
print(f"Filtered sequence sum: {filtered_data['sequence'].sum():.2f}")
```

## Pileup Generation

Generate differentiable pileups from aligned reads:

```python
from diffbio.operators import DifferentiablePileup, PileupConfig

# Configure pileup generator
config = PileupConfig(
    use_quality_weights=True,
    reference_length=50
)
pileup_op = DifferentiablePileup(config)

# Simulate some aligned reads
num_reads = 10
read_length = 20
reads = jax.random.uniform(
    jax.random.PRNGKey(0),
    (num_reads, read_length, 4)
)
reads = jax.nn.softmax(reads, axis=-1)  # Normalize to distributions

positions = jax.random.randint(
    jax.random.PRNGKey(1),
    (num_reads,),
    minval=0,
    maxval=30
)

quality = jax.random.uniform(
    jax.random.PRNGKey(2),
    (num_reads, read_length),
    minval=10.0,
    maxval=40.0
)

# Generate pileup
data = {"reads": reads, "positions": positions, "quality": quality}
result, _, _ = pileup_op.apply(data, {}, None)

print(f"Pileup shape: {result['pileup'].shape}")  # (reference_length, 4)
```

## End-to-End Pipeline

Combine operators into a differentiable pipeline:

```python
import jax
from diffbio.operators import (
    DifferentiableQualityFilter, QualityFilterConfig,
    SmoothSmithWaterman, SmithWatermanConfig,
)
from diffbio.operators.alignment import create_dna_scoring_matrix

def pipeline(params, seq1, seq2, quality1, quality2):
    # Step 1: Quality filtering
    filter_config = QualityFilterConfig(initial_threshold=params['threshold'])
    filter_op = DifferentiableQualityFilter(filter_config)

    filtered1, _, _ = filter_op.apply(
        {"sequence": seq1, "quality_scores": quality1}, {}, None
    )
    filtered2, _, _ = filter_op.apply(
        {"sequence": seq2, "quality_scores": quality2}, {}, None
    )

    # Step 2: Alignment
    align_config = SmithWatermanConfig(temperature=params['temperature'])
    aligner = SmoothSmithWaterman(align_config, scoring_matrix=params['scoring'])

    result = aligner.align(
        filtered1['sequence'],
        filtered2['sequence']
    )

    return result.score

# Initialize parameters
params = {
    'threshold': 20.0,
    'temperature': 1.0,
    'scoring': create_dna_scoring_matrix(match=2.0, mismatch=-1.0),
}

# Compute gradients through the entire pipeline
grad_fn = jax.grad(pipeline)
quality1 = jnp.ones(6) * 30.0
quality2 = jnp.ones(6) * 30.0
grads = grad_fn(params, seq1, seq2, quality1, quality2)

print("Gradients computed through entire pipeline!")
print(f"Threshold gradient: {grads['threshold']:.6f}")
print(f"Temperature gradient: {grads['temperature']:.6f}")
```

## Next Steps

- Learn about [Core Concepts](core-concepts.md) behind differentiable bioinformatics
- Explore the [User Guide](../user-guide/concepts/differentiable-bioinformatics.md) for in-depth documentation
- See [Examples](../examples/overview.md) for more use cases
- Read the [API Reference](../api/core/base.md) for complete API details
