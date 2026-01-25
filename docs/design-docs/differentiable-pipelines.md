# Design: End-to-End Differentiable Pipelines

This document describes the design and architecture of DiffBio's differentiable bioinformatics pipelines.

## Overview

DiffBio provides differentiable implementations of bioinformatics algorithms, enabling gradient-based optimization of entire analysis pipelines. This document covers the design principles, architecture decisions, and implementation strategies.

## Goals

1. **Differentiability**: All operations must support gradient computation
2. **Composability**: Operators should be easily composed into pipelines
3. **Performance**: Leverage JAX for GPU acceleration and JIT compilation
4. **Compatibility**: Integrate with the Datarax framework
5. **Extensibility**: Easy to add new operators and pipelines

## Non-Goals

1. Drop-in replacement for production bioinformatics tools
2. Processing raw sequencing files (FASTQ, BAM) directly
3. Matching exact output of traditional algorithms

## Architecture

### Operator Hierarchy

```text
datarax.core.operator.OperatorModule
    │
    ├── diffbio.operators.DifferentiableQualityFilter
    │
    ├── diffbio.operators.alignment.SmoothSmithWaterman
    │
    ├── diffbio.operators.variant.DifferentiablePileup
    │
    ├── diffbio.operators.variant.VariantClassifier
    │
    └── diffbio.pipelines.VariantCallingPipeline
```

### Data Flow

```text
Input Data (dict)
    │
    ▼
┌───────────────────┐
│ OperatorModule    │
│ .apply(data, ...) │
└────────┬──────────┘
         │
         ▼
Output Data (dict)
```

### Configuration Pattern

Each operator has a corresponding configuration dataclass:

```python
@dataclass
class MyOperatorConfig(OperatorConfig):
    my_param: float = 1.0
    stochastic: bool = False
    stream_name: str | None = None
```

## Differentiability Techniques

### 1. Logsumexp Relaxation

Replace discrete `max` with smooth `logsumexp`:

$$\text{softmax}(x_1, \ldots, x_n) = \tau \log \sum_i e^{x_i/\tau}$$

**Application**: Smith-Waterman alignment recurrence

### 2. Sigmoid Thresholds

Replace hard thresholds with sigmoid:

$$\text{soft\_threshold}(x, t) = \sigma(x - t)$$

**Application**: Quality filtering

### 3. Segment Sum Aggregation

Replace discrete counting with weighted accumulation:

```python
# Differentiable aggregation
pileup = jax.ops.segment_sum(weighted_reads, positions, num_segments=ref_len)
```

**Application**: Pileup generation

### 4. Temperature Control

All smooth approximations use temperature parameter $\tau$:

- $\tau \to 0$: Approaches discrete behavior
- $\tau \to \infty$: Uniform/smooth behavior

## Key Design Decisions

### Decision 1: Inherit from Datarax OperatorModule

**Context**: Need consistent interface for composable operators

**Decision**: All DiffBio operators inherit from `datarax.core.operator.OperatorModule`

**Rationale**:

- Consistent `apply()` interface
- Batch processing via `apply_batch()`
- Integration with Datarax composition utilities

**Trade-offs**:

- Dependency on Datarax
- Must follow Datarax conventions

### Decision 2: Use Flax NNX for Parameters

**Context**: Need mutable state management for learnable parameters

**Decision**: Use `flax.nnx.Param` for all learnable parameters

**Rationale**:

- Clean separation of parameters and logic
- Integration with JAX transformations
- Familiar patterns for deep learning users

**Trade-offs**:

- Requires Flax dependency
- NNX is newer, less documentation than Linen

### Decision 3: One-Hot Sequence Encoding

**Context**: Need differentiable sequence representation

**Decision**: Sequences are one-hot encoded (length, alphabet_size)

**Rationale**:

- Enables gradient flow through sequence operations
- Works naturally with scoring matrices
- Supports soft/probabilistic sequences

**Trade-offs**:

- Higher memory than integer encoding
- Requires conversion from string sequences

### Decision 4: Dictionary-Based Data Flow

**Context**: Operators need flexible input/output

**Decision**: Data is passed as dictionaries with string keys

**Rationale**:

- Flexible, extensible format
- Easy to add new fields
- Works well with Datarax

**Trade-offs**:

- Runtime key lookup overhead
- Less type safety than named tuples

## Implementation Details

### Smith-Waterman Implementation

Uses scan-based dynamic programming:

```python
def fill_row(i, H):
    def cell_update(H_prev_col, j_idx):
        # Compute candidates
        diag = H[i, j] + score[i, j]
        up = H[i, j+1] + gap_penalty
        left = H_prev_col + gap_penalty

        # Smooth max
        h_new = temp * logsumexp([0, diag, up, left] / temp)
        return h_new, h_new

    _, new_row = jax.lax.scan(cell_update, 0.0, jnp.arange(len2))
    return H.at[i+1, 1:].set(new_row)

H = jax.lax.fori_loop(0, len1, fill_row, H)
```

### Pileup Implementation

Uses segment_sum for efficient aggregation:

```python
# Compute absolute positions for all bases
absolute_positions = positions[:, None] + jnp.arange(read_length)

# Flatten and aggregate
flat_positions = absolute_positions.reshape(-1)
flat_reads = weighted_reads.reshape(-1, 4)

pileup = jax.ops.segment_sum(flat_reads, flat_positions, num_segments=ref_len)
```

## Performance Considerations

### JIT Compilation

All operators are designed to work with `jax.jit`:

```python
@jax.jit
def fast_pipeline(data):
    return pipeline.apply(data, {}, None)
```

### Vectorization

Operators support `jax.vmap` for batch processing:

```python
batch_align = jax.vmap(lambda s1, s2: aligner.align(s1, s2))
```

### Memory Management

For large datasets:

- Process in chunks
- Use gradient checkpointing if needed
- Consider float16 for memory-constrained scenarios

## Future Work

1. **Additional Operators**:
   - Needleman-Wunsch global alignment
   - Multiple sequence alignment
   - De Bruijn graph assembly

2. **Performance Optimizations**:
   - Custom CUDA kernels for alignment
   - Sparse operations for large references
   - Gradient checkpointing

3. **Integration**:
   - Direct BAM/VCF parsing
   - Integration with genomics databases
   - Cloud deployment patterns

## References

1. Petti et al. (2023). "End-to-end learning of multiple sequence alignments with differentiable Smith-Waterman."

2. Mensch & Blondel (2018). "Differentiable Dynamic Programming for Structured Prediction and Attention."

3. Datarax documentation: <https://github.com/mahdi-shafiei/datarax>
