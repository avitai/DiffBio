# Differentiable Bioinformatics

Differentiable bioinformatics is the application of automatic differentiation to biological sequence analysis, enabling gradient-based optimization of parameters throughout genomic analysis pipelines.

## The Problem with Traditional Pipelines

Traditional bioinformatics pipelines consist of discrete, non-differentiable operations:

```
Reads → Alignment → Pileup → Variant Calling → Variants
         (BWA)      (samtools)  (GATK/DeepVariant)
```

Each step involves hard decisions:

- **Alignment**: "This read maps to position 1234" (discrete choice)
- **Pileup**: "5 reads support A, 3 support G" (integer counts)
- **Variant Calling**: "This is a variant" (binary decision)

These discrete operations **block gradient flow**, preventing end-to-end optimization.

## The DiffBio Approach

DiffBio replaces discrete operations with differentiable approximations:

```
Reads → Soft Alignment → Soft Pileup → Soft Variant Scores → Variants
         (DiffBio)       (DiffBio)      (DiffBio)
```

### Key Techniques

#### 1. Logsumexp Relaxation

Replace `max` with temperature-scaled logsumexp:

$$
\text{soft\_max}(x_1, x_2, \ldots, x_n) = \tau \cdot \log\sum_{i=1}^{n} e^{x_i/\tau}
$$

As $\tau \to 0$, this approaches the hard maximum.

```python
def smooth_max(*args, temperature):
    stacked = jnp.stack(args, axis=-1)
    return temperature * jax.scipy.special.logsumexp(stacked / temperature, axis=-1)
```

#### 2. Softmax for Selection

Replace `argmax` with softmax to get soft selections:

$$
\text{softmax}(x_i) = \frac{e^{x_i/\tau}}{\sum_j e^{x_j/\tau}}
$$

This produces a probability distribution over choices rather than a single selection.

#### 3. Sigmoid for Thresholds

Replace hard thresholds with sigmoid:

$$
\text{soft\_threshold}(x, t) = \sigma(x - t) = \frac{1}{1 + e^{-(x-t)}}
$$

This smoothly transitions from 0 to 1 around the threshold.

#### 4. Segment Sum for Aggregation

Replace discrete counting with weighted accumulation:

```python
# Hard counting (non-differentiable)
counts = jnp.bincount(positions, minlength=reference_length)

# Soft accumulation (differentiable)
weighted_counts = jax.ops.segment_sum(weights, positions, num_segments=reference_length)
```

## Applications

### Learned Alignment Parameters

Traditional approach: Use fixed scoring matrices (BLOSUM62, etc.)

DiffBio approach: Learn optimal scoring parameters for your specific data:

```python
# Scoring matrix is a learnable parameter
aligner.scoring_matrix  # nnx.Param, shape (4, 4)

# Optimize for your objective
grads = jax.grad(loss_fn)(params)
```

### Adaptive Quality Filtering

Traditional approach: Use fixed Phred threshold (e.g., Q20)

DiffBio approach: Learn optimal threshold for your pipeline:

```python
# Threshold adapts during training
filter_op.threshold  # nnx.Param, learned from data
```

### End-to-End Variant Calling

Traditional approach: Each pipeline stage optimized separately

DiffBio approach: Optimize entire pipeline jointly:

```python
def variant_calling_loss(params, reads, true_variants):
    # Gradients flow through entire pipeline
    predicted = pipeline(params, reads)
    return loss(predicted, true_variants)

# Joint optimization of all parameters
grads = jax.grad(variant_calling_loss)(all_params, reads, variants)
```

## Trade-offs

### Accuracy vs Trainability

The temperature parameter controls this trade-off:

| Temperature | Accuracy | Gradient Quality |
|-------------|----------|------------------|
| Low ($\tau \to 0$) | High (near-exact) | Poor (vanishing gradients) |
| Medium | Good | Good |
| High ($\tau \to \infty$) | Low (uniform) | Excellent (smooth gradients) |

**Recommendation**: Start with high temperature, anneal during training.

### Computational Cost

Differentiable operations can be more expensive than discrete equivalents:

- **Dynamic programming**: O(n·m) for both, but differentiable version maintains full matrix
- **Memory**: Need to store intermediate values for backpropagation
- **JIT compilation**: Initial compilation overhead, but fast subsequent runs

DiffBio mitigates this with:

- JAX's XLA compilation for efficient execution
- GPU acceleration for parallel computation
- Efficient scan-based implementations

## Biological Validity

A key concern: Do differentiable approximations produce biologically valid results?

### Validation Approaches

1. **Temperature Annealing**: As $\tau \to 0$, results approach traditional algorithms
2. **Post-hoc Discretization**: Apply hard decisions after soft training
3. **Regularization**: Encourage sparse, interpretable solutions

### Empirical Evidence

Research shows differentiable sequence alignment (SMURF, etc.) achieves:

- Comparable accuracy to traditional methods
- Better generalization to novel sequences
- Ability to learn domain-specific scoring schemes

## References

1. Petti et al. "End-to-end learning of multiple sequence alignments with differentiable Smith-Waterman." *Bioinformatics* 39(1), 2023.

2. Mensch & Blondel. "Differentiable Dynamic Programming for Structured Prediction and Attention." *ICML*, 2018.

3. Morton et al. "Protein Sequence Alignment with Reinforcement Learning." *NeurIPS*, 2020.
