# Pileup Generation

A pileup aggregates aligned sequencing reads at each position of a reference
genome, producing a per-position summary that variant callers use to identify
mutations. DiffBio provides a differentiable pileup operator that replaces
integer counts with continuous, quality-weighted distributions.

---

## What a Pileup Represents

After alignment, each read covers a contiguous region of the reference. A
pileup stacks all reads at each position to count base observations:

```
Reference:  A  C  G  T  A  C  G  T
Read 1:     A  C  G  T  A  .  .  .
Read 2:     .  C  G  T  A  C  G  .
Read 3:     .  .  G  A  A  C  G  T
            ────────────────────────
Position:   1  2  3  4  5  6  7  8
```

At position 4, the reference is T but one read shows A — a potential variant.
The pileup at position 4 is {A: 1, T: 2} with coverage 3.

### Why Pileup Matters

Pileup is the bridge between raw alignment data and variant calling. The
quality of the pileup directly determines variant calling accuracy:

- **Coverage** — how many reads support each position
- **Base composition** — which nucleotides are observed
- **Quality weighting** — how reliable each observation is

---

## Hard vs Soft Pileup

### Hard Pileup (Traditional)

Traditional tools (samtools) produce integer counts at each position:

| Position | A | C | G | T | Coverage |
|---|---|---|---|---|---|
| 3 | 0 | 0 | 3 | 0 | 3 |
| 4 | 1 | 0 | 0 | 2 | 3 |

This is non-differentiable: a small change in read quality or alignment
probability does not change the integer counts.

### Soft Pileup (DiffBio)

`DifferentiablePileup` produces continuous, quality-weighted distributions:

| Position | A | C | G | T | Soft Coverage |
|---|---|---|---|---|---|
| 3 | 0.001 | 0.002 | 2.987 | 0.010 | 3.0 |
| 4 | 0.994 | 0.003 | 0.001 | 1.992 | 2.99 |

Each read's contribution is weighted by its quality score. Small changes in
quality produce small changes in the pileup — gradients flow.

---

## Quality Weighting

Phred quality scores express the probability that a base call is wrong:

$$
Q = -10 \cdot \log_{10}(P_{\text{error}})
$$

DiffBio converts this to a reliability weight:

$$
w = 1 - 10^{-Q/10}
$$

| Phred Score | Error Rate | Weight |
|---|---|---|
| Q10 | 10% | 0.90 |
| Q20 | 1% | 0.99 |
| Q30 | 0.1% | 0.999 |
| Q40 | 0.01% | 0.9999 |

High-quality bases contribute more to the pileup. This weighting is
continuous and differentiable — gradients flow from the pileup output through
the quality weights back to any upstream parameters that affect read quality.

---

## The Aggregation Technique

DiffBio uses `jax.ops.segment_sum` for efficient, differentiable aggregation:

$$
\text{pileup}[i, j] = \sum_{r : \text{read } r \text{ covers } i} w_r \cdot s_{r,j}
$$

Where:

- $i$ is the reference position
- $j$ is the nucleotide index (A=0, C=1, G=2, T=3)
- $w_r$ is the quality-derived weight for read $r$
- $s_{r,j}$ is the one-hot encoded base call

This is a single scatter-add operation on GPU — fast and differentiable.

---

## Pileup in Context

The pileup sits between alignment and variant calling in a genomics pipeline:

```
Reads → Quality Filter → Alignment → Pileup → Variant Classifier
                                       ↑
                               Quality weights
                               flow backward
```

Because the pileup is differentiable, gradients from the variant classifier
propagate backward through the pileup into the quality filter and alignment
parameters. This is how DiffBio enables end-to-end training of variant
calling pipelines.

---

## Practical Considerations

**Coverage depth**: High coverage (30x+) gives stable pileup estimates. Low
coverage makes the soft distribution noisier but still differentiable.

**Memory**: Storing the full pileup requires $O(\text{reference\_length} \times 4)$.
For large references, process in windows.

**Read length**: All reads are padded to maximum length. Positions beyond the
read's actual length contribute zero weight automatically.

---

## Further Reading

- [Pileup Operator](../operators/pileup.md) — configuration, code examples
- [Pileup API](../../api/operators/pileup.md) — full API reference
- [Variant Calling](variant-calling.md) — how pileup feeds into variant calling
- [Pileup Example](../../examples/basic/pileup-generation.md) — runnable example

### References

1. Li et al. "The Sequence Alignment/Map format and SAMtools."
   *Bioinformatics* 25(16), 2009.
