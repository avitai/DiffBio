# Differentiable Bioinformatics

Differentiable bioinformatics applies automatic differentiation to biological
sequence analysis, enabling gradient-based optimization of parameters across
entire genomic and cellular analysis pipelines.

---

## The Problem with Traditional Pipelines

Traditional bioinformatics pipelines chain independently-optimized tools:

```
Reads → BWA (align) → samtools (pileup) → GATK (call) → Variants
```

Each tool makes hard decisions — "this read maps to position 1234", "5 reads
support A at this locus", "this position is a variant." These discrete
operations block gradient flow, so parameters at each stage cannot learn from
errors at later stages.

The same pattern appears in single-cell analysis:

```
Counts → scanpy (normalize) → PCA → Leiden (cluster) → Labels
```

PCA is not learned for clustering. Clustering thresholds are not informed by
downstream biological conclusions. Each step optimizes its own objective in
isolation.

---

## How DiffBio Makes It Differentiable

DiffBio replaces each discrete operation with a temperature-controlled smooth
approximation:

| Discrete Operation | Smooth Approximation | Gradient Behavior |
|---|---|---|
| `max(a, b)` | $\tau \cdot \log(e^{a/\tau} + e^{b/\tau})$ | Flows to both branches, weighted by magnitude |
| `argmax(x)` | $\text{softmax}(x / \tau)$ | Distributes gradient across all candidates |
| `threshold(x > t)` | $\sigma((x - t) / \tau)$ | Smooth gradient near the threshold |
| Integer counting | Weighted accumulation via `segment_sum` | Continuous gradient through weights |
| Hard clustering | Soft assignment probabilities | Gradient flows to cluster parameters |

The temperature $\tau$ controls how closely the smooth version matches the
discrete original:

- **$\tau \to 0$**: Recovers exact discrete behavior (but gradients vanish)
- **$\tau \to \infty$**: Fully smooth (strong gradients, but output is blurred)
- **$\tau \approx 1$**: Practical sweet spot for training

---

## What This Enables

### Learned Parameters

Instead of fixed scoring matrices (BLOSUM62), gap penalties (hand-tuned), or
quality thresholds (Phred 20 by convention), DiffBio makes these parameters
learnable. Gradients from a downstream loss update upstream operator parameters
through the chain.

### Joint Optimization

A loss defined at the end of a pipeline produces gradients for every operator
in the chain. Alignment parameters, imputation diffusion times, clustering
temperatures, and annotation heads all update together to minimize one
objective.

### End-to-End Training

Combine biological priors (operator structure) with data-driven adaptation
(gradient descent). The operator architecture encodes domain knowledge — what
a pileup is, how Smith-Waterman works — while the parameters adapt to the
specific dataset.

---

## Trade-Offs

### Accuracy vs Trainability

Low temperature gives accurate discrete approximations but poor gradients.
High temperature gives smooth gradients but blurred outputs. The standard
practice is **temperature annealing**: start warm and cool over training.

### Computational Cost

Differentiable operators maintain intermediate values for backpropagation,
increasing memory usage relative to their discrete counterparts. DiffBio
mitigates this through:

- JAX's XLA compilation for fused, efficient operations
- GPU acceleration for parallel computation
- Scan-based DP implementations that avoid materializing full matrices

### Biological Validity

A core concern: do smooth approximations produce biologically meaningful
results? Three validation strategies:

1. **Temperature annealing** — as $\tau \to 0$, outputs converge to
   traditional algorithms
2. **Post-hoc discretization** — train with soft outputs, evaluate with hard
   decisions
3. **Calibrax metrics** — measure biological quality (ARI, silhouette,
   batch mixing) directly on soft outputs

---

## Where It Applies in DiffBio

| Domain | Traditional Tools | DiffBio Operators | What Becomes Learnable |
|---|---|---|---|
| Alignment | BWA, Bowtie | SmoothSmithWaterman, ProfileHMM | Scoring matrix, gap penalties |
| Pileup | samtools | DifferentiablePileup | Quality weighting |
| Variant Calling | GATK, DeepVariant | VariantClassifier, CNVSegmentation | Classification boundaries |
| Single-Cell QC | CellBender, Scrublet | AmbientRemoval, DoubletScorer | Contamination model, doublet threshold |
| Normalization | scVI | VAENormalizer | Encoder/decoder, dispersion |
| Clustering | Leiden, k-means | SoftKMeansClustering | Centroids, assignments |
| Batch Correction | Harmony, scANVI | Harmony, MMD, WGAN | Correction model |
| Trajectory | Palantir, Monocle | Pseudotime, FateProbability | Diffusion components |
| Spatial | STAGATE, PASTE | SpatialDomain, PASTEAlignment | Graph attention, transport plan |

Every operator in the table follows the same `apply()` contract and supports
`jax.grad` and `jax.jit`. See [Operators](../operators/overview.md) for usage.

---

## Further Reading

- [Core Concepts](../../getting-started/core-concepts.md) — operator contract,
  data representations, ecosystem overview
- [Sequence Alignment](sequence-alignment.md) — smooth Smith-Waterman in depth
- [Single-Cell Analysis](single-cell-analysis.md) — dropout, batch effects,
  trajectory inference
- [Variant Calling](variant-calling.md) — pileup-to-genotype pipeline

### References

1. Petti et al. "End-to-end learning of multiple sequence alignments with
   differentiable Smith-Waterman." *Bioinformatics* 39(1), 2023.
2. Mensch & Blondel. "Differentiable Dynamic Programming for Structured
   Prediction and Attention." *ICML*, 2018.
