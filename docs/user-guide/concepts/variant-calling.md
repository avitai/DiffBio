# Variant Calling

Variant calling identifies positions where a sample genome differs from a
reference genome. DiffBio provides differentiable components for each stage
of a variant calling pipeline, enabling joint optimization from raw reads to
genotype predictions.

---

## What Is a Variant?

A genetic variant is a position where the sequenced sample differs from the
reference:

| Type | Change | Example | Size |
|---|---|---|---|
| **SNP** | Single base substitution | A → G | 1 bp |
| **Insertion** | Bases added | AT → AGT | 1+ bp |
| **Deletion** | Bases removed | AGT → AT | 1+ bp |
| **CNV** | Copy number change | 2 copies → 4 copies | 1+ kb |

SNPs are the most common. CNVs are the hardest to detect because they span
large regions and require different statistical approaches.

---

## The Traditional Pipeline

```
Reads → Alignment → Pileup → Statistical Model → Variant Calls
         (BWA)     (samtools)  (GATK/DeepVariant)
```

Each stage is optimized independently:

- BWA minimizes alignment edit distance
- samtools counts exact base observations
- GATK applies a Bayesian genotyper with fixed priors

Errors cascade forward without correction. A misaligned read produces a wrong
pileup count, which the variant caller cannot distinguish from a real variant.

---

## The Differentiable Pipeline

DiffBio replaces each hard decision with a soft, differentiable operation:

```
Reads → Soft Filter → Soft Alignment → Soft Pileup → Neural Classifier
         (sigmoid)     (logsumexp)       (segment_sum)  (MLP)
```

The key difference: gradients from the classifier's loss flow backward through
the entire chain. If a quality threshold is too strict and filters out reads
supporting a real variant, the gradient signal pushes the threshold lower.

### DiffBio Operators in the Pipeline

| Stage | Operator | What It Learns |
|---|---|---|
| Quality filtering | `DifferentiableQualityFilter` | Optimal quality threshold |
| Alignment | `SmoothSmithWaterman` | Scoring matrix, gap penalties |
| Pileup | `DifferentiablePileup` | Quality-to-weight mapping |
| SNP calling | `VariantClassifier` | Classification boundaries |
| CNV detection | `EnhancedCNVSegmentation` | Segmentation breakpoints, state transitions |

---

## Genotype Representation

For diploid organisms, each position has a genotype:

| Genotype | Meaning | Label |
|---|---|---|
| 0/0 | Homozygous reference | Both alleles match reference |
| 0/1 | Heterozygous | One reference, one alternate |
| 1/1 | Homozygous alternate | Both alleles differ |

The variant classifier outputs a probability distribution over these three
classes at each position. During training, cross-entropy loss against known
genotypes drives the learning.

---

## Why End-to-End Helps

### Error Correction Across Stages

In a traditional pipeline, a borderline-quality read is either kept or
discarded based on a fixed threshold. If discarded, downstream stages never
see it. In DiffBio's soft pipeline, the same read contributes with a reduced
weight — and if that contribution improves the final genotype prediction, the
gradient signal encourages keeping it.

### Class Imbalance

Most genomic positions are reference (no variant). Traditional callers handle
this with hand-tuned priors. DiffBio can learn the prior from data through
the loss function — for example, using focal loss to down-weight easy
reference predictions and focus on variant positions.

### Copy Number Variants

CNV detection requires integrating signals across large regions — depth
changes, B-allele frequency shifts, breakpoint patterns. DiffBio's
`EnhancedCNVSegmentation` uses a differentiable HMM with pyramidal smoothing
to detect these multi-scale patterns with learnable transition probabilities.

---

## Evaluation

Variant calling quality is measured by:

| Metric | Definition | Ideal |
|---|---|---|
| **Precision** | TP / (TP + FP) | 1.0 |
| **Recall** | TP / (TP + FN) | 1.0 |
| **F1** | Harmonic mean of precision and recall | 1.0 |

Evaluation should be **stratified** — measured separately for SNPs,
insertions, deletions, and CNVs, since callers often excel at one type and
struggle with others.

DiffBio supports differentiable AUROC (`DifferentiableAUROC`) for training
and exact metrics (`ExactAUROC` via calibrax) for evaluation.

---

## Further Reading

- [Variant Operators](../operators/variant.md) — VariantClassifier, CNVSegmentation usage
- [Pileup Generation](pileup-generation.md) — how pileup feeds into variant calling
- [Quality Filter](../operators/quality-filter.md) — soft quality filtering
- [Variant Calling Pipeline](../pipelines/variant-calling.md) — end-to-end pipeline
- [Variant Calling Example](../../examples/advanced/variant-calling.md) — runnable example

### References

1. Poplin et al. "A universal SNP and small-indel variant caller using deep
   neural networks." *Nature Biotechnology* 36(10), 2018.
2. Luo et al. "Exploring the limit of using a deep neural network on pileup
   data for germline variant calling." *Nature Machine Intelligence* 2(4), 2020.
