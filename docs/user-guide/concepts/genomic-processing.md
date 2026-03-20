# Genomic Processing

Raw sequencing data requires extensive processing before biological analysis
can begin -- adapter contamination must be removed, duplicates identified,
errors corrected, reads mapped to a reference, and in some cases assembled
de novo. DiffBio provides 8 differentiable operators covering the full
processing pipeline from raw reads to assembled genomes, plus specialized
operators for CRISPR guide design and population genetics.

---

## The Sequencing Pipeline

A typical next-generation sequencing experiment produces millions of short
reads (100-300 bp) that must be processed through a series of steps:

```
Raw Reads
    |
    v
Adapter Removal (trim technical sequences)
    |
    v
Error Correction (fix sequencing mistakes)
    |
    v
Duplicate Weighting (handle PCR amplification bias)
    |
    v
Read Mapping (align to reference genome)
    |
    v
Downstream Analysis (variant calling, expression, etc.)
```

Each step traditionally uses hard decisions -- trim or keep, duplicate or
unique, mapped or unmapped. DiffBio replaces these binary decisions with
soft, differentiable operations.

---

## Adapter Contamination

Sequencing library preparation ligates adapter sequences to DNA fragments.
When the fragment is shorter than the read length, the sequencer reads into
the adapter, contaminating the biological sequence.

`SoftAdapterRemoval` uses differentiable Smith-Waterman alignment to find
adapter matches, then applies sigmoid-weighted soft trimming based on the
match position. Instead of hard clipping at a fixed threshold:

- The alignment score determines match confidence (continuous, not binary)
- Soft trimming gradually down-weights bases near the adapter boundary
- Temperature controls the sharpness of the trimming transition

Both the match threshold and trimming temperature are learnable, allowing
the operator to adapt to dataset-specific adapter contamination patterns.

---

## PCR Duplicate Handling

PCR amplification before sequencing creates identical copies of DNA fragments.
These duplicates inflate apparent read depth and can bias variant calling.

Traditional tools (Picard MarkDuplicates, samtools) perform hard binary
classification: each read is either a duplicate (removed) or unique (kept).
This discards information -- some reads marked as duplicates may represent
independent observations of the same genomic position.

`DifferentiableDuplicateWeighting` replaces binary removal with probabilistic
weighting:

| Approach | Decision | Gradient Flow |
|---|---|---|
| Traditional | Binary: remove or keep | None |
| DiffBio | Continuous weight in [0, 1] | Through similarity computation |

The operator learns sequence embeddings, computes pairwise similarity, and
assigns weights inversely proportional to soft cluster size. Highly similar
reads receive lower weights; unique reads receive full weight. Temperature
controls the sharpness of the clustering.

---

## Sequencing Error Correction

Sequencing platforms introduce errors at characteristic rates -- Illumina
produces approximately 0.1-1% substitution errors, with error rates increasing
toward read ends. These errors propagate to downstream analysis as false
variants or misalignments.

`SoftErrorCorrection` uses a neural network (MLP) to predict corrected base
probabilities from local sequence context:

1. A sliding window of size $w$ (default 11) captures the local context
   around each position
2. Both sequence (one-hot) and quality score features are concatenated
3. The MLP predicts a probability distribution over the 4 bases
4. Temperature-controlled softmax blends the original and corrected calls

The operator is inspired by the DeepConsensus approach for consensus calling.
Quality scores serve as prior confidence -- positions with low quality are
corrected more aggressively.

---

## Read Mapping

Read mapping aligns short reads to a reference genome to determine their
genomic origin. `NeuralReadMapper` implements a cross-attention approach:

1. **Sequence encoding**: Both the read and reference window are encoded
   using a shared transformer encoder with positional embeddings
2. **Cross-attention**: Multi-head attention between read and reference
   embeddings computes soft alignment scores at each position
3. **Position prediction**: A softmax over reference positions produces a
   mapping probability distribution

The output is a soft mapping -- each read has a probability distribution
over reference positions rather than a single hard alignment. This is
valuable for multi-mapping reads (repetitive regions) where the true origin
is ambiguous.

---

## Genome Assembly

When no reference genome is available, reads must be assembled de novo into
contiguous sequences (contigs). Assembly algorithms construct an overlap graph
(or de Bruijn graph) where nodes represent reads (or k-mers) and edges
represent overlaps.

`GNNAssemblyNavigator` uses graph attention networks to navigate assembly
graphs:

1. **Message passing**: GATv2 attention layers propagate information between
   connected nodes, learning which overlaps are most reliable
2. **Edge scoring**: Each edge receives a traversal score based on the node
   representations of its endpoints
3. **Soft path selection**: Temperature-controlled softmax selects the
   next edge to traverse, enabling gradient flow through the assembly path

This differentiable approach enables joint optimization of assembly decisions
-- the GNN learns to prefer paths that produce longer, more accurate contigs.

---

## Metagenomic Binning

Environmental samples contain DNA from many organisms mixed together.
Metagenomic binning groups assembled contigs by their organism of origin.

`DifferentiableMetagenomicBinner` implements a VAMB-style VAE approach:

1. **Input features**: Tetranucleotide frequencies (TNF, 136 canonical
   4-mers) and abundance profiles across samples
2. **VAE encoding**: An encoder maps combined features to a latent space
3. **Latent clustering**: Contigs from the same genome cluster together
   in latent space
4. **Soft bin assignment**: Temperature-controlled softmax assigns contigs
   to bins with continuous probabilities

The beta-VAE objective balances reconstruction quality against latent space
regularity, producing clusters that correspond to individual genomes.

---

## CRISPR Guide Design

CRISPR-Cas9 genome editing requires a guide RNA (gRNA) that directs the Cas9
nuclease to a specific genomic target. Guide efficiency varies dramatically
based on sequence context -- some guides cut efficiently while others fail.

`DifferentiableCRISPRScorer` predicts on-target efficiency using a
DeepCRISPR-inspired CNN architecture:

| Input | Shape | Description |
|---|---|---|
| Guide sequence | (23, 4) | One-hot encoded 20nt guide + 3nt PAM |
| Epigenetic features | (23, $k$) | Optional chromatin accessibility, etc. |

The 1D CNN extracts local sequence patterns predictive of cutting efficiency,
followed by fully connected layers that produce a score in [0, 1]. Because
the scoring is differentiable, it can be integrated into guide optimization
pipelines -- gradients indicate which sequence positions most affect the
predicted efficiency.

---

## Population Genetics

`DifferentiableAncestryEstimator` implements a Neural ADMIXTURE-style model
for estimating ancestry proportions from genotype data. Given a genotype
vector of $n$ SNPs and $K$ ancestral populations:

1. An autoencoder maps the genotype to a latent representation
2. A softmax layer produces ancestry proportions
   $\mathbf{q} \in \Delta^{K-1}$ (summing to 1)
3. A decoder reconstructs genotype frequencies from the ancestry estimate

Temperature controls the sharpness of ancestry assignments -- lower
temperature produces more confident (peaky) estimates, higher temperature
allows more admixture. The entire model is differentiable, enabling gradient-
based optimization of the number of populations $K$ and integration with
downstream association studies.

---

## Why Differentiability Matters for Genomic Processing

Traditional genomic processing tools make hard decisions at each step. A
read is either trimmed or not. A duplicate is either removed or kept. A
read maps to one position or is discarded. These binary decisions cannot
be revisited in light of downstream evidence.

DiffBio's differentiable operators enable:

1. **Adaptive preprocessing**: Quality thresholds, trimming parameters,
   and error correction sensitivity adjust to the specific dataset through
   gradient-based optimization
2. **Joint preprocessing-analysis**: A variant calling loss propagates
   gradients back through mapping, error correction, and duplicate
   weighting, learning preprocessing parameters that maximize variant
   detection accuracy
3. **Soft decisions preserve information**: Probabilistic duplicate weights
   and soft mapping positions retain uncertainty that hard decisions discard,
   improving downstream statistical power
4. **Learned assembly strategies**: The GNN assembly navigator learns
   traversal preferences from training data, adapting to genome-specific
   repeat structures and error profiles

---

## Further Reading

- [Preprocessing Operators](../operators/preprocessing.md) -- adapter removal, deduplication, error correction
- [Assembly & Mapping Operators](../operators/assembly-mapping.md) -- neural read mapping and GNN assembly
- [CRISPR Operators](../operators/crispr.md) -- guide RNA scoring
- [Population Operators](../operators/population.md) -- ancestry estimation
- [Preprocessing Pipeline](../pipelines/preprocessing.md) -- end-to-end read processing
- [Genomic Processing API](../../api/operators/preprocessing.md) -- full API reference

### References

1. Dias et al. "Neural ADMIXTURE for rapid genomic clustering."
   *Nature Computational Science* 2, 2022.
2. Nissen et al. "Improved metagenome binning and assembly using deep
   variational autoencoders." *Nature Biotechnology* 39, 2021.
3. Chuai et al. "DeepCRISPR: optimized CRISPR guide RNA design by deep
   learning." *Genome Biology* 19, 2018.
4. Baid et al. "DeepConsensus improves the accuracy of sequences with a
   gap-aware sequence transformer." *Nature Biotechnology* 41, 2023.
