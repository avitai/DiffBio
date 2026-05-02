# Multi-Omics and Regulatory Genomics

Modern biology measures multiple molecular layers -- transcriptomics, epigenomics,
proteomics, metabolomics, and spatial organization -- each providing a different
view of the same biological system. DiffBio provides 11 differentiable operators
for integrating these modalities, analyzing epigenomic regulation, quantifying
splicing, discovering sequence motifs, and comparing mass spectra.

---

## Why Multi-Omics Integration Matters

No single measurement captures the full state of a cell. RNA-seq measures
transcript abundance but misses post-translational regulation. ATAC-seq reveals
chromatin accessibility but not which genes are actively transcribed. Proteomics
quantifies protein levels but lacks the resolution of spatial methods.

Each modality is an incomplete projection of the underlying biology. Integration
recovers information that no single modality can provide alone:

- Gene expression + chromatin accessibility = regulatory potential
- Spatial coordinates + expression = tissue architecture
- RNA + protein = post-transcriptional regulation

---

## Product-of-Experts Fusion

`DifferentiableMultiOmicsVAE` integrates multiple modalities through a shared
latent space using Product-of-Experts (PoE) fusion:

1. **Per-modality encoders** map each data type to a posterior distribution
   $q_m(z | x_m) = \mathcal{N}(\mu_m, \sigma_m^2)$
2. **PoE combines posteriors** by multiplying precision-weighted means:
   $\sigma_{\text{joint}}^{-2} = \sum_m \sigma_m^{-2}$,
   $\mu_{\text{joint}} = \sigma_{\text{joint}}^2 \sum_m \mu_m / \sigma_m^2$
3. **Reparameterized sampling** draws $z$ from the joint posterior
4. **Per-modality decoders** reconstruct each input from the shared $z$

PoE fusion naturally handles missing modalities -- if a modality is absent, its
encoder simply does not contribute to the joint posterior. This makes the model
robust to incomplete data, which is common when combining assays from different
experimental protocols.

---

## Spatial Transcriptomics

Spatial methods (Visium, MERFISH, Slide-seq) preserve the physical location of
gene expression within tissue. Two operators address key spatial analysis tasks:

**Cell type deconvolution** (`SpatialDeconvolution`): Each spatial spot may
contain multiple cell types. This operator learns spot embeddings that account
for spatial context via attention mechanisms, then performs soft assignment to
reference cell type profiles. The output is a per-spot cell type proportion
vector -- fully differentiable for joint optimization with downstream tissue
analysis.

**Spatial gene detection** (`DifferentiableSpatialGeneDetector`): Identifies
genes whose expression varies spatially beyond what random fluctuation would
produce. Uses Gaussian process regression with RBF kernels (following the
SpatialDE approach) to decompose expression variance into spatial and
non-spatial components. The Fraction of Spatial Variance (FSV) quantifies how
much of each gene's variability is explained by spatial structure.

---

## Chromatin and Epigenomic Analysis

Epigenomic marks -- histone modifications, DNA methylation, chromatin
accessibility -- control which genes can be transcribed. Four operators provide
differentiable analysis of these regulatory signals:

| Operator | Input Data | Method | Output |
|---|---|---|---|
| `DifferentiablePeakCaller` | ChIP-seq / ATAC-seq signal | CNN + sigmoid thresholding | Peak regions + scores |
| `FNOPeakCaller` | ChIP-seq / ATAC-seq signal | Fourier neural operator | Peak regions + scores |
| `ChromatinStateAnnotator` | Histone modification profiles | HMM with Bernoulli emissions | State assignments |
| `ContextualEpigenomicsOperator` | Multi-track epigenomic signals | Context-aware encoder + task heads | Joint epigenomic predictions |

### Peak Calling

`DifferentiablePeakCaller` replaces the hard thresholds of MACS2-style peak
callers with learned CNN filters and temperature-controlled sigmoid decisions.
Multi-scale convolution kernels detect peaks at different widths. An optional
VAE denoising stage (inspired by SCALE) encodes the coverage signal into a
latent space before peak detection, separating true signal from noise.

### Chromatin State Annotation

`ChromatinStateAnnotator` implements a ChromHMM-style model where genomic
regions are assigned to discrete chromatin states (active promoter, enhancer,
repressed, etc.) based on combinations of histone marks. The HMM uses Bernoulli
emissions for mark presence/absence, with temperature-controlled soft Viterbi
decoding for differentiability. Optional cell-type conditioning learns
per-cell-type emission parameters.

---

## 3D Genome Organization

`HiCContactAnalysis` analyzes Hi-C contact matrices -- pairwise chromatin
interaction frequencies that reveal 3D genome organization. The operator learns
bin embeddings from contact patterns using a neural encoder, then predicts:

- **Compartments**: A/B compartment assignments (active vs repressed chromatin)
- **TAD boundaries**: Topologically Associating Domain boundaries where
  contact frequency drops sharply

Attention over neighboring bins captures the local contact structure that
defines TAD boundaries, while global patterns reveal compartment identity.

---

## Alternative Splicing

A single gene can produce multiple transcript isoforms through alternative
splicing -- selecting different combinations of exons. `SplicingPSI` computes
the Percent Spliced In (PSI) for each exon:

$$
\text{PSI} = \frac{\text{inclusion reads}}{\text{inclusion reads} + \text{exclusion reads}}
$$

The operator adds pseudocounts for numerical stability and computes confidence
scores based on read depth. Both the pseudocount and the temperature parameter
are learnable, allowing the PSI calculation to adapt to dataset-specific noise
characteristics.

---

## Sequence Motif Discovery

`DifferentiableMotifDiscovery` learns Position Weight Matrices (PWMs) that
represent recurring sequence patterns -- transcription factor binding sites,
splice signals, or regulatory elements. The operator implements a differentiable
MEME-style approach:

1. Learnable PWMs are initialized (shape: motif width $\times$ alphabet size)
2. Sequences are scanned against PWMs using softmax-weighted scoring
3. The best-matching positions contribute to a motif likelihood
4. Gradients from a reconstruction loss update the PWM entries

This is fully differentiable -- PWMs can be jointly optimized with upstream
peak calling or downstream expression prediction.

---

## Mass Spectrometry

`DifferentiableSpectralSimilarity` compares tandem mass spectra (MS/MS) to
predict structural similarity between metabolites. The operator implements an
MS2DeepScore-style Siamese architecture:

1. A shared neural encoder maps binned mass spectra to 200-dimensional embeddings
2. Cosine similarity between embeddings predicts structural similarity
   (Tanimoto score between molecular fingerprints)
3. Monte Carlo dropout provides uncertainty estimates

This enables differentiable metabolite identification: gradients flow from
similarity predictions back through the spectral encoder, learning which
spectral features are most informative for structural comparison.

---

## Why Differentiability Matters for Multi-Omics

Traditional multi-omics analysis pipelines process each modality independently,
then combine results post-hoc. Peak calling, splicing quantification, and
expression normalization are each optimized in isolation.

DiffBio's differentiable operators enable:

1. **Joint multi-modal learning**: A reconstruction loss on all modalities
   simultaneously updates the shared latent space, ensuring it captures
   information relevant across data types
2. **End-to-end regulatory analysis**: Gradients from gene expression
   prediction flow back through chromatin state annotation and peak calling,
   learning which epigenomic features best predict transcriptional output
3. **Adaptive spatial analysis**: Deconvolution parameters adapt to spatial
   gene detection results, jointly optimizing cell type estimates and spatial
   variability
4. **Learnable motif-expression coupling**: Motif discovery and expression
   prediction are jointly optimized, discovering motifs that actually predict
   expression changes rather than just sequence conservation

---

## Further Reading

- [Multi-Omics Operators](../operators/multiomics.md) -- integration, spatial, and Hi-C operators
- [Epigenomics Operators](../operators/epigenomics.md) -- peak calling and chromatin states
- [RNA-seq Operators](../operators/rnaseq.md) -- splicing PSI and motif discovery
- [Metabolomics Operators](../operators/metabolomics.md) -- spectral similarity
- [Multi-Omics API](../../api/operators/multiomics.md) -- full API reference
- [Epigenomics API](../../api/operators/epigenomics.md) -- full API reference

### References

1. Ashuach et al. "MultiVI: deep generative model for the integration of
   multimodal data." *Nature Methods* 20, 2023.
2. Svensson et al. "SpatialDE: identification of spatially variable genes."
   *Nature Methods* 15, 2018.
3. Ernst & Kellis. "ChromHMM: automating chromatin-state discovery."
   *Nature Methods* 9, 2012.
4. Huber et al. "MS2DeepScore: a novel deep learning similarity measure to
   compare tandem mass spectra." *Journal of Cheminformatics* 13, 2021.
