# Examples Overview

Practical examples demonstrating DiffBio's differentiable bioinformatics operators, from single-operator basics to full pipeline composition with ecosystem integration.

## Example Tiers

<div class="nav-grid">
<div class="nav-card">
  <a href="basic/operator-pattern/">
    <div class="nav-title">13 Basic</div>
    <div class="nav-description">Single-operator patterns (5-10 min)</div>
  </a>
</div>
<div class="nav-card">
  <a href="intermediate/imputation/">
    <div class="nav-title">5 Intermediate</div>
    <div class="nav-description">Multi-operator workflows (10-20 min)</div>
  </a>
</div>
<div class="nav-card">
  <a href="advanced/spatial-analysis/">
    <div class="nav-title">14 Advanced</div>
    <div class="nav-description">Pipelines & ecosystem integration (20-40 min)</div>
  </a>
</div>
</div>

---

## Basic

Single-operator examples. One config, one `apply()`, one gradient check. Target audience: first-time DiffBio users.

| Example | Duration | Key Operators | Description |
|---------|----------|---------------|-------------|
| [Operator Pattern](basic/operator-pattern.md) | 5 min | SoftKMeansClustering | The universal Config -> Construct -> Apply pattern |
| [MolNet Data Loading](basic/molnet-data-loading.md) | 5 min | MolNetSource | Load MoleculeNet benchmark datasets |
| [Molecular Fingerprints](basic/molecular-fingerprints.md) | 10 min | CircularFingerprint | Generate ECFP and neural fingerprints |
| [Molecular Similarity](basic/molecular-similarity.md) | 5 min | TanimotoSimilarity | Compare molecules with similarity metrics |
| [Scaffold Splitting](basic/scaffold-splitting.md) | 5 min | ScaffoldSplitter | Proper train/test splits for drug discovery |
| [DNA Encoding](basic/dna-encoding.md) | 5 min | one-hot encoding | One-hot encode DNA sequences |
| [Sequence Alignment](basic/simple-alignment.md) | 10 min | SmoothSmithWaterman | Smith-Waterman local alignment |
| [Pileup Generation](basic/pileup-generation.md) | 10 min | DifferentiablePileup | Generate pileups from aligned reads |
| [Single-Cell Clustering](basic/single-cell-clustering.md) | 10 min | SoftKMeansClustering | Soft k-means with training loop |
| [RNA Structure](basic/rna-structure.md) | 10 min | McCaskill algorithm | Predict RNA secondary structure |
| [Protein Structure](basic/protein-structure.md) | 10 min | DSSP prediction | Predict protein secondary structure |
| [HMM Sequence Model](basic/hmm-sequence-model.md) | 10 min | DifferentiableHMM | Hidden Markov Models for sequences |
| [Preprocessing](basic/preprocessing.md) | 10 min | QualityFilter, AdapterRemoval | Read preprocessing pipeline |

## Intermediate

Multi-operator workflows with two or three operators chained, parameter sweeps, or evaluation against ground truth. Target audience: users building custom pipelines.

| Example | Duration | Key Operators | Description |
|---------|----------|---------------|-------------|
| [Imputation](intermediate/imputation.md) | 15 min | DifferentiableDiffusionImputer | MAGIC-style diffusion imputation for dropout recovery |
| [Trajectory](intermediate/trajectory.md) | 20 min | DifferentiablePseudotime, FateProbability, SwitchDE | Pseudotime ordering and fate probability estimation |
| [Cell Annotation](intermediate/cell-annotation.md) | 15 min | DifferentiableCellAnnotator (3 modes) | Cell type annotation: celltypist, cellassign, scanvi |
| [Doublet Detection](intermediate/doublet-detection.md) | 15 min | DoubletScorer, SoloDetector | Scrublet-style and Solo-style doublet detection |
| [Batch Correction](intermediate/batch-correction.md) | 20 min | Harmony, MMD, WGAN | Three batch correction strategies compared |

## Advanced

Full pipeline composition with ecosystem integration (calibrax metrics, artifex losses, opifex training). Training loops, benchmarking, and multi-operator chains. Target audience: researchers adapting DiffBio for their data.

| Example | Duration | Key Operators | Description |
|---------|----------|---------------|-------------|
| [Spatial Analysis](advanced/spatial-analysis.md) | 25 min | SpatialDomain, PASTEAlignment | STAGATE domain identification and PASTE slice alignment |
| [GRN Inference](advanced/grn-inference.md) | 25 min | DifferentiableGRN | Gene regulatory network inference via GATv2 attention |
| [Single-Cell Pipeline](advanced/singlecell-pipeline.md) | 30 min | Simulator, AmbientRemoval, Imputer, Clustering, Pseudotime | Five-operator end-to-end pipeline |
| [Calibrax Metrics](advanced/calibrax-metrics.md) | 25 min | SoftKMeansClustering, DifferentiableAUROC | Training vs evaluation metric split with calibrax |
| [scVI Benchmark](advanced/scvi-benchmark.md) | 30 min | VAENormalizer, MultiOmicsVAE | scVI-style VAE training with calibrax evaluation |
| [Drug Discovery Workflow](advanced/drug-discovery-workflow.md) | 30 min | CircularFingerprint, PropertyPredictor | End-to-end drug discovery pipeline |
| [ADMET Prediction](advanced/admet-prediction.md) | 25 min | ADMETPredictor | Multi-task ADMET property prediction |
| [AttentiveFP GNN](advanced/attentive-fp.md) | 25 min | AttentiveFPOperator | Attention-based molecular fingerprints |
| [Variant Calling Pipeline](advanced/variant-calling.md) | 30 min | Full variant calling pipeline | End-to-end variant calling with CNN classifier |
| [Single-Cell Batch Correction](advanced/singlecell-batch-correction.md) | 20 min | DifferentiableHarmony | Harmony-style batch correction |
| [Differential Expression](advanced/differential-expression.md) | 25 min | NB-GLM | DESeq2-style statistical testing |
| [RNA Velocity](advanced/rna-velocity.md) | 25 min | Neural ODE velocity | RNA velocity trajectory inference |
| [Epigenomics Analysis](advanced/epigenomics-analysis.md) | 25 min | Peak calling, chromatin states | ChIP-seq and ATAC-seq analysis |
| [Multi-omics Integration](advanced/multiomics-integration.md) | 30 min | Spatial deconvolution, Hi-C | Multi-omics data integration |

## Running Examples

All examples are self-contained Python scripts that generate synthetic data and produce verifiable outputs.

```bash
# Setup
./setup.sh
source ./activate.sh

# Run any example
uv run python examples/basics/operator_pattern.py
uv run python examples/singlecell/clustering.py
uv run python examples/ecosystem/scvi_benchmark.py
```

## Key Features Demonstrated

All examples showcase DiffBio's core capabilities:

1. **Differentiability** -- every operator supports `jax.grad` for gradient computation
2. **JIT Compilation** -- all operators work with `jax.jit` for accelerated execution
3. **apply() Contract** -- consistent `result, state, metadata = operator.apply(data, {}, None)` interface
4. **Synthetic Data** -- self-contained examples with no external data dependencies
5. **Ecosystem Integration** -- calibrax metrics, artifex losses, and opifex training utilities

## Contributing Examples

See the [Contributing Guide](../development/contributing.md) and the [Example Documentation Design Guide](../development/example-documentation-design.md) for details on adding new examples.
