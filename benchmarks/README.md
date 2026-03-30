# DiffBio Benchmarks

Comprehensive benchmarks evaluating DiffBio's differentiable bioinformatics operators across all domains. Each benchmark tests **correctness**, **differentiability** (gradient flow), and **throughput**.

## Directory Structure

```
benchmarks/
    _common.py                    # Shared utilities (data generators, gradient checks, timing)
    schema.py                     # Unified result envelope (BenchmarkEnvelope)
    dashboard.py                  # Terminal dashboard rendering (tabulate)
    regression.py                 # Regression detection against baselines
    run_all.py                    # Master runner (discovery + execution + dashboard)
    generate_report.py            # Markdown report for docs/

    alignment/                    # Sequence alignment
        alignment_benchmark.py    # SmoothSmithWaterman
    assembly/                     # Genome assembly
        assembly_benchmark.py     # GNNAssemblyNavigator, MetagenomicBinner
    drug_discovery/               # Molecular property prediction
        molnet_benchmark.py       # MoleculeNet datasets (BBBP, ESOL, etc.)
        fingerprint_benchmark.py  # ECFP4, MACCS, Neural fingerprints
        circular_fingerprint_benchmark.py  # ECFP4 vs DeepChem
    epigenomics/                  # Chromatin analysis
        epigenomics_benchmark.py  # PeakCaller (CNN/FNO), ChromatinStateAnnotator
    language_models/              # Biological language models
        language_model_benchmark.py  # TransformerEncoder, FoundationModel
    molecular_dynamics/           # MD simulations
        molecular_dynamics_benchmark.py  # ForceField, MDIntegrator
    multiomics/                   # Multi-omics integration
        multiomics_benchmark.py   # MultiOmicsVAE, SpatialDeconv, HiC, SpatialGene
    normalization/                # Dimensionality reduction
        dimreduction_benchmark.py # DifferentiableUMAP, DifferentiablePHATE
    preprocessing/                # Read preprocessing
        preprocessing_benchmark.py  # AdapterRemoval, DuplicateFilter, ErrorCorrection
    protein/                      # Protein structure
        protein_structure_benchmark.py  # SecondaryStructure (DSSP-style)
    rna_structure/                # RNA folding
        rna_structure_benchmark.py  # RNAFold (McCaskill partition function)
    singlecell/                   # Single-cell analysis
        singlecell_benchmark.py   # Harmony, SoftKMeans
        scvi_benchmark.py         # VAENormalizer (scVI-style)
        trajectory_benchmark.py   # Pseudotime, Fate, Velocity, OT
        grn_benchmark.py          # GRN inference, SINDy, CellCommunication
    specialized/                  # Domain-specific operators
        crispr_benchmark.py       # CRISPR guide scoring
        population_benchmark.py   # Ancestry estimation
        metabolomics_benchmark.py # Spectral similarity
    statistical/                  # Statistical models
        statistical_benchmark.py  # HMM, NB-GLM, EM quantification
    variant/                      # Variant calling
        variant_calling_benchmark.py  # Pileup, Classifier, CNV segmentation

    results/                      # JSON output (gitignored)
    baselines/                    # Baseline snapshots for regression detection
```

## Capabilities Matrix

| Domain | Operators Benchmarked | Key Metrics |
|--------|----------------------|-------------|
| Alignment | SmoothSmithWaterman | Score accuracy, gradient flow, temperature sweep |
| Assembly | GNNAssemblyNavigator, MetagenomicBinner | Edge selection accuracy, binning ARI |
| Drug Discovery | ECFP4, MACCS, Neural FP, MolNet | ROC-AUC, RMSE, Tanimoto similarity |
| Epigenomics | PeakCaller (CNN/FNO), ChromatinState | Peak precision/recall, state accuracy |
| Language Models | TransformerEncoder, FoundationModel | Embedding quality, gradient flow |
| Molecular Dynamics | ForceField, MDIntegrator | Energy conservation, force accuracy |
| Multi-omics | MultiOmicsVAE, SpatialDeconv, HiC, SpatialGene | Latent quality, deconvolution accuracy |
| Normalization | DifferentiableUMAP, PHATE | Trustworthiness, cluster separation |
| Preprocessing | AdapterRemoval, DuplicateFilter, ErrorCorrection | Detection accuracy, gradient flow |
| Protein | SecondaryStructure | Q3 accuracy, H-bond matrix quality |
| RNA Structure | RNAFold | Base pair recovery, partition function |
| Single-Cell | Harmony, SoftKMeans, VAE, Pseudotime, Velocity, GRN | Batch mixing, silhouette, ARI, NMI |
| Specialized | CRISPR, Ancestry, Spectral | Scoring correlation, proportion accuracy |
| Statistical | HMM, NB-GLM, EM | Parameter recovery, abundance correlation |
| Variant | Pileup, Classifier, CNV | Classification F1, breakpoint sensitivity |

## scBench / spatialBench Task Coverage

DiffBio operators cover all 8 evaluation task types from scBench and spatialBench:

| Task Type | DiffBio Operator | Benchmark |
|-----------|-----------------|-----------|
| qc_filtering | DifferentiableQualityFilter | preprocessing/ |
| normalization | VAENormalizer | singlecell/scvi |
| clustering | SoftKMeansClustering | singlecell/ |
| cell_typing | CellAnnotator, SoftKMeans | singlecell/ |
| differential_expression | NB-GLM, DE Pipeline | statistical/ |
| batch_correction | DifferentiableHarmony | singlecell/ |
| trajectory | Pseudotime, FateProbability | singlecell/trajectory |
| spatial_analysis | SpatialDomain, SpatialDeconv | multiomics/ |

## Running Benchmarks

### Prerequisites

```bash
source activate.sh  # Activate DiffBio environment
```

### Run All Benchmarks

```bash
python benchmarks/run_all.py
```

### Run Specific Domain

```bash
python benchmarks/run_all.py --domains variant,singlecell
```

### Quick Mode (CI)

```bash
python benchmarks/run_all.py --quick
```

### Run Single Benchmark

```bash
python benchmarks/alignment/alignment_benchmark.py
python benchmarks/variant/variant_calling_benchmark.py
python benchmarks/singlecell/trajectory_benchmark.py
```

### Regression Detection

```bash
# Save a baseline
python benchmarks/run_all.py --save-baseline

# Check against baseline
python benchmarks/run_all.py --check-regression benchmarks/baselines/baseline_20260330.json
```

### Generate Documentation Report

```bash
python benchmarks/generate_report.py
# Output: docs/development/benchmark-results.md
```

## Result Format

All benchmarks save results as JSON to `benchmarks/results/<domain>/`. The unified envelope schema:

```json
{
  "schema_version": "1.0",
  "benchmark_id": "variant/variant_calling_benchmark",
  "domain": "variant",
  "operators_tested": ["DifferentiablePileup", "VariantClassifier"],
  "timestamp": "2026-03-30T14:22:01",
  "status": "pass",
  "correctness": {"passed": true, "tests": [...]},
  "differentiability": {"passed": true, "gradient_norm": 1.234},
  "performance": {"throughput": 5000.0, "throughput_unit": "positions/sec"},
  "evaluation_task_types": ["qc_filtering"]
}
```

## Adding New Benchmarks

1. Create `benchmarks/<domain>/<name>_benchmark.py`
2. Follow the established pattern:
   - Frozen `@dataclass` for results
   - `run_benchmark(quick: bool = False)` function
   - Use `benchmarks._common` utilities (gradient checks, throughput, data generators)
3. Add the benchmark to `_BENCHMARK_REGISTRY` in `run_all.py`
4. Test: `python benchmarks/<domain>/<name>_benchmark.py`
