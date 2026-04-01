# Benchmarks

DiffBio includes a benchmark suite that evaluates operators on **real datasets**
with **field-standard metrics** and **comparison tables** against published SOTA
methods.

---

## Running Benchmarks

```bash
# CI tier (~1 min, subsampled datasets)
uv run python benchmarks/run_all.py --tier ci --quick

# Nightly tier (~30 min, full Tier 1+2 benchmarks)
uv run python benchmarks/run_all.py --tier nightly

# Full suite (~2 hours on GPU)
uv run python benchmarks/run_all.py --tier full

# Filter by domain
uv run python benchmarks/run_all.py --tier nightly --domains singlecell

# Single benchmark
uv run python benchmarks/singlecell/bench_batch_correction.py --quick
```

---

## Benchmark Suite

### Single-Cell (6 benchmarks)

| Benchmark | Operator | Dataset | Metrics | Baselines |
|-----------|----------|---------|---------|-----------|
| Foundation Annotation | LinearEmbeddingProbe on native or imported embeddings | immune_human | Accuracy, macro-F1, train loss | DiffBio native, Geneformer precomputed, scGPT precomputed |
| Batch Correction | DifferentiableHarmony | immune_human (33K cells, 10 batches) | Full scib-metrics (aggregate, silhouette, NMI, ARI, iLISI, cLISI) | scVI, Harmony (R), Scanorama, BBKNN |
| Clustering | SoftKMeansClustering | immune_human (16 cell types) | ARI, NMI, silhouette | Leiden, Louvain, sklearn k-means |
| VAE Integration | VAENormalizer (ZINB) | immune_human | ELBO + scib-metrics suite | scVI, scANVI |
| Trajectory | Pseudotime + Velocity | pancreas (3.7K cells) | Pseudotime range, velocity shape | scVelo, DPT, Monocle3 |
| GRN Inference | DifferentiableGRN | benGRN mESC (11.6K edges) | AUPRC, precision, recall | GENIE3, GRNBoost2, pySCENIC |

### Genomics (3 scaffold benchmarks)

| Benchmark | Operator | Dataset | Metrics | Baselines |
|-----------|----------|---------|---------|-----------|
| Promoter Classification | LinearEmbeddingProbe on native or imported sequence embeddings | synthetic_genomics scaffold | Accuracy, macro-F1, train loss | DiffBio native, sequence precomputed adapters |
| TFBS Classification | LinearEmbeddingProbe on native or imported sequence embeddings | synthetic_genomics scaffold | Accuracy, macro-F1, train loss | DiffBio native, sequence precomputed adapters |
| Splice-Site Classification | LinearEmbeddingProbe on native or imported sequence embeddings | synthetic_genomics scaffold | Accuracy, macro-F1, train loss | DiffBio native, sequence precomputed adapters |

### Alignment (2 benchmarks)

| Benchmark | Operator | Dataset | Metrics | Baselines |
|-----------|----------|---------|---------|-----------|
| MSA | SoftProgressiveMSA | balifam100 (59 families) | SP score, TC score | MAFFT, ClustalW, MUSCLE, T-Coffee |
| Pairwise | SmoothSmithWaterman | balifam100 | Alignment score | BLAST, SSEARCH, FASTA |

### Structure Prediction (2 benchmarks)

| Benchmark | Operator | Dataset | Metrics | Baselines |
|-----------|----------|---------|---------|-----------|
| RNA Folding | DifferentiableRNAFold | ArchiveII | Sensitivity, PPV, F1 | ViennaRNA, LinearFold, EternaFold |
| Protein SS | DifferentiableSecondaryStructure | Ideal backbones | Q3 accuracy | DSSP, STRIDE, KAKSI |

### Molecular Dynamics (1 benchmark)

| Benchmark | Operator | Dataset | Metrics | Baselines |
|-----------|----------|---------|---------|-----------|
| LJ Fluid | ForceFieldOperator + MDIntegrator | 64K LJ system | Steps/sec, energy drift | jax-md (direct), LAMMPS |

### Statistical (1 benchmark)

| Benchmark | Operator | Dataset | Metrics | Baselines |
|-----------|----------|---------|---------|-----------|
| DE Analysis | DifferentiableNBGLM | immune_human (2 cell types) | Concordance with t-test | DESeq2, edgeR, Wilcoxon |

---

## scBench / spatialBench Context

DiffBio operators cover all evaluation task types from
[scBench](https://github.com/your-org/scbench) (394 single-cell evaluations)
and [spatialBench](https://github.com/your-org/spatialbench) (146 spatial
evaluations).

### scBench Leaderboard (by task category)

| Task | Best Model | Accuracy | DiffBio Operator |
|------|-----------|----------|-----------------|
| QC | Claude Opus 4.5 | 63.9% | DifferentiableQualityFilter |
| Normalization | Claude Opus 4.5 | 83.8% | VAENormalizer |
| Clustering | Claude Opus 4.6 | 52.7% | SoftKMeansClustering |
| Cell Typing | Claude Opus 4.6 | 48.2% | CellAnnotator |
| Diff. Expression | Claude Opus 4.6 | 41.4% | NB-GLM, DE Pipeline |
| Trajectory | Claude Opus 4.5 | 61.9% | Pseudotime |

### spatialBench Leaderboard (by task category)

| Task | Best Model | Accuracy | DiffBio Operator |
|------|-----------|----------|-----------------|
| Normalization | GPT-5.2 | 76.2% | VAENormalizer |
| Clustering | Claude Opus 4.5 (CC) | 60.3% | SoftKMeansClustering |
| Spatial Analysis | Claude Opus 4.5 (CC) | 66.7% | SpatialDomain |
| Diff. Expression | Claude Opus 4.5 (CC) | 46.2% | NB-GLM |

*Source: scBench and spatialBench published results (3 runs per model, 95% CI).*

---

## Architecture

Built on the Datarax, Artifex, Opifex, and Calibrax ecosystem:

- **datarax**: `DataSourceModule` and execution patterns for loading and
  iterating real datasets
- **artifex**: benchmark and model-adapter reference patterns for model-facing
  integrations
- **opifex**: scientific-training and optimization surfaces used by
  benchmarked operators
- **calibrax**: `BenchmarkResult`, `Metric`, `Point`, `TimingCollector`,
  `Store`, regression detection, comparison, and publication export
- **DiffBio-specific**: `check_gradient_flow()` for verifying operator
  differentiability

### Benchmark Pattern

Every benchmark inherits from `DiffBioBenchmark` and implements only
`_run_core()`:

```python
class MyBenchmark(DiffBioBenchmark):
    def _run_core(self) -> dict[str, Any]:
        # 1. Load data via DataSource
        source = MyDataSource(MyConfig(data_dir=self.data_dir))
        data = source.load()

        # 2. Create operator
        operator = MyOperator(config, rngs=nnx.Rngs(42))

        # 3. Train operator (for operators with learnable parameters)
        #    Use an unsupervised loss and gradient descent via optax.
        #    Physics-based operators (e.g. Smith-Waterman, DSSP) that
        #    have no learnable params can skip this step.
        opt = nnx.Optimizer(operator, optax.adam(1e-3), wrt=nnx.Param)
        for step in range(n_steps):
            loss, grads = nnx.value_and_grad(unsupervised_loss)(operator)
            opt.update(operator, grads)

        # 4. Evaluate trained operator
        result, _, _ = operator.apply(data, {}, None)

        # 5. Compute metrics
        metrics = evaluate_my_domain(result, ground_truth)

        # 6. Return standard dict
        return {
            "metrics": metrics,
            "operator": operator,
            "input_data": data,
            "loss_fn": lambda m, d: jnp.sum(m.apply(d, {}, None)[0]["output"]),
            "n_items": len(source),
            "iterate_fn": lambda: operator.apply(data, {}, None),
            "baselines": MY_BASELINES,
            "dataset_info": {"n_items": len(source)},
            "operator_name": "MyOperator",
            "dataset_name": "my_dataset",
        }
```

The base class handles: gradient flow check, throughput measurement,
comparison table printing, `BenchmarkResult` construction, and CLI.

### Standard Benchmark Tags

Every `DiffBioBenchmark` result emits these baseline Calibrax tags:

- `framework`: always `diffbio`
- `operator`: operator or pipeline name
- `dataset`: dataset identifier used by the benchmark
- `task`: canonical task slug derived from the benchmark name unless overridden

When a benchmark's `_run_core()` returns raw operator output under
`result_data`, the base class also promotes canonical foundation-model metadata
into Calibrax tags and result metadata:

- `model_family`
- `adapter_mode`
- `artifact_id`
- `preprocessing_version`

These values are decoded from the operator's `foundation_model` payload and
stored once in the shared benchmark layer. The corresponding result metadata
also includes `foundation_model` and `comparison_axes` so regression and
comparison tooling can group by dataset, task, and artifact identity without
benchmark-specific code.

### Imported Foundation-Model Benchmarks

The current stable imported-model path is **precomputed embeddings**. For
single-cell workloads, benchmarks consume a `SingleCellPrecomputedAdapter`
implementation, align artifact rows by `cell_ids`, and then run the normal
DiffBio downstream benchmark.

The first supported imported adapters are `GeneformerPrecomputedAdapter` and
`ScGPTPrecomputedAdapter`. This remains deliberately narrower than generic
checkpoint support:

- supported: external embedding artifacts with explicit `cell_ids`
- supported: benchmark tagging by `model_family`, `adapter_mode`,
  `artifact_id`, and `preprocessing_version`
- supported: deterministic single-cell quick-suite reports across native
  DiffBio, Geneformer, and scGPT adapters for annotation and batch correction
- supported: explicit scGPT batch-context metadata in comparison reports via
  `requires_batch_context`, `batch_key`, and `context_version`
- supported: a shared `SequencePrecomputedAdapter` contract plus a genomics
  quick-suite scaffold for promoter, TFBS, and splice-site tasks
- supported: `DNABERT2PrecomputedAdapter` and
  `NucleotideTransformerPrecomputedAdapter` for aligned precomputed genomics
  artifacts
- not yet supported: arbitrary Geneformer checkpoint loading into DiffBio
- not yet supported: frozen in-process DNABERT-2 or Nucleotide Transformer
  encoder imports in stable APIs
- not yet supported: tokenizer interchangeability claims across upstream models

> **Important**: Operators with learnable parameters (neural networks,
> learnable centroids, GLM coefficients) must be trained before
> evaluation. Comparing untrained random weights against optimised
> baselines produces misleading results. Use an unsupervised loss
> appropriate to the domain (e.g. reconstruction error, compactness,
> log-likelihood).

---

## Datasets

### Required Downloads

| Dataset | Size | Path | Download |
|---------|------|------|----------|
| immune_human | 2.0 GB | `/media/mahdi/ssd23/Data/scib/Immune_ALL_human.h5ad` | [Figshare](https://ndownloader.figshare.com/files/25717328) |
| pancreas | 51 MB | `/media/mahdi/ssd23/Data/scvelo/endocrinogenesis_day15.h5ad` | [GitHub](https://github.com/theislab/scvelo_notebooks/raw/master/data/Pancreas/endocrinogenesis_day15.h5ad) |

### From Cloned Repos

| Dataset | Repo | Used By |
|---------|------|---------|
| balifam100 | `../balifam/` | MSA, pairwise alignment |
| ArchiveII | `../RNAFoldAssess/` | RNA folding |
| mESC ground truth | `../benGRN/` | GRN inference |

---

## Test Environment

| Component | Value |
|-----------|-------|
| Platform | Linux 6.8.0 (Ubuntu) |
| Python | 3.12.6 |
| JAX | 0.9.0.1 |
| GPU | NVIDIA GeForce RTX 4090 (24 GB) |
| Backend | CUDA |
