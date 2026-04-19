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
| Promoter Classification | LinearEmbeddingProbe on native, frozen, or imported sequence embeddings | synthetic_genomics scaffold | Accuracy, macro-F1, train loss | DiffBio native, DiffBio frozen encoder, sequence precomputed adapters |
| TFBS Classification | LinearEmbeddingProbe on native, frozen, or imported sequence embeddings | synthetic_genomics scaffold | Accuracy, macro-F1, train loss | DiffBio native, DiffBio frozen encoder, sequence precomputed adapters |
| Splice-Site Classification | LinearEmbeddingProbe on native, frozen, or imported sequence embeddings | synthetic_genomics scaffold | Accuracy, macro-F1, train loss | DiffBio native, DiffBio frozen encoder, sequence precomputed adapters |

### Drug Discovery (3 benchmarks)

| Benchmark | Operator | Dataset | Metrics | Baselines |
|-----------|----------|---------|---------|-----------|
| MolNet BBBP | CircularFingerprintOperator + MLP | bbbp | Test ROC-AUC, train ROC-AUC | GCN, AttentiveFP, D-MPNN |
| Davis DTI Scaffold | DTIFeatureProbe on paired contract features | davis | RMSE, Pearson, Spearman | non-differentiable fingerprint, differentiable drug encoder |
| BioSNAP DTI Scaffold | DTIFeatureProbe on paired contract features | biosnap | ROC-AUC, PR-AUC, MRR, Recall@1, Recall@5 | non-differentiable fingerprint, differentiable drug encoder |

### Epigenomics (3 benchmarks)

| Benchmark | Operator | Dataset | Metrics | Baselines |
|-----------|----------|---------|---------|-----------|
| Peak Calling | DifferentiablePeakCaller | ENCODE_CTCF_K562 | Precision, recall, F1, Jaccard | MACS2, HOMER, Genrich |
| Contextual Peak Calling | ContextualEpigenomicsOperator ablation suite | synthetic_contextual_epigenomics | Precision, recall, F1, chromatin consistency | sequence-only, `+TF`, `+TF+chromatin` |
| Chromatin-State Prediction | ContextualEpigenomicsOperator ablation suite | synthetic_contextual_epigenomics | Accuracy, chromatin consistency | sequence-only, `+TF`, `+TF+chromatin` |

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
also includes `foundation_model`, `comparison_axes`, and one deterministic
`comparison_key` so regression and comparison tooling can group by dataset,
task, and artifact identity without benchmark-specific code.

Shared foundation-suite reports preserve the same contract. Each task report
now exposes:

- `comparison_axes`: the canonical ordering used for provenance-aware grouping
- `foundation_model`: normalized per-model provenance when present
- `comparison_key`: one deterministic row keyed by `comparison_axes`, with
  `None` for axes that do not apply to a given model

Each full foundation-suite report also stores:

- `regression_expectations`: the canonical `comparison_axes`, `task_order`,
  per-task `required_models` ordering, Calibrax-native `metric_defs`, and the
  stored `calibrax` baseline/threshold policy used for regression checks
- `deferred_tasks`: planned-but-unverified tasks that stay outside the current
  stable promotion scope, with the required follow-on harness or evidence made
  explicit in the saved report and Calibrax run metadata

Use `build_foundation_promotion_report()` from `benchmarks._foundation_models`
to convert a stored suite report plus an optional Calibrax `GuardResult` into a
deterministic promotion-review artifact. That artifact keeps the in-scope task
list, deferred scope, required models, threshold policy, and any missing
promotion evidence in one machine-readable record.

Use `save_foundation_suite_report()` from `benchmarks._foundation_models` to
persist these deterministic suite reports as canonical JSON.
Use `save_foundation_suite_run()` to mirror the same suite report into a
Calibrax `Store`, and `check_foundation_suite_regressions()` to run the stored
suite against the `main` baseline with the persisted threshold policy.
Use `save_foundation_promotion_report()` to persist the promotion-review record
as canonical JSON once the relevant regression check has been attached.
For single-cell promotion review, use
`benchmarks.singlecell.foundation_suite.build_singlecell_foundation_promotion_report()`;
it attaches the Calibrax guard result before building the shared promotion
artifact and fails closed unless an existing baseline is available or baseline
bootstrap is requested explicitly.
For genomics promotion review, use
`benchmarks.genomics.foundation_suite.build_genomics_foundation_promotion_report()`;
it follows the same fail-closed guard path while preserving the current Phase 4
scaffold provenance boundaries in the stored suite report.

### Imported Foundation-Model Benchmarks

The current stable imported-model path is **precomputed embeddings**. For
single-cell workloads, benchmarks consume a `SingleCellPrecomputedAdapter`
implementation, align artifact rows by `cell_ids`, and then run the normal
DiffBio downstream benchmark.

Imported foundation-model benchmarks must expose one shared metadata contract
at the benchmark layer. In addition to benchmark tags, the promoted
`foundation_model` metadata now carries:

- `dataset`
- `task`
- `model_family`
- `adapter_mode`
- `artifact_id`
- `preprocessing_version`

This keeps artifact identity and benchmark scenario identity on one shared
schema for comparison, regression, and provenance tooling.

Genomics Phase 4 scaffold reports also carry `dataset_provenance` so synthetic
interface validation cannot be mistaken for biological validation. The
`synthetic_genomics` scaffold is recorded with:

- `dataset_name`: `synthetic_genomics`
- `source_type`: `scaffold`
- `curation_status`: `synthetic`
- `provenance_label`: `deterministic_motif_scaffold`
- `biological_validation`: `interface_validation_only`
- `promotion_eligible`: `false`

Any custom or curated genomics source must provide its own `dataset_provenance`
payload before it can be reported through the foundation suite.

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
- supported: canonical single-cell deferral metadata that keeps
  `grn_transfer` outside Phase 3 stable promotion until a dedicated
  foundation-aware GRN harness exists
- Phase 4 scaffold: a shared `SequencePrecomputedAdapter` contract plus a
  genomics quick-suite scaffold for promoter, TFBS, and splice-site tasks
- Phase 4 scaffold: `FrozenSequenceEncoderAdapter` for in-process frozen
  sequence encoder benchmarking under `adapter_mode=frozen_encoder`
- Phase 4 scaffold: `DNABERT2PrecomputedAdapter` and
  `NucleotideTransformerPrecomputedAdapter` for aligned precomputed genomics
  artifacts, pending genomics realism and promotion evidence
- supported: deterministic DTI source contracts for Davis affinity regression
  and BioSNAP binary interaction scaffolds, including paired-input batching and
  metric packaging for regression, classification, and ranking
- supported: DTI benchmark metadata that exposes the shared paired-input
  required keys, dataset/split provenance, synthetic-scaffold promotion status,
  and metric groups for regression, classification, and ranking outputs
- supported: a shared contextual epigenomics source contract with canonical
  `sequence`, `tf_context`, `chromatin_contacts`, and `targets` keys
- supported: contextual target-semantics validation for `binary_peak_mask` and
  `chromatin_state_id`, including per-task output-class counts in benchmark
  metadata and suite reports
- supported: `ContextualEpigenomicsOperator` with one configurable code path
  for sequence-only, `+TF`, and `+TF+chromatin` modes, backed by an
  Artifex transformer and an optional structured chromatin-guidance loss
- supported: deterministic contextual epigenomics ablation benchmarks and
  suite reports for peak calling and chromatin-state prediction across
  `sequence_only`, `tf_context`, and `tf_plus_chromatin`
- supported: Calibrax-stored contextual epigenomics ablation comparisons using
  `dataset`, `task`, and `contextual_variant` as comparison axes, with explicit
  metric semantics for task quality and chromatin consistency
- not yet supported: arbitrary Geneformer checkpoint loading into DiffBio
- not yet supported: external frozen DNABERT-2 or Nucleotide Transformer
  checkpoint imports in stable APIs
- not yet supported: tokenizer interchangeability claims across upstream models
- not yet supported: protein-LM and differentiable drug-encoder integration in
  the DTI benchmark family
- not yet supported: real cell-type-resolved epigenomics datasets for the
  contextual benchmark family
- not yet supported: stable biological promotion of contextual epigenomics
  ablation gains from the synthetic contextual source alone

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
