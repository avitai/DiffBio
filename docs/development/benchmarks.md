# Benchmarks

DiffBio includes a comprehensive benchmark suite that evaluates **22 benchmarks
across 15 operator domains**, testing correctness, differentiability (gradient
flow), and throughput. The suite integrates with
[scBench](https://github.com/your-org/scbench) and
[spatialBench](https://github.com/your-org/spatialbench) evaluation task types.

---

## Quick Start

```bash
# Activate environment (required for GPU support)
source ./activate.sh

# Run all benchmarks (full mode, ~30-60 min)
python benchmarks/run_all.py

# Run in quick mode (~5 min, reduced data sizes for CI)
python benchmarks/run_all.py --quick

# Run a single domain
python benchmarks/run_all.py --domains variant

# Run multiple domains
python benchmarks/run_all.py --domains variant,singlecell,statistical

# Run a single benchmark standalone
python benchmarks/variant/variant_calling_benchmark.py
python benchmarks/variant/variant_calling_benchmark.py --quick
```

---

## Benchmark Architecture

### Directory Layout

Benchmarks are organized into **domain subdirectories** that mirror
`src/diffbio/operators/`:

```
benchmarks/
    _common.py          # Shared utilities (gradient checks, throughput, data generators)
    schema.py           # Unified BenchmarkEnvelope result format
    dashboard.py        # Terminal dashboard rendering
    regression.py       # Baseline management and regression detection
    run_all.py          # Master runner
    generate_report.py  # Markdown report generator for docs

    alignment/          # SmoothSmithWaterman
    assembly/           # GNNAssemblyNavigator, MetagenomicBinner
    drug_discovery/     # ECFP4, MACCS, Neural FP, MolNet datasets
    epigenomics/        # PeakCaller (CNN/FNO), ChromatinState
    language_models/    # TransformerEncoder, FoundationModel
    molecular_dynamics/ # ForceField, MDIntegrator
    multiomics/         # MultiOmicsVAE, SpatialDeconv, HiC, SpatialGene
    normalization/      # DifferentiableUMAP, DifferentiablePHATE
    preprocessing/      # AdapterRemoval, DuplicateFilter, ErrorCorrection
    protein/            # SecondaryStructure (DSSP-style)
    rna_structure/      # RNAFold (McCaskill partition function)
    singlecell/         # Harmony, SoftKMeans, VAE, Pseudotime, Velocity, GRN
    specialized/        # CRISPR, Ancestry, SpectralSimilarity
    statistical/        # HMM, NB-GLM, EM quantification
    variant/            # Pileup, VariantClassifier, CNVSegmentation

    results/            # JSON output (gitignored)
    baselines/          # Baseline snapshots for regression detection
```

### Shared Infrastructure (`_common.py`)

All benchmarks share utilities to enforce consistency (DRY):

| Utility | Purpose |
|---------|---------|
| `BaseBenchmarkResult` | Frozen dataclass with common result fields |
| `GradientFlowResult` | Return type for gradient checks |
| `check_gradient_flow(loss_fn, model, *args)` | Verify gradients flow through an nnx.Module |
| `measure_throughput(fn, args, n_iterations, warmup)` | Timing with JIT warmup and `block_until_ready()` |
| `save_benchmark_result(result, domain, name)` | Save JSON to `results/<domain>/` |
| `collect_platform_info()` | JAX version, device, Python version |
| `generate_synthetic_expression(...)` | Synthetic scRNA-seq data with batch effects |
| `generate_synthetic_sequences(...)` | One-hot encoded random DNA/RNA sequences |
| `generate_synthetic_coverage(...)` | Coverage signal with known peaks |

### Benchmark Pattern

Every benchmark follows the same structure:

```python
from __future__ import annotations

from dataclasses import dataclass
from benchmarks._common import (
    check_gradient_flow,
    measure_throughput,
    save_benchmark_result,
)

@dataclass(frozen=True, kw_only=True)
class MyBenchmarkResult:
    """Frozen result with all metrics."""
    shape_correct: bool
    gradient_norm: float
    gradient_nonzero: bool
    throughput: float
    # ... domain-specific fields

def run_benchmark(*, quick: bool = False) -> MyBenchmarkResult:
    """Run the benchmark. Use quick=True for CI."""
    n_items = 50 if quick else 500
    # ... create operator, run tests, return result

def main() -> None:
    result = run_benchmark()
    save_benchmark_result(asdict(result), "my_domain", "my_benchmark")

if __name__ == "__main__":
    main()
```

---

## Capabilities Matrix

### All 22 Benchmarks by Domain

| Domain | Benchmark | Operators Tested | Key Metrics |
|--------|-----------|-----------------|-------------|
| **Alignment** | `alignment_benchmark.py` | SmoothSmithWaterman | Score accuracy, gradient flow, temperature sweep, throughput |
| **Assembly** | `assembly_benchmark.py` | GNNAssemblyNavigator, MetagenomicBinner | Edge selection, binning quality, gradient flow |
| **Drug Discovery** | `molnet_benchmark.py` | ECFP4, MACCS on MoleculeNet | ROC-AUC, RMSE, R^2 on BBBP/ESOL/Lipophilicity |
| | `fingerprint_benchmark.py` | ECFP4, MACCS, Neural FP | Bit density, correlation, gradient flow |
| | `circular_fingerprint_benchmark.py` | ECFP4 vs DeepChem | Tanimoto, bit agreement, speedup |
| **Epigenomics** | `epigenomics_benchmark.py` | PeakCaller (CNN), FNOPeakCaller, ChromatinState | Peak precision/recall, CNN vs FNO, state accuracy |
| **Language Models** | `language_model_benchmark.py` | TransformerEncoder, FoundationModel | Embedding quality, gradient flow, throughput |
| **Molecular Dynamics** | `molecular_dynamics_benchmark.py` | ForceField, MDIntegrator | Energy conservation, force accuracy, trajectory |
| **Multi-omics** | `multiomics_benchmark.py` | MultiOmicsVAE, SpatialDeconv, HiC, SpatialGene | Latent quality, proportion accuracy, TAD detection |
| **Normalization** | `dimreduction_benchmark.py` | DifferentiableUMAP, DifferentiablePHATE | Cluster separation, gradient flow, throughput |
| **Preprocessing** | `preprocessing_benchmark.py` | AdapterRemoval, DuplicateFilter, ErrorCorrection | Detection accuracy, gradient flow through chain |
| **Protein** | `protein_structure_benchmark.py` | SecondaryStructure | Q3 accuracy (helix/strand/coil), H-bond quality |
| **RNA Structure** | `rna_structure_benchmark.py` | RNAFold | Base pair recovery, partition function, temperature sweep |
| **Single-Cell** | `singlecell_benchmark.py` | Harmony, SoftKMeans | Batch mixing, silhouette, clustering inertia |
| | `scvi_benchmark.py` | VAENormalizer (ZINB) | ELBO, reconstruction MSE, ARI, NMI, batch ASW |
| | `trajectory_benchmark.py` | Pseudotime, Fate, Velocity, OTTrajectory | Pseudotime ordering, fate sums, velocity shape |
| | `grn_benchmark.py` | GRN, SINDy, CellCommunication | Edge recovery, coefficient sparsity, communication scores |
| **Specialized** | `crispr_benchmark.py` | CRISPRScorer | Score shape, gradient-guided optimization |
| | `population_benchmark.py` | AncestryEstimator | Proportion accuracy, row sums |
| | `metabolomics_benchmark.py` | SpectralSimilarity | Similarity scores, embedding quality |
| **Statistical** | `statistical_benchmark.py` | HMM, NB-GLM, EM | Log-likelihood, parameter recovery, abundances |
| **Variant** | `variant_calling_benchmark.py` | Pileup, Classifier, CNVSegmentation | Pileup accuracy, classification F1, breakpoint detection |

---

## scBench / spatialBench Task Coverage

DiffBio operators cover all **8 evaluation task types** from the
[scBench](https://github.com/your-org/scbench) (30 canonical single-cell
evaluations) and [spatialBench](https://github.com/your-org/spatialbench) (146
spatial transcriptomics evaluations) benchmarks:

| Task Type | DiffBio Operator | Benchmark File |
|-----------|-----------------|----------------|
| `qc_filtering` | `DifferentiableQualityFilter` | `preprocessing/` |
| `normalization` | `VAENormalizer` | `singlecell/scvi_benchmark.py` |
| `clustering` | `SoftKMeansClustering` | `singlecell/singlecell_benchmark.py` |
| `cell_typing` | `CellAnnotator`, `SoftKMeans` | `singlecell/singlecell_benchmark.py` |
| `differential_expression` | `NB-GLM`, `DE Pipeline` | `statistical/statistical_benchmark.py` |
| `batch_correction` | `DifferentiableHarmony` | `singlecell/singlecell_benchmark.py` |
| `trajectory` | `Pseudotime`, `FateProbability` | `singlecell/trajectory_benchmark.py` |
| `spatial_analysis` | `SpatialDomain`, `SpatialDeconv` | `multiomics/multiomics_benchmark.py` |

Both scBench and spatialBench use the same 5 grader types that DiffBio's
evaluation harness (`src/diffbio/evaluation/`) already supports:

- `numeric_tolerance` -- numeric answers with tolerance windows
- `multiple_choice` -- discrete interpretation questions
- `marker_gene_precision_recall` -- gene list recovery (P@K, R@K)
- `distribution_comparison` -- cell type proportion comparison
- `label_set_jaccard` -- set matching via Jaccard index

---

## Running Benchmarks

### Master Runner (`run_all.py`)

The master runner discovers and executes all 22 benchmarks, renders a terminal
dashboard, and optionally manages baselines:

```bash
# Full suite
python benchmarks/run_all.py

# Quick mode (reduced data sizes for CI, ~5 min)
python benchmarks/run_all.py --quick

# Filter by domain(s)
python benchmarks/run_all.py --domains variant,statistical,protein

# Custom output directory
python benchmarks/run_all.py --output-dir /tmp/bench_results
```

The dashboard shows a capabilities matrix, scBench/spatialBench task coverage,
and a pass/fail/error summary:

```
================================================================================
                     DiffBio Benchmark Dashboard
                     0.1.0 | cuda:0
================================================================================

CAPABILITIES MATRIX
+--------------------+---------------------+---------+------+--------+---------+
| Domain             | Operator            | Correct | Diff | Status | Thru.   |
+--------------------+---------------------+---------+------+--------+---------+
| alignment          | SmoothSmithWaterman | 4/4     | PASS | PASS   | 6.2/s   |
| variant            | Pileup, Classifier  | 3/3     | PASS | PASS   | 5k/s    |
| ...                | ...                 | ...     | ...  | ...    | ...     |
+--------------------+---------------------+---------+------+--------+---------+

PASS: 22/22 | FAIL: 0 | ERROR: 0 | TIME: 8m 12s
```

### Single Benchmark

Each benchmark can be run standalone:

```bash
# Default (full data sizes)
python benchmarks/statistical/statistical_benchmark.py

# Quick mode
python benchmarks/statistical/statistical_benchmark.py --quick

# MolNet with specific config
python benchmarks/drug_discovery/molnet_benchmark.py --dataset bbbp --featurizer ecfp

# All MolNet combinations
python benchmarks/drug_discovery/molnet_benchmark.py --all
```

### Quick Mode

Quick mode reduces data sizes for fast CI runs (~5 min total instead of
~30-60 min). Typical reductions:

| Parameter | Full | Quick |
|-----------|------|-------|
| Cells | 500 | 50 |
| Genes | 200 | 20 |
| Sequence length | 500-1000 | 50-100 |
| Training epochs | 50-100 | 5-20 |
| Throughput iterations | 100 | 10-20 |
| Particles (MD) | 64 | 8 |

---

## Regression Detection

### Saving a Baseline

After a successful run, save the results as a baseline:

```bash
python benchmarks/run_all.py --save-baseline
# Output: benchmarks/baselines/baseline_20260330.json
```

### Checking for Regressions

Compare current results against a saved baseline:

```bash
python benchmarks/run_all.py --check-regression benchmarks/baselines/baseline_20260330.json
```

Regression thresholds (in `regression.py`):

| Check | Threshold | Severity |
|-------|-----------|----------|
| Correctness test flips pass → fail | Any flip | Error (blocks) |
| Throughput drops | > 10% | Error (blocks) |
| Gradient goes to zero | Any | Error (blocks) |

Exit code is non-zero if any error-severity regressions are detected, making
this suitable for CI.

---

## Generating Reports

### Markdown Report

Generate a Markdown report for the documentation site:

```bash
python benchmarks/generate_report.py
# Output: benchmarks/results/benchmark-report.md (gitignored)

# Or specify a custom path
python benchmarks/generate_report.py --output /tmp/report.md
```

The report includes:

- Environment table (DiffBio version, JAX, device, Python)
- Capabilities matrix with pass/fail icons
- scBench/spatialBench task coverage
- Per-domain detail sections (correctness tests, gradient norms, throughput)

### JSON Results

All benchmarks save timestamped JSON to `benchmarks/results/<domain>/`:

```json
{
  "timestamp": "2026-03-30T10:35:32",
  "hmm_log_likelihood_finite": true,
  "hmm_posteriors_sum_to_one": true,
  "hmm_gradient_norm": 10.32,
  "hmm_gradient_nonzero": true,
  "hmm_throughput_per_item_ms": 172.29,
  "hmm_throughput_items_per_sec": 5.8,
  ...
}
```

---

## Benchmark Methodology

### What Each Benchmark Tests

Every benchmark evaluates three aspects of each operator:

1. **Correctness** -- Output shapes, value ranges, mathematical properties
   (probabilities sum to 1, energies are finite, etc.)

2. **Differentiability** -- Gradient flow through the operator via
   `check_gradient_flow()`. This verifies that `nnx.grad` produces non-zero
   gradients through learnable parameters, confirming the operator is usable in
   end-to-end differentiable pipelines.

3. **Throughput** -- Items processed per second via `measure_throughput()` with
   JIT warmup and `block_until_ready()` for accurate GPU timing.

### Synthetic Data

Benchmarks use synthetic data for **reproducibility** and **zero external
dependencies**. The data generators in `_common.py` produce realistic inputs:

- **`generate_synthetic_expression()`** -- scRNA-seq counts with per-type
  expression profiles, batch effects, and negative binomial sampling. Used by
  single-cell, trajectory, GRN, and dimensionality reduction benchmarks.

- **`generate_synthetic_sequences()`** -- One-hot encoded DNA/RNA sequences.
  Used by alignment, preprocessing, and language model benchmarks.

- **`generate_synthetic_coverage()`** -- Poisson background + Gaussian peaks.
  Used by the epigenomics benchmark.

Domain-specific generators create additional data (assembly graphs, molecular
dynamics lattices, Hi-C contact matrices, etc.) within each benchmark file.

### Reproducibility

- Fixed random seeds (`jax.random.key(42)`, `nnx.Rngs(42)`)
- Deterministic data generation
- JIT warmup before timing
- `block_until_ready()` for accurate GPU measurements

---

## Adding a New Benchmark

1. **Create the file**: `benchmarks/<domain>/<name>_benchmark.py`

2. **Follow the pattern**:
    - Frozen `@dataclass(frozen=True, kw_only=True)` for results
    - `run_benchmark(*, quick: bool = False)` entry point
    - Import shared utilities from `benchmarks._common`
    - Print results during execution
    - `main()` with `--quick` flag support

3. **Register it**: Add to `_BENCHMARK_REGISTRY` in `benchmarks/run_all.py`:
    ```python
    ("my_domain", "benchmarks.my_domain.my_benchmark"),
    ```

4. **Test it**:
    ```bash
    python benchmarks/my_domain/my_benchmark.py --quick
    ```

5. **Update docs**: Add a row to the capabilities matrix table above.

### Example: Minimal Benchmark

```python
#!/usr/bin/env python3
"""Minimal benchmark template."""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from datetime import datetime

import jax.numpy as jnp
from flax import nnx

from benchmarks._common import (
    check_gradient_flow,
    measure_throughput,
    save_benchmark_result,
)


@dataclass(frozen=True, kw_only=True)
class MyResult:
    """Benchmark results."""
    timestamp: str
    shape_correct: bool
    gradient_norm: float
    gradient_nonzero: bool
    throughput_items_per_sec: float


def run_benchmark(*, quick: bool = False) -> MyResult:
    """Run the benchmark."""
    from diffbio.operators.my_domain import MyOperator, MyConfig

    n = 50 if quick else 500
    config = MyConfig(...)
    op = MyOperator(config, rngs=nnx.Rngs(42))

    # Correctness
    data = {"input": jnp.ones((n, 10))}
    result, _, _ = op.apply(data, {}, None)
    shape_ok = result["output"].shape == (n, 10)

    # Gradient flow
    def loss_fn(model, x):
        r, _, _ = model.apply(x, {}, None)
        return jnp.sum(r["output"])

    grad = check_gradient_flow(loss_fn, op, data)

    # Throughput
    tp = measure_throughput(
        lambda x: op.apply(x, {}, None),
        (data,),
        n_iterations=10 if quick else 100,
    )

    return MyResult(
        timestamp=datetime.now().isoformat(),
        shape_correct=shape_ok,
        gradient_norm=grad.gradient_norm,
        gradient_nonzero=grad.gradient_nonzero,
        throughput_items_per_sec=tp["items_per_sec"],
    )


def main() -> None:
    quick = "--quick" in sys.argv
    result = run_benchmark(quick=quick)
    print(f"Shape correct: {result.shape_correct}")
    print(f"Gradient nonzero: {result.gradient_nonzero}")
    print(f"Throughput: {result.throughput_items_per_sec:.1f}/s")
    save_benchmark_result(
        asdict(result), "my_domain", "my_benchmark"
    )


if __name__ == "__main__":
    main()
```

---

## Unified Result Schema (`schema.py`)

For integration with the dashboard and regression detection, benchmarks can
optionally produce a `BenchmarkEnvelope`:

```python
from benchmarks.schema import BenchmarkEnvelope

envelope = BenchmarkEnvelope(
    benchmark_id="variant/variant_calling",
    domain="variant",
    operators_tested=["DifferentiablePileup", "VariantClassifier"],
    timestamp="2026-03-30T14:22:01",
    platform=collect_platform_info(),
    status="pass",
    correctness={
        "passed": True,
        "tests": [
            {"name": "pileup_shape", "value": 1.0, "passed": True},
            {"name": "prob_sum_one", "value": 1.0, "passed": True},
        ],
    },
    differentiability={
        "passed": True,
        "gradient_norm": 4.12,
        "gradient_nonzero": True,
    },
    performance={
        "throughput": 5000.0,
        "throughput_unit": "positions/sec",
        "latency_ms": 0.2,
    },
    evaluation_task_types=["qc_filtering"],
)
```

---

## Results from Latest Run

!!! note "Auto-generated results"
    Run `python benchmarks/generate_report.py` to generate a report at
    `benchmarks/results/benchmark-report.md` from the latest JSON output.

### Test Environment

| Component | Value |
|-----------|-------|
| Platform | Linux 6.8.0 (Ubuntu) |
| Python | 3.12.6 |
| JAX | 0.9.0.1 |
| GPU | NVIDIA GeForce RTX 4090 (24 GB) |
| Backend | CUDA (gpu) |
