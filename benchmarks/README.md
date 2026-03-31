# DiffBio Benchmarks

Real-data benchmarks evaluating DiffBio operators against published SOTA methods.
Every benchmark uses real datasets, field-standard metrics, and comparison tables
against established baselines.

## Quick Start

```bash
source ./activate.sh

# CI tier (~1 min, subsampled data)
python benchmarks/run_all.py --tier ci --quick

# Nightly tier (~30 min, all Tier 1+2)
python benchmarks/run_all.py --tier nightly

# Full suite (~2 hours)
python benchmarks/run_all.py --tier full

# Single benchmark
python benchmarks/singlecell/bench_batch_correction.py --quick
```

## Benchmarks

| Domain | Benchmark | Dataset | SOTA Baselines |
|--------|-----------|---------|----------------|
| Single-Cell | batch_correction | immune_human (33K cells) | scVI, Harmony, Scanorama |
| Single-Cell | clustering | immune_human | Leiden, Louvain, k-means |
| Single-Cell | vae_integration | immune_human | scVI, scANVI |
| Single-Cell | trajectory | pancreas (3.7K cells) | scVelo, DPT, Monocle3 |
| Single-Cell | grn | benGRN mESC (11.6K edges) | GENIE3, GRNBoost2 |
| Alignment | msa | balifam100 (59 families) | MAFFT, ClustalW, MUSCLE |
| Alignment | pairwise | balifam100 | BLAST, SSEARCH |
| RNA Structure | rna_fold | ArchiveII | ViennaRNA, LinearFold |
| Protein | secondary_structure | ideal backbones | DSSP, STRIDE |
| Mol. Dynamics | lj | 64K LJ (vs jax-md) | jax-md, LAMMPS |
| Statistical | de | immune_human | DESeq2, edgeR, Wilcoxon |

## Architecture

```
benchmarks/
    _base.py              # DiffBioBenchmark ABC (all benchmarks inherit)
    _gradient.py           # Gradient flow verification (DiffBio-specific)
    _metrics/              # Domain-specific metric bridges
        scib_bridge.py     #   scib-metrics (10+ integration metrics)
        grn.py             #   AUPRC, precision, recall
        alignment.py       #   SP score, TC score
        structure.py       #   Sensitivity, PPV, F1
    _baselines/            # Published SOTA numbers (calibrax Points)
    run_all.py             # Tier-based runner with calibrax Store
    singlecell/            # 5 single-cell benchmarks
    alignment/             # 2 alignment benchmarks
    rna_structure/         # RNA folding benchmark
    protein/               # Protein SS benchmark
    molecular_dynamics/    # LJ simulation benchmark
    statistical/           # DE benchmark
```

Every benchmark:
1. Inherits from `DiffBioBenchmark` (shared gradient check, throughput, result construction)
2. Loads real data via a datarax `DataSourceModule`
3. Computes field-standard quality metrics
4. Compares against published SOTA baselines (calibrax `Point` objects)
5. Returns a `calibrax.core.result.BenchmarkResult`

## Datasets

Download datasets before running:

```bash
# Single-cell (immune_human, 2GB)
mkdir -p /media/mahdi/ssd23/Data/scib
wget -O /media/mahdi/ssd23/Data/scib/Immune_ALL_human.h5ad \
    "https://ndownloader.figshare.com/files/25717328"

# Trajectory (pancreas, 51MB)
mkdir -p /media/mahdi/ssd23/Data/scvelo
wget -O /media/mahdi/ssd23/Data/scvelo/endocrinogenesis_day15.h5ad \
    "https://github.com/theislab/scvelo_notebooks/raw/master/data/Pancreas/endocrinogenesis_day15.h5ad"
```

Other datasets (balifam, ArchiveII, benGRN) are loaded from locally
cloned repos under `../`.

## Adding a New Benchmark

1. Create `benchmarks/<domain>/bench_<name>.py`
2. Inherit from `DiffBioBenchmark`, implement `_run_core()`
3. Create a DataSource in `src/diffbio/sources/` if needed
4. Add baselines to `_baselines/<domain>.py`
5. Write tests in `tests/benchmarks/test_bench_<name>.py`
6. Register in `run_all.py` tier registry
