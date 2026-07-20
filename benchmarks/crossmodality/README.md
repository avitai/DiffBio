# Cross-modality reproduction benchmarks

Scripts that reproduce the experiments in the DiffBio manuscript
(`paper/diffbio.tex`) beyond the single-cell annotation result (which lives in
`benchmarks/singlecell/`). Every script uses **real data**, fits all preprocessing on
the **training split only**, reports **held-out** metrics over 3 seeds, and writes its
summary to `benchmarks/results/crossmodality/`.

Each script reads its dataset path from an environment variable with a default under
`/mnt/ssd2/Data`; override the variable to point at your own copy.

## Data sources

| Pipeline | Dataset | Source | Env var |
|---|---|---|---|
| DNA regulatory elements | Genomic Benchmarks `human_ocr_ensembl` | HuggingFace `katarinagresova/Genomic_Benchmarks_human_ocr_ensembl` (DOI 10.1186/s12863-023-01123-8) | `HF_HOME` |
| scATAC annotation | CATLAS GSE184462 | GEO `GSE184462_RAW.tar` + `GSE184462_metadata.tsv.gz` (Zhang 2021, 10.1016/j.cell.2021.10.024) | `DIFFBIO_CATLAS_DIR` |
| Perturb-seq identity | Virtual Cell Challenge Perturb-seq | challenge `adata_Training.h5ad` | `DIFFBIO_VCC_NPZ` |
| Variant calling | GIAB HG002 chr20 | `HG002.chr20.bam` + GIAB v4.2.1 truth VCF + GRCh38 chr20 FASTA (Zook 2019, 10.1038/s41587-019-0074-6) | `DIFFBIO_GIAB_DIR` |

## Data preparation

Prep scripts materialize the model-ready arrays the benchmarks consume:

```bash
python benchmarks/crossmodality/prepare_scatac.py       # CATLAS fragments -> cell x bin npz
python benchmarks/crossmodality/prepare_variant.py       # GIAB BAM+truth -> labelled pileup windows
python benchmarks/crossmodality/prepare_perturbation.py  # VCC h5ad -> target-masked npz (chunked)
```

DNA needs no prep (the HuggingFace dataset is featurized on the fly by the k-mer operator).

## Reproducing paper results

```bash
source ./activate.sh

# Table 3 / Figure 4 -- the joint gain across modalities and machineries (gain vs k)
python benchmarks/crossmodality/dna_classification.py         # DNA, +7.5pp @ k=5
python benchmarks/crossmodality/scatac_annotation.py          # scATAC, +12.2pp @ k=5
python benchmarks/crossmodality/perturbation_identity.py      # Perturb-seq, +8.3pp @ k=5
python benchmarks/crossmodality/variant_read_pooling.py       # variant (set pooling), +16.9pp @ k=5

# Figure 5a / section 4.4 depth -- scaling crossover and domain baselines
python benchmarks/crossmodality/scatac_deepening.py           # scATAC scaling + LSI+LogReg/kNN baseline
python benchmarks/crossmodality/dna_deepening.py              # DNA scaling + 6-mer+linear baseline

# Section 4.5 -- knob attribution on Perturb-seq (the projection carries the gain)
python benchmarks/crossmodality/perturbation_ablation.py

# Section 4.6 -- trainable and orthonormal reduction bases (MatrixFreePCA, Stiefel projection)
python benchmarks/crossmodality/trainable_basis.py

# Rigor audits (data leakage / overfitting)
python benchmarks/crossmodality/audit_leakfree_scatac.py      # held-out donors + train-only bins, train/test F1
python benchmarks/crossmodality/audit_overfitting.py          # 3-way train/val/test, DNA + Perturb-seq
python benchmarks/crossmodality/audit_regularization_scatac.py  # weight-decay / early-stop / Stiefel vs joint overfit
```

## Data splits (no fitting on held-out data)

Every benchmark fits **all** preprocessing (PCA / LSI / normalization / linear
baselines) on the **training split only**, then applies the fitted transform to the
held-out data. Splits:

| Pipeline | Split |
|---|---|
| scRNA annotation | 80/20 stratified by cell type |
| DNA regulatory elements | official Genomic Benchmarks disjoint train/test sequences (+15% validation carved from train for `audit_overfitting.py`) |
| Perturb-seq identity | held-out experimental **batches** (target-gene columns masked) |
| scATAC annotation | stratified 70/30 (headline); **held-out donors + train-only bin selection** for `audit_leakfree_scatac.py` |
| Variant calling | disjoint **genomic regions** (chr20:2–22 Mb train / 30–40 Mb test) |

Each script prints a per-`k` (or per-arm) summary and writes a JSON to
`benchmarks/results/crossmodality/`. Numbers are means over 3 seeds; small differences
from the paper are expected from GPU non-determinism in the convolution/scan kernels.
