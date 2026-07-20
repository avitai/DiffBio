"""Prepare the VCC Perturb-seq atlas for case study D (chunked, memory-lean).

Two backed passes over 20k-cell chunks: pass 1 accumulates per-gene dispersion stats;
pass 2 densifies the top-2000 dispersion genes. Target-gene columns are dropped
(target-masked). Never materializes the full matrix or its COO.
"""

import os

import anndata as ad
import numpy as np
import scipy.sparse as sp


N_GENES = 2000
CHUNK = 20000
_DATA = os.environ.get("DIFFBIO_DATA_ROOT", "/mnt/ssd2/Data")
OUT = f"{_DATA}/Virtual-Cell-Challenge/vcc_masked.npz"

adata = ad.read_h5ad(f"{_DATA}/Virtual-Cell-Challenge/vcc_data/adata_Training.h5ad", backed="r")
var_names = adata.var_names.astype(str).to_numpy()
target = adata.obs["target_gene"].astype(str).to_numpy()
batch = adata.obs["batch"].astype(str).to_numpy()
n_cells = adata.shape[0]

target_genes = {t for t in np.unique(target) if t != "non-targeting"}
keep_cols = np.array([g not in target_genes for g in var_names])
n_kept = int(keep_cols.sum())
print(
    f"cells {n_cells}, kept genes {n_kept} (dropped {int((~keep_cols).sum())} targets)", flush=True
)


def _chunk(start: int) -> sp.csr_matrix:
    block = adata.X[start : start + CHUNK]
    block = block.tocsr() if sp.issparse(block) else sp.csr_matrix(block)
    return block[:, keep_cols].tocsr()


gene_sum = np.zeros(n_kept)
gene_sumsq = np.zeros(n_kept)
for start in range(0, n_cells, CHUNK):
    block = _chunk(start)
    library = np.asarray(block.sum(1)).ravel()
    library[library == 0] = 1.0
    coo = block.tocoo()
    vals = np.log1p(coo.data * (1.0e4 / library)[coo.row])
    gene_sum += np.bincount(coo.col, weights=vals, minlength=n_kept)
    gene_sumsq += np.bincount(coo.col, weights=vals**2, minlength=n_kept)
print("pass 1 (dispersion) done", flush=True)

mean = gene_sum / n_cells
variance = gene_sumsq / n_cells - mean**2
dispersion = variance / np.where(mean == 0.0, 1.0, mean)
selected = np.sort(np.argsort(dispersion)[::-1][:N_GENES])

blocks = [
    np.asarray(_chunk(start)[:, selected].todense(), dtype=np.float32)
    for start in range(0, n_cells, CHUNK)
]
counts_dense = np.vstack(blocks)
print(f"pass 2 (densify) done -> {counts_dense.shape}", flush=True)

pert_uniq, pert_labels = np.unique(target, return_inverse=True)
batch_uniq, batch_labels = np.unique(batch, return_inverse=True)
np.savez(
    OUT,
    counts=counts_dense,
    labels=pert_labels.astype(np.int32),
    batch=batch_labels.astype(np.int32),
    n_types=len(pert_uniq),
    n_batches=len(batch_uniq),
)
print(f"saved {OUT}: {len(pert_uniq)} perturbations, {len(batch_uniq)} batches", flush=True)
