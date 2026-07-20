"""Case study A build: real CATLAS (GSE184462) scATAC fragments -> cell x bin matrix.

Pools the largest labeled samples (each with a matching fragment file), builds a 5kb
genome-wide tile matrix per sample with snapatac2 (the domain-standard scATAC pipeline),
concatenates, selects the top-N most-accessible bins, keeps cell types with enough cells,
and saves a sparse cell x bin count matrix + cell-type labels + sample ids. Labels come
from the atlas metadata joined on barcode (cellID = ``{sample}+{barcode}``). No synthetic
data: every count is a real Tn5 insertion, every label a real annotated cell type.
"""

import os

import gzip
import re
import tarfile
from collections import Counter

import numpy as np
import scipy.sparse as sp
import snapatac2 as snap


_DATA = os.environ.get("DIFFBIO_DATA_ROOT", "/mnt/ssd2/Data")
TAR = f"{_DATA}/catlas/GSE184462_RAW.tar"
META = f"{_DATA}/catlas/GSE184462_metadata.tsv.gz"
FRAG_DIR = f"{_DATA}/catlas/frags"
H5_DIR = f"{_DATA}/catlas/h5ad"
OUT_COUNTS = f"{_DATA}/catlas/atac_counts.npz"
OUT_META = f"{_DATA}/catlas/atac_meta.npz"

BIN_SIZE = 5000
TOP_BINS = 12000
N_SAMPLES = 16
MIN_TYPE_CELLS = 500


def frag_file_map() -> dict[str, str]:
    """Map metadata sample name -> fragment file member name inside the tar."""
    mapping = {}
    with tarfile.open(TAR) as tar:
        for name in tar.getnames():
            match = re.match(r"GSM\d+_(.+)_rep1_fragments\.bed\.gz", name)
            if match:
                mapping[match.group(1) + "_1"] = name
    return mapping


def sample_barcodes() -> dict[str, dict[str, str]]:
    """Return {sample: {barcode: cell_type}} for every labeled cell."""
    per_sample: dict[str, dict[str, str]] = {}
    with gzip.open(META, "rt") as handle:
        next(handle)
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            sample, barcode = fields[1], fields[0].split("+", 1)[1]
            per_sample.setdefault(sample, {})[barcode] = fields[6]
    return per_sample


def build() -> None:
    """Build and save the pooled scATAC cell x bin matrix with labels."""
    frag_map = frag_file_map()
    per_sample = sample_barcodes()
    usable = {s: bc for s, bc in per_sample.items() if s in frag_map}
    chosen = sorted(usable, key=lambda s: len(usable[s]), reverse=True)[:N_SAMPLES]
    print(
        f"chosen {len(chosen)} samples, {sum(len(usable[s]) for s in chosen)} labeled cells",
        flush=True,
    )

    members = [frag_map[s] for s in chosen]
    with tarfile.open(TAR) as tar:
        for member in members:
            tar.extract(member, path=FRAG_DIR, filter="data")
    print("extracted fragment files", flush=True)

    blocks, labels, samples, var_names = [], [], [], None
    for sample in chosen:
        member = frag_map[sample]
        frag_path = f"{FRAG_DIR}/{member}"
        barcodes = usable[sample]
        adata = snap.pp.import_fragments(
            frag_path,
            chrom_sizes=snap.genome.hg38,
            sorted_by_barcode=True,
            whitelist=list(barcodes),
            min_num_fragments=200,
            file=f"{H5_DIR}/{sample}.h5ad",
        )
        snap.pp.add_tile_matrix(adata, bin_size=BIN_SIZE)
        counts = adata.X[:].tocsr()
        cell_barcodes = np.asarray(adata.obs_names)
        current_var = np.asarray(adata.var_names)
        adata.close()
        if var_names is None:
            var_names = current_var
        elif not np.array_equal(current_var, var_names):
            raise ValueError(f"bin grid mismatch for {sample}")
        blocks.append(counts)
        labels.extend(barcodes[b] for b in cell_barcodes)
        samples.extend([sample] * len(cell_barcodes))
        print(f"  {sample}: {counts.shape[0]} cells", flush=True)

    matrix = sp.vstack(blocks, format="csr")
    labels = np.asarray(labels)
    samples = np.asarray(samples)
    print(f"pooled matrix {matrix.shape}", flush=True)

    # Keep cell types with enough cells for a stable classification benchmark.
    type_counts = Counter(labels)
    keep_type = np.array([type_counts[t] >= MIN_TYPE_CELLS for t in labels])
    matrix, labels, samples = matrix[keep_type], labels[keep_type], samples[keep_type]
    print(
        f"after cell-type filter (>= {MIN_TYPE_CELLS}): {matrix.shape}, "
        f"{len(set(labels))} cell types",
        flush=True,
    )

    # Unsupervised feature selection: the most broadly accessible bins.
    accessibility = np.asarray((matrix > 0).sum(0)).ravel()
    top = np.sort(np.argsort(accessibility)[::-1][:TOP_BINS])
    matrix = matrix[:, top].tocsr()
    print(f"selected top {TOP_BINS} bins -> {matrix.shape}", flush=True)

    type_names, label_ids = np.unique(labels, return_inverse=True)
    sample_names, sample_ids = np.unique(samples, return_inverse=True)
    sp.save_npz(OUT_COUNTS, matrix)
    np.savez(
        OUT_META,
        labels=label_ids.astype(np.int32),
        samples=sample_ids.astype(np.int32),
        type_names=type_names,
        sample_names=sample_names,
        bins=var_names[top],
    )
    print(
        f"saved {OUT_COUNTS} + {OUT_META}: {matrix.shape[0]} cells, "
        f"{len(type_names)} types, {len(sample_names)} samples",
        flush=True,
    )
    print("ATAC BUILD DONE", flush=True)


if __name__ == "__main__":
    build()
