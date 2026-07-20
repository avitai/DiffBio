"""Leak-free scATAC audit: held-out DONORS + train-only feature selection + train/test F1.

The headline scATAC run selected the top-12000 bins on all cells and split cells randomly
(train and test share donors). This rerun removes both: it splits by donor (test donors are
never seen in training), selects the top bins using TRAIN cells only, fits the LSI on train
only, and reports TRAIN and TEST macro-F1 for the frozen and joint arms so any overfitting
is visible. If the gain-vs-k signature survives here, it is not a leakage or overfitting
artifact.
"""

import glob
import gzip
import json
import os
import re
import tempfile

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import snapatac2 as snap
from calibrax.metrics.functional.classification import f1_score
from flax import nnx

from benchmarks.singlecell._gate2_arms import (
    _embedding_probe,
    _probe_forward,
    _project_probe_forward,
    _ProjectionProbe,
)
from diffbio.operators.normalization.learnable_projection import (
    LearnableProjection,
    LearnableProjectionConfig,
)
from diffbio.pipelines.minibatch_training import MiniBatchConfig, train_minibatch
from diffbio.reductions import fit_tfidf_reduction


_DATA = os.environ.get("DIFFBIO_DATA_ROOT", "/mnt/ssd2/Data")
META_TSV = f"{_DATA}/catlas/GSE184462_metadata.tsv.gz"
FRAG_DIR = f"{_DATA}/catlas/frags"
SAMPLE_NAMES = [str(s) for s in np.load(f"{_DATA}/catlas/atac_meta.npz")["sample_names"]]
OUT = "benchmarks/results/crossmodality/audit_atac_leakfree.json"
TOP_BINS = 12000
K_VALUES = [5, 10, 20]
SEEDS = (0, 1, 2)
TEST_DONORS = {SAMPLE_NAMES[i] for i in (2, 6, 10, 14)}  # 4 held-out donors, spread across tissues
MIN_TRAIN, MIN_TEST = 200, 50


def barcode_types() -> dict[str, str]:
    """Map cellID (``sample+barcode``) -> cell type for the chosen donors."""
    keep = set(SAMPLE_NAMES)
    mapping: dict[str, str] = {}
    with gzip.open(META_TSV, "rt") as handle:
        next(handle)
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if fields[1] in keep:
                mapping[fields[0]] = fields[6]
    return mapping


def _frag_map() -> dict[str, str]:
    """Map sample name -> extracted fragment file path."""
    mapping = {}
    for path in glob.glob(f"{FRAG_DIR}/*.bed.gz"):
        match = re.match(r"GSM\d+_(.+)_rep1_fragments\.bed\.gz", os.path.basename(path))
        if match:
            mapping[match.group(1) + "_1"] = path
    return mapping


def load_matrix():
    """Rebuild the tile matrices in-process from fragments; return (counts, labels, donor)."""
    cell_types = barcode_types()
    frags = _frag_map()
    per_sample: dict[str, list[str]] = {}
    for cell_id in cell_types:
        sample, barcode = cell_id.split("+", 1)
        per_sample.setdefault(sample, []).append(barcode)

    blocks, labels, donors = [], [], []
    for sample in SAMPLE_NAMES:
        adata = snap.pp.import_fragments(
            frags[sample],
            chrom_sizes=snap.genome.hg38,
            sorted_by_barcode=True,
            whitelist=per_sample[sample],
            min_num_fragments=200,
            file=os.path.join(tempfile.gettempdir(), f"_audit_{sample}.h5ad"),
        )
        snap.pp.add_tile_matrix(adata, bin_size=5000)
        counts = adata.X[:].tocsr()
        barcodes = np.asarray(adata.obs_names)
        adata.close()
        blocks.append(counts)
        labels.extend(cell_types[f"{sample}+{bc}"] for bc in barcodes)
        donors.extend([sample] * len(barcodes))
    return sp.vstack(blocks, format="csr"), np.asarray(labels), np.asarray(donors)


def macro(pred, true):
    """Return the macro-averaged F1 of predictions against ground truth."""
    return float(
        f1_score(jnp.asarray(np.asarray(pred)), jnp.asarray(np.asarray(true)), average="macro")
    )


def main() -> None:
    """Run the leak-free donor-split scATAC benchmark with train/test reporting."""
    counts, labels, donors = load_matrix()
    test_mask = np.isin(donors, list(TEST_DONORS))
    print(
        f"cells {counts.shape} | train {int((~test_mask).sum())} test {int(test_mask.sum())} "
        f"| test donors {sorted(TEST_DONORS)}",
        flush=True,
    )

    # Cell types present with enough cells in BOTH train and test donors (else unclassifiable).
    tr_lab, te_lab = labels[~test_mask], labels[test_mask]
    import collections

    tr_ct, te_ct = collections.Counter(tr_lab), collections.Counter(te_lab)
    keep_types = {t for t in tr_ct if tr_ct[t] >= MIN_TRAIN and te_ct.get(t, 0) >= MIN_TEST}
    keep = np.isin(labels, list(keep_types))
    counts, labels, test_mask = counts[keep], labels[keep], test_mask[keep]
    type_names, label_ids = np.unique(labels, return_inverse=True)
    n_types = len(type_names)
    tr_counts, te_counts = counts[~test_mask], counts[test_mask]
    tr_y, te_y = label_ids[~test_mask], label_ids[test_mask]
    print(
        f"after shared-type filter: {counts.shape}, {n_types} types "
        f"(train {tr_counts.shape[0]} test {te_counts.shape[0]})",
        flush=True,
    )

    # TRAIN-ONLY feature selection: top bins by accessibility over TRAIN cells only.
    accessibility = np.asarray((tr_counts > 0).sum(0)).ravel()
    top = np.sort(np.argsort(accessibility)[::-1][:TOP_BINS])
    tr_counts, te_counts = tr_counts[:, top].tocsr(), te_counts[:, top].tocsr()
    print(f"selected top {TOP_BINS} bins on TRAIN only -> {tr_counts.shape}", flush=True)

    reduction = fit_tfidf_reduction(tr_counts, max(K_VALUES))
    tr_scaled, te_scaled = reduction.scaled(tr_counts), reduction.scaled(te_counts)
    tr_dense = jnp.asarray(np.asarray(tr_scaled.todense(), np.float32))
    te_dense = jnp.asarray(np.asarray(te_scaled.todense(), np.float32))
    n_features = reduction.loadings.shape[0]

    results = {
        k: {"frozen_tr": [], "frozen_te": [], "joint_tr": [], "joint_te": []} for k in K_VALUES
    }
    for seed in SEEDS:
        cfg = MiniBatchConfig(
            batch_size=2048, n_epochs=60, learning_rate=1e-2, weight_decay=5e-2, seed=seed
        )
        for k in K_VALUES:
            loadings_k = reduction.loadings[:, :k]
            xtr_f = jnp.asarray(np.asarray(tr_scaled @ loadings_k, np.float32))
            xte_f = jnp.asarray(np.asarray(te_scaled @ loadings_k, np.float32))
            probe = _embedding_probe(k, n_types, 128, seed)
            train_minibatch(probe, _probe_forward, xtr_f, tr_y, n_classes=n_types, config=cfg)
            results[k]["frozen_tr"].append(
                macro(jnp.argmax(_probe_forward(probe, xtr_f), -1), tr_y)
            )
            results[k]["frozen_te"].append(
                macro(jnp.argmax(_probe_forward(probe, xte_f), -1), te_y)
            )

            proj = LearnableProjection(
                LearnableProjectionConfig(n_genes=n_features, n_components=k),
                init_loadings=loadings_k,
                rngs=nnx.Rngs(seed),
            )
            model = _ProjectionProbe(proj, _embedding_probe(k, n_types, 128, seed))
            train_minibatch(
                model, _project_probe_forward, tr_dense, tr_y, n_classes=n_types, config=cfg
            )
            results[k]["joint_tr"].append(
                macro(jnp.argmax(_project_probe_forward(model, tr_dense), -1), tr_y)
            )
            results[k]["joint_te"].append(
                macro(jnp.argmax(_project_probe_forward(model, te_dense), -1), te_y)
            )
            print(
                f"  [seed {seed}] k={k:2d} frozen tr {results[k]['frozen_tr'][-1]:.3f}/te "
                f"{results[k]['frozen_te'][-1]:.3f}  joint tr {results[k]['joint_tr'][-1]:.3f}/te "
                f"{results[k]['joint_te'][-1]:.3f}",
                flush=True,
            )

    print("=== LEAK-FREE scATAC (held-out donors, train-only bins) SUMMARY ===", flush=True)
    summary = {}
    for k in K_VALUES:
        ftr, fte = np.mean(results[k]["frozen_tr"]), np.mean(results[k]["frozen_te"])
        jtr, jte = np.mean(results[k]["joint_tr"]), np.mean(results[k]["joint_te"])
        summary[k] = {
            "frozen_train": ftr,
            "frozen_test": fte,
            "joint_train": jtr,
            "joint_test": jte,
            "gain_pp": 100 * (jte - fte),
        }
        print(
            f"k={k:2d}: frozen train {ftr:.3f} test {fte:.3f} (gap {100 * (ftr - fte):.1f}pp) | "
            f"joint train {jtr:.3f} test {jte:.3f} (gap {100 * (jtr - jte):.1f}pp) | "
            f"GAIN {100 * (jte - fte):+.1f}pp",
            flush=True,
        )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump({str(k): v for k, v in summary.items()}, handle, indent=2)
    print("LEAKFREE DONE", flush=True)


if __name__ == "__main__":
    main()
