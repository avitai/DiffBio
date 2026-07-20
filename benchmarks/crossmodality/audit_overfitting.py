"""Overfitting audit: 3-way train / validation / test macro-F1 for frozen vs joint arms.

For DNA (Genomic Benchmarks, disjoint train/test sequences) and Perturb-seq (VCC, split by
experimental batch), we carve a validation set held out from training, fit all preprocessing
on the training portion only, and report macro-F1 on train, validation, and test for both
the frozen-PCA and joint-projection arms. A large train-vs-{val,test} gap signals overfitting;
comparing the joint arm's gap to the frozen arm's shows whether learning the projection
overfits more than the fixed baseline.
"""

import json
import os

import jax.numpy as jnp
import numpy as np
from calibrax.metrics.functional.classification import f1_score
from flax import nnx

_DATA = os.environ.get("DIFFBIO_DATA_ROOT", "/mnt/ssd2/Data")
os.environ.setdefault("HF_HOME", f"{_DATA}/huggingface")
from datasets import load_dataset  # noqa: E402

from benchmarks.singlecell._gate2_arms import (  # noqa: E402
    _embedding_probe,
    _probe_forward,
    _project_probe_forward,
    _ProjectionProbe,
)
from benchmarks.singlecell.frozen_annotation_baseline import fit_frozen_preprocess  # noqa: E402
from diffbio.operators.normalization.learnable_projection import (  # noqa: E402
    LearnableProjection,
    LearnableProjectionConfig,
)
from diffbio.pipelines.minibatch_training import MiniBatchConfig, train_minibatch  # noqa: E402
from diffbio.reductions import fit_pca_reduction  # noqa: E402
from diffbio.sequences.kmer import kmer_featurize  # noqa: E402

OUT = "benchmarks/results/crossmodality/audit_overfit.json"
K = 5
SEEDS = (0, 1, 2)


def macro(pred, true):
    """Return the macro-averaged F1 of predictions against ground truth."""
    return float(
        f1_score(jnp.asarray(np.asarray(pred)), jnp.asarray(np.asarray(true)), average="macro")
    )


def _frozen(reduction, splits, ys, n_classes, seed):
    """Frozen-PCA probe: return {split: macro-F1} over train/val/test."""
    loadings_k = reduction.loadings[:, :K]
    proj = {
        name: jnp.asarray(
            np.asarray((reduction.scaled(x) - reduction.pca_mean) @ loadings_k, np.float32)
        )
        for name, x in splits.items()
    }
    probe = _embedding_probe(K, n_classes, 128, seed)
    cfg = MiniBatchConfig(
        batch_size=1024, n_epochs=60, learning_rate=1e-2, weight_decay=5e-2, seed=seed
    )
    train_minibatch(
        probe, _probe_forward, proj["train"], ys["train"], n_classes=n_classes, config=cfg
    )
    return {n: macro(jnp.argmax(_probe_forward(probe, proj[n]), -1), ys[n]) for n in splits}


def _joint(reduction, splits, ys, n_classes, seed, n_features):
    """Joint-projection probe: return {split: macro-F1} over train/val/test."""
    loadings_k = reduction.loadings[:, :K]
    centered = {
        name: jnp.asarray(reduction.scaled(x) - reduction.pca_mean) for name, x in splits.items()
    }
    model = _ProjectionProbe(
        LearnableProjection(
            LearnableProjectionConfig(n_genes=n_features, n_components=K),
            init_loadings=loadings_k,
            rngs=nnx.Rngs(seed),
        ),
        _embedding_probe(K, n_classes, 128, seed),
    )
    cfg = MiniBatchConfig(
        batch_size=1024, n_epochs=60, learning_rate=1e-2, weight_decay=5e-2, seed=seed
    )
    train_minibatch(
        model,
        _project_probe_forward,
        centered["train"],
        ys["train"],
        n_classes=n_classes,
        config=cfg,
    )
    return {
        n: macro(jnp.argmax(_project_probe_forward(model, centered[n]), -1), ys[n]) for n in splits
    }


def run_pipeline(name, splits_x, ys, n_classes, reduction_fn):
    """Fit the reduction on the train split only, then evaluate both arms on all splits."""
    print(
        f"--- {name}: train {splits_x['train'].shape[0]} val {splits_x['val'].shape[0]} "
        f"test {splits_x['test'].shape[0]} | {n_classes} classes ---",
        flush=True,
    )
    out = {"frozen": {s: [] for s in splits_x}, "joint": {s: [] for s in splits_x}}
    for seed in SEEDS:
        reduction = reduction_fn(splits_x["train"])
        n_features = reduction.loadings.shape[0]
        fr = _frozen(reduction, splits_x, ys, n_classes, seed)
        jo = _joint(reduction, splits_x, ys, n_classes, seed, n_features)
        for s in splits_x:
            out["frozen"][s].append(fr[s])
            out["joint"][s].append(jo[s])
        print(
            f"  [seed {seed}] frozen tr {fr['train']:.3f}/val {fr['val']:.3f}/te {fr['test']:.3f}  "
            f"joint tr {jo['train']:.3f}/val {jo['val']:.3f}/te {jo['test']:.3f}",
            flush=True,
        )
    summary = {
        arm: {s: float(np.mean(out[arm][s])) for s in splits_x} for arm in ("frozen", "joint")
    }
    for arm in ("frozen", "joint"):
        tr, va, te = summary[arm]["train"], summary[arm]["val"], summary[arm]["test"]
        print(
            f"  {name} {arm}: train {tr:.3f} val {va:.3f} test {te:.3f} "
            f"(train-test gap {100 * (tr - te):.1f}pp)",
            flush=True,
        )
    return summary


def dna_splits():
    """DNA: HF disjoint train/test; carve a stratified 15% validation from train."""
    ds = load_dataset("katarinagresova/Genomic_Benchmarks_human_ocr_ensembl")  # nosec B615
    ytr = np.asarray(ds["train"]["label"], np.int32)
    yte = np.asarray(ds["test"]["label"], np.int32)
    xtr = kmer_featurize(list(ds["train"]["seq"]), 6, canonical=True)
    xte = kmer_featurize(list(ds["test"]["seq"]), 6, canonical=True)
    rng = np.random.default_rng(0)
    val_mask = np.zeros(len(ytr), bool)
    for c in np.unique(ytr):
        idx = np.where(ytr == c)[0]
        rng.shuffle(idx)
        val_mask[idx[: int(0.15 * len(idx))]] = True
    splits = {
        "train": np.asarray(xtr)[~val_mask],
        "val": np.asarray(xtr)[val_mask],
        "test": np.asarray(xte),
    }
    ys = {"train": ytr[~val_mask], "val": ytr[val_mask], "test": yte}
    return splits, ys, int(max(ytr.max(), yte.max())) + 1


def vcc_splits():
    """Perturb-seq: split by experimental batch into disjoint train/val/test batch sets."""
    data = np.load(f"{_DATA}/Virtual-Cell-Challenge/vcc_masked.npz")
    counts = np.asarray(data["counts"], np.float32)
    labels = np.asarray(data["labels"], np.int32)
    batch = np.asarray(data["batch"], np.int32)
    n_batches = int(data["n_batches"])
    test_b = set(range(0, n_batches, 5))
    val_b = set(range(2, n_batches, 5))
    test_m = np.isin(batch, list(test_b))
    val_m = np.isin(batch, list(val_b)) & ~test_m
    train_m = ~test_m & ~val_m
    splits = {"train": counts[train_m], "val": counts[val_m], "test": counts[test_m]}
    ys = {"train": labels[train_m], "val": labels[val_m], "test": labels[test_m]}
    return splits, ys, int(data["n_types"])


def main() -> None:
    """Run the DNA and Perturb-seq 3-way overfitting audits."""
    results = {}
    dna_x, dna_y, dna_nc = dna_splits()
    results["DNA"] = run_pipeline("DNA", dna_x, dna_y, dna_nc, lambda x: fit_pca_reduction(x, K))
    vcc_x, vcc_y, vcc_nc = vcc_splits()
    results["Perturb-seq"] = run_pipeline(
        "Perturb-seq",
        vcc_x,
        vcc_y,
        vcc_nc,
        lambda x: fit_frozen_preprocess(x, n_top_genes=2000, n_components=50),
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(results, handle, indent=2)
    print("OVERFIT AUDIT DONE", flush=True)


if __name__ == "__main__":
    main()
