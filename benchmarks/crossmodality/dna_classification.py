"""Case study B: genomic sequence classification, frozen PCA vs learnable projection.

Featurize sequences as canonical 6-mer spectra (the frozen frontend), then compare a
frozen-PCA reduction against a PCA-initialized learnable projection, jointly trained with
the classifier -- the exact two-arm test from the scRNA result, on DNA sequences.
"""

import json
import os
import sys

import jax.numpy as jnp
import numpy as np
from calibrax.metrics.functional.classification import f1_score
from flax import nnx

_DATA = os.environ.get("DIFFBIO_DATA_ROOT", "/mnt/ssd2/Data")
os.environ.setdefault("HF_HOME", f"{_DATA}/huggingface")
from datasets import load_dataset  # noqa: E402

from diffbio.reductions import fit_pca_reduction  # noqa: E402
from benchmarks.singlecell._gate2_arms import (  # noqa: E402
    _embedding_probe,
    _probe_forward,
    _project_probe_forward,
    _ProjectionProbe,
)
from diffbio.operators.normalization.learnable_projection import (  # noqa: E402
    LearnableProjection,
    LearnableProjectionConfig,
)
from diffbio.pipelines.minibatch_training import MiniBatchConfig, train_minibatch  # noqa: E402
from diffbio.sequences.kmer import kmer_featurize  # noqa: E402


DATASET = sys.argv[1] if len(sys.argv) > 1 else "demo_human_or_worm"
K = 6
SEEDS = (0, 1, 2)
K_VALUES = [5, 10, 20, 50]

ds = load_dataset(f"katarinagresova/Genomic_Benchmarks_{DATASET}")  # nosec B615
train_seq = list(ds["train"]["seq"])
test_seq = list(ds["test"]["seq"])
train_y = np.asarray(ds["train"]["label"], np.int32)
test_y = np.asarray(ds["test"]["label"], np.int32)
n_classes = int(max(train_y.max(), test_y.max())) + 1
print(f"[{DATASET}] train {len(train_seq)} test {len(test_seq)} classes {n_classes}", flush=True)

x_train = kmer_featurize(train_seq, K, canonical=True)
x_test = kmer_featurize(test_seq, K, canonical=True)
print(f"featurized: train {x_train.shape} test {x_test.shape}", flush=True)


def macro(pred, true):
    """Return the macro-averaged F1 of predictions against ground truth."""
    return float(f1_score(jnp.asarray(pred), jnp.asarray(true), average="macro"))


frozen = {k: [] for k in K_VALUES}
joint = {k: [] for k in K_VALUES}
for seed in SEEDS:
    reduction = fit_pca_reduction(x_train, 50)
    tr_c = reduction.scaled(x_train) - reduction.pca_mean
    te_c = reduction.scaled(x_test) - reduction.pca_mean
    n_features = reduction.loadings.shape[0]
    cfg = MiniBatchConfig(
        batch_size=1024, n_epochs=60, learning_rate=1e-2, weight_decay=5e-2, seed=seed
    )
    for k in K_VALUES:
        loadings_k = reduction.loadings[:, :k]
        xtr_f = jnp.asarray(tr_c @ loadings_k)
        xte_f = jnp.asarray(te_c @ loadings_k)
        probe_f = _embedding_probe(k, n_classes, 128, seed)
        train_minibatch(probe_f, _probe_forward, xtr_f, train_y, n_classes=n_classes, config=cfg)
        frozen[k].append(macro(np.asarray(jnp.argmax(_probe_forward(probe_f, xte_f), -1)), test_y))
        proj = LearnableProjection(
            LearnableProjectionConfig(n_genes=n_features, n_components=k),
            init_loadings=loadings_k,
            rngs=nnx.Rngs(seed),
        )
        model = _ProjectionProbe(proj, _embedding_probe(k, n_classes, 128, seed))
        train_minibatch(
            model,
            _project_probe_forward,
            jnp.asarray(tr_c),
            train_y,
            n_classes=n_classes,
            config=cfg,
        )
        joint[k].append(
            macro(
                np.asarray(jnp.argmax(_project_probe_forward(model, jnp.asarray(te_c)), -1)), test_y
            )
        )
        print(
            f"  [seed {seed}] k={k:2d} frozen {frozen[k][-1]:.4f} joint {joint[k][-1]:.4f}",
            flush=True,
        )

print(f"=== {DATASET} SUMMARY ===", flush=True)
for k in K_VALUES:
    fm, jm = np.mean(frozen[k]), np.mean(joint[k])
    print(
        f"k={k:2d}: frozen {fm:.4f}±{np.std(frozen[k]):.4f}  "
        f"joint {jm:.4f}±{np.std(joint[k]):.4f}  gain {100 * (jm - fm):+.1f}pp",
        flush=True,
    )

out = f"benchmarks/results/crossmodality/gate_sequence_{DATASET}.json"
with open(out, "w") as h:
    json.dump(
        {
            "k_values": K_VALUES,
            "frozen": {str(k): [float(np.mean(v)), float(np.std(v))] for k, v in frozen.items()},
            "joint": {str(k): [float(np.mean(v)), float(np.std(v))] for k, v in joint.items()},
        },
        h,
        indent=2,
    )
print("SEQ DONE", flush=True)
