"""Demonstration: differentiable / orthonormal PCA bases as trainable reductions.

Validates the two new operators end-to-end on real DNA regulatory-element data, against
the frozen-PCA baseline and the existing unconstrained learnable projection, at aggressive
reduction (k=5):

  Section 1 (mini-batch, directly comparable to the paper's two-arm protocol):
    - frozen PCA                    : fixed top-k loadings, only the probe trains.
    - LearnableProjection           : PCA-initialized unconstrained residual (existing).
    - LearnableOrthogonalProjection : PCA-initialized residual kept ORTHONORMAL by a QR
                                      retraction (Stiefel manifold; eigensolver-free).

  Section 2 (full-batch, the "trainable basis" future-work item):
    - frozen MatrixFreePCA          : basis computed once from the (unscaled) data.
    - learnable-scale + MatrixFreePCA: a learnable per-feature scale reshapes the data and
                                      the PCA basis is RECOMPUTED differentiably each step
                                      (matrix-free subspace iteration), so gradients flow
                                      through the eigensolver to the scale -- the basis
                                      itself is trainable, with no fixed anchor.
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

from benchmarks.singlecell._gate2_arms import _embedding_probe, _probe_forward  # noqa: E402
from diffbio.operators.normalization.learnable_orthogonal_projection import (  # noqa: E402
    LearnableOrthogonalProjection,
    LearnableOrthogonalProjectionConfig,
)
from diffbio.operators.normalization.learnable_projection import (  # noqa: E402
    LearnableProjection,
    LearnableProjectionConfig,
)
from diffbio.operators.normalization.matrix_free_pca import (  # noqa: E402
    MatrixFreePCA,
    MatrixFreePCAConfig,
)
from diffbio.pipelines.minibatch_training import MiniBatchConfig, train_minibatch  # noqa: E402
from diffbio.reductions import fit_pca_reduction  # noqa: E402
from diffbio.sequences.kmer import kmer_featurize  # noqa: E402


OUT = "benchmarks/results/crossmodality/gate_diffpca.json"
DATASET = "human_ocr_ensembl"
KMER, K, SEEDS = 6, 5, (0, 1, 2)
FULL_BATCH_CELLS = 20000


def macro(pred, true):
    """Return the macro-averaged F1 of predictions against ground truth."""
    return float(
        f1_score(jnp.asarray(np.asarray(pred)), jnp.asarray(np.asarray(true)), average="macro")
    )


class _ProjProbe(nnx.Module):
    """Any projection operator (writes ``projection``) followed by the classifier probe."""

    def __init__(self, projection: nnx.Module, probe: nnx.Module) -> None:
        self.projection = projection
        self.probe = probe

    def logits(self, features: jnp.ndarray) -> jnp.ndarray:
        projected = self.projection.apply({"features": features}, {}, None)[0]["projection"]
        return _probe_forward(self.probe, projected)


def _proj_forward(model: _ProjProbe, features: jnp.ndarray) -> jnp.ndarray:
    return model.logits(features)


class _DiffPCAProbe(nnx.Module):
    """Learnable per-feature scale -> differentiable matrix-free PCA basis -> probe."""

    def __init__(self, n_features: int, learnable: bool, probe: nnx.Module) -> None:
        cls = nnx.Param if learnable else nnx.Variable
        self.log_scale = cls(jnp.zeros(n_features, jnp.float32))
        self.pca = MatrixFreePCA(MatrixFreePCAConfig(n_components=K, num_iterations=2))
        self.probe = probe

    def scored(self, features: jnp.ndarray) -> jnp.ndarray:
        scaled = features * jnp.exp(self.log_scale[...])
        return self.pca.apply({"features": scaled}, {}, None)[0]["pca"]

    def logits(self, features: jnp.ndarray) -> jnp.ndarray:
        return _probe_forward(self.probe, self.scored(features))


def _diff_forward(model: _DiffPCAProbe, features: jnp.ndarray) -> jnp.ndarray:
    return model.logits(features)


def main() -> None:
    """Featurize, then run Section 1 (mini-batch) and Section 2 (full-batch)."""
    ds = load_dataset(f"katarinagresova/Genomic_Benchmarks_{DATASET}")  # nosec B615
    y_tr = np.asarray(ds["train"]["label"], np.int32)
    y_te = np.asarray(ds["test"]["label"], np.int32)
    n_classes = int(max(y_tr.max(), y_te.max())) + 1
    x_tr = kmer_featurize(list(ds["train"]["seq"]), KMER, canonical=True)
    x_te = kmer_featurize(list(ds["test"]["seq"]), KMER, canonical=True)
    n_features = x_tr.shape[1]
    print(f"[{DATASET}] train {x_tr.shape} test {x_te.shape} | {n_classes} classes", flush=True)

    arms = {a: [] for a in ("frozen", "learnable_proj", "orthogonal_proj", "diff_pca")}
    for seed in SEEDS:
        reduction = fit_pca_reduction(x_tr, K)
        tr_c = reduction.scaled(x_tr) - reduction.pca_mean
        te_c = reduction.scaled(x_te) - reduction.pca_mean
        loadings_k = reduction.loadings[:, :K]  # orthonormal (PCA components)
        cfg = MiniBatchConfig(
            batch_size=1024, n_epochs=60, learning_rate=1e-2, weight_decay=5e-2, seed=seed
        )

        # --- Section 1: mini-batch projection arms ---
        xtr_f = jnp.asarray(tr_c @ loadings_k)
        xte_f = jnp.asarray(te_c @ loadings_k)
        probe = _embedding_probe(K, n_classes, 128, seed)
        train_minibatch(probe, _probe_forward, xtr_f, y_tr, n_classes=n_classes, config=cfg)
        arms["frozen"].append(macro(jnp.argmax(_probe_forward(probe, xte_f), -1), y_te))

        proj = LearnableProjection(
            LearnableProjectionConfig(n_genes=n_features, n_components=K),
            init_loadings=loadings_k,
            rngs=nnx.Rngs(seed),
        )
        model = _ProjProbe(proj, _embedding_probe(K, n_classes, 128, seed))
        train_minibatch(
            model, _proj_forward, jnp.asarray(tr_c), y_tr, n_classes=n_classes, config=cfg
        )
        arms["learnable_proj"].append(macro(jnp.argmax(model.logits(jnp.asarray(te_c)), -1), y_te))

        ortho = LearnableOrthogonalProjection(
            LearnableOrthogonalProjectionConfig(n_features=n_features, n_components=K),
            init_loadings=loadings_k,
            rngs=nnx.Rngs(seed),
        )
        omodel = _ProjProbe(ortho, _embedding_probe(K, n_classes, 128, seed))
        train_minibatch(
            omodel, _proj_forward, jnp.asarray(tr_c), y_tr, n_classes=n_classes, config=cfg
        )
        arms["orthogonal_proj"].append(
            macro(jnp.argmax(omodel.logits(jnp.asarray(te_c)), -1), y_te)
        )

        # --- Section 2: full-batch differentiable matrix-free PCA basis ---
        rng = np.random.default_rng(seed)
        sub = rng.permutation(x_tr.shape[0])[:FULL_BATCH_CELLS]
        xtr_sub = jnp.asarray(np.asarray(reduction.scaled(x_tr[sub]), np.float32))
        xte_sc = jnp.asarray(np.asarray(reduction.scaled(x_te), np.float32))
        y_sub = y_tr[sub]
        full_cfg = MiniBatchConfig(
            batch_size=None, n_epochs=200, learning_rate=5e-2, weight_decay=1e-3, seed=seed
        )
        for name, learnable in (("frozen_mf", False), ("diff_pca", True)):
            dmodel = _DiffPCAProbe(n_features, learnable, _embedding_probe(K, n_classes, 128, seed))
            train_minibatch(
                dmodel, _diff_forward, xtr_sub, y_sub, n_classes=n_classes, config=full_cfg
            )
            # eval: project the test set onto the basis of the (scaled) training subset.
            components = dmodel.pca.apply(
                {"features": xtr_sub * jnp.exp(dmodel.log_scale[...])}, {}, None
            )[0]["pca_components"]
            test_scores = (xte_sc * jnp.exp(dmodel.log_scale[...])) @ components.T
            pred = jnp.argmax(_probe_forward(dmodel.probe, test_scores), -1)
            key = "diff_pca" if learnable else "frozen_mf"
            arms.setdefault(key, []).append(macro(pred, y_te))
        print(
            f"  [seed {seed}] frozen {arms['frozen'][-1]:.4f} "
            f"learn {arms['learnable_proj'][-1]:.4f} "
            f"ortho {arms['orthogonal_proj'][-1]:.4f} | frozen_mf {arms['frozen_mf'][-1]:.4f} "
            f"diff_pca {arms['diff_pca'][-1]:.4f}",
            flush=True,
        )

    print("=== differentiable/orthonormal PCA demonstration (DNA, k=5) SUMMARY ===", flush=True)
    for name, vals in arms.items():
        print(f"{name:>16}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(
            {n: [float(np.mean(v)), float(np.std(v))] for n, v in arms.items()}, handle, indent=2
        )
    print("DIFFPCA DEMO DONE", flush=True)


if __name__ == "__main__":
    main()
