"""Case study C, rebuilt: variant calling with a REAL learnable-reduction bottleneck.

The first C attempt (per-pixel learnable pileup encoding) had no reduction bottleneck on
the gradient path -- the CNN saw every read -- so it did not instantiate the hypothesis.
This rebuild does: each read is featurized (base one-hot + quality + ref-mismatch + strand
+ mapq), then a genuine k-bottleneck compresses per-read features (248 -> k) and a
permutation-invariant mean-pool over reads aggregates the evidence, before a classifier.
Frozen PCA of per-read features vs a learnable projection initialized from it (init-at-
frozen), swept over k. Set-pooling is genuinely different machinery from the single-vector
linear reductions of the other case studies, but with the same lossy k-bottleneck, so the
hypothesis predicts the gain-vs-k signature. Real GIAB HG002 chr20 windows (no synthetic).
"""

import os

import json

import jax.numpy as jnp
import numpy as np
from calibrax.metrics.functional.classification import balanced_accuracy, f1_score
from flax import nnx

from benchmarks.singlecell._gate2_arms import _embedding_probe, _probe_forward
from diffbio.operators.normalization.learnable_projection import (
    LearnableProjection,
    LearnableProjectionConfig,
)
from diffbio.pipelines.minibatch_training import MiniBatchConfig, train_minibatch
from diffbio.reductions import fit_pca_reduction


_DATA = os.environ.get("DIFFBIO_DATA_ROOT", "/mnt/ssd2/Data")
DATA = f"{_DATA}/giab/variant_windows.npz"
OUT = "benchmarks/results/crossmodality/gate_variant_pool.json"
SEEDS = (0, 1, 2)
K_VALUES = [3, 5, 10, 20]
QUALITY_MAX, MAPQ_MAX = 40.0, 60.0


def read_features(split: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build per-read feature vectors ``(N, 50, 248)`` and a real-read mask ``(N, 50)``."""
    reads = split["reads"].astype(np.float32)  # (N, 50, 41, 4)
    n, max_reads, window, _ = reads.shape
    qual = split["base_qualities"].astype(np.float32) / QUALITY_MAX  # (N, 50, 41)
    ref = split["reference"].astype(np.float32)  # (N, 41, 4)
    mismatch = 1.0 - (reads * ref[:, None, :, :]).sum(-1)  # (N, 50, 41)
    strand = split["strands"].astype(np.float32)[..., None]  # (N, 50, 1)
    mapq = split["mapping_qualities"].astype(np.float32)[..., None] / MAPQ_MAX  # (N, 50, 1)
    base_flat = reads.reshape(n, max_reads, window * 4)  # (N, 50, 164)
    features = np.concatenate([base_flat, qual, mismatch, strand, mapq], axis=-1)
    mask = (reads.sum(axis=(2, 3)) > 0).astype(np.float32)  # (N, 50) real (non-padding) reads
    return features.astype(np.float32), mask


class ReadPoolProbe(nnx.Module):
    """Learnable per-read projection -> masked mean-pool over reads -> classifier probe."""

    def __init__(self, projection: LearnableProjection, probe: nnx.Module) -> None:
        self.projection = projection
        self.probe = probe

    def logits(self, batch: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Project each read to k dims, mean-pool over real reads, then classify."""
        features, mask = batch["features"], batch["mask"]  # (B, 50, 248), (B, 50)
        weights = self.projection.basis[...] + self.projection.delta[...]  # (248, k)
        projected = features @ weights + self.projection.projection_bias[...]  # (B, 50, k)
        denom = jnp.clip(mask.sum(axis=1, keepdims=True), 1.0, None)
        pooled = (projected * mask[..., None]).sum(axis=1) / denom  # (B, k)
        return _probe_forward(self.probe, pooled)


def _pool_forward(model: ReadPoolProbe, batch: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return model.logits(batch)


def _frozen_pooled(
    features: np.ndarray, mask: np.ndarray, loadings_k: np.ndarray, pca_mean: np.ndarray, scaler
) -> jnp.ndarray:
    """Frozen: standardize + PCA-project each read, then masked mean-pool over reads."""
    n, max_reads, dim = features.shape
    scaled = scaler(features.reshape(n * max_reads, dim)) - pca_mean
    projected = (scaled @ loadings_k).reshape(n, max_reads, loadings_k.shape[1])
    denom = np.clip(mask.sum(axis=1, keepdims=True), 1.0, None)
    pooled = (projected * mask[..., None]).sum(axis=1) / denom
    return jnp.asarray(pooled.astype(np.float32))


def macro(pred: np.ndarray, true: np.ndarray) -> float:
    """Return the macro-averaged F1 of predictions against ground truth."""
    return float(f1_score(jnp.asarray(pred), jnp.asarray(true), average="macro"))


def main() -> None:
    """Fit the frozen per-read PCA once, then sweep k for frozen vs learnable projection."""
    data = np.load(DATA)
    train = {
        k: data[f"train_{k}"]
        for k in ("reads", "reference", "base_qualities", "mapping_qualities", "strands")
    }
    test = {
        k: data[f"test_{k}"]
        for k in ("reads", "reference", "base_qualities", "mapping_qualities", "strands")
    }
    tr_y, te_y = data["train_labels"].astype(np.int32), data["test_labels"].astype(np.int32)
    tr_feat, tr_mask = read_features(train)
    te_feat, te_mask = read_features(test)
    dim = tr_feat.shape[-1]
    print(f"train {tr_feat.shape} test {te_feat.shape} | per-read dim {dim}", flush=True)

    # Fit frozen per-read PCA on real training reads (deterministic -> fit once).
    real_reads = tr_feat[tr_mask > 0]
    reduction = fit_pca_reduction(real_reads, max(K_VALUES))
    print(f"per-read PCA fit on {real_reads.shape[0]} reads", flush=True)
    # Centered, standardized per-read features for the learnable arm (init-at-frozen).
    n_tr = tr_feat.shape[0]
    tr_centered = (reduction.scaled(tr_feat.reshape(-1, dim)) - reduction.pca_mean).reshape(
        n_tr, tr_feat.shape[1], dim
    )
    te_centered = (reduction.scaled(te_feat.reshape(-1, dim)) - reduction.pca_mean).reshape(
        te_feat.shape[0], te_feat.shape[1], dim
    )
    tr_learn = {
        "features": jnp.asarray(tr_centered.astype(np.float32)),
        "mask": jnp.asarray(tr_mask),
    }
    te_learn = {
        "features": jnp.asarray(te_centered.astype(np.float32)),
        "mask": jnp.asarray(te_mask),
    }

    frozen, joint, bal = {k: [] for k in K_VALUES}, {k: [] for k in K_VALUES}, {"f": [], "j": []}
    for seed in SEEDS:
        config = MiniBatchConfig(
            batch_size=256, n_epochs=60, learning_rate=1e-2, weight_decay=5e-2, seed=seed
        )
        for k in K_VALUES:
            loadings_k = reduction.loadings[:, :k]
            xtr = _frozen_pooled(tr_feat, tr_mask, loadings_k, reduction.pca_mean, reduction.scaled)
            xte = _frozen_pooled(te_feat, te_mask, loadings_k, reduction.pca_mean, reduction.scaled)
            probe_f = _embedding_probe(k, 2, 128, seed)
            train_minibatch(probe_f, _probe_forward, xtr, tr_y, n_classes=2, config=config)
            pred_f = np.asarray(jnp.argmax(_probe_forward(probe_f, xte), -1))
            frozen[k].append(macro(pred_f, te_y))

            proj = LearnableProjection(
                LearnableProjectionConfig(n_genes=dim, n_components=k),
                init_loadings=loadings_k,
                rngs=nnx.Rngs(seed),
            )
            model = ReadPoolProbe(proj, _embedding_probe(k, 2, 128, seed))
            train_minibatch(model, _pool_forward, tr_learn, tr_y, n_classes=2, config=config)
            pred_j = np.asarray(jnp.argmax(model.logits(te_learn), -1))
            joint[k].append(macro(pred_j, te_y))
            if k == 5:
                bal["f"].append(float(balanced_accuracy(jnp.asarray(pred_f), jnp.asarray(te_y))))
                bal["j"].append(float(balanced_accuracy(jnp.asarray(pred_j), jnp.asarray(te_y))))
            print(
                f"  [seed {seed}] k={k:2d} frozen {frozen[k][-1]:.4f} joint {joint[k][-1]:.4f}",
                flush=True,
            )

    print("=== variant read-pool (frozen per-read PCA vs learnable) SUMMARY ===", flush=True)
    for k in K_VALUES:
        fm, jm = np.mean(frozen[k]), np.mean(joint[k])
        print(
            f"k={k:2d}: frozen {fm:.4f}+/-{np.std(frozen[k]):.4f}  "
            f"joint {jm:.4f}+/-{np.std(joint[k]):.4f}  gain {100 * (jm - fm):+.1f}pp",
            flush=True,
        )
    print(
        f"balanced-acc @k=5: frozen {np.mean(bal['f']):.4f} joint {np.mean(bal['j']):.4f}",
        flush=True,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(
            {
                "k_values": K_VALUES,
                "frozen": {
                    str(k): [float(np.mean(v)), float(np.std(v))] for k, v in frozen.items()
                },
                "joint": {str(k): [float(np.mean(v)), float(np.std(v))] for k, v in joint.items()},
            },
            handle,
            indent=2,
        )
    print("VARIANT POOL DONE", flush=True)


if __name__ == "__main__":
    main()
