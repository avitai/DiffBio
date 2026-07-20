"""Case study D full-stack knob ablation: which learnable knob carries the joint gain?

Mirrors the single-cell knob-attribution ablation on the Perturb-seq perturbation-identity
task. Starting from the frozen transform's scaled features, we make each preprocessing knob
learnable one at a time -- a per-gene soft gate (feature selection, like SoftHVG), a per-gene
affine (feature scaling), and the projection directions (the reduction) -- each initialized
to reproduce the frozen pipeline exactly, and trained jointly with the probe. A knob that is
"off" is stored as a frozen buffer, so only the intended parameters are optimized. Expected,
as in single cells: the projection carries essentially the entire gain.

Real Virtual Cell Challenge Perturb-seq (target-gene-masked, split by experimental batch).
"""

import os

import json

import jax.numpy as jnp
import numpy as np
from calibrax.metrics.functional.classification import f1_score
from flax import nnx

from benchmarks.singlecell._gate2_arms import _embedding_probe, _probe_forward
from benchmarks.singlecell.frozen_annotation_baseline import fit_frozen_preprocess
from diffbio.pipelines.minibatch_training import MiniBatchConfig, train_minibatch


_DATA = os.environ.get("DIFFBIO_DATA_ROOT", "/mnt/ssd2/Data")
DATA = f"{_DATA}/Virtual-Cell-Challenge/vcc_masked.npz"
OUT = "benchmarks/results/crossmodality/gate_vcc_ablation.json"
SEEDS = (0, 1, 2)
K = 10
ARMS = ["frozen", "gate", "scale", "proj", "all"]


class KnobModel(nnx.Module):
    """Scaled features -> soft gate -> affine scale -> projection -> probe.

    Each knob is an ``nnx.Param`` when learnable and an ``nnx.Variable`` (frozen buffer)
    otherwise; all knobs start at their frozen-pipeline values so the frozen arm reproduces
    the frozen-PCA baseline exactly and every learnable arm starts there too.
    """

    def __init__(self, n_features, loadings, learn, n_types, seed):
        n_out = loadings.shape[1]
        gate_cls = nnx.Param if "gate" in learn else nnx.Variable
        scale_cls = nnx.Param if "scale" in learn else nnx.Variable
        proj_cls = nnx.Param if "proj" in learn else nnx.Variable
        self.gate = gate_cls(jnp.ones(n_features, jnp.float32))
        self.scale_a = scale_cls(jnp.ones(n_features, jnp.float32))
        self.scale_b = scale_cls(jnp.zeros(n_features, jnp.float32))
        self.basis = nnx.Variable(jnp.asarray(loadings, jnp.float32))
        self.delta = proj_cls(jnp.zeros_like(jnp.asarray(loadings, jnp.float32)))
        self.proj_bias = proj_cls(jnp.zeros(n_out, jnp.float32))
        self.probe = _embedding_probe(n_out, n_types, 128, seed)

    def logits(self, features):
        """Apply the gate, affine, and projection, then classify."""
        gated = features * self.gate[...]
        scaled = gated * self.scale_a[...] + self.scale_b[...]
        projected = scaled @ (self.basis[...] + self.delta[...]) + self.proj_bias[...]
        return _probe_forward(self.probe, projected)


def _forward(model, features):
    return model.logits(features)


def macro(pred, true):
    """Return the macro-averaged F1 of predictions against ground truth."""
    return float(
        f1_score(jnp.asarray(np.asarray(pred)), jnp.asarray(np.asarray(true)), average="macro")
    )


def main() -> None:
    """Run the knob ablation across seeds on the batch-split Perturb-seq task."""
    data = np.load(DATA)
    counts = np.asarray(data["counts"], np.float32)
    labels = np.asarray(data["labels"], np.int32)
    batch = np.asarray(data["batch"], np.int32)
    n_types = int(data["n_types"])
    test_batches = set(range(0, int(data["n_batches"]), 5))
    test_mask = np.isin(batch, list(test_batches))
    tr_counts, te_counts = counts[~test_mask], counts[test_mask]
    tr_y, te_y = labels[~test_mask], labels[test_mask]
    print(f"train {tr_counts.shape} test {te_counts.shape} | {n_types} perturbations", flush=True)

    results = {arm: [] for arm in ARMS}
    for seed in SEEDS:
        transform = fit_frozen_preprocess(tr_counts, n_top_genes=2000, n_components=50)
        tr_scaled = jnp.asarray(transform.scaled(tr_counts) - transform.pca_mean)
        te_scaled = jnp.asarray(transform.scaled(te_counts) - transform.pca_mean)
        n_features = transform.loadings.shape[0]
        loadings_k = transform.loadings[:, :K]
        config = MiniBatchConfig(
            batch_size=4096, n_epochs=100, learning_rate=1e-2, weight_decay=5e-2, seed=seed
        )
        learn_sets = {
            "frozen": set(),
            "gate": {"gate"},
            "scale": {"scale"},
            "proj": {"proj"},
            "all": {"gate", "scale", "proj"},
        }
        for arm in ARMS:
            model = KnobModel(n_features, loadings_k, learn_sets[arm], n_types, seed)
            train_minibatch(model, _forward, tr_scaled, tr_y, n_classes=n_types, config=config)
            pred = np.asarray(jnp.argmax(model.logits(te_scaled), -1))
            results[arm].append(macro(pred, te_y))
        print(
            f"  [seed {seed}] " + " ".join(f"{a} {results[a][-1]:.4f}" for a in ARMS),
            flush=True,
        )

    print("=== VCC knob ablation (k=10) SUMMARY ===", flush=True)
    for arm in ARMS:
        print(f"{arm:>7}: {np.mean(results[arm]):.4f} +/- {np.std(results[arm]):.4f}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(
            {a: [float(np.mean(v)), float(np.std(v))] for a, v in results.items()}, handle, indent=2
        )
    print("VCC ABLATION DONE", flush=True)


if __name__ == "__main__":
    main()
