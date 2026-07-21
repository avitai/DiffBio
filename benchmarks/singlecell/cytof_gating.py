"""Case study: CyTOF cell-type gating, frozen ``arcsinh(x/5)`` vs learnable cofactor.

New modality (mass cytometry), same two-arm frozen-vs-joint machinery as the scRNA/scATAC
moat. The differentiable degree of freedom here is the per-channel arcsinh cofactor -- one
parameter per marker, initialized at the conventional 5 -- trained jointly with the gating
classifier:

- Frozen arm: fixed ``arcsinh(x / 5)`` transform, then a gating probe (the cofactor is not
  a parameter, so it cannot move -- the honest frozen baseline).
- Joint arm: :class:`ArcsinhCofactor` (trainable, init 5) composed with an identically
  initialized probe, trained on the raw intensities. At step 0 the cofactor is exactly 5,
  so the joint arm reproduces the frozen transform (init-at-frozen); any gain comes from
  the cofactor adapting per channel.

Datasets (HDCytoData; staged via ``_prep_cytof.py``):

- ``samusik``  (primary): 10 samples -> held-out-DONOR split.
- ``levine32`` (secondary): 2 patients -> stratified held-out-cell split.

Weight decay is 0 for both arms: AdamW decoupled weight decay is uniform across parameters,
and decaying a physical cofactor toward the softplus origin would corrupt the transform
independently of the data. Both arms share the choice, keeping the comparison apples-to-apples.

Run: ``python -m benchmarks.singlecell.cytof_gating`` (needs staged .npz), or
``python -m benchmarks.singlecell.cytof_gating --smoke`` to exercise the pipeline on
synthetic data with no staged inputs.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from calibrax.metrics.functional.classification import balanced_accuracy, f1_score
from calibrax.statistics.significance import paired_significance_test
from flax import nnx

from benchmarks.singlecell._gate2_arms import _embedding_probe, _probe_forward
from diffbio.operators.normalization.arcsinh_cofactor import (
    ArcsinhCofactor,
    ArcsinhCofactorConfig,
    arcsinh_transform,
)
from diffbio.pipelines.minibatch_training import MiniBatchConfig, train_minibatch

_DATA = os.environ.get("DIFFBIO_DATA_ROOT", "/mnt/ssd2/Data")
_HD = f"{_DATA}/hdcytodata"
OUT = "benchmarks/results/singlecell/cytof_gating.json"
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
_COFACTOR = 5.0
_HIDDEN = 64
_FRAC_TEST = 0.3


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """A CyTOF dataset and the held-out split it supports.

    Attributes:
        key: Short dataset name used in output.
        npz: Path to the staged ``.npz`` (see ``_prep_cytof.py``).
        split: ``"donor"`` (held-out sample/donor) or ``"stratified"`` (held-out cells).
    """

    key: str
    npz: str
    split: str


DATASETS = (
    DatasetSpec("samusik", f"{_HD}/samusik.npz", "donor"),
    DatasetSpec("levine32", f"{_HD}/levine32.npz", "stratified"),
)


class _ArcsinhProbe(nnx.Module):
    """A learnable arcsinh-cofactor transform composed with a gating probe."""

    def __init__(self, cofactor: ArcsinhCofactor, probe: nnx.Module) -> None:
        """Store the cofactor transform and probe submodules.

        Args:
            cofactor: The learnable per-channel arcsinh cofactor stage.
            probe: The gating classifier head.
        """
        self.cofactor = cofactor
        self.probe = probe


def _arcsinh_probe_forward(model: _ArcsinhProbe, intensities: jnp.ndarray) -> jnp.ndarray:
    """Transform raw intensities with the learnable cofactor then return probe logits."""
    transformed = model.cofactor.apply({"intensities": intensities}, {}, None)[0]["transformed"]
    return model.probe.apply({"embeddings": transformed}, {}, None)[0]["logits"]


def macro_f1(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return the macro-averaged F1 of predictions against ground truth."""
    return float(f1_score(jnp.asarray(predictions), jnp.asarray(targets), average="macro"))


def balanced_acc(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return the balanced accuracy of predictions against ground truth."""
    return float(balanced_accuracy(jnp.asarray(predictions), jnp.asarray(targets)))


def donor_split(
    donor: np.ndarray, frac_test: float, seed: int
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return train/test masks holding out whole donors, plus the held-out donor ids."""
    rng = np.random.default_rng(seed)
    donors = np.unique(donor)
    rng.shuffle(donors)
    n_test = max(1, int(round(len(donors) * frac_test)))
    test_donors = sorted(int(value) for value in donors[:n_test])
    test_mask = np.isin(donor, test_donors)
    return ~test_mask, test_mask, test_donors


def stratified_split(
    labels: np.ndarray, frac_test: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test masks with each class split by ``frac_test``."""
    rng = np.random.default_rng(seed)
    test_mask = np.zeros(len(labels), dtype=bool)
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * frac_test))
        test_mask[idx[:n_test]] = True
    return ~test_mask, test_mask


def _train_config(seed: int, n_epochs: int, batch_size: int) -> MiniBatchConfig:
    """Build the mini-batch config shared by both arms (weight decay 0; see module docstring)."""
    return MiniBatchConfig(
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=1.0e-2,
        weight_decay=0.0,
        seed=seed,
    )


def run_arrays(
    intensities: np.ndarray,
    labels: np.ndarray,
    donor: np.ndarray,
    type_names: np.ndarray,
    *,
    split: str,
    n_epochs: int = 40,
    batch_size: int = 2048,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, object]:
    """Run the frozen-vs-joint arcsinh-cofactor study on one CyTOF dataset.

    Args:
        intensities: ``(n_cells, n_markers)`` raw marker intensities.
        labels: ``(n_cells,)`` gated population ids.
        donor: ``(n_cells,)`` sample/donor ids.
        type_names: ``(n_types,)`` population names.
        split: ``"donor"`` or ``"stratified"``.
        n_epochs: Training epochs per arm.
        batch_size: Mini-batch size.
        seeds: Training seeds; the split itself is fixed (seed 0).

    Returns:
        A results dict with per-arm macro-F1/balanced-accuracy summaries, the paired
        significance of the frozen-vs-joint macro-F1 difference, and the learned cofactors.
    """
    intensities = np.asarray(intensities, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    n_types = int(len(type_names))
    n_markers = int(intensities.shape[1])

    if split == "donor":
        train_mask, test_mask, held = donor_split(donor, _FRAC_TEST, seed=0)
        split_desc = f"held-out donors {held}"
    else:
        train_mask, test_mask = stratified_split(labels, _FRAC_TEST, seed=0)
        split_desc = "stratified held-out cells"

    x_train, x_test = intensities[train_mask], intensities[test_mask]
    y_train, y_test = labels[train_mask], labels[test_mask]
    # Frozen features are deterministic -> precompute once (mirrors the frozen-PCA arm).
    xf_train = jnp.asarray(np.asarray(arcsinh_transform(x_train, _COFACTOR), dtype=np.float32))
    xf_test = jnp.asarray(np.asarray(arcsinh_transform(x_test, _COFACTOR), dtype=np.float32))
    xr_train, xr_test = jnp.asarray(x_train), jnp.asarray(x_test)
    print(
        f"  cells {intensities.shape} | train {x_train.shape[0]} test {x_test.shape[0]} "
        f"| {n_types} types {n_markers} markers | {split_desc}",
        flush=True,
    )

    frozen_f1: list[float] = []
    joint_f1: list[float] = []
    frozen_bal: list[float] = []
    joint_bal: list[float] = []
    learned_cofactor = np.full(n_markers, _COFACTOR, dtype=np.float32)
    for seed in seeds:
        config = _train_config(seed, n_epochs, batch_size)

        probe_frozen = _embedding_probe(n_markers, n_types, _HIDDEN, seed)
        train_minibatch(
            probe_frozen, _probe_forward, xf_train, y_train, n_classes=n_types, config=config
        )
        pred_frozen = np.asarray(jnp.argmax(_probe_forward(probe_frozen, xf_test), -1))
        frozen_f1.append(macro_f1(pred_frozen, y_test))
        frozen_bal.append(balanced_acc(pred_frozen, y_test))

        cofactor = ArcsinhCofactor(
            ArcsinhCofactorConfig(num_channels=n_markers, cofactor_init=_COFACTOR, trainable=True),
            rngs=nnx.Rngs(seed),
        )
        model = _ArcsinhProbe(cofactor, _embedding_probe(n_markers, n_types, _HIDDEN, seed))
        train_minibatch(
            model, _arcsinh_probe_forward, xr_train, y_train, n_classes=n_types, config=config
        )
        pred_joint = np.asarray(jnp.argmax(_arcsinh_probe_forward(model, xr_test), -1))
        joint_f1.append(macro_f1(pred_joint, y_test))
        joint_bal.append(balanced_acc(pred_joint, y_test))
        learned_cofactor = np.asarray(jax.nn.softplus(model.cofactor.raw_cofactor[...]))
        print(
            f"  [seed {seed}] frozen macroF1 {frozen_f1[-1]:.4f}  joint {joint_f1[-1]:.4f}  "
            f"gain {100 * (joint_f1[-1] - frozen_f1[-1]):+.2f}pp",
            flush=True,
        )

    significance = paired_significance_test(frozen_f1, joint_f1)
    return {
        "split": split_desc,
        "n_cells": int(intensities.shape[0]),
        "n_types": n_types,
        "n_markers": n_markers,
        "frozen_macro_f1": [float(np.mean(frozen_f1)), float(np.std(frozen_f1))],
        "joint_macro_f1": [float(np.mean(joint_f1)), float(np.std(joint_f1))],
        "frozen_balanced_acc": [float(np.mean(frozen_bal)), float(np.std(frozen_bal))],
        "joint_balanced_acc": [float(np.mean(joint_bal)), float(np.std(joint_bal))],
        "gain_pp": float(100 * (np.mean(joint_f1) - np.mean(frozen_f1))),
        "paired_p_value": float(significance.p_value),
        "paired_significant": bool(significance.significant),
        "learned_cofactor": {
            "mean": float(np.mean(learned_cofactor)),
            "min": float(np.min(learned_cofactor)),
            "max": float(np.max(learned_cofactor)),
        },
    }


def _synthetic_dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a small label-structured CyTOF-like dataset for the smoke run."""
    rng = np.random.default_rng(seed)
    n_types, n_markers, per_type_per_donor, n_donors = 5, 12, 120, 4
    centers = rng.uniform(2.0, 8.0, size=(n_types, n_markers))
    intensities, labels, donor = [], [], []
    for donor_id in range(n_donors):
        for type_id in range(n_types):
            base = centers[type_id] + rng.normal(0.0, 0.3, size=n_markers)
            cells = 5.0 * np.sinh(base) + rng.gamma(2.0, 5.0, size=(per_type_per_donor, n_markers))
            intensities.append(cells.astype(np.float32))
            labels.append(np.full(per_type_per_donor, type_id, dtype=np.int32))
            donor.append(np.full(per_type_per_donor, donor_id, dtype=np.int32))
    type_names = np.asarray([f"pop{i}" for i in range(n_types)], dtype=np.str_)
    return (
        np.concatenate(intensities),
        np.concatenate(labels),
        np.concatenate(donor),
        type_names,
    )


def _smoke() -> None:
    """Exercise the full two-arm pipeline on synthetic data with no staged inputs."""
    intensities, labels, donor, type_names = _synthetic_dataset()

    # Init-at-frozen check: at cofactor=5 with a shared seed, both arms compute identical
    # logits before any training, so the joint arm genuinely starts at the frozen baseline.
    n_markers, n_types = intensities.shape[1], len(type_names)
    probe_a = _embedding_probe(n_markers, n_types, _HIDDEN, seed=0)
    probe_b = _embedding_probe(n_markers, n_types, _HIDDEN, seed=0)
    cofactor = ArcsinhCofactor(
        ArcsinhCofactorConfig(num_channels=n_markers, cofactor_init=_COFACTOR, trainable=True),
        rngs=nnx.Rngs(0),
    )
    sample = jnp.asarray(intensities[:16])
    frozen_logits = _probe_forward(
        probe_a, jnp.asarray(arcsinh_transform(intensities[:16], _COFACTOR))
    )
    joint_logits = _arcsinh_probe_forward(_ArcsinhProbe(cofactor, probe_b), sample)
    np.testing.assert_allclose(
        np.asarray(frozen_logits),
        np.asarray(joint_logits),
        atol=1e-5,
        err_msg="init-at-frozen violated: joint arm does not start at the frozen transform",
    )
    print("init-at-frozen OK: joint == frozen logits at initialization", flush=True)

    result = run_arrays(
        intensities,
        labels,
        donor,
        type_names,
        split="donor",
        n_epochs=6,
        batch_size=256,
        seeds=(0, 1),
    )
    print(json.dumps(result, indent=2), flush=True)
    print("CYTOF SMOKE DONE", flush=True)


def main() -> None:
    """Run the CyTOF gating study over the staged datasets (or the synthetic smoke run)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="run on synthetic data, no staged inputs"
    )
    args = parser.parse_args()

    if args.smoke:
        _smoke()
        return

    results: dict[str, object] = {}
    for spec in DATASETS:
        if not os.path.exists(spec.npz):
            print(f"SKIP {spec.key}: missing {spec.npz} (stage it with _prep_cytof.py)", flush=True)
            continue
        print(f"=== CyTOF gating: {spec.key} ===", flush=True)
        data = np.load(spec.npz, allow_pickle=False)
        result = run_arrays(
            data["intensities"],
            data["labels"],
            data["donor"],
            data["type_names"],
            split=spec.split,
        )
        results[spec.key] = result
        print(
            f"{spec.key}: frozen {result['frozen_macro_f1'][0]:.4f} "  # type: ignore[index]
            f"joint {result['joint_macro_f1'][0]:.4f} gain {result['gain_pp']:+.2f}pp "
            f"p={result['paired_p_value']:.3f}",
            flush=True,
        )

    if results:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"CYTOF GATE DONE -> {OUT}", flush=True)
    else:
        print("CYTOF GATE: no datasets staged; nothing run.", flush=True)


if __name__ == "__main__":
    main()
