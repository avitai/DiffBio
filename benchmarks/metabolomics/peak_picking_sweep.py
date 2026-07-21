"""How much of the peak-picking joint gain is default-recovery vs intrinsic?

The headline peak-picking study (``peak_picking``) used a deliberately suboptimal frozen S/N
threshold, so its +60pp is a proof-of-mechanism, not a natural effect size. This sweep varies
the shared initial threshold from crippling down to reasonable and reports frozen accuracy,
joint accuracy, and their gap at each. A gain that collapses as the default improves means the
joint picker is mostly recovering from bad defaults; a gain that persists at a good default
means end-to-end training has intrinsic value on this (synthetic) task.
"""

from __future__ import annotations

import json
import os

import jax.numpy as jnp
import numpy as np
from flax import nnx

from benchmarks._classification import stratified_label_split
from benchmarks.metabolomics.peak_picking import (
    _MAX_MZ,
    _MIN_MZ,
    _N_POINTS,
    _PEAK_WIDTH_INIT,
    _MSPipeline,
    _accuracy,
    _pipeline_forward,
    synthesize,
)
from benchmarks.singlecell._gate2_arms import _embedding_probe
from diffbio.operators.metabolomics.isotope_envelope import (
    SoftIsotopeEnvelope,
    SoftIsotopeEnvelopeConfig,
)
from diffbio.operators.metabolomics.soft_centroiding import SoftCentroider, SoftCentroiderConfig
from diffbio.pipelines.minibatch_training import MiniBatchConfig, train_minibatch

OUT = "benchmarks/results/metabolomics/peak_picking_sweep.json"
SEEDS = (0, 1, 2, 3, 4)
SNTHRESH_VALUES = (6.0, 3.0, 1.5)
_DECAY_INIT = 0.6  # reasonable averagine, not the crippled 0.4 of the headline study
_HIDDEN = 64


def _build(n_compounds: int, *, snthresh: float, trainable: bool, seed: int) -> _MSPipeline:
    """Build the pipeline at a given S/N threshold; ``trainable`` toggles the knobs."""
    centroider = SoftCentroider(
        SoftCentroiderConfig(
            min_mz=_MIN_MZ,
            max_mz=_MAX_MZ,
            n_points=_N_POINTS,
            snthresh_init=snthresh,
            peak_width_init=_PEAK_WIDTH_INIT,
            trainable=trainable,
        ),
        rngs=nnx.Rngs(seed),
    )
    isotoper = SoftIsotopeEnvelope(
        SoftIsotopeEnvelopeConfig(
            min_mz=_MIN_MZ,
            max_mz=_MAX_MZ,
            n_points=_N_POINTS,
            charges=(1, 2, 3),
            decay_init=_DECAY_INIT,
            trainable=trainable,
        ),
        rngs=nnx.Rngs(seed),
    )
    return _MSPipeline(
        centroider, isotoper, _embedding_probe(_N_POINTS, n_compounds, _HIDDEN, seed)
    )


def run_snthresh(
    x_train: jnp.ndarray,
    x_test: jnp.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_compounds: int,
    *,
    snthresh: float,
    seeds: tuple[int, ...],
    n_epochs: int,
    batch_size: int,
) -> dict[str, object]:
    """Frozen vs joint at one shared initial S/N threshold, across seeds."""
    frozen_acc: list[float] = []
    joint_acc: list[float] = []
    for seed in seeds:
        config = MiniBatchConfig(
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=1.0e-2,
            weight_decay=0.0,
            seed=seed,
        )
        frozen = _build(n_compounds, snthresh=snthresh, trainable=False, seed=seed)
        train_minibatch(
            frozen, _pipeline_forward, x_train, y_train, n_classes=n_compounds, config=config
        )
        frozen_acc.append(
            _accuracy(np.asarray(jnp.argmax(_pipeline_forward(frozen, x_test), -1)), y_test)
        )

        joint = _build(n_compounds, snthresh=snthresh, trainable=True, seed=seed)
        train_minibatch(
            joint, _pipeline_forward, x_train, y_train, n_classes=n_compounds, config=config
        )
        joint_acc.append(
            _accuracy(np.asarray(jnp.argmax(_pipeline_forward(joint, x_test), -1)), y_test)
        )
        print(
            f"  snthresh={snthresh:.1f} [seed {seed}] frozen {frozen_acc[-1]:.4f} "
            f"joint {joint_acc[-1]:.4f} gain {100 * (joint_acc[-1] - frozen_acc[-1]):+.2f}pp",
            flush=True,
        )
    return {
        "snthresh": snthresh,
        "frozen_accuracy": float(np.mean(frozen_acc)),
        "joint_accuracy": float(np.mean(joint_acc)),
        "gain_pp": float(100 * (np.mean(joint_acc) - np.mean(frozen_acc))),
    }


def main() -> None:
    """Sweep the frozen S/N threshold and report the joint gain at each."""
    spectra, labels, n_compounds = synthesize(12, 8000, seed=0)
    train_idx, test_idx = stratified_label_split(
        labels, train_fraction=0.8, seed=0, minimum_count_name="spectra"
    )
    x_train, x_test = jnp.asarray(spectra[train_idx]), jnp.asarray(spectra[test_idx])
    y_train, y_test = labels[train_idx], labels[test_idx]

    print("=== MS peak-picking: joint gain vs frozen S/N threshold quality ===", flush=True)
    rows = [
        run_snthresh(
            x_train,
            x_test,
            y_train,
            y_test,
            n_compounds,
            snthresh=snthresh,
            seeds=SEEDS,
            n_epochs=30,
            batch_size=512,
        )
        for snthresh in SNTHRESH_VALUES
    ]
    for row in rows:
        print(
            f"snthresh={row['snthresh']:.1f}: frozen {row['frozen_accuracy']:.4f} "
            f"joint {row['joint_accuracy']:.4f} gain {row['gain_pp']:+.2f}pp",
            flush=True,
        )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump({"decay_init": _DECAY_INIT, "by_snthresh": rows}, handle, indent=2)
    print(f"PEAK PICKING SWEEP DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
