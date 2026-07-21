"""Case study: back-propagating a compound-ID loss into the mass-spec peak picker.

Existing MS identification tools (DeepMS, Alpha-XIC) start from already-centroided,
deisotoped input -- the lossy peak-picking knobs are frozen upstream of the model. This
study makes the whole front end differentiable and asks whether training it end-to-end helps:

- Frozen arm: fixed (deliberately suboptimal) centroiding + deisotoping knobs feed a probe;
  only the probe trains.
- Joint arm: the same pipeline and initialization, but the SoftCentroider S/N threshold +
  peak width and the SoftIsotopeEnvelope decay are learnable and trained through the ID loss.

Truth is synthetic (the report's first-listed source): compounds are simulated isotope
envelopes with known monoisotopic m/z and charge, placed on a noisy profile grid with low-SNR
trailing isotopes that a too-aggressive frozen threshold discards. The task is to identify the
compound. A real ProteomeXchange confirmation is deferred as a follow-on.

Run: ``python -m benchmarks.metabolomics.peak_picking`` or ``--smoke``.
"""

from __future__ import annotations

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from calibrax.metrics.functional.classification import balanced_accuracy
from calibrax.statistics.significance import paired_significance_test
from flax import nnx

from benchmarks._classification import stratified_label_split
from benchmarks.singlecell._gate2_arms import _embedding_probe
from diffbio.operators.metabolomics.isotope_envelope import (
    SoftIsotopeEnvelope,
    SoftIsotopeEnvelopeConfig,
)
from diffbio.operators.metabolomics.soft_centroiding import (
    SoftCentroider,
    SoftCentroiderConfig,
    mz_grid,
)
from diffbio.pipelines.minibatch_training import MiniBatchConfig, train_minibatch

OUT = "benchmarks/results/metabolomics/peak_picking.json"
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
_MIN_MZ, _MAX_MZ, _N_POINTS = 100.0, 300.0, 2000
_SPACING = 1.00235
# Deliberately suboptimal frozen knobs: the threshold is high enough to clip low-SNR
# isotopes and the decay is mismatched, so end-to-end tuning has something to recover.
_SNTHRESH_INIT = 6.0
_PEAK_WIDTH_INIT = 4.0
_DECAY_INIT = 0.4
_HIDDEN = 64


class _MSPipeline(nnx.Module):
    """SoftCentroider -> SoftIsotopeEnvelope -> annotation probe."""

    def __init__(
        self, centroider: SoftCentroider, isotoper: SoftIsotopeEnvelope, probe: nnx.Module
    ) -> None:
        """Store the peak-picking front end and the probe."""
        self.centroider = centroider
        self.isotoper = isotoper
        self.probe = probe


def _pipeline_forward(model: _MSPipeline, spectra: jnp.ndarray) -> jnp.ndarray:
    """Centroid, deisotope, and classify a batch of profile spectra."""

    def deisotope_one(spectrum: jnp.ndarray) -> jnp.ndarray:
        centroided = model.centroider.apply({"intensity": spectrum}, {}, None)[0]["centroided"]
        return model.isotoper.apply({"intensity": centroided}, {}, None)[0]["deisotoped"]

    features = jax.vmap(deisotope_one)(spectra)
    return model.probe.apply({"embeddings": features}, {}, None)[0]["logits"]


def _build_pipeline(n_compounds: int, *, trainable: bool, seed: int) -> _MSPipeline:
    """Build the pipeline with shared (suboptimal) init; ``trainable`` toggles the knobs."""
    centroider = SoftCentroider(
        SoftCentroiderConfig(
            min_mz=_MIN_MZ,
            max_mz=_MAX_MZ,
            n_points=_N_POINTS,
            snthresh_init=_SNTHRESH_INIT,
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
    probe = _embedding_probe(_N_POINTS, n_compounds, _HIDDEN, seed)
    return _MSPipeline(centroider, isotoper, probe)


def synthesize(n_compounds: int, n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Simulate profile spectra of known compounds (isotope envelopes) on a noisy grid."""
    rng = np.random.default_rng(seed)
    grid = np.asarray(mz_grid(_MIN_MZ, _MAX_MZ, _N_POINTS))
    # Compounds SHARE monoisotopic positions across charge states, so identity can only be
    # resolved from the isotope spacing (charge) -- which the deisotoping front end infers.
    # A too-aggressive frozen threshold clips the low-SNR trailing isotopes that carry the
    # charge signal, so the frozen picker confuses same-m/z compounds the joint picker can split.
    n_bases = max(2, n_compounds // 3)
    base_mz = rng.uniform(_MIN_MZ + 20.0, _MAX_MZ - 20.0, size=n_bases).astype(np.float32)
    mono = np.array([base_mz[c % n_bases] for c in range(n_compounds)], dtype=np.float32)
    charge = np.array([(c // n_bases) % 3 + 1 for c in range(n_compounds)], dtype=np.int64)
    ratios = np.array([0.6, 1.0, 0.7, 0.35, 0.15], dtype=np.float32)

    spectra = np.zeros((n_samples, _N_POINTS), dtype=np.float32)
    labels = rng.integers(0, n_compounds, size=n_samples)
    for sample in range(n_samples):
        compound = labels[sample]
        scale = float(rng.uniform(20.0, 60.0))
        for i, ratio in enumerate(ratios):
            center = mono[compound] + i * _SPACING / charge[compound]
            jitter = float(rng.normal(0.0, 0.01))
            spectra[sample] += scale * ratio * np.exp(-0.5 * ((grid - center - jitter) / 0.06) ** 2)
        spectra[sample] += rng.uniform(0.0, 8.0, size=_N_POINTS).astype(np.float32)
    return spectra, labels.astype(np.int32), n_compounds


def _accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float(balanced_accuracy(jnp.asarray(pred), jnp.asarray(true)))


def run_study(
    spectra: np.ndarray,
    labels: np.ndarray,
    n_compounds: int,
    *,
    seeds: tuple[int, ...] = SEEDS,
    n_epochs: int = 40,
    batch_size: int = 512,
) -> dict[str, object]:
    """Run the frozen-vs-joint peak-picking study and report identification accuracy."""
    train_idx, test_idx = stratified_label_split(
        labels, train_fraction=0.8, seed=0, minimum_count_name="spectra"
    )
    x_train, x_test = jnp.asarray(spectra[train_idx]), jnp.asarray(spectra[test_idx])
    y_train, y_test = labels[train_idx], labels[test_idx]

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
        frozen = _build_pipeline(n_compounds, trainable=False, seed=seed)
        train_minibatch(
            frozen, _pipeline_forward, x_train, y_train, n_classes=n_compounds, config=config
        )
        pred_f = np.asarray(jnp.argmax(_pipeline_forward(frozen, x_test), -1))
        frozen_acc.append(_accuracy(pred_f, y_test))

        joint = _build_pipeline(n_compounds, trainable=True, seed=seed)
        train_minibatch(
            joint, _pipeline_forward, x_train, y_train, n_classes=n_compounds, config=config
        )
        pred_j = np.asarray(jnp.argmax(_pipeline_forward(joint, x_test), -1))
        joint_acc.append(_accuracy(pred_j, y_test))
        print(
            f"  [seed {seed}] frozen {frozen_acc[-1]:.4f}  joint {joint_acc[-1]:.4f}  "
            f"gain {100 * (joint_acc[-1] - frozen_acc[-1]):+.2f}pp",
            flush=True,
        )

    significance = paired_significance_test(frozen_acc, joint_acc)
    return {
        "n_compounds": n_compounds,
        "frozen_accuracy": [float(np.mean(frozen_acc)), float(np.std(frozen_acc))],
        "joint_accuracy": [float(np.mean(joint_acc)), float(np.std(joint_acc))],
        "gain_pp": float(100 * (np.mean(joint_acc) - np.mean(frozen_acc))),
        "paired_p_value": float(significance.p_value),
        "paired_significant": bool(significance.significant),
    }


def _smoke() -> None:
    """Exercise the differentiable pipeline end to end on a small synthetic set."""
    spectra, labels, n_compounds = synthesize(6, 1500, seed=0)

    # The ID loss must reach the peak-picker knobs -- verify non-zero gradients there.
    model = _build_pipeline(n_compounds, trainable=True, seed=0)

    def loss_fn(m: _MSPipeline) -> jax.Array:
        logits = _pipeline_forward(m, jnp.asarray(spectra[:32]))
        return jnp.mean((logits - 0.0) ** 2)

    grads = nnx.grad(loss_fn)(model)
    picker_grads = [
        grads.centroider.raw_snthresh.value,
        grads.centroider.raw_peak_width.value,
        grads.isotoper.raw_decay.value,
    ]
    assert all(jnp.isfinite(g).all() for g in picker_grads)
    assert any(jnp.any(g != 0.0) for g in picker_grads)
    print("ID loss reaches peak-picker knobs: gradients flow into snthresh/width/decay", flush=True)

    result = run_study(spectra, labels, n_compounds, seeds=(0, 1), n_epochs=8, batch_size=256)
    print(json.dumps(result, indent=2), flush=True)
    print("PEAK PICKING SMOKE DONE", flush=True)


def main() -> None:
    """Run the synthetic frozen-vs-joint peak-picking study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="small synthetic run")
    args = parser.parse_args()
    if args.smoke:
        _smoke()
        return

    spectra, labels, n_compounds = synthesize(12, 12000, seed=0)
    print("=== MS peak-picking: frozen knobs vs ID-trained peak picker ===", flush=True)
    result = run_study(spectra, labels, n_compounds)
    print(
        f"frozen {result['frozen_accuracy'][0]:.4f} joint {result['joint_accuracy'][0]:.4f} "  # type: ignore[index]
        f"gain {result['gain_pp']:+.2f}pp p={result['paired_p_value']:.3f}",
        flush=True,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"PEAK PICKING DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
