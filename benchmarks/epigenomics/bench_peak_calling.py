#!/usr/bin/env python3
"""Peak calling benchmark: DifferentiablePeakCaller on ENCODE ChIP-seq.

Evaluates DiffBio's DifferentiablePeakCaller on real ENCODE CTCF
ChIP-seq narrowPeak data from K562 cells. A synthetic coverage signal
is reconstructed from the known peak coordinates and signal values,
then fed through the differentiable peak caller. The CNN filters
are trained via gradient descent on a self-supervised coverage
reconstruction loss before evaluation. Called peaks are compared
against the ENCODE ground truth.

Metrics: precision, recall, F1, Jaccard index.

Results are compared against published baselines: MACS2 (~0.87 F1),
HOMER (~0.82 F1), Genrich (~0.89 F1).

Usage:
    python benchmarks/epigenomics/bench_peak_calling.py
    python benchmarks/epigenomics/bench_peak_calling.py --quick
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from benchmarks._base import (
    DiffBioBenchmark,
    DiffBioBenchmarkConfig,
)
from benchmarks._baselines.epigenomics import PEAK_CALLING_BASELINES
from benchmarks._optimizers import create_benchmark_optimizer
from diffbio.operators.epigenomics.peak_calling import (
    DifferentiablePeakCaller,
    PeakCallerConfig,
)
from diffbio.sources.encode_peaks import (
    ENCODEPeakConfig,
    ENCODEPeakSource,
)

logger = logging.getLogger(__name__)

_CONFIG = DiffBioBenchmarkConfig(
    name="epigenomics/peak_calling",
    domain="epigenomics",
    n_iterations_quick=5,
    n_iterations_full=20,
)

# Coverage signal parameters
_REGION_PADDING = 500  # bp padding around peaks
_BACKGROUND_NOISE_STD = 0.5
_PEAK_SHAPE_STD_FRACTION = 0.25  # Gaussian std as fraction of width


def build_coverage_signal(
    starts: np.ndarray,
    ends: np.ndarray,
    signals: np.ndarray,
    region_start: int,
    region_end: int,
    *,
    noise_std: float = _BACKGROUND_NOISE_STD,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a 1D coverage signal from known peak coordinates.

    Creates a synthetic coverage profile by placing Gaussian-shaped
    peaks at known positions with amplitudes proportional to the
    ENCODE signal values. Background noise is added to simulate
    real ChIP-seq coverage.

    Args:
        starts: Peak start positions (absolute genomic coordinates).
        ends: Peak end positions (absolute genomic coordinates).
        signals: Signal enrichment values for each peak.
        region_start: Start of the genomic region to generate.
        region_end: End of the genomic region to generate.
        noise_std: Standard deviation of background noise.
        seed: Random seed for reproducible noise generation.

    Returns:
        Tuple of (coverage, truth_mask):
            - coverage: 1D float array of length (region_end - region_start).
            - truth_mask: Binary array marking true peak positions.
    """
    length = region_end - region_start
    coverage = np.zeros(length, dtype=np.float32)
    truth_mask = np.zeros(length, dtype=np.float32)
    positions = np.arange(length, dtype=np.float32)

    for start, end, signal in zip(starts, ends, signals):
        # Convert to relative coordinates
        rel_start = int(start - region_start)
        rel_end = int(end - region_start)

        # Skip peaks outside the region
        if rel_end <= 0 or rel_start >= length:
            continue

        # Clip to region boundaries
        rel_start = max(0, rel_start)
        rel_end = min(length, rel_end)

        # Mark truth mask
        truth_mask[rel_start:rel_end] = 1.0

        # Add Gaussian-shaped peak
        center = (rel_start + rel_end) / 2.0
        width = max(rel_end - rel_start, 1)
        std = max(width * _PEAK_SHAPE_STD_FRACTION, 1.0)
        gaussian = signal * np.exp(-0.5 * ((positions - center) / std) ** 2)
        coverage += gaussian

    # Add background noise
    rng = np.random.default_rng(seed)
    noise = np.abs(rng.normal(0.0, noise_std, size=length).astype(np.float32))
    coverage += noise

    return coverage, truth_mask


def compute_peak_metrics(
    predicted_mask: np.ndarray,
    truth_mask: np.ndarray,
) -> dict[str, float]:
    """Compute precision, recall, F1, and Jaccard for peak regions.

    Operates on binary masks where 1 indicates a peak position.
    Metrics are computed at base-pair resolution.

    Args:
        predicted_mask: Binary array of predicted peak positions.
        truth_mask: Binary array of true peak positions.

    Returns:
        Dict with keys: precision, recall, f1, jaccard.
    """
    pred_bool = predicted_mask > 0.5
    truth_bool = truth_mask > 0.5

    true_positives = float(np.sum(pred_bool & truth_bool))
    predicted_positives = float(np.sum(pred_bool))
    actual_positives = float(np.sum(truth_bool))
    union = float(np.sum(pred_bool | truth_bool))

    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    recall = true_positives / actual_positives if actual_positives > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    jaccard = true_positives / union if union > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
    }


def _select_dense_region(
    starts: np.ndarray,
    ends: np.ndarray,
    n_peaks_target: int,
) -> tuple[int, int, np.ndarray]:
    """Select a contiguous genomic region with many peaks.

    Finds a stretch of ``n_peaks_target`` consecutive peaks and
    returns the region spanning from the first peak's start to the
    last peak's end, with padding.

    Args:
        starts: Sorted peak start positions.
        ends: Sorted peak end positions.
        n_peaks_target: Number of peaks to include in the region.

    Returns:
        Tuple of (region_start, region_end, peak_indices) where
        peak_indices are the indices of peaks within the region.
    """
    n_peaks = len(starts)
    if n_peaks <= n_peaks_target:
        region_start = int(starts[0]) - _REGION_PADDING
        region_end = int(ends[-1]) + _REGION_PADDING
        return region_start, region_end, np.arange(n_peaks)

    # Find the densest window of n_peaks_target consecutive peaks
    # (smallest span from first to last peak in the window)
    spans = ends[n_peaks_target - 1 :] - starts[: n_peaks - n_peaks_target + 1]
    best_offset = int(np.argmin(spans))
    indices = np.arange(best_offset, best_offset + n_peaks_target)

    region_start = int(starts[best_offset]) - _REGION_PADDING
    region_end = int(ends[best_offset + n_peaks_target - 1]) + _REGION_PADDING
    return region_start, region_end, indices


class PeakCallingBenchmark(DiffBioBenchmark):
    """Evaluate DifferentiablePeakCaller on ENCODE ChIP-seq data."""

    def __init__(
        self,
        config: DiffBioBenchmarkConfig = _CONFIG,
        *,
        quick: bool = False,
        data_dir: str = "",
    ) -> None:
        super().__init__(config, quick=quick, data_dir=data_dir)

    def _run_core(self) -> dict[str, Any]:
        """Load ENCODE peaks, build coverage, run peak caller."""
        n_peaks_region = 50 if self.quick else 200

        # 1. Load ENCODE peaks
        logger.info("Loading ENCODE narrowPeak data...")
        source_config = ENCODEPeakConfig(
            chromosome="chr22",
            max_peaks=None,
        )
        source = ENCODEPeakSource(source_config)
        data = source.load()

        starts = data["starts"]
        ends = data["ends"]
        signals = data["signal_values"]
        n_total = data["n_peaks"]
        logger.info("  %d peaks on chr22", n_total)

        # 2. Select a dense region for evaluation
        region_start, region_end, region_idx = _select_dense_region(starts, ends, n_peaks_region)
        region_starts = starts[region_idx]
        region_ends = ends[region_idx]
        region_signals = signals[region_idx]
        region_length = region_end - region_start
        logger.info(
            "  Region: %d-%d (%d bp, %d peaks)",
            region_start,
            region_end,
            region_length,
            len(region_idx),
        )

        # 3. Build coverage signal from peak coordinates
        coverage, truth_mask = build_coverage_signal(
            region_starts,
            region_ends,
            region_signals,
            region_start,
            region_end,
        )

        # 4. Create and run operator
        op_config = PeakCallerConfig(
            window_size=200,
            num_filters=32,
            kernel_sizes=(5, 11, 21),
            threshold=0.5,
            temperature=1.0,
        )
        rngs = nnx.Rngs(42)
        operator = DifferentiablePeakCaller(op_config, rngs=rngs)

        coverage_jax = jnp.array(coverage)
        jnp.array(truth_mask)
        input_data = {"coverage": coverage_jax}

        # Train CNN peak detector: self-supervised loss encouraging
        # peak_probabilities to reconstruct the normalised coverage shape.
        # High coverage => high peak probability, low => low.
        n_steps = 200 if self.quick else 500
        logger.info("Training peak caller (%d steps)...", n_steps)
        opt = nnx.Optimizer(
            operator,
            create_benchmark_optimizer(learning_rate=1e-3),
            wrt=nnx.Param,
        )
        cov_norm = coverage_jax / (jnp.max(coverage_jax) + 1e-8)

        @nnx.jit
        def _peak_step(
            model: DifferentiablePeakCaller,
            optimizer: nnx.Optimizer,
            data: dict[str, jax.Array],
            target: jax.Array,
        ) -> jax.Array:
            def _loss(m: DifferentiablePeakCaller) -> jax.Array:
                res, _, _ = m.apply(data, {}, None)
                probs = res["peak_probabilities"]
                return jnp.mean(optax.sigmoid_binary_cross_entropy(probs, target))

            loss, grads = nnx.value_and_grad(_loss)(model)
            optimizer.update(model, grads)
            return loss

        for step in range(n_steps):
            loss_val = _peak_step(operator, opt, input_data, cov_norm)
            if (step + 1) % 50 == 0:
                logger.info("  step %d/%d  loss=%.4f", step + 1, n_steps, float(loss_val))

        result, _, _ = operator.apply(input_data, {}, None)

        # 5. Threshold peak_probabilities to get binary predictions
        peak_probs = np.asarray(result["peak_probabilities"])
        predicted_mask = (peak_probs > 0.5).astype(np.float32)

        # 6. Compute metrics
        quality = compute_peak_metrics(predicted_mask, truth_mask)
        for key, value in sorted(quality.items()):
            logger.info("  %s: %.4f", key, value)

        # 7. Gradient check setup
        def loss_fn(
            model: DifferentiablePeakCaller,
            d: dict[str, Any],
        ) -> jnp.ndarray:
            res, _, _ = model.apply(d, {}, None)
            return jnp.sum(res["peak_probabilities"])

        return {
            "metrics": quality,
            "operator": operator,
            "input_data": input_data,
            "loss_fn": loss_fn,
            "n_items": region_length,
            "iterate_fn": lambda: operator.apply(input_data, {}, None),
            "baselines": PEAK_CALLING_BASELINES,
            "dataset_info": {
                "name": "ENCODE_CTCF_K562",
                "chromosome": "chr22",
                "n_peaks_total": n_total,
                "n_peaks_region": len(region_idx),
                "region_start": region_start,
                "region_end": region_end,
                "region_length": region_length,
            },
            "operator_config": {
                "window_size": op_config.window_size,
                "num_filters": op_config.num_filters,
                "threshold": op_config.threshold,
                "temperature": op_config.temperature,
            },
            "operator_name": "DifferentiablePeakCaller",
            "dataset_name": "ENCODE_CTCF_K562",
        }


def main() -> None:
    """CLI entry point."""
    DiffBioBenchmark.cli_main(
        PeakCallingBenchmark,
        _CONFIG,
    )


if __name__ == "__main__":
    main()
