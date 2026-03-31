#!/usr/bin/env python3
"""RNA folding benchmark: DifferentiableRNAFold on ArchiveII.

Evaluates DiffBio's DifferentiableRNAFold operator on ArchiveII RNA
secondary structures using base pair F1, sensitivity, and PPV.

Results are compared against published baselines: ViennaRNA (~0.72 F1),
LinearFold (~0.73), EternaFold (~0.75), CONTRAfold (~0.70).

Usage:
    python benchmarks/rna_structure/bench_rna_fold.py
    python benchmarks/rna_structure/bench_rna_fold.py --quick
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from benchmarks._base import DiffBioBenchmark, DiffBioBenchmarkConfig
from benchmarks._baselines.rna import RNA_FOLD_BASELINES
from benchmarks._metrics.structure import base_pair_metrics
from diffbio.operators.rna_structure.rna_folding import (
    DifferentiableRNAFold,
    RNAFoldConfig,
)
from diffbio.sources.archive_ii import (
    ArchiveIIConfig,
    ArchiveIISource,
)

logger = logging.getLogger(__name__)

# RNA nucleotide mapping for one-hot encoding
_NUC_TO_IDX: dict[str, int] = {"A": 0, "C": 1, "G": 2, "U": 3}
_RNA_ALPHABET_SIZE = 4

_CONFIG = DiffBioBenchmarkConfig(
    name="rna_structure/rna_fold",
    domain="rna_structure",
    n_iterations_quick=5,
    n_iterations_full=20,
)


def encode_rna_sequence(sequence: str) -> jnp.ndarray:
    """Encode an RNA sequence string as a one-hot JAX array.

    Maps A=0, C=1, G=2, U=3. T is treated as U for DNA/RNA
    compatibility. Unknown nucleotides are encoded as uniform
    distributions (0.25 for each).

    Args:
        sequence: RNA sequence string (e.g. ``"GUCUACC"``).

    Returns:
        One-hot encoded array of shape ``(length, 4)``.
    """
    indices = []
    for nuc in sequence.upper():
        if nuc == "T":
            nuc = "U"
        indices.append(_NUC_TO_IDX.get(nuc, -1))

    idx_array = jnp.array(indices, dtype=jnp.int32)
    one_hot = jax.nn.one_hot(idx_array, _RNA_ALPHABET_SIZE)
    unknown_mask = (idx_array < 0).astype(jnp.float32)
    uniform = jnp.full((_RNA_ALPHABET_SIZE,), 1.0 / _RNA_ALPHABET_SIZE)
    one_hot = jnp.where(
        unknown_mask[:, None] > 0.5,
        uniform[None, :],
        one_hot,
    )
    return one_hot


class RNAFoldBenchmark(DiffBioBenchmark):
    """Evaluate DifferentiableRNAFold on the ArchiveII dataset."""

    def __init__(
        self,
        config: DiffBioBenchmarkConfig = _CONFIG,
        *,
        quick: bool = False,
        data_dir: str = ("/media/mahdi/ssd23/Works/RNAFoldAssess/tutorial/processed_data"),
    ) -> None:
        super().__init__(config, quick=quick, data_dir=data_dir)

    def _run_core(self) -> dict[str, Any]:
        """Fold sequences, compute base pair metrics."""
        max_sequences = 5 if self.quick else None

        # 1. Load dataset
        logger.info("Loading ArchiveII dataset...")
        source = ArchiveIISource(
            ArchiveIIConfig(
                data_dir=self.data_dir,
                max_sequences=max_sequences,
            )
        )
        data = source.load()

        entries = data["entries"]
        n_sequences = data["n_sequences"]
        logger.info("  %d sequences loaded", n_sequences)

        # 2. Create operator
        op_config = RNAFoldConfig(temperature=1.0)
        rngs = nnx.Rngs(42)
        operator = DifferentiableRNAFold(op_config, rngs=rngs)

        # 3. Fold each sequence and compute metrics
        logger.info("Running DifferentiableRNAFold...")
        all_metrics: list[dict[str, float]] = []

        for entry in entries:
            seq_onehot = encode_rna_sequence(entry["sequence"])
            input_data = {"sequence": seq_onehot}
            result, _, _ = operator.apply(input_data, {}, None)
            bp_probs_np = np.asarray(result["bp_probs"])

            metrics = base_pair_metrics(bp_probs_np, entry["structure"])
            all_metrics.append(metrics)

            logger.debug(
                "  %s: F1=%.4f sens=%.4f ppv=%.4f (len=%d, pairs=%d)",
                entry["name"],
                metrics["f1"],
                metrics["sensitivity"],
                metrics["ppv"],
                len(entry["sequence"]),
                metrics["n_true_pairs"],
            )

        # 4. Aggregate metrics
        avg_sensitivity = float(np.mean([m["sensitivity"] for m in all_metrics]))
        avg_ppv = float(np.mean([m["ppv"] for m in all_metrics]))
        avg_f1 = float(np.mean([m["f1"] for m in all_metrics]))
        total_true = int(sum(m["n_true_pairs"] for m in all_metrics))
        total_predicted = int(sum(m["n_predicted_pairs"] for m in all_metrics))

        quality: dict[str, float] = {
            "sensitivity": avg_sensitivity,
            "ppv": avg_ppv,
            "f1": avg_f1,
        }

        for key, value in sorted(quality.items()):
            logger.info("  %s: %.4f", key, value)

        # 5. Gradient check input from first sequence
        first_seq = encode_rna_sequence(entries[0]["sequence"])
        grad_input = {"sequence": first_seq}

        def loss_fn(
            model: DifferentiableRNAFold,
            d: dict[str, Any],
        ) -> jnp.ndarray:
            res, _, _ = model.apply(d, {}, None)
            return jnp.sum(res["bp_probs"])

        return {
            "metrics": quality,
            "operator": operator,
            "input_data": grad_input,
            "loss_fn": loss_fn,
            "n_items": len(entries[0]["sequence"]),
            "iterate_fn": lambda: operator.apply(grad_input, {}, None),
            "baselines": RNA_FOLD_BASELINES,
            "dataset_info": {
                "name": "archiveII",
                "n_sequences": n_sequences,
                "total_true_pairs": total_true,
                "total_predicted_pairs": total_predicted,
            },
            "operator_config": {
                "temperature": op_config.temperature,
                "min_hairpin_loop": op_config.min_hairpin_loop,
                "max_sequences": max_sequences,
            },
            "operator_name": "DifferentiableRNAFold",
            "dataset_name": "archiveII",
        }


def main() -> None:
    """CLI entry point."""
    DiffBioBenchmark.cli_main(
        RNAFoldBenchmark,
        _CONFIG,
        data_dir=("/media/mahdi/ssd23/Works/RNAFoldAssess/tutorial/processed_data"),
    )


if __name__ == "__main__":
    main()
