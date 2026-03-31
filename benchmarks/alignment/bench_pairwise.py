#!/usr/bin/env python3
"""Pairwise alignment benchmark: SmoothSmithWaterman on BAliBASE families.

Evaluates DiffBio's SmoothSmithWaterman operator on balifam100, a curated
subset of BAliBASE protein family alignments. For each family, all pairs
of reference sequences are aligned using the differentiable Smith-Waterman
implementation with BLOSUM62 scoring.

Results are compared against published baselines: BLAST, SSEARCH, and
FASTA (pairwise alignment tools).

Usage:
    python benchmarks/alignment/bench_pairwise.py
    python benchmarks/alignment/bench_pairwise.py --quick
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import jax.numpy as jnp
import numpy as np

from benchmarks._base import DiffBioBenchmark, DiffBioBenchmarkConfig
from benchmarks._baselines.alignment import PAIRWISE_BASELINES
from benchmarks.alignment._encoding import onehot_encode_sequence
from diffbio.operators.alignment.scoring import get_blosum62
from diffbio.operators.alignment.smith_waterman import (
    SmithWatermanConfig,
    SmoothSmithWaterman,
)
from diffbio.sources.balifam import BalifamConfig, BalifamSource

logger = logging.getLogger(__name__)

_CONFIG = DiffBioBenchmarkConfig(
    name="alignment/pairwise",
    domain="alignment",
    n_iterations_quick=5,
    n_iterations_full=10,
)

_MAX_FAMILIES_QUICK = 3
_MAX_FAMILIES_FULL = 10
_MAX_SEQ_LENGTH_QUICK = 50
_MAX_SEQ_LENGTH_FULL = 100


def _strip_gaps(aligned_seq: str) -> str:
    """Remove gap characters from an aligned sequence.

    Args:
        aligned_seq: Aligned sequence with dots as gap chars.

    Returns:
        Unaligned sequence string (uppercase, no gaps).
    """
    return aligned_seq.replace(".", "").upper()


class PairwiseBenchmark(DiffBioBenchmark):
    """Evaluate SmoothSmithWaterman on BAliBASE pairwise alignments."""

    def __init__(
        self,
        config: DiffBioBenchmarkConfig = _CONFIG,
        *,
        quick: bool = False,
        data_dir: str = "/media/mahdi/ssd23/Works/balifam",
    ) -> None:
        super().__init__(config, quick=quick, data_dir=data_dir)

    def _run_core(self) -> dict[str, Any]:
        """Align all sequence pairs per family, compute scores."""
        if self.quick:
            max_families = _MAX_FAMILIES_QUICK
            max_seq_length = _MAX_SEQ_LENGTH_QUICK
        else:
            max_families = _MAX_FAMILIES_FULL
            max_seq_length = _MAX_SEQ_LENGTH_FULL

        # 1. Load dataset
        logger.info("Loading balifam100 families...")
        source = BalifamSource(
            BalifamConfig(
                data_dir=self.data_dir,
                tier=100,
                max_families=max_families,
            )
        )
        families = source.load()
        n_families = len(families)
        logger.info("  %d families loaded", n_families)

        # 2. Create operator
        scoring_matrix = get_blosum62()
        op_config = SmithWatermanConfig(
            temperature=1.0,
            gap_open=-10.0,
            gap_extend=-1.0,
        )
        operator = SmoothSmithWaterman(
            op_config, scoring_matrix=scoring_matrix
        )

        # 3. Evaluate pairwise alignments for each family
        logger.info("Running SmoothSmithWaterman on sequence pairs...")
        all_scores: list[float] = []
        n_pairs_total = 0

        for family in families:
            family_id = family["family_id"]
            ref_seqs = [
                (name, _strip_gaps(seq))
                for name, seq in family["reference"]
            ]

            if len(ref_seqs) < 2:
                logger.info(
                    "Skipping %s: fewer than 2 ref sequences",
                    family_id,
                )
                continue

            family_scores: list[float] = []
            pairs = list(itertools.combinations(ref_seqs, 2))

            for (name_a, seq_a), (name_b, seq_b) in pairs:
                truncated_a = seq_a[:max_seq_length]
                truncated_b = seq_b[:max_seq_length]

                encoded_a = onehot_encode_sequence(
                    truncated_a, len(truncated_a)
                )
                encoded_b = onehot_encode_sequence(
                    truncated_b, len(truncated_b)
                )

                input_data = {"seq1": encoded_a, "seq2": encoded_b}
                result_data, _, _ = operator.apply(
                    input_data, {}, None
                )

                score = float(result_data["score"])
                family_scores.append(score)

            all_scores.extend(family_scores)
            n_pairs_total += len(pairs)
            avg_family = (
                sum(family_scores) / len(family_scores)
                if family_scores
                else 0.0
            )
            logger.info(
                "  %s: avg_score=%.4f (%d pairs)",
                family_id,
                avg_family,
                len(pairs),
            )

        # 4. Aggregate metrics
        avg_score = (
            sum(all_scores) / len(all_scores)
            if all_scores
            else 0.0
        )
        n_finite = sum(
            1 for s in all_scores if np.isfinite(s)
        )
        logger.info(
            "  Mean alignment score: %.4f (%d/%d finite)",
            avg_score,
            n_finite,
            len(all_scores),
        )

        # 5. Prepare gradient check input from first pair
        first_family = families[0]
        sample_seqs = [
            _strip_gaps(seq)[:max_seq_length]
            for _, seq in first_family["reference"][:2]
        ]
        # Ensure we have at least 2 sequences for gradient check
        if len(sample_seqs) < 2:
            sample_seqs = [sample_seqs[0], sample_seqs[0]]

        sample_a = onehot_encode_sequence(
            sample_seqs[0], len(sample_seqs[0])
        )
        sample_b = onehot_encode_sequence(
            sample_seqs[1], len(sample_seqs[1])
        )
        sample_input = {"seq1": sample_a, "seq2": sample_b}

        def loss_fn(
            model: SmoothSmithWaterman,
            data: dict[str, Any],
        ) -> jnp.ndarray:
            """Scalar loss for gradient flow check."""
            res, _, _ = model.apply(data, {}, None)
            return res["score"]

        metrics = {
            "avg_alignment_score": avg_score,
            "n_pairs_evaluated": float(n_pairs_total),
            "alignment_scores_finite": float(n_finite),
        }

        return {
            "metrics": metrics,
            "operator": operator,
            "input_data": sample_input,
            "loss_fn": loss_fn,
            "n_items": n_pairs_total,
            "iterate_fn": lambda: operator.apply(
                sample_input, {}, None
            ),
            "baselines": PAIRWISE_BASELINES,
            "dataset_info": {
                "name": "balifam100",
                "n_families": n_families,
                "n_pairs": n_pairs_total,
            },
            "operator_config": {
                "temperature": op_config.temperature,
                "gap_open": op_config.gap_open,
                "gap_extend": op_config.gap_extend,
                "max_seq_length": max_seq_length,
                "max_families": max_families,
            },
            "operator_name": "SmoothSmithWaterman",
            "dataset_name": "balifam100",
        }


def main() -> None:
    """CLI entry point."""
    DiffBioBenchmark.cli_main(
        PairwiseBenchmark,
        _CONFIG,
        data_dir="/media/mahdi/ssd23/Works/balifam",
    )


if __name__ == "__main__":
    main()
