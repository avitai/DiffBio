#!/usr/bin/env python3
"""MSA benchmark: SoftProgressiveMSA on BAliBASE reference alignments.

Evaluates DiffBio's SoftProgressiveMSA operator on balifam100, a curated
subset of BAliBASE protein family alignments, using SP (Sum of Pairs)
and TC (Total Column) scores. The sequence encoder is trained via
gradient descent on alignment quality (unsupervised) before evaluation.

Results are compared against published baselines: MAFFT, ClustalW,
MUSCLE, and T-Coffee.

Usage:
    python benchmarks/alignment/bench_msa.py
    python benchmarks/alignment/bench_msa.py --quick
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from benchmarks._base import DiffBioBenchmark, DiffBioBenchmarkConfig
from benchmarks._baselines.alignment import MSA_BASELINES
from benchmarks._metrics.alignment import sp_score, tc_score
from benchmarks._optimizers import create_benchmark_optimizer
from diffbio.losses.alignment_losses import AlignmentScoreLoss
from benchmarks.alignment._encoding import onehot_encode_sequence
from diffbio.operators.alignment import PROTEIN_ALPHABET
from diffbio.operators.alignment.soft_msa import (
    SoftProgressiveMSA,
    SoftProgressiveMSAConfig,
)
from diffbio.sources.balifam import BalifamConfig, BalifamSource

logger = logging.getLogger(__name__)

_CONFIG = DiffBioBenchmarkConfig(
    name="alignment/msa",
    domain="alignment",
    n_iterations_quick=5,
    n_iterations_full=10,
)


def _decode_alignment(
    aligned_sequences: jnp.ndarray,
    seq_lengths: list[int],
    seq_names: list[str],
) -> list[tuple[str, str]]:
    """Decode soft-aligned one-hot sequences back to strings.

    Takes the argmax of each position to recover discrete residue
    identities. Positions where the max probability is below a
    threshold are treated as gaps.

    Args:
        aligned_sequences: Array (n_seqs, aligned_len, alphabet).
        seq_lengths: Original (unpadded) lengths for each sequence.
        seq_names: Sequence names.

    Returns:
        List of (name, aligned_sequence_string) tuples.
    """
    aligned_np = np.asarray(aligned_sequences)
    n_seqs, aligned_len, _ = aligned_np.shape
    gap_threshold = 0.05

    decoded: list[tuple[str, str]] = []
    for seq_idx in range(n_seqs):
        chars: list[str] = []
        for pos in range(aligned_len):
            row = aligned_np[seq_idx, pos]
            max_val = float(np.max(row))
            if max_val < gap_threshold:
                chars.append(".")
            else:
                aa_idx = int(np.argmax(row))
                chars.append(PROTEIN_ALPHABET[aa_idx])
        decoded.append((seq_names[seq_idx], "".join(chars)))

    return decoded


def _extract_ref_sequences_from_input(
    family: dict[str, Any],
) -> list[tuple[str, str]]:
    """Extract unaligned sequences matching reference names.

    Reference alignments use short names (e.g., ``IF2G_HUMAN``)
    while input files use full UniProt identifiers with ranges
    (e.g., ``A0A452HWX8_9SAUR/30-374``). This function extracts
    the raw (unaligned) sequences from the reference alignment
    by stripping gap characters, since the reference names may
    not directly appear in the input file.

    Args:
        family: Family dict from BalifamSource.

    Returns:
        List of (name, unaligned_sequence) tuples for reference
        sequences only, with gaps and case differences removed.
    """
    result: list[tuple[str, str]] = []
    for name, aligned_seq in family["reference"]:
        # Strip gaps to recover unaligned sequence
        unaligned = aligned_seq.replace(".", "").upper()
        result.append((name, unaligned))
    return result


class MSABenchmark(DiffBioBenchmark):
    """Evaluate SoftProgressiveMSA on BAliBASE reference alignments."""

    def __init__(
        self,
        config: DiffBioBenchmarkConfig = _CONFIG,
        *,
        quick: bool = False,
        data_dir: str = "/media/mahdi/ssd23/Works/balifam",
    ) -> None:
        super().__init__(config, quick=quick, data_dir=data_dir)

    def _run_core(self) -> dict[str, Any]:
        """Align families, compute SP/TC scores."""
        max_families = 3 if self.quick else None
        max_seq_length = 50 if self.quick else 200

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
        op_config = SoftProgressiveMSAConfig(
            max_seq_length=max_seq_length,
            hidden_dim=64,
            num_layers=2,
            alphabet_size=20,
            temperature=1.0,
            gap_open_penalty=-10.0,
            gap_extend_penalty=-1.0,
        )
        rngs = nnx.Rngs(42)
        operator = SoftProgressiveMSA(op_config, rngs=rngs)

        # 2b. Train sequence encoder on alignment quality (unsupervised).
        # Use the first family's sequences to optimise the encoder so
        # that the alignment score is maximised.
        train_family = families[0]
        train_seqs = _extract_ref_sequences_from_input(train_family)
        if len(train_seqs) >= 2:
            train_raw = [seq[:max_seq_length] for _, seq in train_seqs]
            train_lens = [len(s) for s in train_raw]
            train_padded = max(train_lens)
            train_encoded = jnp.stack(
                [onehot_encode_sequence(s, train_padded, 20) for s in train_raw]
            )
            train_input = {"sequences": train_encoded}

            n_steps = 100 if self.quick else 300
            logger.info("Training MSA encoder (%d steps)...", n_steps)
            AlignmentScoreLoss(rngs=rngs)
            opt = nnx.Optimizer(
                operator,
                create_benchmark_optimizer(learning_rate=1e-3),
                wrt=nnx.Param,
            )

            @nnx.jit
            def _msa_step(
                model: SoftProgressiveMSA,
                optimizer: nnx.Optimizer,
                data: dict[str, jax.Array],
            ) -> jax.Array:
                def _loss(m: SoftProgressiveMSA) -> jax.Array:
                    res, _, _ = m.apply(data, {}, None)
                    # Maximise alignment score (negate for minimisation)
                    return -jnp.mean(res["alignment_scores"])

                loss, grads = nnx.value_and_grad(_loss)(model)
                optimizer.update(model, grads)
                return loss

            for step in range(n_steps):
                loss_val = _msa_step(operator, opt, train_input)
                if (step + 1) % 50 == 0:
                    logger.info(
                        "  step %d/%d  loss=%.4f",
                        step + 1,
                        n_steps,
                        float(loss_val),
                    )

        # 3. Evaluate on each family
        logger.info("Running SoftProgressiveMSA on families...")
        sp_scores: list[float] = []
        tc_scores: list[float] = []

        for family in families:
            family_id = family["family_id"]
            ref_seqs = _extract_ref_sequences_from_input(family)

            if len(ref_seqs) < 2:
                logger.info(
                    "Skipping %s: fewer than 2 ref sequences",
                    family_id,
                )
                continue

            names = [name for name, _ in ref_seqs]
            raw_seqs = [seq[:max_seq_length] for _, seq in ref_seqs]
            seq_lengths = [len(s) for s in raw_seqs]
            padded_len = max(seq_lengths)

            encoded = jnp.stack([onehot_encode_sequence(seq, padded_len, 20) for seq in raw_seqs])

            input_data = {"sequences": encoded}
            result_data, _, _ = operator.apply(input_data, {}, None)

            aligned = result_data["aligned_sequences"]
            predicted = _decode_alignment(aligned, seq_lengths, names)

            sp = sp_score(predicted, family["reference"])
            tc = tc_score(predicted, family["reference"])

            sp_scores.append(sp)
            tc_scores.append(tc)

            logger.info(
                "  %s: SP=%.4f, TC=%.4f (%d seqs)",
                family_id,
                sp,
                tc,
                len(ref_seqs),
            )

        # 4. Aggregate scores
        mean_sp = sum(sp_scores) / len(sp_scores) if sp_scores else 0.0
        mean_tc = sum(tc_scores) / len(tc_scores) if tc_scores else 0.0
        logger.info("  Mean SP: %.4f, Mean TC: %.4f", mean_sp, mean_tc)

        # 5. Prepare gradient check input from first family
        sample_family = families[0]
        sample_seqs = _extract_ref_sequences_from_input(sample_family)
        sample_seqs = sample_seqs[:5]
        sample_raw = [s[:max_seq_length] for _, s in sample_seqs]
        sample_len = max(len(s) for s in sample_raw)
        sample_encoded = jnp.stack([onehot_encode_sequence(s, sample_len, 20) for s in sample_raw])
        sample_input = {"sequences": sample_encoded}

        def loss_fn(model: SoftProgressiveMSA, d: dict[str, Any]) -> jnp.ndarray:
            res, _, _ = model.apply(d, {}, None)
            return jnp.sum(res["aligned_sequences"])

        quality = {
            "sp_score": mean_sp,
            "tc_score": mean_tc,
        }

        return {
            "metrics": quality,
            "operator": operator,
            "input_data": sample_input,
            "loss_fn": loss_fn,
            "n_items": len(sample_seqs),
            "iterate_fn": lambda: operator.apply(sample_input, {}, None),
            "baselines": MSA_BASELINES,
            "dataset_info": {
                "name": "balifam100",
                "n_families": n_families,
                "n_evaluated": len(sp_scores),
            },
            "operator_config": {
                "max_seq_length": max_seq_length,
                "hidden_dim": op_config.hidden_dim,
                "num_layers": op_config.num_layers,
                "alphabet_size": op_config.alphabet_size,
                "temperature": op_config.temperature,
                "max_families": max_families,
            },
            "operator_name": "SoftProgressiveMSA",
            "dataset_name": "balifam100",
        }


def main() -> None:
    """CLI entry point."""
    DiffBioBenchmark.cli_main(
        MSABenchmark,
        _CONFIG,
        data_dir="/media/mahdi/ssd23/Works/balifam",
    )


if __name__ == "__main__":
    main()
