#!/usr/bin/env python3
"""MSA benchmark: SoftProgressiveMSA on BAliBASE reference alignments.

Evaluates DiffBio's SoftProgressiveMSA operator on balifam100, a curated
subset of BAliBASE protein family alignments, using SP (Sum of Pairs)
and TC (Total Column) scores.

Results are compared against published baselines: MAFFT, ClustalW,
MUSCLE, and T-Coffee.

Usage:
    python benchmarks/alignment/bench_msa.py
    python benchmarks/alignment/bench_msa.py --quick
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import jax.numpy as jnp
from calibrax.core.models import Metric
from calibrax.core.result import BenchmarkResult
from calibrax.profiling.timing import TimingCollector
from flax import nnx

from benchmarks._baselines.alignment import MSA_BASELINES
from benchmarks._gradient import check_gradient_flow
from benchmarks._metrics.alignment import sp_score, tc_score
from diffbio.operators.alignment import PROTEIN_ALPHABET
from diffbio.operators.alignment.soft_msa import (
    SoftProgressiveMSA,
    SoftProgressiveMSAConfig,
)
from diffbio.sources.balifam import BalifamConfig, BalifamSource

logger = logging.getLogger(__name__)

_ALPHABET_INDEX: dict[str, int] = {
    aa: i for i, aa in enumerate(PROTEIN_ALPHABET)
}


def _onehot_encode_sequence(
    sequence: str,
    max_length: int,
    alphabet_size: int = 20,
) -> jnp.ndarray:
    """One-hot encode a protein sequence, padded to max_length.

    Unknown residues (not in the standard 20 amino acid alphabet)
    are encoded as uniform distributions over all residues.

    Args:
        sequence: Amino acid sequence string (uppercase).
        max_length: Pad/truncate to this length.
        alphabet_size: Size of amino acid alphabet.

    Returns:
        One-hot array of shape (max_length, alphabet_size).
    """
    import numpy as np  # noqa: PLC0415

    result = np.zeros((max_length, alphabet_size), dtype=np.float32)
    for i, aa in enumerate(sequence[:max_length]):
        idx = _ALPHABET_INDEX.get(aa.upper())
        if idx is not None:
            result[i, idx] = 1.0
        else:
            # Unknown residue: uniform
            result[i, :] = 1.0 / alphabet_size
    return jnp.array(result)


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
    import numpy as np  # noqa: PLC0415

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


class MSABenchmark:
    """Evaluate SoftProgressiveMSA on BAliBASE reference alignments.

    Computes SP and TC scores across balifam100 families and
    compares against published MSA tool baselines.

    Args:
        quick: If True, use only 3 families with short sequences.
        data_dir: Root directory of the balifam repository.
    """

    def __init__(
        self,
        *,
        quick: bool = False,
        data_dir: str = "/media/mahdi/ssd23/Works/balifam",
    ) -> None:
        self.quick = quick
        self.data_dir = data_dir

    def run(self) -> BenchmarkResult:
        """Execute the benchmark and return a calibrax result."""
        max_families = 3 if self.quick else None
        max_seq_length = 50 if self.quick else 200

        # 1. Load dataset via DataSource
        print("Loading balifam100 families...")
        source_config = BalifamConfig(
            data_dir=self.data_dir,
            tier=100,
            max_families=max_families,
        )
        source = BalifamSource(source_config)
        families = source.load()
        n_families = len(families)
        print(f"  {n_families} families loaded")

        # 2. Create operator
        config = SoftProgressiveMSAConfig(
            max_seq_length=max_seq_length,
            hidden_dim=64,
            num_layers=2,
            alphabet_size=20,
            temperature=1.0,
            gap_open_penalty=-10.0,
            gap_extend_penalty=-1.0,
        )
        rngs = nnx.Rngs(42)
        operator = SoftProgressiveMSA(config, rngs=rngs)

        # 3. Evaluate on each family
        print("Running SoftProgressiveMSA on families...")
        sp_scores: list[float] = []
        tc_scores: list[float] = []
        family_results: list[dict[str, Any]] = []

        for family in families:
            family_id = family["family_id"]
            ref_seqs = _extract_ref_sequences_from_input(family)

            # Skip families with fewer than 2 reference sequences
            if len(ref_seqs) < 2:
                logger.info(
                    "Skipping %s: fewer than 2 ref sequences",
                    family_id,
                )
                continue

            # Truncate sequences to max_seq_length
            names = [name for name, _ in ref_seqs]
            raw_seqs = [seq[:max_seq_length] for _, seq in ref_seqs]
            seq_lengths = [len(s) for s in raw_seqs]

            # Pad all sequences to the same length
            padded_len = max(seq_lengths)

            # One-hot encode
            encoded = jnp.stack([
                _onehot_encode_sequence(seq, padded_len, 20)
                for seq in raw_seqs
            ])

            # Run operator
            input_data = {"sequences": encoded}
            result_data, _, _ = operator.apply(input_data, {}, None)

            # Decode alignment
            aligned = result_data["aligned_sequences"]
            predicted = _decode_alignment(aligned, seq_lengths, names)

            # Compute metrics against reference
            sp = sp_score(predicted, family["reference"])
            tc = tc_score(predicted, family["reference"])

            sp_scores.append(sp)
            tc_scores.append(tc)
            family_results.append({
                "family_id": family_id,
                "sp_score": sp,
                "tc_score": tc,
                "n_sequences": len(ref_seqs),
            })

            print(
                f"  {family_id}: SP={sp:.4f}, TC={tc:.4f} "
                f"({len(ref_seqs)} seqs)"
            )

        # 4. Aggregate scores
        mean_sp = (
            sum(sp_scores) / len(sp_scores) if sp_scores else 0.0
        )
        mean_tc = (
            sum(tc_scores) / len(tc_scores) if tc_scores else 0.0
        )
        print(f"\n  Mean SP: {mean_sp:.4f}, Mean TC: {mean_tc:.4f}")

        # 5. Check gradient flow
        print("Checking gradient flow...")
        sample_family = families[0]
        sample_seqs = _extract_ref_sequences_from_input(sample_family)
        sample_seqs = sample_seqs[:5]  # Limit for speed
        sample_names = [n for n, _ in sample_seqs]
        sample_raw = [s[:max_seq_length] for _, s in sample_seqs]
        sample_len = max(len(s) for s in sample_raw)
        sample_encoded = jnp.stack([
            _onehot_encode_sequence(s, sample_len, 20)
            for s in sample_raw
        ])
        sample_input = {"sequences": sample_encoded}

        def loss_fn(
            model: SoftProgressiveMSA, d: dict[str, Any]
        ) -> jnp.ndarray:
            res, _, _ = model.apply(d, {}, None)
            return jnp.sum(res["aligned_sequences"])

        grad = check_gradient_flow(loss_fn, operator, sample_input)
        print(
            f"  Gradient norm: {grad.gradient_norm:.4f}, "
            f"nonzero: {grad.gradient_nonzero}"
        )

        # 6. Measure throughput
        print("Measuring throughput...")
        n_iterations = 5 if self.quick else 10
        collector = TimingCollector(warmup_iterations=1)
        timing = collector.measure_iteration(
            iterator=iter(range(n_iterations)),
            num_batches=n_iterations,
            process_fn=lambda _: operator.apply(
                sample_input, {}, None
            ),
            count_fn=lambda _: len(sample_seqs),
        )
        seqs_per_sec = timing.num_elements / timing.wall_clock_sec
        print(f"  {seqs_per_sec:.0f} seqs/sec")

        # 7. Print comparison table
        print("\nComparison Table:")
        header = f"{'Method':<20} {'SP Score':>10} {'TC Score':>10}"
        print(header)
        print("-" * len(header))
        print(
            f"{'DiffBio MSA':<20} "
            f"{mean_sp:>10.4f} "
            f"{mean_tc:>10.4f}"
        )
        for name, point in MSA_BASELINES.items():
            sp_val = point.metrics.get(
                "sp_score", Metric(value=0)
            ).value
            tc_val = point.metrics.get(
                "tc_score", Metric(value=0)
            ).value
            print(f"{name:<20} {sp_val:>10.4f} {tc_val:>10.4f}")

        # 8. Build calibrax BenchmarkResult
        metrics = {
            "sp_score": Metric(value=mean_sp),
            "tc_score": Metric(value=mean_tc),
            "gradient_norm": Metric(value=grad.gradient_norm),
            "gradient_nonzero": Metric(
                value=1.0 if grad.gradient_nonzero else 0.0
            ),
            "seqs_per_sec": Metric(value=seqs_per_sec),
        }

        return BenchmarkResult(
            name="alignment/msa",
            domain="diffbio_benchmarks",
            tags={
                "operator": "SoftProgressiveMSA",
                "dataset": "balifam100",
                "framework": "diffbio",
            },
            timing=timing,
            metrics=metrics,
            config={
                "max_seq_length": max_seq_length,
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
                "alphabet_size": config.alphabet_size,
                "temperature": config.temperature,
                "quick": self.quick,
                "max_families": max_families,
            },
            metadata={
                "dataset_info": {
                    "name": "balifam100",
                    "n_families": n_families,
                    "n_evaluated": len(sp_scores),
                },
                "family_results": family_results,
                "baselines": {
                    name: {
                        k: v.value
                        for k, v in p.metrics.items()
                    }
                    for name, p in MSA_BASELINES.items()
                },
            },
        )


def main() -> None:
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)
    quick = "--quick" in sys.argv

    print("=" * 60)
    print("DiffBio Benchmark: Multiple Sequence Alignment")
    print(
        f"Mode: {'quick (3 families)' if quick else 'full (all)'}"
    )
    print("=" * 60)

    bench = MSABenchmark(quick=quick)
    result = bench.run()

    # Save result
    from pathlib import Path  # noqa: PLC0415

    output_dir = Path("benchmarks/results/alignment")
    output_dir.mkdir(parents=True, exist_ok=True)
    result.save(output_dir / "msa.json")
    print(f"\nResult saved to: {output_dir / 'msa.json'}")

    print("\n" + "=" * 60)
    print(
        f"SP score: {result.metrics['sp_score'].value:.4f}, "
        f"TC score: {result.metrics['tc_score'].value:.4f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
