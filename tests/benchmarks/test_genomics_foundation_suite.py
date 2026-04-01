"""Tests for the genomics foundation benchmark suite harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from benchmarks.genomics.foundation_suite import (
    build_genomics_foundation_suite_report,
    run_genomics_foundation_suite,
)
from diffbio.operators.foundation_models import (
    DNABERT2PrecomputedAdapter,
    NucleotideTransformerPrecomputedAdapter,
)
from diffbio.sequences.dna import encode_dna_string


class _SyntheticSource:
    """Minimal source wrapper for genomics suite tests."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def load(self) -> dict[str, Any]:
        return self._data


def _make_synthetic_sequence_data() -> dict[str, Any]:
    """Create a simple synthetic genomics classification dataset."""
    sequences = [
        "AAAAAACCCC",
        "AAAATACCCC",
        "AAAAGACCCC",
        "CCCCCCAAAA",
        "CCCCTCAAAA",
        "CCCCGCAAAA",
        "GGGGGGTTTT",
        "GGGGAGTTTT",
        "GGGGCGTTTT",
    ]
    labels = np.repeat(np.arange(3, dtype=np.int32), 3)
    one_hot_sequences = jnp.asarray(
        np.stack(
            [np.asarray(encode_dna_string(sequence), dtype=np.float32) for sequence in sequences]
        ),
        dtype=jnp.float32,
    )
    return {
        "sequence_ids": [f"seq_{index}" for index in range(len(sequences))],
        "sequences": sequences,
        "one_hot_sequences": one_hot_sequences,
        "labels": labels,
    }


class TestGenomicsFoundationSuiteHarness:
    """Tests for the quick genomics suite across planned tasks."""

    def test_suite_report_is_reproducible_and_task_aware(self, tmp_path: Path) -> None:
        task_data = {
            "promoter": _make_synthetic_sequence_data(),
            "tfbs": _make_synthetic_sequence_data(),
            "splice_site": _make_synthetic_sequence_data(),
        }
        reversed_indices = np.arange(8, -1, -1)

        dnabert2_path = tmp_path / "dnabert2_embeddings.npz"
        np.savez(
            dnabert2_path,
            embeddings=np.asarray(task_data["promoter"]["one_hot_sequences"]).mean(axis=1)[
                reversed_indices
            ],
            sequence_ids=np.asarray(task_data["promoter"]["sequence_ids"])[reversed_indices],
        )
        nucleotide_transformer_path = tmp_path / "nucleotide_transformer_embeddings.npz"
        np.savez(
            nucleotide_transformer_path,
            embeddings=np.asarray(task_data["promoter"]["one_hot_sequences"]).sum(axis=1)[
                reversed_indices
            ],
            sequence_ids=np.asarray(task_data["promoter"]["sequence_ids"])[reversed_indices],
        )

        adapters = {
            "dnabert2_precomputed": DNABERT2PrecomputedAdapter(
                artifact_path=dnabert2_path,
            ),
            "nucleotide_transformer_precomputed": NucleotideTransformerPrecomputedAdapter(
                artifact_path=nucleotide_transformer_path,
            ),
        }

        source_factories = {
            task_name: (lambda data: lambda subsample: _SyntheticSource(data))(data)
            for task_name, data in task_data.items()
        }

        results_a = run_genomics_foundation_suite(
            quick=True,
            source_factories=source_factories,
            adapters=adapters,
        )
        results_b = run_genomics_foundation_suite(
            quick=True,
            source_factories=source_factories,
            adapters=adapters,
        )

        report_a = build_genomics_foundation_suite_report(results_a)
        report_b = build_genomics_foundation_suite_report(results_b)

        assert report_a == report_b
        assert tuple(report_a["task_order"]) == ("promoter", "tfbs", "splice_site")
        for task_name in ("promoter", "tfbs", "splice_site"):
            assert tuple(report_a["tasks"][task_name]["model_order"]) == (
                "diffbio_native",
                "dnabert2_precomputed",
                "nucleotide_transformer_precomputed",
            )

        assert (
            report_a["tasks"]["promoter"]["models"]["dnabert2_precomputed"]["tags"]["artifact_id"]
            == "dnabert2.v1"
        )
        assert (
            report_a["tasks"]["splice_site"]["models"]["nucleotide_transformer_precomputed"][
                "tags"
            ]["preprocessing_version"]
            == "bpe_v1"
        )
