"""Tests for the genomics sequence-classification scaffold helpers."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks._base import DiffBioBenchmarkConfig
from benchmarks.genomics._sequence_classification import (
    SequenceClassificationBenchmark,
    SequenceTaskSpec,
    _SyntheticSequenceClassificationSource,
    _validate_sequence_dataset,
    build_synthetic_sequence_classification_dataset,
)


def _make_task_spec() -> SequenceTaskSpec:
    """Create a compact task spec for scaffold-only tests."""
    return SequenceTaskSpec(
        benchmark_name="genomics/promoter",
        task_name="promoter",
        quick_samples_per_class=2,
        full_samples_per_class=3,
        sequence_length=12,
    )


class TestSyntheticSequenceClassificationDataset:
    """Tests for deterministic synthetic genomics scaffold generation."""

    def test_build_dataset_returns_expected_contract(self) -> None:
        dataset = build_synthetic_sequence_classification_dataset(
            task_name="promoter",
            samples_per_class=2,
            sequence_length=12,
        )

        assert len(dataset["sequence_ids"]) == 6
        assert len(dataset["sequences"]) == 6
        assert dataset["one_hot_sequences"].shape == (6, 12, 4)
        assert dataset["labels"].shape == (6,)
        assert dataset["sequence_ids"][0] == "promoter_0_0"
        assert dataset["sequence_ids"][-1] == "promoter_2_1"
        assert dataset["sequences"][0][3:9] == "TATAAA"
        assert dataset["sequences"][2][3:9] == "CGCGCG"
        assert dataset["sequences"][4][3:9] == "AATTAA"

    def test_synthetic_source_loads_the_same_dataset_contract(self) -> None:
        source = _SyntheticSequenceClassificationSource(
            task_name="tfbs",
            samples_per_class=2,
            sequence_length=12,
        )

        dataset = source.load()

        assert len(dataset["sequence_ids"]) == 6
        assert dataset["one_hot_sequences"].shape == (6, 12, 4)
        assert dataset["labels"].tolist() == [0, 0, 1, 1, 2, 2]


class TestSequenceDatasetValidation:
    """Tests for shared genomics dataset validation."""

    DatasetMutator = Callable[[dict[str, object]], object]

    def test_accepts_valid_dataset(self) -> None:
        dataset = build_synthetic_sequence_classification_dataset(
            task_name="splice_site",
            samples_per_class=2,
            sequence_length=12,
        )

        _validate_sequence_dataset(dataset)

    @pytest.mark.parametrize(
        ("mutator", "message"),
        [
            (
                lambda data: data.pop("labels"),
                "missing required keys",
            ),
            (
                lambda data: data.__setitem__("sequence_ids", data["sequence_ids"][:-1]),
                "same leading dimension",
            ),
            (
                lambda data: data.__setitem__(
                    "sequence_ids",
                    [data["sequence_ids"][0], *data["sequence_ids"][1:-1], data["sequence_ids"][0]],
                ),
                "must be unique",
            ),
            (
                lambda data: data.__setitem__(
                    "one_hot_sequences",
                    jnp.asarray(np.zeros((len(data["sequence_ids"]), 12), dtype=np.float32)),
                ),
                "must have shape",
            ),
            (
                lambda data: data.__setitem__(
                    "labels",
                    np.asarray(data["labels"], dtype=np.int32)[:, None],
                ),
                "must be rank-1",
            ),
        ],
    )
    def test_rejects_invalid_dataset_variants(
        self,
        mutator: DatasetMutator,
        message: str,
    ) -> None:
        dataset = build_synthetic_sequence_classification_dataset(
            task_name="promoter",
            samples_per_class=2,
            sequence_length=12,
        )

        mutator(dataset)

        with pytest.raises(ValueError, match=message):
            _validate_sequence_dataset(dataset)


class TestSequenceClassificationBenchmarkDefaults:
    """Tests for benchmark default source behavior."""

    def test_default_source_factory_uses_quick_sample_count(self) -> None:
        benchmark = SequenceClassificationBenchmark(
            DiffBioBenchmarkConfig(name="genomics/promoter", domain="genomics"),
            task_spec=_make_task_spec(),
            quick=True,
        )

        source = benchmark._default_source_factory(None)
        dataset = source.load()

        assert len(dataset["sequence_ids"]) == 6
        assert dataset["one_hot_sequences"].shape == (6, 12, 4)

    def test_default_source_factory_uses_full_sample_count(self) -> None:
        benchmark = SequenceClassificationBenchmark(
            DiffBioBenchmarkConfig(name="genomics/promoter", domain="genomics"),
            task_spec=_make_task_spec(),
            quick=False,
        )

        source = benchmark._default_source_factory(None)
        dataset = source.load()

        assert len(dataset["sequence_ids"]) == 9
        assert dataset["one_hot_sequences"].shape == (9, 12, 4)
