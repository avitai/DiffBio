"""Tests for PerturbationBatchSampler."""

from __future__ import annotations

import numpy as np
import pytest

from diffbio.samplers.perturbation_sampler import (
    PerturbationBatchSampler,
    PerturbationSamplerConfig,
)


@pytest.fixture()
def group_codes() -> np.ndarray:
    """Create sample group codes: 4 groups, 100 cells each."""
    return np.repeat(np.arange(4, dtype=np.int32), 100)


class TestPerturbationBatchSampler:
    """Tests for PerturbationBatchSampler."""

    def test_all_indices_yielded(self, group_codes: np.ndarray) -> None:
        config = PerturbationSamplerConfig(sentence_size=50, seed=42)
        sampler = PerturbationBatchSampler(config, group_codes)
        all_indices: list[int] = []
        for batch in sampler:
            all_indices.extend(batch)
        assert set(all_indices) == set(range(400))

    def test_sentence_grouping(self, group_codes: np.ndarray) -> None:
        config = PerturbationSamplerConfig(sentence_size=25, seed=42)
        sampler = PerturbationBatchSampler(config, group_codes)
        batches = list(sampler)
        # Each batch should be a list of indices
        for batch in batches:
            assert isinstance(batch, list)
            assert len(batch) > 0

    def test_within_sentence_same_group(self, group_codes: np.ndarray) -> None:
        config = PerturbationSamplerConfig(sentence_size=25, seed=42)
        sampler = PerturbationBatchSampler(config, group_codes)
        batches = list(sampler)
        # Within each sentence (consecutive 25 indices), all should be same group
        for batch in batches:
            for start in range(0, len(batch), 25):
                sentence = batch[start : start + 25]
                groups = {group_codes[i] for i in sentence}
                assert len(groups) == 1, f"Mixed groups in sentence: {groups}"

    def test_epoch_shuffle_changes_order(self, group_codes: np.ndarray) -> None:
        config = PerturbationSamplerConfig(sentence_size=50, seed=42)
        sampler = PerturbationBatchSampler(config, group_codes)

        batches_epoch0 = [b.copy() for b in sampler]
        sampler.set_epoch(1)
        batches_epoch1 = [b.copy() for b in sampler]

        # Same total indices but potentially different order
        flat0 = [i for b in batches_epoch0 for i in b]
        flat1 = [i for b in batches_epoch1 for i in b]
        assert set(flat0) == set(flat1)
        # Order should differ (with high probability)
        assert flat0 != flat1

    def test_len(self, group_codes: np.ndarray) -> None:
        config = PerturbationSamplerConfig(
            sentence_size=50, sentences_per_batch=2, seed=42
        )
        sampler = PerturbationBatchSampler(config, group_codes)
        batches = list(sampler)
        assert len(sampler) == len(batches)

    def test_downsample_cells(self, group_codes: np.ndarray) -> None:
        config = PerturbationSamplerConfig(
            sentence_size=10, downsample_cells=20, seed=42
        )
        sampler = PerturbationBatchSampler(config, group_codes)
        all_indices: list[int] = []
        for batch in sampler:
            all_indices.extend(batch)
        # With 4 groups, max 20 cells each = max 80 total
        assert len(all_indices) <= 80

    def test_drop_last(self, group_codes: np.ndarray) -> None:
        config_drop = PerturbationSamplerConfig(
            sentence_size=50, sentences_per_batch=2, drop_last=True, seed=42
        )
        config_keep = PerturbationSamplerConfig(
            sentence_size=50, sentences_per_batch=2, drop_last=False, seed=42
        )
        sampler_drop = PerturbationBatchSampler(config_drop, group_codes)
        sampler_keep = PerturbationBatchSampler(config_keep, group_codes)
        # drop_last should yield fewer or equal batches
        assert len(sampler_drop) <= len(sampler_keep)
