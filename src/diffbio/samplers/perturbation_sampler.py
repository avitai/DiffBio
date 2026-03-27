"""Perturbation-aware batch sampler for single-cell experiments.

Groups cells by (cell_type, perturbation) into "sentences", then combines
sentences into batches. Uses integer group codes for fast grouping.

References:
    - cell-load/src/cell_load/data_modules/samplers.py (PerturbationBatchSampler)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from datarax.core.config import StructuralConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerturbationSamplerConfig(StructuralConfig):
    """Configuration for PerturbationBatchSampler.

    Attributes:
        sentence_size: Number of cells per "sentence" (same perturbation
            and cell type).
        sentences_per_batch: Number of sentences combined into one batch.
        seed: Random seed for shuffling.
        drop_last: Whether to drop the last incomplete batch.
        downsample_cells: Maximum cells per (cell_type, perturbation) group.
            None means no downsampling.
    """

    sentence_size: int = 512
    sentences_per_batch: int = 1
    seed: int = 42
    drop_last: bool = False
    downsample_cells: int | None = None


class PerturbationBatchSampler:
    """Groups cells by (cell_type, perturbation) into sentence-based batches.

    Creates "sentences" where all cells share the same group code (typically
    encoding cell_type and perturbation). Sentences are then combined into
    batches. Supports epoch-aware shuffling and cell downsampling.

    Args:
        config: Sampler configuration.
        group_codes: Per-cell integer group codes (from
            ``PerturbationAnnDataSource.get_group_codes()``).
    """

    def __init__(
        self,
        config: PerturbationSamplerConfig,
        group_codes: np.ndarray,
    ) -> None:
        self._config = config
        self._group_codes = group_codes
        self._epoch = 0
        self._sentences = self._create_sentences()

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of cell indices.

        Each batch contains ``sentences_per_batch`` sentences, where each
        sentence is a group of ``sentence_size`` cells from the same
        (cell_type, perturbation) group.

        Yields:
            Lists of cell indices forming each batch.
        """
        rng = np.random.default_rng(self._config.seed + self._epoch)

        # Shuffle sentence order (not within sentences)
        sentence_order = rng.permutation(len(self._sentences))

        spb = self._config.sentences_per_batch
        n_batches = len(self._sentences) // spb

        for batch_idx in range(n_batches):
            batch_indices: list[int] = []
            for s_offset in range(spb):
                s_idx = sentence_order[batch_idx * spb + s_offset]
                batch_indices.extend(self._sentences[s_idx])
            yield batch_indices

        # Handle remainder unless drop_last
        remainder_start = n_batches * spb
        if not self._config.drop_last and remainder_start < len(self._sentences):
            batch_indices = []
            for s_idx in sentence_order[remainder_start:]:
                batch_indices.extend(self._sentences[s_idx])
            if batch_indices:
                yield batch_indices

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        spb = self._config.sentences_per_batch
        n_full = len(self._sentences) // spb
        has_remainder = (
            not self._config.drop_last
            and len(self._sentences) % spb > 0
        )
        return n_full + int(has_remainder)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling.

        Args:
            epoch: Current epoch number.
        """
        self._epoch = epoch

    def _create_sentences(self) -> list[list[int]]:
        """Group cell indices by group code and split into sentences."""
        rng = np.random.default_rng(self._config.seed)
        sentence_size = self._config.sentence_size
        downsample = self._config.downsample_cells

        # Group indices by group code
        groups: dict[int, list[int]] = defaultdict(list)
        for idx, code in enumerate(self._group_codes):
            groups[int(code)].append(idx)

        sentences: list[list[int]] = []

        for _, indices in sorted(groups.items()):
            cell_indices = np.array(indices)

            # Apply cell downsampling
            if downsample is not None and len(cell_indices) > downsample:
                cell_indices = rng.choice(
                    cell_indices, size=downsample, replace=False
                )

            # Split into sentences
            for start in range(0, len(cell_indices), sentence_size):
                sentence = cell_indices[start : start + sentence_size].tolist()
                sentences.append(sentence)

        return sentences
