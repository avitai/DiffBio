"""Mixin for stateful index-based batching in data sources."""

import logging

import jax

from datarax.typing import Element

from diffbio.sources._batch_iteration import next_batch, reset_iteration_state

logger = logging.getLogger(__name__)


class IndexedBatchSourceMixin:
    """Reusable `reset` and `get_batch` logic for indexable data sources."""

    def _batch_total_size(self) -> int:
        raise NotImplementedError

    def _batch_element(self, idx: int) -> Element:
        raise NotImplementedError

    def reset(self, seed: int | None = None) -> None:
        """Reset iteration state, optionally with a new seed."""
        reset_iteration_state(self, seed)

    def get_batch(self, batch_size: int, key: jax.Array | None = None) -> list[Element]:
        """Return the next batch of elements, advancing the internal index."""
        batch, self._current_idx = next_batch(
            batch_size=batch_size,
            key=key,
            current_idx=self._current_idx,
            total_size=self._batch_total_size(),
            get_element=self._batch_element,
        )
        return batch
