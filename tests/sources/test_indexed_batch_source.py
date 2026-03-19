"""Tests for IndexedBatchSourceMixin."""

import jax

from datarax.typing import Element

from diffbio.sources._indexed_batch_source import IndexedBatchSourceMixin


class _ToyIndexedSource(IndexedBatchSourceMixin):
    def __init__(self, size: int):
        self._current_idx = 0
        self._items = [
            Element(data={"idx": i}, state={}, metadata={})  # pyright: ignore[reportArgumentType]
            for i in range(size)
        ]

    def _batch_total_size(self) -> int:
        return len(self._items)

    def _batch_element(self, idx: int) -> Element:
        return self._items[idx]


def test_get_batch_advances_index_and_stops_at_end():
    source = _ToyIndexedSource(size=4)

    batch1 = source.get_batch(2)
    assert [int(elem.data["idx"]) for elem in batch1] == [0, 1]
    assert source._current_idx == 2

    batch2 = source.get_batch(10)
    assert [int(elem.data["idx"]) for elem in batch2] == [2, 3]
    assert source._current_idx == 4


def test_reset_restores_iteration_state_and_key_is_accepted():
    source = _ToyIndexedSource(size=3)

    _ = source.get_batch(1, key=jax.random.key(0))
    assert source._current_idx == 1

    source.reset(seed=123)
    assert source._current_idx == 0
