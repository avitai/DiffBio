"""Shared mock classes for DiffBio tests."""

from dataclasses import dataclass

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from flax import nnx


@dataclass
class MockSourceConfig(StructuralConfig):
    """Configuration for MockDataSource used across tests."""


class MockDataSource(DataSourceModule):
    """Simple mock data source for testing.

    Wraps a list of elements and provides DataSourceModule interface
    with indexing, length, and iteration support.
    """

    _data: list = nnx.data()

    def __init__(self, config: MockSourceConfig, data: list, *, rngs=None, name=None) -> None:
        super().__init__(config, rngs=rngs, name=name)
        self._data = data
        self._current_idx = 0

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx):
        if 0 <= idx < len(self._data):
            return self._data[idx]
        return None

    def __iter__(self):
        self._current_idx = 0
        return self

    def __next__(self):
        if self._current_idx >= len(self._data):
            raise StopIteration
        elem = self._data[self._current_idx]
        self._current_idx += 1
        return elem
