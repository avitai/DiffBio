"""IndexedViewSource - Lazy-loading view into a data source.

This module provides IndexedViewSource, which wraps an existing DataSourceModule
and provides access only to elements at specified indices. Elements are loaded
on-demand from the underlying source, enabling lazy loading for large datasets.
"""

from dataclasses import dataclass
from typing import Iterator

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.typing import Element


@dataclass
class IndexedViewSourceConfig(StructuralConfig):
    """Configuration for IndexedViewSource.

    Attributes:
        shuffle: Whether to shuffle the view indices on initialization and reset
        seed: Random seed for shuffling (optional)
    """

    shuffle: bool = False
    seed: int | None = None


class IndexedViewSource(DataSourceModule):
    """Lazy-loading view into a data source using index mapping.

    This source wraps an existing DataSourceModule and provides access
    only to elements at specified indices. Elements are loaded ON-DEMAND
    from the underlying source, enabling lazy loading for large datasets.

    Key Features:
        - LAZY LOADING: Elements fetched from underlying source only when accessed
        - Memory efficient: Only stores indices, not actual data
        - Preserves underlying source's lazy loading behavior
        - Supports shuffling of view indices (not underlying data)

    Example:
        >>> # Create view of first 1000 elements
        >>> indices = jnp.arange(1000)
        >>> config = IndexedViewSourceConfig()
        >>> view = IndexedViewSource(config, original_source, indices)
        >>> view[0]  # Fetches original_source[indices[0]] lazily

    Args:
        config: Configuration for the view source
        source: Underlying data source to wrap
        indices: Array of indices into the source to expose
        rngs: Random number generators for shuffling
        name: Optional name for the module
    """

    def __init__(
        self,
        config: IndexedViewSourceConfig,
        source: DataSourceModule,
        indices: jnp.ndarray,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize IndexedViewSource.

        Args:
            config: Configuration for the view source
            source: Underlying data source to wrap
            indices: Array of indices into the source to expose
            rngs: Random number generators for shuffling
            name: Optional name for the module
        """
        super().__init__(config, rngs=rngs, name=name)
        self._source = source
        self._indices = indices
        self._view_indices = jnp.arange(len(indices))  # Local view ordering
        self._current_idx = 0

        # Apply initial shuffle if configured
        if config.shuffle:
            self._shuffle_view()

    def _shuffle_view(self) -> None:
        """Shuffle the view indices (not the underlying data)."""
        if self.rngs is not None and "shuffle" in self.rngs:
            key = self.rngs.shuffle()
        elif self.config.seed is not None:
            key = jax.random.key(self.config.seed)
        else:
            key = jax.random.key(0)

        self._view_indices = jax.random.permutation(key, self._view_indices)

    def __len__(self) -> int:
        """Return number of elements in the view."""
        return len(self._indices)

    def __getitem__(self, idx: int) -> Element | None:
        """Get element at view index (LAZY - fetches from underlying source).

        Args:
            idx: Index into the VIEW (0 to len(view)-1)

        Returns:
            Element from underlying source at mapped index, or None if out of bounds
        """
        if idx < 0 or idx >= len(self._indices):
            return None

        # Map view index -> shuffled view index -> original source index
        view_idx = int(self._view_indices[idx])
        source_idx = int(self._indices[view_idx])

        # LAZY: Fetch from underlying source only now
        return self._source[source_idx]

    def __iter__(self) -> Iterator[Element]:
        """Iterate over view elements (LAZY - fetches on demand)."""
        self._current_idx = 0
        return self

    def __next__(self) -> Element:
        """Get next element (LAZY)."""
        if self._current_idx >= len(self._indices):
            raise StopIteration

        element = self[self._current_idx]
        self._current_idx += 1

        if element is None:
            raise StopIteration

        return element

    def reset(self, seed: int | None = None) -> None:
        """Reset iteration and optionally reshuffle.

        Args:
            seed: Optional new seed for shuffling
        """
        self._current_idx = 0

        if self.config.shuffle:
            if seed is not None:
                # Update seed and reshuffle
                key = jax.random.key(seed)
                self._view_indices = jax.random.permutation(
                    key, jnp.arange(len(self._indices))
                )
            else:
                self._shuffle_view()

    def get_batch(self, batch_size: int, key: jax.Array | None = None) -> list[Element]:
        """Get next batch of elements (LAZY).

        Args:
            batch_size: Number of elements to fetch
            key: Optional RNG key (unused, for interface compatibility)

        Returns:
            List of elements
        """
        batch = []
        for _ in range(batch_size):
            if self._current_idx >= len(self._indices):
                break
            element = self[self._current_idx]
            if element is not None:
                batch.append(element)
            self._current_idx += 1
        return batch

    @property
    def underlying_source(self) -> DataSourceModule:
        """Access the underlying data source."""
        return self._source

    @property
    def source_indices(self) -> jnp.ndarray:
        """Get the indices into the underlying source."""
        return self._indices
