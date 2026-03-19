"""Utilities for simple stateful batch iteration in data sources."""

from collections.abc import Callable

import jax

from datarax.typing import Element


def reset_iteration_state(source: object, seed: int | None = None) -> None:
    """Reset a source object exposing `_current_idx` iteration state."""
    del seed  # API compatibility with DataSourceModule reset signature
    source._current_idx = 0


def next_batch(
    *,
    batch_size: int,
    key: jax.Array | None,
    current_idx: int,
    total_size: int,
    get_element: Callable[[int], Element],
) -> tuple[list[Element], int]:
    """Collect up to `batch_size` elements starting from `current_idx`."""
    del key  # API compatibility with DataSourceModule.get_batch signature

    batch: list[Element] = []
    idx = current_idx
    for _ in range(batch_size):
        if idx >= total_size:
            break
        batch.append(get_element(idx))
        idx += 1
    return batch, idx
