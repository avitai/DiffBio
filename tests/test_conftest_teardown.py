"""The per-test teardown leaves a CPU test's garbage to CPython's own collector.

A full ``gc.collect()`` walks the whole heap, which the JIT caches kept across tests make larger
as a session runs, so collecting after every test costs time that grows with the session. Only a
GPU-marked test collects, with its caches cleared, so its device buffers are released before the
next one.
"""

from __future__ import annotations

import types

import conftest
import jax
import pytest


class _Item:
    """A test item carrying the given markers, as ``pytest_runtest_teardown`` reads one."""

    def __init__(self, *markers: str) -> None:
        self._markers = markers

    def get_closest_marker(self, name: str) -> pytest.Mark | None:
        """Return the named marker when the item carries it."""
        return getattr(pytest.mark, name).mark if name in self._markers else None


@pytest.fixture
def collections(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, ...]]:
    """Record the collections the conftest asks for, through its own ``gc`` reference only."""
    calls: list[tuple[int, ...]] = []

    def record(*generation: int) -> int:
        calls.append(generation)
        return 0

    monkeypatch.setattr(conftest, "gc", types.SimpleNamespace(collect=record))
    return calls


def test_an_unmarked_test_is_not_followed_by_a_collection(
    collections: list[tuple[int, ...]],
) -> None:
    conftest.pytest_runtest_teardown(_Item(), None)

    assert collections == []


@pytest.mark.parametrize("marker", ["gpu", "cuda"])
def test_a_gpu_test_clears_the_caches_then_collects(
    marker: str, collections: list[tuple[int, ...]], monkeypatch: pytest.MonkeyPatch
) -> None:
    cleared: list[bool] = []
    monkeypatch.setattr(jax, "clear_caches", lambda: cleared.append(True))

    conftest.pytest_runtest_teardown(_Item(marker), None)

    assert cleared == [True]
    assert collections == [()]
