"""Tests for benchmarks._base DiffBioBenchmark base class.

TDD: Define expected behavior of the base class before implementation.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from benchmarks._base import DiffBioBenchmarkConfig


class TestDiffBioBenchmarkConfig:
    """Tests for the benchmark config dataclass."""

    def test_frozen(self) -> None:
        config = DiffBioBenchmarkConfig(
            name="test", domain="test_domain"
        )
        with pytest.raises(FrozenInstanceError):
            config.name = "changed"  # type: ignore[misc]

    def test_required_fields(self) -> None:
        config = DiffBioBenchmarkConfig(
            name="test/bench", domain="test"
        )
        assert config.name == "test/bench"
        assert config.domain == "test"

    def test_quick_subsample_default(self) -> None:
        config = DiffBioBenchmarkConfig(
            name="test", domain="test"
        )
        assert config.quick_subsample == 2000

    def test_kw_only(self) -> None:
        """Config should require keyword arguments."""
        with pytest.raises(TypeError):
            DiffBioBenchmarkConfig("test", "domain")  # type: ignore[misc]
