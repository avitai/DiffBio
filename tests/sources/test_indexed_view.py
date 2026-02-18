"""Tests for IndexedViewSource.

Test-driven development: These tests define the expected behavior.
Implementation must pass these tests without modification.
"""

import jax.numpy as jnp
import pytest
from datarax.core.config import FrozenInstanceError
from flax import nnx


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_elements():
    """Create sample elements for testing."""
    from datarax.typing import Element

    return [
        Element(data={"value": jnp.array(i), "name": f"item_{i}"}, state={}, metadata={"idx": i})  # pyright: ignore[reportArgumentType]
        for i in range(10)
    ]


@pytest.fixture
def mock_data_source(sample_elements):
    """Create a mock DataSourceModule for testing."""
    from tests.mocks import MockDataSource, MockSourceConfig

    config = MockSourceConfig()
    return MockDataSource(config, sample_elements)


# =============================================================================
# Tests for IndexedViewSource
# =============================================================================


class TestIndexedViewSourceBasic:
    """Basic functionality tests for IndexedViewSource."""

    def test_import(self):
        """Test that IndexedViewSource can be imported."""
        from diffbio.sources import IndexedViewSource, IndexedViewSourceConfig

        assert IndexedViewSource is not None
        assert IndexedViewSourceConfig is not None

    def test_initialization(self, mock_data_source):
        """Test IndexedViewSource initialization."""
        from diffbio.sources import IndexedViewSource, IndexedViewSourceConfig

        indices = jnp.array([0, 2, 4, 6, 8])
        config = IndexedViewSourceConfig()
        view = IndexedViewSource(config, mock_data_source, indices)

        assert len(view) == 5
        assert view.underlying_source is mock_data_source

    def test_getitem_basic(self, mock_data_source):
        """Test basic indexing into view."""
        from diffbio.sources import IndexedViewSource, IndexedViewSourceConfig

        indices = jnp.array([0, 2, 4, 6, 8])
        config = IndexedViewSourceConfig()
        view = IndexedViewSource(config, mock_data_source, indices)

        # Index 0 in view should map to index 0 in source
        elem = view[0]
        assert elem is not None
        assert int(elem.data["value"]) == 0

        # Index 1 in view should map to index 2 in source
        elem = view[1]
        assert elem is not None
        assert int(elem.data["value"]) == 2

    def test_getitem_out_of_bounds(self, mock_data_source):
        """Test out of bounds indexing returns None."""
        from diffbio.sources import IndexedViewSource, IndexedViewSourceConfig

        indices = jnp.array([0, 2, 4])
        config = IndexedViewSourceConfig()
        view = IndexedViewSource(config, mock_data_source, indices)

        assert view[-1] is None
        assert view[10] is None

    def test_len(self, mock_data_source):
        """Test length returns correct view size."""
        from diffbio.sources import IndexedViewSource, IndexedViewSourceConfig

        indices = jnp.array([1, 3, 5, 7])
        config = IndexedViewSourceConfig()
        view = IndexedViewSource(config, mock_data_source, indices)

        assert len(view) == 4

    def test_source_indices_property(self, mock_data_source):
        """Test source_indices property returns the original indices."""
        from diffbio.sources import IndexedViewSource, IndexedViewSourceConfig

        indices = jnp.array([1, 3, 5])
        config = IndexedViewSourceConfig()
        view = IndexedViewSource(config, mock_data_source, indices)

        assert jnp.array_equal(view.source_indices, indices)


class TestIndexedViewSourceIteration:
    """Tests for iteration functionality."""

    def test_iteration_basic(self, mock_data_source):
        """Test basic iteration over view."""
        from diffbio.sources import IndexedViewSource, IndexedViewSourceConfig

        indices = jnp.array([0, 2, 4])
        config = IndexedViewSourceConfig()
        view = IndexedViewSource(config, mock_data_source, indices)

        values = [int(elem.data["value"]) for elem in view]
        assert values == [0, 2, 4]

    def test_iteration_empty_view(self, mock_data_source):
        """Test iteration over empty view."""
        from diffbio.sources import IndexedViewSource, IndexedViewSourceConfig

        indices = jnp.array([], dtype=jnp.int32)
        config = IndexedViewSourceConfig()
        view = IndexedViewSource(config, mock_data_source, indices)

        values = list(view)
        assert values == []

    def test_iteration_resets(self, mock_data_source):
        """Test that iteration can be performed multiple times."""
        from diffbio.sources import IndexedViewSource, IndexedViewSourceConfig

        indices = jnp.array([1, 3])
        config = IndexedViewSourceConfig()
        view = IndexedViewSource(config, mock_data_source, indices)

        values_1 = [int(elem.data["value"]) for elem in view]
        values_2 = [int(elem.data["value"]) for elem in view]

        assert values_1 == values_2 == [1, 3]


class TestIndexedViewSourceShuffle:
    """Tests for shuffling functionality."""

    def test_shuffle_config(self, mock_data_source):
        """Test shuffle configuration."""
        from diffbio.sources import IndexedViewSource, IndexedViewSourceConfig

        indices = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        config = IndexedViewSourceConfig(shuffle=True, seed=42)
        rngs = nnx.Rngs(42, shuffle=42)
        view = IndexedViewSource(config, mock_data_source, indices, rngs=rngs)

        # With shuffle, iteration order should differ from original indices
        values = [int(elem.data["value"]) for elem in view]

        # Values should contain all original values but potentially in different order
        assert sorted(values) == list(range(10))

    def test_shuffle_reproducibility(self, mock_data_source):
        """Test that shuffle with same seed produces same order."""
        from diffbio.sources import IndexedViewSource, IndexedViewSourceConfig

        indices = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        config1 = IndexedViewSourceConfig(shuffle=True, seed=42)
        rngs1 = nnx.Rngs(42, shuffle=42)
        view1 = IndexedViewSource(config1, mock_data_source, indices.copy(), rngs=rngs1)

        config2 = IndexedViewSourceConfig(shuffle=True, seed=42)
        rngs2 = nnx.Rngs(42, shuffle=42)
        view2 = IndexedViewSource(config2, mock_data_source, indices.copy(), rngs=rngs2)

        values_1 = [int(elem.data["value"]) for elem in view1]
        values_2 = [int(elem.data["value"]) for elem in view2]

        assert values_1 == values_2

    def test_no_shuffle_preserves_order(self, mock_data_source):
        """Test that without shuffle, order matches indices."""
        from diffbio.sources import IndexedViewSource, IndexedViewSourceConfig

        indices = jnp.array([9, 7, 5, 3, 1])
        config = IndexedViewSourceConfig(shuffle=False)
        view = IndexedViewSource(config, mock_data_source, indices)

        values = [int(elem.data["value"]) for elem in view]
        assert values == [9, 7, 5, 3, 1]

    def test_reset_with_new_seed(self, mock_data_source):
        """Test reset with new seed changes shuffle order."""
        from diffbio.sources import IndexedViewSource, IndexedViewSourceConfig

        indices = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        config = IndexedViewSourceConfig(shuffle=True, seed=42)
        rngs = nnx.Rngs(42, shuffle=42)
        view = IndexedViewSource(config, mock_data_source, indices, rngs=rngs)

        values_1 = [int(elem.data["value"]) for elem in view]
        view.reset(seed=123)  # Reset with different seed
        values_2 = [int(elem.data["value"]) for elem in view]

        # Both should contain all values but potentially different order
        assert sorted(values_1) == sorted(values_2) == list(range(10))


class TestIndexedViewSourceBatching:
    """Tests for batch retrieval functionality."""

    def test_get_batch_basic(self, mock_data_source):
        """Test basic batch retrieval."""
        from diffbio.sources import IndexedViewSource, IndexedViewSourceConfig

        indices = jnp.array([0, 1, 2, 3, 4])
        config = IndexedViewSourceConfig()
        view = IndexedViewSource(config, mock_data_source, indices)

        batch = view.get_batch(3)
        assert len(batch) == 3

        values = [int(elem.data["value"]) for elem in batch]
        assert values == [0, 1, 2]

    def test_get_batch_remainder(self, mock_data_source):
        """Test batch retrieval at end returns remaining elements."""
        from diffbio.sources import IndexedViewSource, IndexedViewSourceConfig

        indices = jnp.array([0, 1, 2, 3, 4])
        config = IndexedViewSourceConfig()
        view = IndexedViewSource(config, mock_data_source, indices)

        # Get first batch
        _ = view.get_batch(3)
        # Get remaining
        batch = view.get_batch(10)
        assert len(batch) == 2

    def test_get_batch_empty_after_exhaustion(self, mock_data_source):
        """Test batch is empty after view is exhausted."""
        from diffbio.sources import IndexedViewSource, IndexedViewSourceConfig

        indices = jnp.array([0, 1, 2])
        config = IndexedViewSourceConfig()
        view = IndexedViewSource(config, mock_data_source, indices)

        _ = view.get_batch(5)  # Exhaust all elements
        batch = view.get_batch(3)
        assert len(batch) == 0


class TestIndexedViewSourceConfig:
    """Tests for IndexedViewSourceConfig."""

    def test_config_defaults(self):
        """Test config default values."""
        from diffbio.sources import IndexedViewSourceConfig

        config = IndexedViewSourceConfig()
        assert config.shuffle is False
        assert config.seed is None

    def test_config_custom_values(self):
        """Test config with custom values."""
        from diffbio.sources import IndexedViewSourceConfig

        config = IndexedViewSourceConfig(shuffle=True, seed=42)
        assert config.shuffle is True
        assert config.seed == 42

    def test_config_frozen(self):
        """Test that StructuralConfig is frozen."""
        from diffbio.sources import IndexedViewSourceConfig

        config = IndexedViewSourceConfig(shuffle=True, seed=42)

        # Attempting to modify should raise
        with pytest.raises(FrozenInstanceError):
            config.shuffle = False
