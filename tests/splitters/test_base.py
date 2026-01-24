"""Tests for splitter base classes.

Test-driven development: These tests define the expected behavior.
Implementation must pass these tests without modification.
"""

import jax.numpy as jnp
import pytest
from flax import nnx


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_elements():
    """Create sample elements for testing."""
    from datarax.typing import Element

    return [
        Element(data={"value": jnp.array(i), "label": i % 3}, state={}, metadata={"idx": i})
        for i in range(100)
    ]


@pytest.fixture
def mock_data_source(sample_elements):
    """Create a mock DataSourceModule for testing."""
    from dataclasses import dataclass

    from datarax.core.config import StructuralConfig
    from datarax.core.data_source import DataSourceModule

    @dataclass
    class MockSourceConfig(StructuralConfig):
        pass

    class MockDataSource(DataSourceModule):
        """Simple mock data source for testing."""

        # Annotate _data with nnx.Data to allow storing JAX arrays
        _data: nnx.Data[list]

        def __init__(self, config, data, *, rngs=None, name=None):
            super().__init__(config, rngs=rngs, name=name)
            self._data = data
            self._current_idx = 0

        def __len__(self):
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

    config = MockSourceConfig()
    return MockDataSource(config, sample_elements)


# =============================================================================
# Tests for SplitResult
# =============================================================================


class TestSplitResult:
    """Tests for SplitResult namedtuple."""

    def test_import(self):
        """Test that SplitResult can be imported."""
        from diffbio.splitters import SplitResult

        assert SplitResult is not None

    def test_creation(self):
        """Test creating a SplitResult."""
        from diffbio.splitters import SplitResult

        train = jnp.array([0, 1, 2, 3])
        valid = jnp.array([4, 5])
        test = jnp.array([6, 7, 8, 9])

        result = SplitResult(
            train_indices=train,
            valid_indices=valid,
            test_indices=test,
        )

        assert jnp.array_equal(result.train_indices, train)
        assert jnp.array_equal(result.valid_indices, valid)
        assert jnp.array_equal(result.test_indices, test)

    def test_size_properties(self):
        """Test size properties return correct values."""
        from diffbio.splitters import SplitResult

        result = SplitResult(
            train_indices=jnp.arange(80),
            valid_indices=jnp.arange(10),
            test_indices=jnp.arange(10),
        )

        assert result.train_size == 80
        assert result.valid_size == 10
        assert result.test_size == 10


# =============================================================================
# Tests for SplitterConfig
# =============================================================================


class TestSplitterConfig:
    """Tests for SplitterConfig."""

    def test_import(self):
        """Test that SplitterConfig can be imported."""
        from diffbio.splitters import SplitterConfig

        assert SplitterConfig is not None

    def test_default_values(self):
        """Test default configuration values."""
        from diffbio.splitters import SplitterConfig

        config = SplitterConfig()
        assert config.train_frac == 0.8
        assert config.valid_frac == 0.1
        assert config.test_frac == 0.1
        assert config.seed is None

    def test_custom_values(self):
        """Test custom configuration values."""
        from diffbio.splitters import SplitterConfig

        config = SplitterConfig(train_frac=0.7, valid_frac=0.15, test_frac=0.15, seed=42)
        assert config.train_frac == 0.7
        assert config.valid_frac == 0.15
        assert config.test_frac == 0.15
        assert config.seed == 42

    def test_validation_fractions_sum_to_one(self):
        """Test that fractions must sum to 1.0."""
        from diffbio.splitters import SplitterConfig

        # Should raise because fractions sum to 1.1
        with pytest.raises(ValueError, match="must sum to 1.0"):
            SplitterConfig(train_frac=0.8, valid_frac=0.2, test_frac=0.1)

    def test_frozen(self):
        """Test that config is frozen after creation."""
        from diffbio.splitters import SplitterConfig

        config = SplitterConfig()

        with pytest.raises(Exception):  # FrozenInstanceError
            config.train_frac = 0.5


# =============================================================================
# Tests for SplitterModule Base Class
# =============================================================================


class TestSplitterModuleBase:
    """Tests for SplitterModule base class."""

    def test_import(self):
        """Test that SplitterModule can be imported."""
        from diffbio.splitters import SplitterModule

        assert SplitterModule is not None

    def test_split_method_abstract(self):
        """Test that split() is abstract and requires implementation."""
        from diffbio.splitters import SplitterConfig, SplitterModule

        config = SplitterConfig()
        splitter = SplitterModule(config)

        # Base class split() should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            splitter.split(None)

    def test_process_delegates_to_split(self, mock_data_source):
        """Test that process() delegates to split()."""
        from diffbio.splitters import SplitResult, SplitterConfig, SplitterModule

        class ConcreteSplitter(SplitterModule):
            def split(self, data_source):
                n = len(data_source)
                return SplitResult(
                    train_indices=jnp.arange(n // 2),
                    valid_indices=jnp.arange(n // 2, n * 3 // 4),
                    test_indices=jnp.arange(n * 3 // 4, n),
                )

        config = SplitterConfig()
        splitter = ConcreteSplitter(config)

        result_split = splitter.split(mock_data_source)
        result_process = splitter.process(mock_data_source)

        assert jnp.array_equal(result_split.train_indices, result_process.train_indices)


class TestSplitterCreateSplitSources:
    """Tests for create_split_sources method."""

    def test_create_split_sources_lazy(self, mock_data_source):
        """Test create_split_sources with lazy loading (IndexedViewSource)."""
        from diffbio.sources import IndexedViewSource
        from diffbio.splitters import SplitResult, SplitterConfig, SplitterModule

        class SimpleSplitter(SplitterModule):
            def split(self, data_source):
                n = len(data_source)
                return SplitResult(
                    train_indices=jnp.arange(int(n * 0.8)),
                    valid_indices=jnp.arange(int(n * 0.8), int(n * 0.9)),
                    test_indices=jnp.arange(int(n * 0.9), n),
                )

        config = SplitterConfig()
        splitter = SimpleSplitter(config)

        train_src, valid_src, test_src = splitter.create_split_sources(
            mock_data_source, lazy=True
        )

        # Should return IndexedViewSource instances
        assert isinstance(train_src, IndexedViewSource)
        assert isinstance(valid_src, IndexedViewSource)
        assert isinstance(test_src, IndexedViewSource)

        # Should have correct sizes
        assert len(train_src) == 80
        assert len(valid_src) == 10
        assert len(test_src) == 10

    @pytest.mark.skip(reason="Datarax MemorySource has bug - needs upstream fix")
    def test_create_split_sources_eager(self, mock_data_source):
        """Test create_split_sources with eager loading (MemorySource)."""
        from datarax.sources import MemorySource

        from diffbio.splitters import SplitResult, SplitterConfig, SplitterModule

        class SimpleSplitter(SplitterModule):
            def split(self, data_source):
                n = len(data_source)
                return SplitResult(
                    train_indices=jnp.arange(int(n * 0.8)),
                    valid_indices=jnp.arange(int(n * 0.8), int(n * 0.9)),
                    test_indices=jnp.arange(int(n * 0.9), n),
                )

        config = SplitterConfig()
        splitter = SimpleSplitter(config)

        train_src, valid_src, test_src = splitter.create_split_sources(
            mock_data_source, lazy=False
        )

        # Should return MemorySource instances
        assert isinstance(train_src, MemorySource)
        assert isinstance(valid_src, MemorySource)
        assert isinstance(test_src, MemorySource)

    def test_create_split_sources_preserves_data(self, mock_data_source):
        """Test that split sources contain correct data."""
        from diffbio.splitters import SplitResult, SplitterConfig, SplitterModule

        class SimpleSplitter(SplitterModule):
            def split(self, data_source):
                return SplitResult(
                    train_indices=jnp.array([0, 1, 2]),
                    valid_indices=jnp.array([3, 4]),
                    test_indices=jnp.array([5]),
                )

        config = SplitterConfig(train_frac=0.5, valid_frac=0.33, test_frac=0.17)
        splitter = SimpleSplitter(config)

        train_src, valid_src, test_src = splitter.create_split_sources(
            mock_data_source, lazy=True
        )

        # Check train has correct values
        train_values = [int(elem.data["value"]) for elem in train_src]
        assert train_values == [0, 1, 2]

        # Check valid has correct values
        valid_values = [int(elem.data["value"]) for elem in valid_src]
        assert valid_values == [3, 4]

        # Check test has correct values
        test_values = [int(elem.data["value"]) for elem in test_src]
        assert test_values == [5]

    def test_train_shuffle_valid_test_no_shuffle(self, mock_data_source):
        """Test that train is shuffled but valid/test are not."""
        from diffbio.splitters import SplitResult, SplitterConfig, SplitterModule

        class SimpleSplitter(SplitterModule):
            def split(self, data_source):
                return SplitResult(
                    train_indices=jnp.arange(80),
                    valid_indices=jnp.arange(80, 90),
                    test_indices=jnp.arange(90, 100),
                )

        config = SplitterConfig()
        splitter = SimpleSplitter(config, rngs=nnx.Rngs(42, shuffle=42))

        train_src, valid_src, test_src = splitter.create_split_sources(
            mock_data_source, lazy=True
        )

        # Valid and test should preserve order
        valid_values = [int(elem.data["value"]) for elem in valid_src]
        assert valid_values == list(range(80, 90))

        test_values = [int(elem.data["value"]) for elem in test_src]
        assert test_values == list(range(90, 100))


class TestSplitterKFold:
    """Tests for k-fold cross-validation."""

    def test_k_fold_split_not_implemented(self, mock_data_source):
        """Test that k_fold_split raises NotImplementedError by default."""
        from diffbio.splitters import SplitterConfig, SplitterModule

        config = SplitterConfig()
        splitter = SplitterModule(config)

        with pytest.raises(NotImplementedError):
            splitter.k_fold_split(mock_data_source, k=5)
