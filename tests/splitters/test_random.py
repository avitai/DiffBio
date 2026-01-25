"""Tests for RandomSplitter and StratifiedSplitter.

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
def imbalanced_elements():
    """Create imbalanced class distribution for stratified splitting tests."""
    from datarax.typing import Element

    elements = []
    # 70 class 0, 20 class 1, 10 class 2
    for i in range(70):
        elements.append(
            Element(data={"value": jnp.array(i), "y": 0}, state={}, metadata={"idx": i})
        )
    for i in range(20):
        elements.append(
            Element(data={"value": jnp.array(70 + i), "y": 1}, state={}, metadata={"idx": 70 + i})
        )
    for i in range(10):
        elements.append(
            Element(data={"value": jnp.array(90 + i), "y": 2}, state={}, metadata={"idx": 90 + i})
        )
    return elements


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


@pytest.fixture
def imbalanced_data_source(imbalanced_elements):
    """Create a mock DataSourceModule with imbalanced classes."""
    from dataclasses import dataclass

    from datarax.core.config import StructuralConfig
    from datarax.core.data_source import DataSourceModule

    @dataclass
    class MockSourceConfig(StructuralConfig):
        pass

    class MockDataSource(DataSourceModule):
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
    return MockDataSource(config, imbalanced_elements)


# =============================================================================
# Tests for RandomSplitter
# =============================================================================


class TestRandomSplitter:
    """Tests for RandomSplitter."""

    def test_import(self):
        """Test that RandomSplitter can be imported."""
        from diffbio.splitters import RandomSplitter, RandomSplitterConfig

        assert RandomSplitter is not None
        assert RandomSplitterConfig is not None

    def test_split_default_fractions(self, mock_data_source):
        """Test split with default fractions (80/10/10)."""
        from diffbio.splitters import RandomSplitter, RandomSplitterConfig

        config = RandomSplitterConfig(seed=42)
        splitter = RandomSplitter(config)
        result = splitter.split(mock_data_source)

        assert result.train_size == 80
        assert result.valid_size == 10
        assert result.test_size == 10

    def test_split_custom_fractions(self, mock_data_source):
        """Test split with custom fractions."""
        from diffbio.splitters import RandomSplitter, RandomSplitterConfig

        config = RandomSplitterConfig(train_frac=0.7, valid_frac=0.15, test_frac=0.15, seed=42)
        splitter = RandomSplitter(config)
        result = splitter.split(mock_data_source)

        assert result.train_size == 70
        assert result.valid_size == 15
        assert result.test_size == 15

    def test_split_no_overlap(self, mock_data_source):
        """Test that splits have no overlapping indices."""
        from diffbio.splitters import RandomSplitter, RandomSplitterConfig

        config = RandomSplitterConfig(seed=42)
        splitter = RandomSplitter(config)
        result = splitter.split(mock_data_source)

        all_indices = jnp.concatenate(
            [result.train_indices, result.valid_indices, result.test_indices]
        )

        # All indices should be unique
        assert len(jnp.unique(all_indices)) == len(all_indices)

    def test_split_covers_all_data(self, mock_data_source):
        """Test that splits cover all data points."""
        from diffbio.splitters import RandomSplitter, RandomSplitterConfig

        config = RandomSplitterConfig(seed=42)
        splitter = RandomSplitter(config)
        result = splitter.split(mock_data_source)

        all_indices = jnp.concatenate(
            [result.train_indices, result.valid_indices, result.test_indices]
        )

        assert len(all_indices) == len(mock_data_source)
        assert set(all_indices.tolist()) == set(range(len(mock_data_source)))

    def test_split_reproducibility(self, mock_data_source):
        """Test that same seed produces same split."""
        from diffbio.splitters import RandomSplitter, RandomSplitterConfig

        config1 = RandomSplitterConfig(seed=42)
        splitter1 = RandomSplitter(config1)
        result1 = splitter1.split(mock_data_source)

        config2 = RandomSplitterConfig(seed=42)
        splitter2 = RandomSplitter(config2)
        result2 = splitter2.split(mock_data_source)

        assert jnp.array_equal(result1.train_indices, result2.train_indices)
        assert jnp.array_equal(result1.valid_indices, result2.valid_indices)
        assert jnp.array_equal(result1.test_indices, result2.test_indices)

    def test_split_different_seeds_different_results(self, mock_data_source):
        """Test that different seeds produce different splits."""
        from diffbio.splitters import RandomSplitter, RandomSplitterConfig

        config1 = RandomSplitterConfig(seed=42)
        splitter1 = RandomSplitter(config1)
        result1 = splitter1.split(mock_data_source)

        config2 = RandomSplitterConfig(seed=123)
        splitter2 = RandomSplitter(config2)
        result2 = splitter2.split(mock_data_source)

        # At least one split should differ
        assert not jnp.array_equal(result1.train_indices, result2.train_indices)


class TestRandomSplitterKFold:
    """Tests for RandomSplitter k-fold cross-validation."""

    def test_k_fold_returns_correct_number_of_folds(self, mock_data_source):
        """Test that k_fold_split returns k folds."""
        from diffbio.splitters import RandomSplitter, RandomSplitterConfig

        config = RandomSplitterConfig(seed=42)
        splitter = RandomSplitter(config)
        folds = splitter.k_fold_split(mock_data_source, k=5)

        assert len(folds) == 5

    def test_k_fold_each_fold_has_train_and_val(self, mock_data_source):
        """Test that each fold has train and val indices."""
        from diffbio.splitters import RandomSplitter, RandomSplitterConfig

        config = RandomSplitterConfig(seed=42)
        splitter = RandomSplitter(config)
        folds = splitter.k_fold_split(mock_data_source, k=5)

        for train_indices, val_indices in folds:
            assert len(train_indices) > 0
            assert len(val_indices) > 0

    def test_k_fold_no_overlap_within_fold(self, mock_data_source):
        """Test that train and val have no overlap within each fold."""
        from diffbio.splitters import RandomSplitter, RandomSplitterConfig

        config = RandomSplitterConfig(seed=42)
        splitter = RandomSplitter(config)
        folds = splitter.k_fold_split(mock_data_source, k=5)

        for train_indices, val_indices in folds:
            train_set = set(train_indices.tolist())
            val_set = set(val_indices.tolist())
            assert len(train_set & val_set) == 0

    def test_k_fold_covers_all_data(self, mock_data_source):
        """Test that each fold covers all data."""
        from diffbio.splitters import RandomSplitter, RandomSplitterConfig

        config = RandomSplitterConfig(seed=42)
        splitter = RandomSplitter(config)
        folds = splitter.k_fold_split(mock_data_source, k=5)

        for train_indices, val_indices in folds:
            all_indices = jnp.concatenate([train_indices, val_indices])
            assert len(all_indices) == len(mock_data_source)

    def test_k_fold_val_sets_partition_data(self, mock_data_source):
        """Test that validation sets across folds partition the data."""
        from diffbio.splitters import RandomSplitter, RandomSplitterConfig

        config = RandomSplitterConfig(seed=42)
        splitter = RandomSplitter(config)
        folds = splitter.k_fold_split(mock_data_source, k=5)

        all_val_indices = []
        for _, val_indices in folds:
            all_val_indices.extend(val_indices.tolist())

        # All indices should appear exactly once across validation sets
        assert len(all_val_indices) == len(mock_data_source)
        assert set(all_val_indices) == set(range(len(mock_data_source)))


# =============================================================================
# Tests for StratifiedSplitter
# =============================================================================


class TestStratifiedSplitter:
    """Tests for StratifiedSplitter."""

    def test_import(self):
        """Test that StratifiedSplitter can be imported."""
        from diffbio.splitters import StratifiedSplitter, StratifiedSplitterConfig

        assert StratifiedSplitter is not None
        assert StratifiedSplitterConfig is not None

    def test_config_label_key(self):
        """Test that config has label_key parameter."""
        from diffbio.splitters import StratifiedSplitterConfig

        config = StratifiedSplitterConfig(label_key="target")
        assert config.label_key == "target"

    def test_split_preserves_class_distribution(self, imbalanced_data_source):
        """Test that split preserves approximate class distribution."""
        from diffbio.splitters import StratifiedSplitter, StratifiedSplitterConfig

        config = StratifiedSplitterConfig(seed=42, label_key="y")
        splitter = StratifiedSplitter(config)
        result = splitter.split(imbalanced_data_source)

        # Get labels for each split
        train_labels = [int(imbalanced_data_source[int(i)].data["y"]) for i in result.train_indices]
        valid_labels = [int(imbalanced_data_source[int(i)].data["y"]) for i in result.valid_indices]
        test_labels = [int(imbalanced_data_source[int(i)].data["y"]) for i in result.test_indices]

        # Calculate class proportions
        def get_proportions(labels):
            total = len(labels)
            if total == 0:
                return {}
            return {c: labels.count(c) / total for c in set(labels)}

        train_props = get_proportions(train_labels)
        valid_props = get_proportions(valid_labels)
        test_props = get_proportions(test_labels)

        # Original proportions: 70% class 0, 20% class 1, 10% class 2
        # Each split should approximately preserve this
        # Allow tolerance due to rounding
        for props in [train_props, valid_props, test_props]:
            if 0 in props:
                assert abs(props[0] - 0.7) < 0.15  # Within 15% of expected
            if 1 in props:
                assert abs(props[1] - 0.2) < 0.15
            if 2 in props:
                assert abs(props[2] - 0.1) < 0.15

    def test_split_no_overlap(self, imbalanced_data_source):
        """Test that stratified splits have no overlapping indices."""
        from diffbio.splitters import StratifiedSplitter, StratifiedSplitterConfig

        config = StratifiedSplitterConfig(seed=42, label_key="y")
        splitter = StratifiedSplitter(config)
        result = splitter.split(imbalanced_data_source)

        all_indices = jnp.concatenate(
            [result.train_indices, result.valid_indices, result.test_indices]
        )

        assert len(jnp.unique(all_indices)) == len(all_indices)

    def test_split_covers_all_data(self, imbalanced_data_source):
        """Test that stratified splits cover all data."""
        from diffbio.splitters import StratifiedSplitter, StratifiedSplitterConfig

        config = StratifiedSplitterConfig(seed=42, label_key="y")
        splitter = StratifiedSplitter(config)
        result = splitter.split(imbalanced_data_source)

        all_indices = jnp.concatenate(
            [result.train_indices, result.valid_indices, result.test_indices]
        )

        assert len(all_indices) == len(imbalanced_data_source)

    def test_split_reproducibility(self, imbalanced_data_source):
        """Test that same seed produces same stratified split."""
        from diffbio.splitters import StratifiedSplitter, StratifiedSplitterConfig

        config1 = StratifiedSplitterConfig(seed=42, label_key="y")
        splitter1 = StratifiedSplitter(config1)
        result1 = splitter1.split(imbalanced_data_source)

        config2 = StratifiedSplitterConfig(seed=42, label_key="y")
        splitter2 = StratifiedSplitter(config2)
        result2 = splitter2.split(imbalanced_data_source)

        assert jnp.array_equal(result1.train_indices, result2.train_indices)
