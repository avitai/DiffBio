"""Tests for sequence identity splitter."""

from dataclasses import dataclass

import pytest
from flax import nnx

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.typing import Element


@dataclass
class MockSourceConfig(StructuralConfig):
    """Mock config for test data source."""

    pass


class MockSequenceSource(DataSourceModule):
    """Mock data source with DNA/protein sequences for testing."""

    _data: nnx.Data[list]

    def __init__(self, sequences: list[str], *, rngs: nnx.Rngs | None = None):
        """Initialize with list of sequences."""
        super().__init__(config=MockSourceConfig(), rngs=rngs)
        self._data = [
            Element(
                data={"sequence": seq, "label": i % 2},
                state={},
                metadata={"idx": i},
            )
            for i, seq in enumerate(sequences)
        ]
        self._current_idx = 0

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Element | None:
        if idx < 0 or idx >= len(self._data):
            return None
        return self._data[idx]

    def __iter__(self):
        self._current_idx = 0
        return self

    def __next__(self) -> Element:
        if self._current_idx >= len(self._data):
            raise StopIteration
        elem = self._data[self._current_idx]
        self._current_idx += 1
        return elem


class TestSequenceIdentitySplitter:
    """Test suite for SequenceIdentitySplitter."""

    def test_import(self):
        """Test that SequenceIdentitySplitter can be imported."""
        from diffbio.splitters import SequenceIdentitySplitter, SequenceIdentitySplitterConfig

        assert SequenceIdentitySplitter is not None
        assert SequenceIdentitySplitterConfig is not None

    def test_config_defaults(self):
        """Test default configuration values."""
        from diffbio.splitters import SequenceIdentitySplitterConfig

        config = SequenceIdentitySplitterConfig()
        assert config.sequence_key == "sequence"
        assert config.identity_threshold == 0.3
        assert config.alignment_method == "simple"
        assert config.train_frac == 0.8
        assert config.valid_frac == 0.1
        assert config.test_frac == 0.1

    def test_config_custom_values(self):
        """Test custom configuration values."""
        from diffbio.splitters import SequenceIdentitySplitterConfig

        config = SequenceIdentitySplitterConfig(
            sequence_key="protein_seq",
            identity_threshold=0.5,
            alignment_method="simple",
            train_frac=0.7,
            valid_frac=0.15,
            test_frac=0.15,
        )
        assert config.sequence_key == "protein_seq"
        assert config.identity_threshold == 0.5
        assert config.train_frac == 0.7

    def test_init(self):
        """Test splitter initialization."""
        from diffbio.splitters import SequenceIdentitySplitter, SequenceIdentitySplitterConfig

        config = SequenceIdentitySplitterConfig()
        splitter = SequenceIdentitySplitter(config)
        assert splitter is not None
        assert splitter.config == config

    def test_split_valid_result(self):
        """Test that split returns valid SplitResult."""
        from diffbio.splitters import (
            SequenceIdentitySplitter,
            SequenceIdentitySplitterConfig,
            SplitResult,
        )

        # Create sequences with some similar ones (should cluster together)
        sequences = [
            "ATGCATGCATGC",  # Group 1 - similar sequences
            "ATGCATGCATGA",  # Similar to sequence 0
            "TTTTTTTTTTTT",  # Group 2 - different from group 1
            "TTTTTTTTTTTA",  # Similar to sequence 2
            "GGGGGGGGGGGG",  # Group 3 - different from others
            "GGGGGGGGGGGC",  # Similar to sequence 4
            "CCCCCCCCCCCC",  # Group 4
            "AAAAAAAAACCC",  # Group 5
            "TACGTACGTACG",  # Group 6
            "AAAAAAAAAAAA",  # Group 7
        ]
        source = MockSequenceSource(sequences)

        config = SequenceIdentitySplitterConfig(identity_threshold=0.7)
        splitter = SequenceIdentitySplitter(config)
        result = splitter.split(source)

        assert isinstance(result, SplitResult)
        assert hasattr(result, "train_indices")
        assert hasattr(result, "valid_indices")
        assert hasattr(result, "test_indices")

    def test_split_no_overlap(self):
        """Test that train/valid/test splits have no overlapping indices."""
        from diffbio.splitters import SequenceIdentitySplitter, SequenceIdentitySplitterConfig

        sequences = [
            "ATGCATGCATGC",
            "TTTTTTTTTTTT",
            "GGGGGGGGGGGG",
            "CCCCCCCCCCCC",
            "AAAAAAAAACCC",
            "TACGTACGTACG",
            "AAAAAAAAAAAA",
            "GCGCGCGCGCGC",
            "TATATATATATA",
            "CGCGCGCGCGCG",
        ]
        source = MockSequenceSource(sequences)

        config = SequenceIdentitySplitterConfig(identity_threshold=0.5)
        splitter = SequenceIdentitySplitter(config)
        result = splitter.split(source)

        train_set = set(int(i) for i in result.train_indices)
        valid_set = set(int(i) for i in result.valid_indices)
        test_set = set(int(i) for i in result.test_indices)

        # Check no overlap between sets
        assert train_set.isdisjoint(valid_set)
        assert train_set.isdisjoint(test_set)
        assert valid_set.isdisjoint(test_set)

    def test_split_covers_all_data(self):
        """Test that all indices are covered by the split."""
        from diffbio.splitters import SequenceIdentitySplitter, SequenceIdentitySplitterConfig

        sequences = [
            "ATGCATGCATGC",
            "TTTTTTTTTTTT",
            "GGGGGGGGGGGG",
            "CCCCCCCCCCCC",
            "AAAAAAAAACCC",
            "TACGTACGTACG",
            "AAAAAAAAAAAA",
            "GCGCGCGCGCGC",
            "TATATATATATA",
            "CGCGCGCGCGCG",
        ]
        source = MockSequenceSource(sequences)

        config = SequenceIdentitySplitterConfig(identity_threshold=0.5)
        splitter = SequenceIdentitySplitter(config)
        result = splitter.split(source)

        all_indices = set(int(i) for i in result.train_indices)
        all_indices.update(int(i) for i in result.valid_indices)
        all_indices.update(int(i) for i in result.test_indices)

        expected_indices = set(range(len(sequences)))
        assert all_indices == expected_indices

    def test_similar_sequences_same_split(self):
        """Test that highly similar sequences end up in the same split."""
        from diffbio.splitters import SequenceIdentitySplitter, SequenceIdentitySplitterConfig

        # Create sequences where some pairs are highly similar
        sequences = [
            "AAAAAAAAAAAAAAAAAAAA",  # 0: Very similar to 1
            "AAAAAAAAAAAAAAAAAAAB",  # 1: Very similar to 0 (95% identity)
            "TTTTTTTTTTTTTTTTTTTT",  # 2: Very similar to 3
            "TTTTTTTTTTTTTTTTTTTC",  # 3: Very similar to 2 (95% identity)
            "GGGGGGGGGGGGGGGGGGGG",  # 4: Very similar to 5
            "GGGGGGGGGGGGGGGGGGGD",  # 5: Very similar to 4 (95% identity)
            "CCCCCCCCCCCCCCCCCCCC",  # 6: Unique
            "AAAAATTTTTGGGGGCCCCC",  # 7: Unique
        ]
        source = MockSequenceSource(sequences)

        # High identity threshold means similar sequences should cluster
        config = SequenceIdentitySplitterConfig(
            identity_threshold=0.9,
            train_frac=0.6,
            valid_frac=0.2,
            test_frac=0.2,
        )
        splitter = SequenceIdentitySplitter(config)
        result = splitter.split(source)

        # Check that similar pairs are in the same split
        train_set = set(int(i) for i in result.train_indices)
        valid_set = set(int(i) for i in result.valid_indices)
        test_set = set(int(i) for i in result.test_indices)

        # Helper function to check if both indices are in the same set
        def in_same_split(idx1: int, idx2: int) -> bool:
            return (
                (idx1 in train_set and idx2 in train_set)
                or (idx1 in valid_set and idx2 in valid_set)
                or (idx1 in test_set and idx2 in test_set)
            )

        # Similar pairs should be in same split
        assert in_same_split(0, 1), "Sequences 0 and 1 should be in same split"
        assert in_same_split(2, 3), "Sequences 2 and 3 should be in same split"
        assert in_same_split(4, 5), "Sequences 4 and 5 should be in same split"

    def test_custom_sequence_key(self):
        """Test splitting with custom sequence key."""
        from diffbio.splitters import SequenceIdentitySplitter, SequenceIdentitySplitterConfig

        # Create source with custom key
        class CustomKeySource(DataSourceModule):
            _data: nnx.Data[list]

            def __init__(self, sequences: list[str], *, rngs: nnx.Rngs | None = None):
                super().__init__(config=MockSourceConfig(), rngs=rngs)
                self._data = [
                    Element(
                        data={"protein_seq": seq},
                        state={},
                        metadata={},
                    )
                    for seq in sequences
                ]
                self._current_idx = 0

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx: int):
                return self._data[idx] if 0 <= idx < len(self._data) else None

            def __iter__(self):
                self._current_idx = 0
                return self

            def __next__(self):
                if self._current_idx >= len(self._data):
                    raise StopIteration
                elem = self._data[self._current_idx]
                self._current_idx += 1
                return elem

        sequences = ["ATGC" * 5, "TTTT" * 5, "GGGG" * 5, "CCCC" * 5, "AAAA" * 5]
        source = CustomKeySource(sequences)

        config = SequenceIdentitySplitterConfig(
            sequence_key="protein_seq",
            identity_threshold=0.3,
        )
        splitter = SequenceIdentitySplitter(config)
        result = splitter.split(source)

        # Should work without errors
        total = len(result.train_indices) + len(result.valid_indices) + len(result.test_indices)
        assert total == len(sequences)

    def test_identity_computation(self):
        """Test the sequence identity computation."""
        from diffbio.splitters import SequenceIdentitySplitter, SequenceIdentitySplitterConfig

        config = SequenceIdentitySplitterConfig()
        splitter = SequenceIdentitySplitter(config)

        # Test identical sequences
        assert splitter._compute_identity("ATGC", "ATGC") == 1.0

        # Test completely different sequences
        assert splitter._compute_identity("AAAA", "TTTT") == 0.0

        # Test 50% identical
        assert splitter._compute_identity("AATT", "AACT") == 0.75

        # Test different lengths
        assert splitter._compute_identity("ATGC", "AT") == 1.0  # Truncates to shorter

    def test_empty_sequences(self):
        """Test handling of empty sequences."""
        from diffbio.splitters import SequenceIdentitySplitter, SequenceIdentitySplitterConfig

        config = SequenceIdentitySplitterConfig()
        splitter = SequenceIdentitySplitter(config)

        # Empty sequence identity
        assert splitter._compute_identity("", "") == 0.0
        assert splitter._compute_identity("ATGC", "") == 0.0


class TestSequenceIdentitySplitterEdgeCases:
    """Edge case tests for SequenceIdentitySplitter."""

    def test_single_sequence(self):
        """Test splitting with single sequence."""
        from diffbio.splitters import SequenceIdentitySplitter, SequenceIdentitySplitterConfig

        source = MockSequenceSource(["ATGCATGCATGC"])

        config = SequenceIdentitySplitterConfig()
        splitter = SequenceIdentitySplitter(config)
        result = splitter.split(source)

        # Single sequence should go to train
        total = len(result.train_indices) + len(result.valid_indices) + len(result.test_indices)
        assert total == 1

    def test_all_identical_sequences(self):
        """Test splitting when all sequences are identical."""
        from diffbio.splitters import SequenceIdentitySplitter, SequenceIdentitySplitterConfig

        # All identical sequences should form a single cluster
        sequences = ["ATGCATGC"] * 10
        source = MockSequenceSource(sequences)

        config = SequenceIdentitySplitterConfig(identity_threshold=0.5)
        splitter = SequenceIdentitySplitter(config)
        result = splitter.split(source)

        # All should be in one split (train)
        assert len(result.train_indices) == 10
        assert len(result.valid_indices) == 0
        assert len(result.test_indices) == 0

    def test_all_different_sequences(self):
        """Test splitting when all sequences are completely different."""
        from diffbio.splitters import SequenceIdentitySplitter, SequenceIdentitySplitterConfig

        # All completely different (no similarity)
        sequences = [
            "A" * 20,
            "T" * 20,
            "G" * 20,
            "C" * 20,
            "AT" * 10,
            "GC" * 10,
            "AG" * 10,
            "TC" * 10,
            "AC" * 10,
            "TG" * 10,
        ]
        source = MockSequenceSource(sequences)

        config = SequenceIdentitySplitterConfig(identity_threshold=0.5)
        splitter = SequenceIdentitySplitter(config)
        result = splitter.split(source)

        # Each sequence is its own cluster, distributed across splits
        total = len(result.train_indices) + len(result.valid_indices) + len(result.test_indices)
        assert total == 10

    def test_invalid_alignment_method(self):
        """Test that invalid alignment method raises error."""
        from diffbio.splitters import SequenceIdentitySplitter, SequenceIdentitySplitterConfig

        sequences = ["ATGC"] * 5
        source = MockSequenceSource(sequences)

        config = SequenceIdentitySplitterConfig(alignment_method="invalid_method")
        splitter = SequenceIdentitySplitter(config)

        with pytest.raises(ValueError, match="Unknown alignment method"):
            splitter.split(source)

    def test_mmseqs2_not_implemented(self):
        """Test that mmseqs2 method raises NotImplementedError."""
        from diffbio.splitters import SequenceIdentitySplitter, SequenceIdentitySplitterConfig

        sequences = ["ATGC"] * 5
        source = MockSequenceSource(sequences)

        config = SequenceIdentitySplitterConfig(alignment_method="mmseqs2")
        splitter = SequenceIdentitySplitter(config)

        with pytest.raises(NotImplementedError, match="MMseqs2"):
            splitter.split(source)

    def test_very_low_threshold(self):
        """Test with very low identity threshold (most sequences cluster)."""
        from diffbio.splitters import SequenceIdentitySplitter, SequenceIdentitySplitterConfig

        sequences = [
            "ATGCATGCATGCATGCATGC",
            "TTTTTTTTTTTTTTTTTTTT",
            "GGGGGGGGGGGGGGGGGGGG",
            "CCCCCCCCCCCCCCCCCCCC",
            "AAAAAAAAAAAAAAAAACCC",
        ]
        source = MockSequenceSource(sequences)

        # Very low threshold - almost all different sequences will be separate
        config = SequenceIdentitySplitterConfig(identity_threshold=0.1)
        splitter = SequenceIdentitySplitter(config)
        result = splitter.split(source)

        # Should still cover all data
        total = len(result.train_indices) + len(result.valid_indices) + len(result.test_indices)
        assert total == 5

    def test_very_high_threshold(self):
        """Test with very high identity threshold (only exact matches cluster)."""
        from diffbio.splitters import SequenceIdentitySplitter, SequenceIdentitySplitterConfig

        sequences = [
            "ATGCATGCATGCATGCATGC",
            "ATGCATGCATGCATGCATGA",  # 95% similar to 0
            "ATGCATGCATGCATGCATGC",  # Identical to 0
            "TTTTTTTTTTTTTTTTTTTT",
            "GGGGGGGGGGGGGGGGGGGG",
        ]
        source = MockSequenceSource(sequences)

        # Very high threshold - only exact matches cluster
        config = SequenceIdentitySplitterConfig(identity_threshold=0.99)
        splitter = SequenceIdentitySplitter(config)
        result = splitter.split(source)

        train_set = set(int(i) for i in result.train_indices)

        # Indices 0 and 2 are identical, should be in same cluster
        if 0 in train_set:
            assert 2 in train_set
        else:
            # Both should be in same split
            valid_set = set(int(i) for i in result.valid_indices)
            test_set = set(int(i) for i in result.test_indices)
            same_split = (
                (0 in valid_set and 2 in valid_set) or (0 in test_set and 2 in test_set)
            )
            assert same_split
