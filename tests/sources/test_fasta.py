"""Tests for FastaSource.

Test-driven development: These tests define the expected behavior.
Implementation must pass these tests without modification.
"""

import tempfile
from pathlib import Path

import jax.numpy as jnp
import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def pyfaidx_available():
    """Check if pyfaidx is available."""
    import importlib.util

    return importlib.util.find_spec("pyfaidx") is not None


@pytest.fixture
def sample_fasta_file():
    """Create a sample FASTA file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = Path(tmpdir) / "test.fasta"

        # Write sample sequences
        fasta_content = """>seq1 Description of sequence 1
ACGTACGTACGTACGTACGT
>seq2 Another sequence
GGGGCCCCAAAATTTT
>seq3 Third sequence with N
ACGTNNNNNACGT
"""
        fasta_path.write_text(fasta_content)

        yield fasta_path


@pytest.fixture
def sample_fasta_file_indexed(sample_fasta_file, pyfaidx_available):
    """Create an indexed FASTA file for testing."""
    if not pyfaidx_available:
        pytest.skip("pyfaidx not installed")

    import pyfaidx

    # Opening with pyfaidx creates the index
    _ = pyfaidx.Fasta(str(sample_fasta_file))

    return sample_fasta_file


# =============================================================================
# Tests for FastaSource
# =============================================================================


class TestFastaSourceImport:
    """Tests for FastaSource module imports."""

    def test_import(self):
        """Test that FastaSource can be imported."""
        from diffbio.sources import FastaSource, FastaSourceConfig

        assert FastaSource is not None
        assert FastaSourceConfig is not None


class TestFastaSourceConfig:
    """Tests for FastaSourceConfig."""

    def test_config_defaults(self):
        """Test config default values."""
        from diffbio.sources import FastaSourceConfig

        config = FastaSourceConfig(file_path=Path("/tmp/test.fasta"))
        assert config.file_path == Path("/tmp/test.fasta")
        assert config.handle_n == "uniform"
        assert config.create_index is True

    def test_config_custom_values(self):
        """Test config with custom values."""
        from diffbio.sources import FastaSourceConfig

        config = FastaSourceConfig(
            file_path=Path("/tmp/test.fasta"),
            handle_n="zero",
            create_index=False,
        )
        assert config.handle_n == "zero"
        assert config.create_index is False

    def test_config_frozen(self):
        """Test that StructuralConfig is frozen."""
        from diffbio.sources import FastaSourceConfig

        config = FastaSourceConfig(file_path=Path("/tmp/test.fasta"))
        with pytest.raises(Exception):  # FrozenInstanceError
            config.handle_n = "zero"


class TestFastaSourceBasic:
    """Basic functionality tests for FastaSource."""

    def test_initialization(self, sample_fasta_file_indexed):
        """Test FastaSource initialization."""
        from diffbio.sources import FastaSource, FastaSourceConfig

        config = FastaSourceConfig(file_path=sample_fasta_file_indexed)
        source = FastaSource(config)

        assert len(source) == 3  # 3 sequences

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        from diffbio.sources import FastaSource, FastaSourceConfig

        config = FastaSourceConfig(file_path=Path("/nonexistent/path.fasta"))
        with pytest.raises(FileNotFoundError):
            FastaSource(config)

    def test_getitem_basic(self, sample_fasta_file_indexed):
        """Test basic indexing."""
        from diffbio.sources import FastaSource, FastaSourceConfig

        config = FastaSourceConfig(file_path=sample_fasta_file_indexed)
        source = FastaSource(config)

        elem = source[0]
        assert elem is not None
        assert "sequence" in elem.data
        assert "sequence_id" in elem.data

    def test_getitem_returns_one_hot_sequence(self, sample_fasta_file_indexed):
        """Test that sequence is one-hot encoded."""
        from diffbio.sources import FastaSource, FastaSourceConfig

        config = FastaSourceConfig(file_path=sample_fasta_file_indexed)
        source = FastaSource(config)

        elem = source[0]
        seq = elem.data["sequence"]

        # Should be (length, 4) one-hot encoding
        assert seq.ndim == 2
        assert seq.shape[1] == 4
        # seq1 is ACGTACGTACGTACGTACGT (20 bp)
        assert seq.shape[0] == 20
        # Each row should sum to 1
        row_sums = jnp.sum(seq, axis=1)
        assert jnp.allclose(row_sums, 1.0)

    def test_handle_n_uniform(self, sample_fasta_file_indexed):
        """Test handling N nucleotides with uniform encoding."""
        from diffbio.sources import FastaSource, FastaSourceConfig

        config = FastaSourceConfig(
            file_path=sample_fasta_file_indexed,
            handle_n="uniform",
        )
        source = FastaSource(config)

        # seq3 has N's: ACGTNNNNNACGT
        elem = source[2]
        seq = elem.data["sequence"]

        # N positions (4-8) should be uniform [0.25, 0.25, 0.25, 0.25]
        for i in range(4, 9):
            assert jnp.allclose(seq[i], jnp.array([0.25, 0.25, 0.25, 0.25]))

    def test_handle_n_zero(self, sample_fasta_file_indexed):
        """Test handling N nucleotides with zero encoding."""
        from diffbio.sources import FastaSource, FastaSourceConfig

        config = FastaSourceConfig(
            file_path=sample_fasta_file_indexed,
            handle_n="zero",
        )
        source = FastaSource(config)

        # seq3 has N's: ACGTNNNNNACGT
        elem = source[2]
        seq = elem.data["sequence"]

        # N positions (4-8) should be zero [0, 0, 0, 0]
        for i in range(4, 9):
            assert jnp.allclose(seq[i], jnp.zeros(4))

    def test_getitem_out_of_bounds(self, sample_fasta_file_indexed):
        """Test out of bounds indexing returns None."""
        from diffbio.sources import FastaSource, FastaSourceConfig

        config = FastaSourceConfig(file_path=sample_fasta_file_indexed)
        source = FastaSource(config)

        assert source[-1] is None
        assert source[100] is None

    def test_metadata_includes_sequence_info(self, sample_fasta_file_indexed):
        """Test that metadata includes sequence information."""
        from diffbio.sources import FastaSource, FastaSourceConfig

        config = FastaSourceConfig(file_path=sample_fasta_file_indexed)
        source = FastaSource(config)

        elem = source[0]
        assert "idx" in elem.metadata
        assert elem.data["sequence_id"] == "seq1"


class TestFastaSourceIteration:
    """Tests for iteration functionality."""

    def test_iteration_basic(self, sample_fasta_file_indexed):
        """Test basic iteration over source."""
        from diffbio.sources import FastaSource, FastaSourceConfig

        config = FastaSourceConfig(file_path=sample_fasta_file_indexed)
        source = FastaSource(config)

        elements = list(source)
        assert len(elements) == 3

    def test_iteration_yields_correct_elements(self, sample_fasta_file_indexed):
        """Test that iteration yields properly formatted elements."""
        from diffbio.sources import FastaSource, FastaSourceConfig

        config = FastaSourceConfig(file_path=sample_fasta_file_indexed)
        source = FastaSource(config)

        for elem in source:
            assert "sequence" in elem.data
            assert "sequence_id" in elem.data
            assert elem.data["sequence"].ndim == 2

    def test_iteration_resets(self, sample_fasta_file_indexed):
        """Test that iteration can be performed multiple times."""
        from diffbio.sources import FastaSource, FastaSourceConfig

        config = FastaSourceConfig(file_path=sample_fasta_file_indexed)
        source = FastaSource(config)

        elements_1 = list(source)
        elements_2 = list(source)

        assert len(elements_1) == len(elements_2)


class TestFastaSourceByName:
    """Tests for accessing sequences by name."""

    def test_get_by_name(self, sample_fasta_file_indexed):
        """Test getting sequence by name."""
        from diffbio.sources import FastaSource, FastaSourceConfig

        config = FastaSourceConfig(file_path=sample_fasta_file_indexed)
        source = FastaSource(config)

        elem = source.get_by_name("seq2")
        assert elem is not None
        assert elem.data["sequence_id"] == "seq2"

    def test_get_by_name_not_found(self, sample_fasta_file_indexed):
        """Test that missing sequence returns None."""
        from diffbio.sources import FastaSource, FastaSourceConfig

        config = FastaSourceConfig(file_path=sample_fasta_file_indexed)
        source = FastaSource(config)

        elem = source.get_by_name("nonexistent")
        assert elem is None

    def test_sequence_names_property(self, sample_fasta_file_indexed):
        """Test sequence_names property."""
        from diffbio.sources import FastaSource, FastaSourceConfig

        config = FastaSourceConfig(file_path=sample_fasta_file_indexed)
        source = FastaSource(config)

        names = source.sequence_names
        assert "seq1" in names
        assert "seq2" in names
        assert "seq3" in names


class TestFastaSourceBatching:
    """Tests for batch retrieval functionality."""

    def test_get_batch_basic(self, sample_fasta_file_indexed):
        """Test basic batch retrieval."""
        from diffbio.sources import FastaSource, FastaSourceConfig

        config = FastaSourceConfig(file_path=sample_fasta_file_indexed)
        source = FastaSource(config)

        batch = source.get_batch(2)
        assert len(batch) == 2

    def test_get_batch_remainder(self, sample_fasta_file_indexed):
        """Test batch retrieval at end returns remaining elements."""
        from diffbio.sources import FastaSource, FastaSourceConfig

        config = FastaSourceConfig(file_path=sample_fasta_file_indexed)
        source = FastaSource(config)

        # Get first batch
        _ = source.get_batch(2)
        # Get remaining
        batch = source.get_batch(10)
        assert len(batch) == 1

    def test_get_batch_empty_after_exhaustion(self, sample_fasta_file_indexed):
        """Test batch is empty after source is exhausted."""
        from diffbio.sources import FastaSource, FastaSourceConfig

        config = FastaSourceConfig(file_path=sample_fasta_file_indexed)
        source = FastaSource(config)

        _ = source.get_batch(5)  # Exhaust all elements
        batch = source.get_batch(3)
        assert len(batch) == 0


class TestFastaSourceReset:
    """Tests for reset functionality."""

    def test_reset_allows_reiteration(self, sample_fasta_file_indexed):
        """Test that reset allows starting iteration from beginning."""
        from diffbio.sources import FastaSource, FastaSourceConfig

        config = FastaSourceConfig(file_path=sample_fasta_file_indexed)
        source = FastaSource(config)

        # Exhaust via batching
        _ = source.get_batch(5)
        batch_empty = source.get_batch(3)
        assert len(batch_empty) == 0

        # Reset and try again
        source.reset()
        batch_after_reset = source.get_batch(2)
        assert len(batch_after_reset) == 2
