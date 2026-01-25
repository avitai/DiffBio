"""Tests for BAMSource.

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
def pysam_available():
    """Check if pysam is available."""
    import importlib.util

    return importlib.util.find_spec("pysam") is not None


@pytest.fixture
def sample_bam_file(pysam_available):
    """Create a sample BAM file for testing."""
    if not pysam_available:
        pytest.skip("pysam not installed")

    import pysam

    with tempfile.TemporaryDirectory() as tmpdir:
        bam_path = Path(tmpdir) / "test.bam"

        # Create a minimal BAM header
        header = pysam.AlignmentHeader.from_dict(
            {
                "HD": {"VN": "1.0"},
                "SQ": [
                    {"LN": 1000, "SN": "chr1"},
                    {"LN": 500, "SN": "chr2"},
                ],
            }
        )

        # Write test reads
        with pysam.AlignmentFile(str(bam_path), "wb", header=header) as outf:
            for i in range(10):
                a = pysam.AlignedSegment()
                a.query_name = f"read_{i}"
                a.query_sequence = "ACGTACGTACGT"  # 12bp read
                a.flag = 0
                a.reference_id = 0  # chr1
                a.reference_start = i * 50
                a.mapping_quality = 60
                a.cigar = [(0, 12)]  # 12M
                a.query_qualities = pysam.qualitystring_to_array("IIIIIIIIIIII")  # Q40
                outf.write(a)

        # Index the BAM file
        pysam.index(str(bam_path))

        yield bam_path


# =============================================================================
# Tests for BAMSource
# =============================================================================


class TestBAMSourceImport:
    """Tests for BAMSource module imports."""

    def test_import(self):
        """Test that BAMSource can be imported."""
        from diffbio.sources import BAMSource, BAMSourceConfig

        assert BAMSource is not None
        assert BAMSourceConfig is not None


class TestBAMSourceConfig:
    """Tests for BAMSourceConfig."""

    def test_config_defaults(self):
        """Test config default values."""
        from diffbio.sources import BAMSourceConfig

        config = BAMSourceConfig(file_path=Path("/tmp/test.bam"))
        assert config.file_path == Path("/tmp/test.bam")
        assert config.reference_path is None
        assert config.include_unmapped is False
        assert config.min_mapping_quality is None
        assert config.region is None

    def test_config_custom_values(self):
        """Test config with custom values."""
        from diffbio.sources import BAMSourceConfig

        config = BAMSourceConfig(
            file_path=Path("/tmp/test.bam"),
            reference_path=Path("/tmp/ref.fa"),
            include_unmapped=True,
            min_mapping_quality=20,
            region="chr1:1000-2000",
        )
        assert config.include_unmapped is True
        assert config.min_mapping_quality == 20
        assert config.region == "chr1:1000-2000"

    def test_config_frozen(self):
        """Test that StructuralConfig is frozen."""
        from diffbio.sources import BAMSourceConfig

        config = BAMSourceConfig(file_path=Path("/tmp/test.bam"))
        with pytest.raises(Exception):  # FrozenInstanceError
            config.include_unmapped = True


class TestBAMSourceBasic:
    """Basic functionality tests for BAMSource."""

    def test_initialization(self, sample_bam_file):
        """Test BAMSource initialization."""
        from diffbio.sources import BAMSource, BAMSourceConfig

        config = BAMSourceConfig(file_path=sample_bam_file)
        source = BAMSource(config)

        assert len(source) == 10  # 10 reads

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        from diffbio.sources import BAMSource, BAMSourceConfig

        config = BAMSourceConfig(file_path=Path("/nonexistent/path.bam"))
        with pytest.raises(FileNotFoundError):
            BAMSource(config)

    def test_getitem_basic(self, sample_bam_file):
        """Test basic indexing."""
        from diffbio.sources import BAMSource, BAMSourceConfig

        config = BAMSourceConfig(file_path=sample_bam_file)
        source = BAMSource(config)

        elem = source[0]
        assert elem is not None
        assert "sequence" in elem.data
        assert "quality_scores" in elem.data
        assert "read_name" in elem.data

    def test_getitem_returns_one_hot_sequence(self, sample_bam_file):
        """Test that sequence is one-hot encoded."""
        from diffbio.sources import BAMSource, BAMSourceConfig

        config = BAMSourceConfig(file_path=sample_bam_file)
        source = BAMSource(config)

        elem = source[0]
        seq = elem.data["sequence"]

        # Should be (length, 4) one-hot encoding
        assert seq.ndim == 2
        assert seq.shape[1] == 4
        # Each row should sum to 1 (or 0.25*4 for N)
        row_sums = jnp.sum(seq, axis=1)
        assert jnp.allclose(row_sums, 1.0)

    def test_getitem_returns_quality_scores(self, sample_bam_file):
        """Test that quality scores are returned as Phred values."""
        from diffbio.sources import BAMSource, BAMSourceConfig

        config = BAMSourceConfig(file_path=sample_bam_file)
        source = BAMSource(config)

        elem = source[0]
        quality = elem.data["quality_scores"]

        # Should be 1D array matching sequence length
        assert quality.ndim == 1
        assert len(quality) == elem.data["sequence"].shape[0]
        # Quality values should be reasonable (0-60 typical range)
        assert jnp.all(quality >= 0)
        assert jnp.all(quality <= 60)

    def test_getitem_out_of_bounds(self, sample_bam_file):
        """Test out of bounds indexing returns None."""
        from diffbio.sources import BAMSource, BAMSourceConfig

        config = BAMSourceConfig(file_path=sample_bam_file)
        source = BAMSource(config)

        assert source[-1] is None
        assert source[100] is None

    def test_metadata_includes_read_info(self, sample_bam_file):
        """Test that metadata includes read information."""
        from diffbio.sources import BAMSource, BAMSourceConfig

        config = BAMSourceConfig(file_path=sample_bam_file)
        source = BAMSource(config)

        elem = source[0]
        assert "idx" in elem.metadata
        assert "reference_name" in elem.metadata or elem.metadata.get("unmapped", False)


class TestBAMSourceIteration:
    """Tests for iteration functionality."""

    def test_iteration_basic(self, sample_bam_file):
        """Test basic iteration over source."""
        from diffbio.sources import BAMSource, BAMSourceConfig

        config = BAMSourceConfig(file_path=sample_bam_file)
        source = BAMSource(config)

        elements = list(source)
        assert len(elements) == 10

    def test_iteration_yields_correct_elements(self, sample_bam_file):
        """Test that iteration yields properly formatted elements."""
        from diffbio.sources import BAMSource, BAMSourceConfig

        config = BAMSourceConfig(file_path=sample_bam_file)
        source = BAMSource(config)

        for elem in source:
            assert "sequence" in elem.data
            assert "quality_scores" in elem.data
            assert elem.data["sequence"].ndim == 2

    def test_iteration_resets(self, sample_bam_file):
        """Test that iteration can be performed multiple times."""
        from diffbio.sources import BAMSource, BAMSourceConfig

        config = BAMSourceConfig(file_path=sample_bam_file)
        source = BAMSource(config)

        elements_1 = list(source)
        elements_2 = list(source)

        assert len(elements_1) == len(elements_2)


class TestBAMSourceFiltering:
    """Tests for filtering functionality."""

    def test_min_mapping_quality_filter(self, sample_bam_file):
        """Test filtering by minimum mapping quality."""
        from diffbio.sources import BAMSource, BAMSourceConfig

        # All reads have MAPQ 60, so this should include all
        config = BAMSourceConfig(
            file_path=sample_bam_file,
            min_mapping_quality=30,
        )
        source = BAMSource(config)
        assert len(source) == 10

        # This should include all as well (MAPQ >= 60)
        config_high = BAMSourceConfig(
            file_path=sample_bam_file,
            min_mapping_quality=60,
        )
        source_high = BAMSource(config_high)
        assert len(source_high) == 10


class TestBAMSourceBatching:
    """Tests for batch retrieval functionality."""

    def test_get_batch_basic(self, sample_bam_file):
        """Test basic batch retrieval."""
        from diffbio.sources import BAMSource, BAMSourceConfig

        config = BAMSourceConfig(file_path=sample_bam_file)
        source = BAMSource(config)

        batch = source.get_batch(3)
        assert len(batch) == 3

    def test_get_batch_remainder(self, sample_bam_file):
        """Test batch retrieval at end returns remaining elements."""
        from diffbio.sources import BAMSource, BAMSourceConfig

        config = BAMSourceConfig(file_path=sample_bam_file)
        source = BAMSource(config)

        # Get first batch
        _ = source.get_batch(8)
        # Get remaining
        batch = source.get_batch(10)
        assert len(batch) == 2

    def test_get_batch_empty_after_exhaustion(self, sample_bam_file):
        """Test batch is empty after source is exhausted."""
        from diffbio.sources import BAMSource, BAMSourceConfig

        config = BAMSourceConfig(file_path=sample_bam_file)
        source = BAMSource(config)

        _ = source.get_batch(15)  # Exhaust all elements
        batch = source.get_batch(3)
        assert len(batch) == 0


class TestBAMSourceReset:
    """Tests for reset functionality."""

    def test_reset_allows_reiteration(self, sample_bam_file):
        """Test that reset allows starting iteration from beginning."""
        from diffbio.sources import BAMSource, BAMSourceConfig

        config = BAMSourceConfig(file_path=sample_bam_file)
        source = BAMSource(config)

        # Exhaust via batching
        _ = source.get_batch(10)
        batch_empty = source.get_batch(3)
        assert len(batch_empty) == 0

        # Reset and try again
        source.reset()
        batch_after_reset = source.get_batch(3)
        assert len(batch_after_reset) == 3
