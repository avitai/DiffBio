"""Tests for BalifamSource data source.

Unit tests validate config and FASTA parsing with synthetic data.
Integration tests require the real balifam repository on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from diffbio.sources.balifam import (
    BalifamConfig,
    BalifamSource,
    _parse_fasta,
)

_DATA_DIR = Path("/media/mahdi/ssd23/Works/balifam")
_DATA_EXISTS = (_DATA_DIR / "balifam100").exists()


class TestParseFasta:
    """Tests for the internal _parse_fasta helper."""

    def test_single_sequence(self, tmp_path: Path) -> None:
        """Parses a single FASTA entry correctly."""
        fasta_file = tmp_path / "single.fasta"
        fasta_file.write_text(">seq1\nACGTACGT\n")
        entries = _parse_fasta(fasta_file)
        assert len(entries) == 1
        assert entries[0] == ("seq1", "ACGTACGT")

    def test_multiple_sequences(self, tmp_path: Path) -> None:
        """Parses multiple FASTA entries."""
        fasta_file = tmp_path / "multi.fasta"
        fasta_file.write_text(">seq1\nACGT\n>seq2\nTGCA\n>seq3\nAAAA\n")
        entries = _parse_fasta(fasta_file)
        assert len(entries) == 3
        assert entries[0][0] == "seq1"
        assert entries[1][0] == "seq2"
        assert entries[2][0] == "seq3"

    def test_multiline_sequence(self, tmp_path: Path) -> None:
        """Multi-line sequences are concatenated."""
        fasta_file = tmp_path / "multiline.fasta"
        fasta_file.write_text(">seq1\nACGT\nTGCA\nAAAA\n")
        entries = _parse_fasta(fasta_file)
        assert len(entries) == 1
        assert entries[0][1] == "ACGTTGCAAAAA"

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file returns empty list."""
        fasta_file = tmp_path / "empty.fasta"
        fasta_file.write_text("")
        entries = _parse_fasta(fasta_file)
        assert entries == []

    def test_name_whitespace_stripped(self, tmp_path: Path) -> None:
        """Leading/trailing whitespace in names is stripped."""
        fasta_file = tmp_path / "whitespace.fasta"
        fasta_file.write_text(">  seq1  \nACGT\n")
        entries = _parse_fasta(fasta_file)
        assert entries[0][0] == "seq1"


class TestBalifamConfig:
    """Tests for BalifamConfig validation."""

    def test_invalid_tier_raises_value_error(self) -> None:
        """Invalid tier value raises ValueError."""
        with pytest.raises(ValueError, match="tier must be one of"):
            BalifamConfig(data_dir=str(_DATA_DIR), tier=500)

    def test_missing_directory_raises_file_not_found(self, tmp_path: Path) -> None:
        """Missing tier directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Balifam tier directory not found"):
            BalifamConfig(data_dir=str(tmp_path), tier=100)

    @pytest.mark.skipif(not _DATA_EXISTS, reason="balifam data not available")
    def test_frozen(self) -> None:
        """Config is immutable."""
        config = BalifamConfig(data_dir=str(_DATA_DIR))
        with pytest.raises(AttributeError):
            config.tier = 1000  # type: ignore[misc]


@pytest.mark.skipif(not _DATA_EXISTS, reason="balifam data not available")
class TestBalifamSourceIntegration:
    """Integration tests requiring the real balifam repository."""

    @pytest.fixture()
    def source(self) -> BalifamSource:
        """Load a small subset of balifam100 families."""
        config = BalifamConfig(
            data_dir=str(_DATA_DIR),
            tier=100,
            max_families=3,
        )
        return BalifamSource(config)

    def test_load_returns_list(self, source: BalifamSource) -> None:
        """load() returns a list."""
        families = source.load()
        assert isinstance(families, list)

    def test_max_families_limits_results(self, source: BalifamSource) -> None:
        """max_families limits the number of families loaded."""
        families = source.load()
        assert len(families) <= 3

    def test_family_has_required_keys(self, source: BalifamSource) -> None:
        """Each family dict has family_id, sequences, reference."""
        families = source.load()
        assert len(families) > 0
        family = families[0]
        assert "family_id" in family
        assert "sequences" in family
        assert "reference" in family

    def test_sequences_are_list_of_tuples(self, source: BalifamSource) -> None:
        """Sequences are (name, sequence) tuples."""
        families = source.load()
        sequences = families[0]["sequences"]
        assert isinstance(sequences, list)
        assert len(sequences) > 0
        name, seq = sequences[0]
        assert isinstance(name, str)
        assert isinstance(seq, str)

    def test_reference_is_list_of_tuples(self, source: BalifamSource) -> None:
        """Reference alignment entries are (name, sequence) tuples."""
        families = source.load()
        reference = families[0]["reference"]
        assert isinstance(reference, list)
        assert len(reference) > 0
        name, seq = reference[0]
        assert isinstance(name, str)
        assert isinstance(seq, str)

    def test_n_sequences_field_correct(self, source: BalifamSource) -> None:
        """n_sequences matches actual sequence count."""
        families = source.load()
        family = families[0]
        assert family["n_sequences"] == len(family["sequences"])

    def test_len_matches_family_count(self, source: BalifamSource) -> None:
        """__len__ matches the number of loaded families."""
        families = source.load()
        assert len(source) == len(families)

    def test_iter_yields_families(self, source: BalifamSource) -> None:
        """Iteration yields each family dict."""
        families = source.load()
        iterated = list(source)
        assert len(iterated) == len(families)
