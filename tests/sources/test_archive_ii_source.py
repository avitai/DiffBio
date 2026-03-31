"""Tests for ArchiveIISource data source.

Unit tests validate config and CSV parsing with synthetic data.
Integration tests require the real ArchiveII dataset on disk.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from diffbio.sources.archive_ii import (
    ArchiveIIConfig,
    ArchiveIISource,
    _parse_csv,
)

_DATA_DIR = Path("/media/mahdi/ssd23/Works/RNAFoldAssess/tutorial/processed_data")
_DATA_EXISTS = (_DATA_DIR / "example_data_structure.csv").exists()


class TestParseCSV:
    """Tests for the internal _parse_csv helper."""

    def _write_csv(
        self,
        path: Path,
        rows: list[dict[str, str]],
    ) -> None:
        """Write a CSV file with ArchiveII-format columns."""
        fieldnames = [
            "name",
            "sequence",
            "ground_truth_type",
            "ground_truth_data",
        ]
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_parses_dbn_rows(self, tmp_path: Path) -> None:
        """Rows with ground_truth_type=DBN are loaded."""
        csv_path = tmp_path / "data.csv"
        self._write_csv(
            csv_path,
            [
                {
                    "name": "rna1",
                    "sequence": "GUCUACC",
                    "ground_truth_type": "DBN",
                    "ground_truth_data": ".((.)).",
                },
            ],
        )
        entries = _parse_csv(csv_path, max_sequences=None)
        assert len(entries) == 1
        assert entries[0]["name"] == "rna1"
        assert entries[0]["sequence"] == "GUCUACC"
        assert entries[0]["structure"] == ".((.))."

    def test_skips_non_dbn_rows(self, tmp_path: Path) -> None:
        """Rows without ground_truth_type=DBN are skipped."""
        csv_path = tmp_path / "data.csv"
        self._write_csv(
            csv_path,
            [
                {
                    "name": "rna1",
                    "sequence": "GUCUACC",
                    "ground_truth_type": "CT",
                    "ground_truth_data": "some_data",
                },
                {
                    "name": "rna2",
                    "sequence": "AUCG",
                    "ground_truth_type": "DBN",
                    "ground_truth_data": "(())",
                },
            ],
        )
        entries = _parse_csv(csv_path, max_sequences=None)
        assert len(entries) == 1
        assert entries[0]["name"] == "rna2"

    def test_max_sequences_limits_output(self, tmp_path: Path) -> None:
        """max_sequences parameter limits the number of entries."""
        csv_path = tmp_path / "data.csv"
        rows = [
            {
                "name": f"rna{i}",
                "sequence": "ACGU",
                "ground_truth_type": "DBN",
                "ground_truth_data": "(())",
            }
            for i in range(10)
        ]
        self._write_csv(csv_path, rows)
        entries = _parse_csv(csv_path, max_sequences=3)
        assert len(entries) == 3

    def test_empty_structure_skipped(self, tmp_path: Path) -> None:
        """Rows with empty ground_truth_data are skipped."""
        csv_path = tmp_path / "data.csv"
        self._write_csv(
            csv_path,
            [
                {
                    "name": "rna1",
                    "sequence": "ACGU",
                    "ground_truth_type": "DBN",
                    "ground_truth_data": "",
                },
            ],
        )
        entries = _parse_csv(csv_path, max_sequences=None)
        assert len(entries) == 0

    def test_sequence_uppercased(self, tmp_path: Path) -> None:
        """Sequences are converted to uppercase."""
        csv_path = tmp_path / "data.csv"
        self._write_csv(
            csv_path,
            [
                {
                    "name": "rna1",
                    "sequence": "gucuacc",
                    "ground_truth_type": "DBN",
                    "ground_truth_data": ".((.)).",
                },
            ],
        )
        entries = _parse_csv(csv_path, max_sequences=None)
        assert entries[0]["sequence"] == "GUCUACC"


class TestArchiveIIConfig:
    """Tests for ArchiveIIConfig validation."""

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """Missing CSV raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="ArchiveII data not found"):
            ArchiveIIConfig(data_dir=str(tmp_path))

    @pytest.mark.skipif(not _DATA_EXISTS, reason="ArchiveII data not available")
    def test_frozen(self) -> None:
        """Config is immutable."""
        config = ArchiveIIConfig(data_dir=str(_DATA_DIR))
        with pytest.raises(AttributeError):
            config.filename = "other.csv"  # type: ignore[misc]


class TestArchiveIISourceUnit:
    """Unit tests with synthetic CSV data (no real dataset needed)."""

    def _create_csv(
        self,
        tmp_path: Path,
        n_entries: int = 5,
    ) -> Path:
        """Create a synthetic ArchiveII CSV file."""
        csv_path = tmp_path / "example_data_structure.csv"
        fieldnames = [
            "name",
            "sequence",
            "ground_truth_type",
            "ground_truth_data",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(n_entries):
                writer.writerow(
                    {
                        "name": f"rna_{i}",
                        "sequence": "GUCUACC",
                        "ground_truth_type": "DBN",
                        "ground_truth_data": ".((.)).",
                    }
                )
        return csv_path

    def test_loads_entries(self, tmp_path: Path) -> None:
        """Source loads the correct number of entries."""
        self._create_csv(tmp_path, n_entries=5)
        config = ArchiveIIConfig(
            data_dir=str(tmp_path),
            filename="example_data_structure.csv",
        )
        source = ArchiveIISource(config)
        data = source.load()
        assert data["n_sequences"] == 5

    def test_entries_have_required_keys(self, tmp_path: Path) -> None:
        """Each entry has name, sequence, structure."""
        self._create_csv(tmp_path, n_entries=2)
        config = ArchiveIIConfig(
            data_dir=str(tmp_path),
            filename="example_data_structure.csv",
        )
        source = ArchiveIISource(config)
        data = source.load()
        entry = data["entries"][0]
        assert "name" in entry
        assert "sequence" in entry
        assert "structure" in entry

    def test_max_sequences_limits_entries(self, tmp_path: Path) -> None:
        """max_sequences config limits how many entries are loaded."""
        self._create_csv(tmp_path, n_entries=10)
        config = ArchiveIIConfig(
            data_dir=str(tmp_path),
            filename="example_data_structure.csv",
            max_sequences=3,
        )
        source = ArchiveIISource(config)
        assert len(source) == 3

    def test_structures_valid_dbn(self, tmp_path: Path) -> None:
        """Loaded structures contain only valid DBN characters."""
        self._create_csv(tmp_path, n_entries=3)
        config = ArchiveIIConfig(
            data_dir=str(tmp_path),
            filename="example_data_structure.csv",
        )
        source = ArchiveIISource(config)
        valid_chars = set(".()")
        for entry in source:
            for char in entry["structure"]:
                assert char in valid_chars, f"Invalid DBN char '{char}' in {entry['name']}"

    def test_len_matches_n_sequences(self, tmp_path: Path) -> None:
        """__len__ returns n_sequences."""
        self._create_csv(tmp_path, n_entries=4)
        config = ArchiveIIConfig(
            data_dir=str(tmp_path),
            filename="example_data_structure.csv",
        )
        source = ArchiveIISource(config)
        assert len(source) == source.load()["n_sequences"]

    def test_no_dbn_entries_raises_value_error(self, tmp_path: Path) -> None:
        """CSV with no DBN rows raises ValueError."""
        csv_path = tmp_path / "example_data_structure.csv"
        fieldnames = [
            "name",
            "sequence",
            "ground_truth_type",
            "ground_truth_data",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "name": "rna1",
                    "sequence": "ACGU",
                    "ground_truth_type": "CT",
                    "ground_truth_data": "some_ct_data",
                }
            )
        config = ArchiveIIConfig(
            data_dir=str(tmp_path),
            filename="example_data_structure.csv",
        )
        with pytest.raises(ValueError, match="No DBN-annotated sequences"):
            ArchiveIISource(config)

    def test_iter_yields_entries(self, tmp_path: Path) -> None:
        """Iterating yields each entry dict."""
        self._create_csv(tmp_path, n_entries=3)
        config = ArchiveIIConfig(
            data_dir=str(tmp_path),
            filename="example_data_structure.csv",
        )
        source = ArchiveIISource(config)
        entries = list(source)
        assert len(entries) == 3


@pytest.mark.skipif(not _DATA_EXISTS, reason="ArchiveII data not available")
class TestArchiveIISourceIntegration:
    """Integration tests with the real ArchiveII dataset."""

    @pytest.fixture()
    def source(self) -> ArchiveIISource:
        """Load a small subset of the real dataset."""
        config = ArchiveIIConfig(data_dir=str(_DATA_DIR), max_sequences=10)
        return ArchiveIISource(config)

    def test_returns_entries(self, source: ArchiveIISource) -> None:
        """Real dataset returns entries."""
        data = source.load()
        assert data["n_sequences"] > 0

    def test_structures_valid_dbn(self, source: ArchiveIISource) -> None:
        """Real structures are valid dot-bracket notation."""
        valid_chars = set(".()")
        for entry in source:
            for char in entry["structure"]:
                assert char in valid_chars, f"Invalid DBN char '{char}' in {entry['name']}"

    def test_max_sequences_applied(self, source: ArchiveIISource) -> None:
        """max_sequences limits the loaded count."""
        assert len(source) <= 10
