"""ArchiveII RNA secondary structure DataSource.

Loads RNA sequences with known secondary structures from the ArchiveII
benchmark dataset (Sloma & Mathews, 2016). Structures are provided in
dot-bracket notation (DBN) and represent experimentally validated RNA
secondary structures from crystal structures and NMR.

The dataset is read from CSV files produced by RNAFoldAssess, with
columns: name, sequence, ground_truth_type, ground_truth_data.

Only rows with ``ground_truth_type == "DBN"`` are loaded; rows without
structure annotations are silently skipped.

References:
    Sloma, M. F. & Mathews, D. H. (2016). Exact calculation of loop
    formation probability identifies folding motifs in RNA secondary
    structures. RNA 22, 1808-1818.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from flax import nnx

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = "/media/mahdi/ssd23/Works/RNAFoldAssess/tutorial/processed_data"
_STRUCTURE_FILENAME = "example_data_structure.csv"


@dataclass(frozen=True, kw_only=True)
class ArchiveIIConfig(StructuralConfig):
    """Configuration for ArchiveIISource.

    Attributes:
        data_dir: Directory containing the ArchiveII CSV file.
        filename: Name of the CSV file with structure annotations.
        max_sequences: Maximum number of sequences to load.
            None means load all available sequences.
    """

    data_dir: str = _DEFAULT_DATA_DIR
    filename: str = _STRUCTURE_FILENAME
    max_sequences: int | None = None

    def __post_init__(self) -> None:
        """Validate that the data file exists."""
        super().__post_init__()
        path = Path(self.data_dir) / self.filename
        if not path.exists():
            raise FileNotFoundError(
                f"ArchiveII data not found: {path}. "
                f"Expected a CSV with columns: name, sequence, "
                f"ground_truth_type, ground_truth_data."
            )


def _parse_csv(
    csv_path: Path,
    max_sequences: int | None,
) -> list[dict[str, str]]:
    """Parse ArchiveII CSV and return DBN-annotated entries.

    Args:
        csv_path: Path to the CSV file.
        max_sequences: Maximum entries to return. None for all.

    Returns:
        List of dicts with keys: name, sequence, structure.
    """
    entries: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gt_type = row.get("ground_truth_type", "").strip()
            gt_data = row.get("ground_truth_data", "").strip()
            if gt_type != "DBN" or not gt_data:
                continue
            sequence = row["sequence"].strip().upper()
            name = row["name"].strip()
            entries.append(
                {
                    "name": name,
                    "sequence": sequence,
                    "structure": gt_data,
                }
            )
            if max_sequences is not None and len(entries) >= max_sequences:
                break
    return entries


class ArchiveIISource(DataSourceModule):
    """DataSource for ArchiveII RNA secondary structures.

    Loads RNA sequences and their known dot-bracket notation (DBN)
    structures from a CSV file. Only entries with ground_truth_type
    ``"DBN"`` are included.

    Each entry is a dict with keys:
        - ``name``: Sequence identifier (e.g. ``"1KXK_chain_0"``)
        - ``sequence``: RNA sequence string (e.g. ``"GUCUACC..."``)
        - ``structure``: DBN string (e.g. ``"....(((...)))..."``)

    Example:
        ```python
        config = ArchiveIIConfig(max_sequences=10)
        source = ArchiveIISource(config)
        data = source.load()
        print(data["n_sequences"])
        print(data["entries"][0]["name"])
        ```
    """

    data: dict[str, Any] = nnx.data()

    def __init__(
        self,
        config: ArchiveIIConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Load ArchiveII RNA structures from CSV.

        Args:
            config: Configuration with data directory and limits.
            rngs: Optional RNG state (unused, for interface compat).
            name: Optional module name.
        """
        super().__init__(config, rngs=rngs, name=name or "ArchiveIISource")
        csv_path = Path(config.data_dir) / config.filename
        entries = _parse_csv(csv_path, config.max_sequences)

        if not entries:
            raise ValueError(
                f"No DBN-annotated sequences found in {csv_path}. "
                f"Ensure the CSV has rows with "
                f"ground_truth_type='DBN'."
            )

        self.data = {
            "entries": entries,
            "n_sequences": len(entries),
        }
        logger.info(
            "Loaded ArchiveII: %d sequences from %s",
            len(entries),
            csv_path,
        )

    def load(self) -> dict[str, Any]:
        """Return the full dataset as a dictionary.

        Returns:
            Dict with keys: entries (list of dicts), n_sequences.
        """
        return self.data

    def __len__(self) -> int:
        """Return the number of loaded sequences."""
        return self.data["n_sequences"]

    def __iter__(self) -> Iterator[dict[str, str]]:
        """Iterate over individual RNA entries."""
        yield from self.data["entries"]
