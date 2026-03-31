"""ENCODE narrowPeak BED data source.

Loads peak calls from ENCODE narrowPeak BED files (gzipped or plain),
such as those produced by the ENCODE ChIP-seq pipeline. Each row in
the BED file encodes one called peak with:

    chr  start  end  name  score  strand  signalValue  pValue  qValue  peak

Only the first ten columns are required. The ``peak`` column (column 10)
gives the offset within the peak region to the summit position.

Peaks can be filtered to a single chromosome for speed and optionally
capped at a maximum count.

Reference:
    ENCODE Project Consortium. "An integrated encyclopedia of DNA
    elements in the human genome." Nature 489, 57-74 (2012).
"""

from __future__ import annotations

import gzip
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from flax import nnx

logger = logging.getLogger(__name__)

_DEFAULT_DATA_PATH = "/media/mahdi/ssd23/Data/encode/CTCF_K562_narrowPeak.bed.gz"


@dataclass(frozen=True, kw_only=True)
class ENCODEPeakConfig(StructuralConfig):
    """Configuration for ENCODEPeakSource.

    Attributes:
        data_path: Path to the narrowPeak BED file (gzipped or plain).
        chromosome: Chromosome to filter peaks to. None loads all.
        max_peaks: Maximum number of peaks to load. None loads all.
    """

    data_path: str = _DEFAULT_DATA_PATH
    chromosome: str | None = "chr22"
    max_peaks: int | None = None

    def __post_init__(self) -> None:
        """Validate that the data file exists."""
        super().__post_init__()
        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(
                f"ENCODE narrowPeak file not found: {path}. "
                f"Download from https://www.encodeproject.org/"
            )


@dataclass(frozen=True, kw_only=True)
class ENCODEPeak:
    """A single ENCODE narrowPeak record.

    Attributes:
        chromosome: Chromosome name (e.g. ``"chr22"``).
        start: 0-based start coordinate.
        end: 0-based end coordinate (exclusive).
        signal_value: Fold-enrichment signal value.
        p_value: -log10 p-value (or -1 if unavailable).
        q_value: -log10 q-value (or -1 if unavailable).
        summit_offset: Offset from ``start`` to the summit position.
    """

    chromosome: str
    start: int
    end: int
    signal_value: float
    p_value: float
    q_value: float
    summit_offset: int


def _parse_narrowpeak(
    file_path: Path,
    chromosome: str | None,
    max_peaks: int | None,
) -> list[ENCODEPeak]:
    """Parse a narrowPeak BED file into ENCODEPeak records.

    Args:
        file_path: Path to gzipped or plain BED file.
        chromosome: Filter to this chromosome. None keeps all.
        max_peaks: Maximum peaks to return. None returns all.

    Returns:
        Sorted list of ENCODEPeak records (by start position).
    """
    peaks: list[ENCODEPeak] = []
    opener = gzip.open if file_path.suffix == ".gz" else open

    with opener(file_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 10:
                continue

            chrom = fields[0]
            if chromosome is not None and chrom != chromosome:
                continue

            peak = ENCODEPeak(
                chromosome=chrom,
                start=int(fields[1]),
                end=int(fields[2]),
                signal_value=float(fields[6]),
                p_value=float(fields[7]),
                q_value=float(fields[8]),
                summit_offset=int(fields[9]),
            )
            peaks.append(peak)

            if max_peaks is not None and len(peaks) >= max_peaks:
                break

    # Sort by genomic position for deterministic ordering
    peaks.sort(key=lambda p: (p.chromosome, p.start))
    return peaks


class ENCODEPeakSource(DataSourceModule):
    """DataSource for ENCODE narrowPeak ChIP-seq peak calls.

    Loads peak positions and signal values from an ENCODE narrowPeak
    BED file. Peaks are optionally filtered to a single chromosome
    and/or capped at a maximum count.

    Each loaded peak provides genomic coordinates, signal enrichment,
    statistical significance, and summit position.

    Example:
        ```python
        config = ENCODEPeakConfig(chromosome="chr22", max_peaks=500)
        source = ENCODEPeakSource(config)
        data = source.load()
        print(data["n_peaks"])       # Number of peaks loaded
        print(data["starts"][:5])    # First 5 start positions
        ```
    """

    data: dict[str, Any] = nnx.data()

    def __init__(
        self,
        config: ENCODEPeakConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Load ENCODE narrowPeak data from BED file.

        Args:
            config: Configuration with file path and filters.
            rngs: Optional RNG state (unused, for interface compat).
            name: Optional module name.
        """
        super().__init__(config, rngs=rngs, name=name or "ENCODEPeakSource")
        file_path = Path(config.data_path)
        peaks = _parse_narrowpeak(file_path, config.chromosome, config.max_peaks)

        if not peaks:
            chrom_msg = f" on {config.chromosome}" if config.chromosome else ""
            raise ValueError(
                f"No peaks found in {file_path}{chrom_msg}. "
                f"Check the file format and chromosome filter."
            )

        starts = np.array([p.start for p in peaks], dtype=np.int64)
        ends = np.array([p.end for p in peaks], dtype=np.int64)
        signals = np.array([p.signal_value for p in peaks], dtype=np.float64)
        summits = np.array(
            [p.start + p.summit_offset for p in peaks],
            dtype=np.int64,
        )

        self.data = {
            "peaks": peaks,
            "starts": starts,
            "ends": ends,
            "signal_values": signals,
            "summit_positions": summits,
            "n_peaks": len(peaks),
            "chromosome": config.chromosome,
        }
        logger.info(
            "Loaded ENCODE peaks: %d peaks from %s%s",
            len(peaks),
            file_path.name,
            f" ({config.chromosome})" if config.chromosome else "",
        )

    def load(self) -> dict[str, Any]:
        """Return the full dataset as a dictionary.

        Returns:
            Dict with keys: peaks, starts, ends, signal_values,
            summit_positions, n_peaks, chromosome.
        """
        return self.data

    def __len__(self) -> int:
        """Return the number of loaded peaks."""
        return self.data["n_peaks"]

    def __iter__(self) -> Iterator[ENCODEPeak]:
        """Iterate over individual peak records."""
        yield from self.data["peaks"]
