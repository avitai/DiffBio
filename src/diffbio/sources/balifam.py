"""BAliBASE reference alignment DataSource (balifam).

Loads protein families from the balifam repository, which provides
curated subsets of BAliBASE reference alignments at three tiers
(100, 1000, 10000 sequences per family).

Each family contains unaligned input sequences and a reference
alignment with mixed-case convention:
- Uppercase residues: core (scored) positions
- Lowercase residues: insert (unscored) positions
- Dots (.): gap characters

The dataset must be available locally. See:
https://github.com/steineggerlab/balifam

Data directory layout::

    balifam100/
        in/       # Unaligned FASTA files (one per family)
        ref/      # Reference alignments (FASTA with gaps)
        info/     # Family ID lists
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from flax import nnx

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = "/media/mahdi/ssd23/Works/balifam"


def _parse_fasta(path: Path) -> list[tuple[str, str]]:
    """Parse a FASTA file into (name, sequence) tuples.

    Handles multi-line sequences by concatenating continuation lines.

    Args:
        path: Path to FASTA file.

    Returns:
        List of (sequence_name, sequence_string) tuples.
    """
    entries: list[tuple[str, str]] = []
    current_name: str | None = None
    current_seq_parts: list[str] = []

    with path.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if current_name is not None:
                    entries.append(
                        (current_name, "".join(current_seq_parts))
                    )
                current_name = line[1:].strip()
                current_seq_parts = []
            elif current_name is not None:
                current_seq_parts.append(line)

    if current_name is not None:
        entries.append((current_name, "".join(current_seq_parts)))

    return entries


@dataclass(frozen=True, kw_only=True)
class BalifamConfig(StructuralConfig):
    """Configuration for BalifamSource.

    Attributes:
        data_dir: Root directory of the balifam repository.
        tier: Family size tier (100, 1000, or 10000 sequences).
        max_families: Maximum number of families to load.
            None loads all available families.
    """

    data_dir: str = _DEFAULT_DATA_DIR
    tier: int = 100
    max_families: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        valid_tiers = {100, 1000, 10000}
        if self.tier not in valid_tiers:
            raise ValueError(
                f"tier must be one of {valid_tiers}, got {self.tier}"
            )
        tier_dir = Path(self.data_dir) / f"balifam{self.tier}"
        if not tier_dir.exists():
            raise FileNotFoundError(
                f"Balifam tier directory not found: {tier_dir}. "
                f"Clone from: https://github.com/steineggerlab/balifam"
            )


class BalifamSource(DataSourceModule):
    """DataSource for BAliBASE reference alignments (balifam).

    Loads protein family alignments from balifam for evaluating
    multiple sequence alignment methods. Each family provides
    unaligned input sequences and a curated reference alignment.

    The reference alignment contains a subset of the input sequences
    with known structural alignment, using mixed-case annotation
    (uppercase = core/scored, lowercase = insert/unscored).

    Example:
        ```python
        config = BalifamConfig(tier=100, max_families=5)
        source = BalifamSource(config)
        families = source.load()
        print(families[0]["family_id"])
        ```
    """

    families: list[dict[str, Any]] = nnx.data()

    def __init__(
        self,
        config: BalifamConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Load balifam families.

        Args:
            config: Configuration with data directory and options.
            rngs: Optional RNG state (unused, for interface compat).
            name: Optional module name.
        """
        super().__init__(config, rngs=rngs, name=name or "BalifamSource")
        self.families = self._load_families(config)
        logger.info(
            "Loaded %d balifam%d families",
            len(self.families),
            config.tier,
        )

    def _load_families(
        self, config: BalifamConfig
    ) -> list[dict[str, Any]]:
        """Load families from disk.

        Args:
            config: Source configuration.

        Returns:
            List of family dicts with keys: family_id, sequences,
            reference, n_sequences, n_reference.
        """
        tier_dir = Path(config.data_dir) / f"balifam{config.tier}"
        in_dir = tier_dir / "in"
        ref_dir = tier_dir / "ref"

        # Discover families from reference directory (authoritative)
        family_files = sorted(ref_dir.iterdir())
        if config.max_families is not None:
            family_files = family_files[: config.max_families]

        families: list[dict[str, Any]] = []
        for ref_path in family_files:
            family_id = ref_path.name
            in_path = in_dir / family_id

            if not in_path.exists():
                logger.warning(
                    "Input file missing for family %s, skipping",
                    family_id,
                )
                continue

            sequences = _parse_fasta(in_path)
            reference = _parse_fasta(ref_path)

            families.append({
                "family_id": family_id,
                "sequences": sequences,
                "reference": reference,
                "n_sequences": len(sequences),
                "n_reference": len(reference),
            })

        return families

    def load(self) -> list[dict[str, Any]]:
        """Return all loaded families.

        Returns:
            List of family dicts, each with keys: family_id,
            sequences, reference, n_sequences, n_reference.
        """
        return self.families

    def __len__(self) -> int:
        """Return the number of loaded families."""
        return len(self.families)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over families."""
        return iter(self.families)
