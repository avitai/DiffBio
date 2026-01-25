"""FASTA file data source for genomics workflows.

This module provides FastaSource for reading DNA/RNA sequences
from FASTA files with lazy loading and efficient indexed access.

Based on best practices from:
- pyfaidx (samtools-compatible FASTA indexing)
- BioPython SeqIO.index patterns
- Google Nucleus FASTA handling

References:
    - https://github.com/mdshw5/pyfaidx
    - https://pythonhosted.org/pyfaidx/
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
from flax import nnx

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.typing import Element

from diffbio.sequences.dna import encode_dna_string


@dataclass
class FastaSourceConfig(StructuralConfig):
    """Configuration for FASTA data source.

    Attributes:
        file_path: Path to FASTA file
        handle_n: How to handle N nucleotides ("uniform" or "zero")
        create_index: Whether to create .fai index if not exists (default: True)
    """

    file_path: Path = None  # type: ignore[assignment]  # Required, validated in post_init
    handle_n: Literal["uniform", "zero"] = "uniform"
    create_index: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        super().__post_init__()
        if self.file_path is None:
            raise ValueError("file_path is required")


class FastaSource(DataSourceModule):
    """FASTA file data source extending Datarax DataSourceModule.

    Provides efficient access to DNA/RNA sequences with:

    - Lazy loading using samtools-compatible .fai index
    - Dictionary-like access by sequence name
    - One-hot encoded sequence output
    - Support for compressed BGZF files

    Inherits from DataSourceModule (StructuralModule) because:

    - Non-parametric: FASTA reading is deterministic
    - Frozen config: file parameters don't change
    - Domain-specific: requires genomics-specific handling

    Example:
        ```python
        config = FastaSourceConfig(file_path=Path("genome.fasta"))
        source = FastaSource(config)
        elem = source.get_by_name("chr1")
        print(elem.data["sequence"].shape)
        ```

    Performance Tips (from pyfaidx best practices):

    - Use indexed FASTA files (.fai) for random access
    - Access regions with slicing for large chromosomes
    - BGZF compression reduces disk space while maintaining random access
    """

    # Annotate data storage for Flax NNX
    _sequence_names: nnx.Data[list]
    _name_to_idx: nnx.Data[dict]

    def __init__(
        self,
        config: FastaSourceConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize FastaSource.

        Args:
            config: FASTA source configuration
            rngs: Random number generators (unused for data loading)
            name: Optional module name

        Raises:
            FileNotFoundError: If FASTA file not found
            ImportError: If pyfaidx is not installed
        """
        super().__init__(config, rngs=rngs, name=name)

        # Import pyfaidx lazily to allow installation without it
        try:
            import pyfaidx

            self._pyfaidx = pyfaidx
        except ImportError as err:
            raise ImportError(
                "pyfaidx is required for FastaSource. Install with: pip install pyfaidx"
            ) from err

        # Validate file exists
        if not config.file_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {config.file_path}")

        # Open FASTA file with pyfaidx (creates index if needed)
        self._fasta = self._pyfaidx.Fasta(
            str(config.file_path),
            build_index=config.create_index,
        )

        # Build sequence name index
        self._sequence_names = list(self._fasta.keys())
        self._name_to_idx = {name: idx for idx, name in enumerate(self._sequence_names)}
        self._current_idx = 0

    def _sequence_to_element(self, idx: int, seq_name: str) -> Element:
        """Convert FASTA sequence to Element with one-hot encoding.

        Args:
            idx: Index of the sequence
            seq_name: Name/ID of the sequence

        Returns:
            Element with one-hot encoded sequence and metadata
        """
        # Get sequence from pyfaidx (lazy loaded)
        fasta_seq = self._fasta[seq_name]
        sequence_str = str(fasta_seq).upper()

        # Encode sequence as one-hot
        sequence = encode_dna_string(sequence_str, handle_n=self.config.handle_n)

        # Get description if available
        description = getattr(fasta_seq, "long_name", seq_name)

        data = {
            "sequence": sequence,
            "sequence_id": seq_name,
            "description": description,
        }

        metadata = {
            "idx": idx,
            "length": len(sequence_str),
            "file_path": str(self.config.file_path),
        }

        return Element(data=data, state={}, metadata=metadata)

    def __len__(self) -> int:
        """Return the number of sequences in the source."""
        return len(self._sequence_names)

    def __getitem__(self, idx: int) -> Element | None:
        """Get sequence by index.

        Args:
            idx: Index of the sequence

        Returns:
            Element at the given index, or None if out of bounds
        """
        if idx < 0 or idx >= len(self._sequence_names):
            return None
        seq_name = self._sequence_names[idx]
        return self._sequence_to_element(idx, seq_name)

    def __iter__(self):
        """Return iterator over sequences."""
        self._current_idx = 0
        return self

    def __next__(self) -> Element:
        """Get next sequence in iteration."""
        if self._current_idx >= len(self._sequence_names):
            raise StopIteration
        seq_name = self._sequence_names[self._current_idx]
        elem = self._sequence_to_element(self._current_idx, seq_name)
        self._current_idx += 1
        return elem

    def reset(self, seed: int | None = None) -> None:  # noqa: ARG002
        """Reset iteration state.

        Args:
            seed: Optional seed (unused, for API compatibility)
        """
        del seed  # Unused, for API compatibility
        self._current_idx = 0

    def get_batch(self, batch_size: int, key: jax.Array | None = None) -> list[Element]:  # noqa: ARG002
        """Get a batch of sequences.

        Args:
            batch_size: Number of sequences to retrieve
            key: Optional JAX random key (unused)

        Returns:
            List of Elements
        """
        del key  # Unused, for API compatibility
        batch = []
        for _ in range(batch_size):
            if self._current_idx >= len(self._sequence_names):
                break
            seq_name = self._sequence_names[self._current_idx]
            elem = self._sequence_to_element(self._current_idx, seq_name)
            batch.append(elem)
            self._current_idx += 1
        return batch

    def get_by_name(self, name: str) -> Element | None:
        """Get sequence by name/ID.

        Args:
            name: Sequence identifier (e.g., "chr1", "seq1")

        Returns:
            Element for the sequence, or None if not found
        """
        if name not in self._name_to_idx:
            return None
        idx = self._name_to_idx[name]
        return self._sequence_to_element(idx, name)

    @property
    def sequence_names(self) -> list[str]:
        """Get list of all sequence names in the FASTA file."""
        return list(self._sequence_names)
