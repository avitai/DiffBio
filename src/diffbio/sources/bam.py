"""BAM/CRAM file data source for genomics workflows.

This module provides BAMSource for reading aligned sequencing reads
from BAM/CRAM files with lazy loading and efficient indexed access.

Based on best practices from:
- pysam (HTSlib Python wrapper)
- Google Nucleus genomics library
- DeepVariant BAM handling patterns

References:
    - https://pysam.readthedocs.io/
    - https://github.com/pysam-developers/pysam
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.typing import Element

from diffbio.sequences.dna import encode_dna_string


@dataclass
class BAMSourceConfig(StructuralConfig):
    """Configuration for BAM/CRAM data source.

    Attributes:
        file_path: Path to BAM/CRAM file
        reference_path: Optional path to reference FASTA (required for CRAM)
        include_unmapped: Whether to include unmapped reads (default: False)
        min_mapping_quality: Minimum mapping quality to include (default: None)
        region: Optional genomic region to query (e.g., "chr1:1000-2000")
        handle_n: How to handle N nucleotides in sequences
    """

    file_path: Path = None  # type: ignore[assignment]  # Required, validated in post_init
    reference_path: Path | None = None
    include_unmapped: bool = False
    min_mapping_quality: int | None = None
    region: str | None = None
    handle_n: Literal["uniform", "zero"] = "uniform"

    def __post_init__(self):
        """Validate configuration after initialization."""
        super().__post_init__()
        if self.file_path is None:
            raise ValueError("file_path is required")


class BAMSource(DataSourceModule):
    """BAM/CRAM file data source extending Datarax DataSourceModule.

    Provides efficient access to aligned sequencing reads with:

    - Lazy loading using pysam iterators
    - Indexed random access via BAI/CRAI files
    - Quality filtering at load time
    - One-hot encoded sequence output

    Inherits from DataSourceModule (StructuralModule) because:

    - Non-parametric: BAM reading is deterministic
    - Frozen config: file parameters don't change
    - Domain-specific: requires genomics-specific handling

    Example:
        ```python
        config = BAMSourceConfig(file_path=Path("sample.bam"))
        source = BAMSource(config)
        for element in source:
            print(element.data["read_name"], element.data["sequence"].shape)
        ```

    Performance Tips (from pysam best practices):

    - Use indexed BAM files for random access
    - Filter by region to reduce data loading
    - Set min_mapping_quality to filter at read time
    """

    # Annotate data storage for Flax NNX
    _reads: list = nnx.data()

    def __init__(
        self,
        config: BAMSourceConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize BAMSource.

        Args:
            config: BAM source configuration
            rngs: Random number generators (unused for data loading)
            name: Optional module name

        Raises:
            FileNotFoundError: If BAM file not found
            ImportError: If pysam is not installed
        """
        super().__init__(config, rngs=rngs, name=name)

        # Import pysam lazily to allow installation without it
        try:
            import pysam

            self._pysam = pysam
        except ImportError as err:
            raise ImportError(
                "pysam is required for BAMSource. Install with: pip install pysam"
            ) from err

        # Validate file exists
        if not config.file_path.exists():
            raise FileNotFoundError(f"BAM file not found: {config.file_path}")

        # Load read index (metadata only, not full sequences)
        self._reads = self._index_reads()
        self._current_idx = 0

    def _index_reads(self) -> list[dict]:
        """Build an index of reads for random access.

        This loads read metadata without fully parsing sequences,
        enabling lazy loading on access.

        Returns:
            List of read metadata dictionaries
        """
        config = self.config
        reads = []

        # Open BAM file
        mode = "rb" if str(config.file_path).endswith(".bam") else "rc"
        reference = str(config.reference_path) if config.reference_path else None

        with self._pysam.AlignmentFile(
            str(config.file_path), mode, reference_filename=reference
        ) as bam:
            # Use region query if specified, otherwise iterate all
            if config.region:
                iterator = bam.fetch(region=config.region)
            else:
                iterator = bam.fetch(until_eof=True)

            for read in iterator:
                # Skip unmapped if configured
                if read.is_unmapped and not config.include_unmapped:
                    continue

                # Skip low quality reads
                if (
                    config.min_mapping_quality is not None
                    and read.mapping_quality < config.min_mapping_quality
                ):
                    continue

                # Store minimal info for indexing
                read_info = {
                    "query_name": read.query_name,
                    "reference_id": read.reference_id,
                    "reference_start": read.reference_start,
                    "is_unmapped": read.is_unmapped,
                    "query_sequence": read.query_sequence,
                    "query_qualities": (
                        list(read.query_qualities) if read.query_qualities is not None else None
                    ),
                    "mapping_quality": read.mapping_quality,
                    "reference_name": (read.reference_name if not read.is_unmapped else None),
                }
                reads.append(read_info)

        return reads

    def _read_to_element(self, idx: int, read_info: dict) -> Element:
        """Convert read info to Element with one-hot encoded sequence.

        Args:
            idx: Index of the read
            read_info: Read metadata dictionary

        Returns:
            Element with sequence, quality scores, and metadata
        """
        # Encode sequence as one-hot
        sequence_str = read_info["query_sequence"] or ""
        if sequence_str:
            sequence = encode_dna_string(sequence_str, handle_n=self.config.handle_n)
        else:
            sequence = jnp.zeros((0, 4), dtype=jnp.float32)

        # Get quality scores
        if read_info["query_qualities"] is not None:
            quality_scores = jnp.array(read_info["query_qualities"], dtype=jnp.float32)
        else:
            # Default quality if not available
            quality_scores = jnp.ones(len(sequence_str), dtype=jnp.float32) * 30.0

        data = {
            "sequence": sequence,
            "quality_scores": quality_scores,
            "read_name": read_info["query_name"],
        }

        metadata = {
            "idx": idx,
            "reference_name": read_info["reference_name"],
            "reference_start": read_info["reference_start"],
            "mapping_quality": read_info["mapping_quality"],
            "unmapped": read_info["is_unmapped"],
        }

        return Element(data=data, state={}, metadata=metadata)  # pyright: ignore[reportArgumentType]

    def __len__(self) -> int:
        """Return the number of reads in the source."""
        return len(self._reads)

    def __getitem__(self, idx: int) -> Element | None:
        """Get read by index.

        Args:
            idx: Index of the read

        Returns:
            Element at the given index, or None if out of bounds
        """
        if idx < 0 or idx >= len(self._reads):
            return None
        return self._read_to_element(idx, self._reads[idx])

    def __iter__(self):
        """Return iterator over reads."""
        self._current_idx = 0
        return self

    def __next__(self) -> Element:
        """Get next read in iteration."""
        if self._current_idx >= len(self._reads):
            raise StopIteration
        elem = self._read_to_element(self._current_idx, self._reads[self._current_idx])
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
        """Get a batch of reads.

        Args:
            batch_size: Number of reads to retrieve
            key: Optional JAX random key (unused)

        Returns:
            List of Elements
        """
        del key  # Unused, for API compatibility
        batch = []
        for _ in range(batch_size):
            if self._current_idx >= len(self._reads):
                break
            elem = self._read_to_element(self._current_idx, self._reads[self._current_idx])
            batch.append(elem)
            self._current_idx += 1
        return batch
