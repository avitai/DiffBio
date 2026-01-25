"""Sequence identity splitter for bioinformatics applications.

This module provides sequence-aware splitting utilities:
- SequenceIdentitySplitter: Split by sequence identity clustering

For genomics/proteomics applications where similar sequences
should not appear in both train and test sets.
"""

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
from flax import nnx

from datarax.core.data_source import DataSourceModule

from diffbio.splitters.base import SplitResult, SplitterConfig, SplitterModule


@dataclass
class SequenceIdentitySplitterConfig(SplitterConfig):
    """Configuration for sequence identity splitter.

    Attributes:
        sequence_key: Key in data element containing sequence string (default: "sequence")
        identity_threshold: Identity threshold for clustering (default: 0.3)
            Sequences with identity > threshold are clustered together.
        alignment_method: Method for identity computation ("simple" or "mmseqs2")
    """

    sequence_key: str = "sequence"
    identity_threshold: float = 0.3
    alignment_method: str = "simple"


class SequenceIdentitySplitter(SplitterModule):
    """Split sequences by identity threshold.

    Groups similar sequences together using identity clustering,
    then assigns clusters to train/valid/test to ensure structural
    diversity between splits. This prevents data leakage from
    similar sequences appearing in different splits.

    Inherits from SplitterModule (StructuralModule) because:

    - Non-parametric: clustering is deterministic
    - Frozen config: splitting strategy doesn't change
    - Domain-specific: requires sequence comparison

    Similar to CD-HIT or MMseqs2 clustering approach.

    Example:
        ```python
        config = SequenceIdentitySplitterConfig(identity_threshold=0.3)
        splitter = SequenceIdentitySplitter(config)
        result = splitter.split(sequence_source)
        ```

    References:
        Li, Weizhong, and Adam Godzik. "Cd-hit: a fast program for clustering
        and comparing large sets of protein or nucleotide sequences."
        Bioinformatics 22.13 (2006): 1658-1659.
    """

    def __init__(
        self,
        config: SequenceIdentitySplitterConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize SequenceIdentitySplitter.

        Args:
            config: Sequence identity splitter configuration
            rngs: Random number generators (unused for identity splitting)
            name: Optional module name
        """
        super().__init__(config, rngs=rngs, name=name)

    def _compute_identity(self, seq1: str, seq2: str) -> float:
        """Compute sequence identity between two sequences.

        Uses simple character matching. For unequal lengths,
        compares up to the length of the shorter sequence.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Identity fraction between 0.0 and 1.0
        """
        if not seq1 or not seq2:
            return 0.0

        # Use shorter sequence for comparison
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]

        if not seq1:
            return 0.0

        matches = sum(c1 == c2 for c1, c2 in zip(seq1, seq2))
        return matches / len(seq1)

    def _cluster_by_identity(self, sequences: Sequence[str]) -> list[list[int]]:
        """Cluster sequences by identity threshold.

        Uses greedy clustering: each sequence joins the first cluster
        where it has identity > threshold with the representative.

        Args:
            sequences: List of sequence strings

        Returns:
            List of clusters, each cluster is a list of sequence indices
        """
        if self.config.alignment_method == "simple":
            return self._simple_clustering(sequences)
        elif self.config.alignment_method == "mmseqs2":
            return self._mmseqs2_clustering(sequences)
        else:
            raise ValueError(f"Unknown alignment method: {self.config.alignment_method}")

    def _simple_clustering(self, sequences: Sequence[str]) -> list[list[int]]:
        """Simple greedy clustering by identity.

        Each sequence joins the first cluster where it has
        identity > threshold with the representative.

        Args:
            sequences: List of sequence strings

        Returns:
            List of clusters
        """
        clusters: list[list[int]] = []
        representatives: list[str] = []

        for idx, seq in enumerate(sequences):
            assigned = False

            for cluster_idx, rep in enumerate(representatives):
                identity = self._compute_identity(seq, rep)
                if identity > self.config.identity_threshold:
                    clusters[cluster_idx].append(idx)
                    assigned = True
                    break

            if not assigned:
                clusters.append([idx])
                representatives.append(seq)

        return clusters

    def _mmseqs2_clustering(self, sequences: Sequence[str]) -> list[list[int]]:
        """Use MMseqs2 for clustering.

        Requires MMseqs2 installation.

        Args:
            sequences: List of sequence strings

        Returns:
            List of clusters

        Raises:
            NotImplementedError: MMseqs2 integration not yet implemented
        """
        raise NotImplementedError(
            "MMseqs2 clustering requires external tool installation. "
            "Use alignment_method='simple' for built-in clustering."
        )

    def split(self, data_source: DataSourceModule) -> SplitResult:
        """Split by sequence identity clustering.

        Clusters sequences by identity, then assigns clusters
        to train/valid/test splits. Largest clusters go to
        train first.

        Args:
            data_source: Datarax DataSourceModule to split

        Returns:
            SplitResult with identity-based train/valid/test indices
        """
        # Extract sequences from data source
        sequences = [data_source[i].data[self.config.sequence_key] for i in range(len(data_source))]

        # Cluster by identity
        clusters = self._cluster_by_identity(sequences)

        # Sort clusters by size (largest first)
        sorted_clusters = sorted(clusters, key=len, reverse=True)

        # Calculate split cutoffs
        n = len(data_source)
        train_cutoff = self.config.train_frac * n
        valid_cutoff = (self.config.train_frac + self.config.valid_frac) * n

        train_inds: list[int] = []
        valid_inds: list[int] = []
        test_inds: list[int] = []

        for cluster in sorted_clusters:
            # Fill train first until we've exceeded the cutoff
            if len(train_inds) < train_cutoff:
                train_inds.extend(cluster)
            # Then fill valid until we've exceeded its cutoff
            elif len(train_inds) + len(valid_inds) < valid_cutoff:
                valid_inds.extend(cluster)
            # Everything else goes to test
            else:
                test_inds.extend(cluster)

        return SplitResult(
            train_indices=jnp.array(train_inds, dtype=jnp.int32),
            valid_indices=jnp.array(valid_inds, dtype=jnp.int32),
            test_indices=jnp.array(test_inds, dtype=jnp.int32),
        )
