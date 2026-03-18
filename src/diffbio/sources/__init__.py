"""DiffBio data sources module.

This module provides data source implementations extending Datarax's DataSourceModule
for bioinformatics and drug discovery applications.

Sources:
    AnnDataSource: AnnData (.h5ad) file reading for single-cell data
    IndexedViewSource: Lazy-loading view into a data source using index mapping
    MolNetSource: MoleculeNet benchmark datasets for drug discovery
    BAMSource: BAM/CRAM file reading for aligned sequencing reads
    FastaSource: FASTA file reading for DNA/RNA sequences
"""

from diffbio.sources.anndata_source import AnnDataSource
from diffbio.sources.bam import BAMSource, BAMSourceConfig
from diffbio.sources.fasta import FastaSource, FastaSourceConfig
from diffbio.sources.indexed_view import IndexedViewSource, IndexedViewSourceConfig
from diffbio.sources.molnet import MolNetSource, MolNetSourceConfig

__all__ = [
    "AnnDataSource",
    "BAMSource",
    "BAMSourceConfig",
    "FastaSource",
    "FastaSourceConfig",
    "IndexedViewSource",
    "IndexedViewSourceConfig",
    "MolNetSource",
    "MolNetSourceConfig",
]
