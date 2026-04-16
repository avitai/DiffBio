"""DiffBio data sources module.

This module provides data source implementations extending Datarax's DataSourceModule
for bioinformatics and drug discovery applications.

Sources:
    AnnDataSource: AnnData (.h5ad) file reading for single-cell data
    ENCODEPeakSource: ENCODE narrowPeak BED file reading for ChIP-seq peaks
    IndexedViewSource: Lazy-loading view into a data source using index mapping
    MolNetSource: MoleculeNet benchmark datasets for drug discovery
    BAMSource: BAM/CRAM file reading for aligned sequencing reads
    BioSNAPDTISource: Deterministic binary DTI benchmark source
    DavisDTISource: Deterministic affinity DTI benchmark source
    FastaSource: FASTA file reading for DNA/RNA sequences

Interop:
    to_anndata: Convert DiffBio data dict to AnnData object
    from_anndata: Convert AnnData object to DiffBio data dict
"""

from diffbio.sources.anndata_interop import from_anndata, to_anndata
from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig
from diffbio.sources.bam import BAMSource, BAMSourceConfig
from diffbio.sources.contextual_epigenomics import (
    CONTEXTUAL_EPIGENOMICS_DATASET_CONTRACT_KEYS,
    CONTEXTUAL_TARGET_SEMANTICS,
    build_synthetic_contextual_epigenomics_dataset,
    validate_contextual_epigenomics_dataset,
)
from diffbio.sources.dti import (
    DTI_DATASET_CONTRACT_KEYS,
    BioSNAPDTISource,
    DTISourceConfig,
    DavisDTISource,
    build_paired_dti_batch,
    deterministic_dti_split,
    validate_dti_dataset,
)
from diffbio.sources.embeddings import (
    EmbeddingArtifactSource,
    EmbeddingArtifactSourceConfig,
)
from diffbio.sources.encode_peaks import ENCODEPeakConfig, ENCODEPeakSource
from diffbio.sources.fasta import FastaSource, FastaSourceConfig
from diffbio.sources.indexed_view import IndexedViewSource, IndexedViewSourceConfig
from diffbio.sources.molnet import MolNetSource, MolNetSourceConfig
from diffbio.sources.perturbation import (
    BatchControlMapping,
    ControlMappingConfig,
    ExperimentConfig,
    GlobalH5MetadataCache,
    H5MetadataCache,
    PerturbationAnnDataSource,
    PerturbationConcatSource,
    PerturbationSourceConfig,
    RandomControlMapping,
    load_experiment_config,
)
from diffbio.sources.singlecell_foundation import (
    SingleCellEmbeddingSource,
    SingleCellEmbeddingSourceConfig,
    align_singlecell_embeddings,
    load_singlecell_embedding_source,
)
from diffbio.sources.sequence_foundation import (
    SequenceEmbeddingSource,
    SequenceEmbeddingSourceConfig,
    align_sequence_embeddings,
    load_sequence_embedding_source,
)

__all__ = [
    "AnnDataSource",
    "AnnDataSourceConfig",
    "BAMSource",
    "BAMSourceConfig",
    "BioSNAPDTISource",
    "CONTEXTUAL_EPIGENOMICS_DATASET_CONTRACT_KEYS",
    "CONTEXTUAL_TARGET_SEMANTICS",
    "DTI_DATASET_CONTRACT_KEYS",
    "DTISourceConfig",
    "DavisDTISource",
    "EmbeddingArtifactSource",
    "EmbeddingArtifactSourceConfig",
    "ENCODEPeakConfig",
    "ENCODEPeakSource",
    "FastaSource",
    "FastaSourceConfig",
    "IndexedViewSource",
    "IndexedViewSourceConfig",
    "MolNetSource",
    "MolNetSourceConfig",
    "from_anndata",
    "to_anndata",
    # Perturbation
    "BatchControlMapping",
    "ControlMappingConfig",
    "ExperimentConfig",
    "GlobalH5MetadataCache",
    "H5MetadataCache",
    "PerturbationAnnDataSource",
    "PerturbationConcatSource",
    "PerturbationSourceConfig",
    "RandomControlMapping",
    "build_synthetic_contextual_epigenomics_dataset",
    "build_paired_dti_batch",
    "deterministic_dti_split",
    "validate_dti_dataset",
    "validate_contextual_epigenomics_dataset",
    "load_experiment_config",
    "SequenceEmbeddingSource",
    "SequenceEmbeddingSourceConfig",
    "align_sequence_embeddings",
    "load_sequence_embedding_source",
    "SingleCellEmbeddingSource",
    "SingleCellEmbeddingSourceConfig",
    "align_singlecell_embeddings",
    "load_singlecell_embedding_source",
]
