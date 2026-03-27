"""Perturbation-aware data loading for single-cell experiments.

This sub-package provides data sources, configuration, and utilities for
loading single-cell perturbation experiment data (e.g., CRISPR screens),
porting and adapting features from the cell-load library to the
JAX/datarax ecosystem.

Sources:
    PerturbationAnnDataSource: Single-file perturbation-aware AnnData source
    PerturbationConcatSource: Multi-dataset concatenation source

Configuration:
    ExperimentConfig: TOML-based experiment configuration
    PerturbationSourceConfig: Source configuration

Control Mapping:
    RandomControlMapping: Random control cell mapping within cell type
    BatchControlMapping: Batch-aware control cell mapping

Utilities:
    H5MetadataCache: Singleton cache for H5 metadata
    GlobalH5MetadataCache: Process-global cache manager
"""

from diffbio.sources.perturbation._types import MappingStrategy, OutputSpaceMode
from diffbio.sources.perturbation.concat_source import PerturbationConcatSource
from diffbio.sources.perturbation.control_mapping import (
    BatchControlMapping,
    ControlMappingConfig,
    RandomControlMapping,
)
from diffbio.sources.perturbation.experiment_config import (
    DatasetEntry,
    ExperimentConfig,
    FewshotEntry,
    ZeroshotEntry,
    load_experiment_config,
)
from diffbio.sources.perturbation.h5_metadata_cache import (
    GlobalH5MetadataCache,
    H5MetadataCache,
)
from diffbio.sources.perturbation.perturbation_source import (
    PerturbationAnnDataSource,
    PerturbationSourceConfig,
)

__all__ = [
    # Types
    "MappingStrategy",
    "OutputSpaceMode",
    # Sources
    "PerturbationAnnDataSource",
    "PerturbationSourceConfig",
    "PerturbationConcatSource",
    # Configuration
    "ExperimentConfig",
    "DatasetEntry",
    "ZeroshotEntry",
    "FewshotEntry",
    "load_experiment_config",
    # Control Mapping
    "ControlMappingConfig",
    "RandomControlMapping",
    "BatchControlMapping",
    # Cache
    "H5MetadataCache",
    "GlobalH5MetadataCache",
]
