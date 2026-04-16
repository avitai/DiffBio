"""SeqFISH cortex spatial transcriptomics DataSource.

Loads the seqFISH mouse cortex dataset (Lohoff et al., Nature
Biotechnology 2022) from a pre-downloaded h5ad file. This dataset
provides spatially resolved gene expression for 19,416 cells with
351 genes and 22 cell types.

The dataset must be pre-downloaded (e.g. via squidpy) to a local
directory. See ``benchmarks/README.md`` for download instructions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from datarax.core.config import StructuralConfig
from flax import nnx

from diffbio.sources._benchmark_source import (
    BenchmarkDataSource,
    encode_label_column,
)
from diffbio.sources._utils import to_dense_float32 as _to_dense

logger = logging.getLogger(__name__)

_FILENAME = "seqfish_cortex.h5ad"


@dataclass(frozen=True, kw_only=True)
class SeqFISHConfig(StructuralConfig):
    """Configuration for SeqFISHSource.

    Attributes:
        data_dir: Directory containing the seqfish_cortex.h5ad file.
        subsample: If set, randomly subsample this many cells.
            Use for quick/CI benchmark runs.
        label_key: Column name in obs for cell type labels.
        spatial_key: Key in obsm for spatial coordinates.
    """

    data_dir: str = "/media/mahdi/ssd23/Data/spatial"
    subsample: int | None = None
    label_key: str = "celltype_mapped_refined"
    spatial_key: str = "spatial"

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        path = Path(self.data_dir) / _FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}. Download via squidpy: squidpy.datasets.seqfish()"
            )


class SeqFISHSource(BenchmarkDataSource):
    """DataSource for the seqFISH mouse cortex dataset.

    Loads spatially resolved gene expression (19,416 cells, 351
    genes, 22 cell types) from a pre-downloaded h5ad file.

    Follows the datarax DataSourceModule pattern: eager loading
    at init, dict-based access via ``load()``, length via
    ``__len__``.

    Example:
        ```python
        config = SeqFISHConfig(data_dir="/path/to/data")
        source = SeqFISHSource(config)
        data = source.load()
        print(data["counts"].shape)  # (19416, 351)
        ```
    """

    iter_static_keys = ("gene_names", "cell_type_names")

    def __init__(
        self,
        config: SeqFISHConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Load the seqFISH cortex dataset.

        Args:
            config: Configuration with data directory and options.
            rngs: Optional RNG state (unused, for interface compat).
            name: Optional module name.
        """
        super().__init__(config, rngs=rngs, name=name or "SeqFISHSource")
        self.data = self._load(config)
        self._log_loaded_summary(logger, "seqfish_cortex", ("n_cells", "n_genes", "n_types"))

    def _load(self, config: SeqFISHConfig) -> dict[str, Any]:
        """Load and preprocess the h5ad file."""
        adata, counts = self._load_benchmark_counts(config, _FILENAME, _to_dense)

        # Encode cell type labels as integer codes
        cell_type_labels, cell_type_names = encode_label_column(
            adata.obs[config.label_key],
            include_names=True,
        )

        # Spatial coordinates
        spatial_coords = jnp.array(np.asarray(adata.obsm[config.spatial_key], dtype=np.float32))

        gene_names = list(adata.var_names)

        return {
            "counts": counts,
            "cell_type_labels": cell_type_labels,
            "cell_type_names": cell_type_names,
            "spatial_coords": spatial_coords,
            "gene_names": gene_names,
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
            "n_types": int(len(np.unique(cell_type_labels))),
        }
