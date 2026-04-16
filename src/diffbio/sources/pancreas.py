"""Pancreas endocrinogenesis DataSource for trajectory benchmarks.

Loads the scVelo pancreas dataset (Bastidas-Ponce et al. 2019,
Bergen et al. 2020) with spliced/unspliced layers for RNA velocity
and precomputed PCA embeddings for pseudotime inference.

Dataset: 3,696 cells, 27,998 genes, 8 cell types, 5 coarse types.
Source: scVelo tutorial dataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.core.config import StructuralConfig

from diffbio.sources._benchmark_source import (
    BenchmarkDataSource,
    encode_label_column,
)
from diffbio.sources._utils import to_dense_float32 as _to_dense

logger = logging.getLogger(__name__)

_FILENAME = "endocrinogenesis_day15.h5ad"


@dataclass(frozen=True, kw_only=True)
class PancreasConfig(StructuralConfig):
    """Configuration for PancreasSource.

    Attributes:
        data_dir: Directory containing the downloaded h5ad file.
        subsample: If set, randomly subsample this many cells.
        cluster_key: Column in obs for cell type labels.
    """

    data_dir: str = "/media/mahdi/ssd23/Data/scvelo"
    subsample: int | None = None
    cluster_key: str = "clusters"

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        path = Path(self.data_dir) / _FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}. Download the scVelo "
                f"pancreas dataset to {self.data_dir}/"
            )


class PancreasSource(BenchmarkDataSource):
    """DataSource for the scVelo pancreas endocrinogenesis dataset.

    Provides counts, spliced/unspliced layers, PCA embeddings,
    and cell type labels for trajectory and velocity benchmarks.
    """

    def __init__(
        self,
        config: PancreasConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Load the pancreas dataset."""
        super().__init__(config, rngs=rngs, name=name or "PancreasSource")
        self.data = self._load(config)
        self._log_loaded_summary(logger, "pancreas", ("n_cells", "n_genes", "n_types"))

    def _load(self, config: PancreasConfig) -> dict[str, Any]:
        """Load and preprocess the h5ad file."""
        adata, counts = self._load_benchmark_counts(config, _FILENAME, _to_dense)

        # Spliced/unspliced for velocity
        spliced = jnp.array(_to_dense(adata.layers["spliced"]))
        unspliced = jnp.array(_to_dense(adata.layers["unspliced"]))

        # Cell type labels
        labels = encode_label_column(adata.obs[config.cluster_key])

        # Embeddings
        embeddings = jnp.array(
            np.asarray(
                adata.obsm.get("X_pca", np.zeros((adata.n_obs, 50))),
                dtype=np.float32,
            )
        )

        return {
            "counts": counts,
            "spliced": spliced,
            "unspliced": unspliced,
            "cell_type_labels": labels,
            "embeddings": embeddings,
            "gene_names": list(adata.var_names),
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
            "n_types": int(len(np.unique(labels))),
        }
