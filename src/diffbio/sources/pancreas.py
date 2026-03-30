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
from datarax.core.data_source import DataSourceModule

logger = logging.getLogger(__name__)

_FILENAME = "endocrinogenesis_day15.h5ad"


@dataclass(frozen=True)
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


class PancreasSource(DataSourceModule):
    """DataSource for the scVelo pancreas endocrinogenesis dataset.

    Provides counts, spliced/unspliced layers, PCA embeddings,
    and cell type labels for trajectory and velocity benchmarks.
    """

    data: dict[str, Any] = nnx.data()

    def __init__(
        self,
        config: PancreasConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Load the pancreas dataset."""
        super().__init__(
            config, rngs=rngs, name=name or "PancreasSource"
        )
        self.data = self._load(config)
        logger.info(
            "Loaded pancreas: %d cells, %d genes, %d types",
            self.data["n_cells"],
            self.data["n_genes"],
            self.data["n_types"],
        )

    def _load(self, config: PancreasConfig) -> dict[str, Any]:
        """Load and preprocess the h5ad file."""
        import anndata  # noqa: PLC0415
        import scipy.sparse  # noqa: PLC0415

        path = Path(config.data_dir) / _FILENAME
        adata = anndata.read_h5ad(path)

        if config.subsample is not None and config.subsample < adata.n_obs:
            rng = np.random.default_rng(42)
            idx = rng.choice(
                adata.n_obs, size=config.subsample, replace=False
            )
            idx.sort()
            adata = adata[idx].copy()

        def _to_dense(x: Any) -> np.ndarray:
            if scipy.sparse.issparse(x):
                return np.asarray(x.toarray(), dtype=np.float32)
            return np.asarray(x, dtype=np.float32)

        counts = jnp.array(_to_dense(adata.X))

        # Spliced/unspliced for velocity
        spliced = jnp.array(_to_dense(adata.layers["spliced"]))
        unspliced = jnp.array(_to_dense(adata.layers["unspliced"]))

        # Cell type labels
        label_col = adata.obs[config.cluster_key]
        if hasattr(label_col, "cat"):
            labels = np.asarray(label_col.cat.codes, dtype=np.int32)
        else:
            _, labels = np.unique(
                np.asarray(label_col), return_inverse=True
            )
            labels = labels.astype(np.int32)

        # Embeddings
        embeddings = jnp.array(
            np.asarray(
                adata.obsm.get(
                    "X_pca", np.zeros((adata.n_obs, 50))
                ),
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

    def load(self) -> dict[str, Any]:
        """Return the full dataset."""
        return self.data

    def __len__(self) -> int:
        """Return the number of cells."""
        return self.data["n_cells"]

    def __iter__(self):
        """Iterate over cells."""
        for i in range(len(self)):
            yield {
                k: v[i] if hasattr(v, "__getitem__")
                and k != "gene_names"
                else v
                for k, v in self.data.items()
            }
