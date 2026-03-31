"""Immune human integration benchmark DataSource.

Loads the human immune cell atlas dataset from the scib benchmark
(Luecken et al., Nature Methods 2022). This is the standard dataset
for evaluating single-cell batch integration methods.

Dataset: 33,506 cells, 12,303 genes, 10 batches, 16 cell types.
Source: https://figshare.com/articles/dataset/12420968

The dataset must be pre-downloaded to a local directory. See
``benchmarks/README.md`` for download instructions.
"""

from __future__ import annotations

from collections.abc import Iterator

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

_FILENAME = "Immune_ALL_human.h5ad"


def _require_anndata() -> Any:
    """Import anndata with a clear error message."""
    try:
        import anndata  # noqa: PLC0415

        return anndata
    except ImportError as err:
        raise ImportError(
            "anndata is required for ImmuneHumanSource. "
            "Install with: uv pip install anndata"
        ) from err


from diffbio.sources._utils import to_dense_float32 as _to_dense


@dataclass(frozen=True, kw_only=True)
class ImmuneHumanConfig(StructuralConfig):
    """Configuration for ImmuneHumanSource.

    Attributes:
        data_dir: Directory containing the downloaded h5ad file.
        subsample: If set, randomly subsample this many cells.
            Use for quick/CI benchmark runs.
        batch_key: Column name in obs for batch labels.
        label_key: Column name in obs for cell type labels.
        embedding_key: Key in obsm for precomputed embeddings.
    """

    data_dir: str = "/media/mahdi/ssd23/Data/scib"
    subsample: int | None = None
    batch_key: str = "batch"
    label_key: str = "final_annotation"
    embedding_key: str = "X_pca"

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        path = Path(self.data_dir) / _FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}. "
                f"Download from: https://ndownloader.figshare.com/"
                f"files/25717328"
            )


class ImmuneHumanSource(DataSourceModule):
    """DataSource for the scib immune human integration benchmark.

    Loads the human immune cell atlas (33,506 cells, 12,303 genes,
    10 batches, 16 cell types) from a pre-downloaded h5ad file.

    Follows the datarax DataSourceModule pattern: eager loading at
    init, dict-based access via ``load()``, length via ``__len__``.

    Example:
        ```python
        config = ImmuneHumanConfig(data_dir="/path/to/data")
        source = ImmuneHumanSource(config)
        data = source.load()
        print(data["counts"].shape)  # (33506, 12303)
        ```
    """

    data: dict[str, Any] = nnx.data()

    def __init__(
        self,
        config: ImmuneHumanConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Load the immune human dataset.

        Args:
            config: Configuration with data directory and options.
            rngs: Optional RNG state (unused, for interface compat).
            name: Optional module name.
        """
        super().__init__(config, rngs=rngs, name=name or "ImmuneHumanSource")
        self.data = self._load(config)
        logger.info(
            "Loaded immune_human: %d cells, %d genes, %d batches, %d types",
            self.data["n_cells"],
            self.data["n_genes"],
            self.data["n_batches"],
            self.data["n_types"],
        )

    def _load(self, config: ImmuneHumanConfig) -> dict[str, Any]:
        """Load and preprocess the h5ad file."""
        anndata_mod = _require_anndata()
        path = Path(config.data_dir) / _FILENAME
        adata = anndata_mod.read_h5ad(path)

        if config.subsample is not None and config.subsample < adata.n_obs:
            rng = np.random.default_rng(42)
            indices = rng.choice(
                adata.n_obs, size=config.subsample, replace=False
            )
            indices.sort()
            adata = adata[indices].copy()

        counts = jnp.array(_to_dense(adata.X))

        # Encode categorical labels as integer codes
        batch_col = adata.obs[config.batch_key]
        if hasattr(batch_col, "cat"):
            batch_labels = np.asarray(batch_col.cat.codes, dtype=np.int32)
        else:
            _, batch_labels = np.unique(
                np.asarray(batch_col), return_inverse=True
            )
            batch_labels = batch_labels.astype(np.int32)

        label_col = adata.obs[config.label_key]
        if hasattr(label_col, "cat"):
            cell_type_labels = np.asarray(
                label_col.cat.codes, dtype=np.int32
            )
        else:
            _, cell_type_labels = np.unique(
                np.asarray(label_col), return_inverse=True
            )
            cell_type_labels = cell_type_labels.astype(np.int32)

        # Get embeddings (PCA from obsm, or compute on the fly)
        if config.embedding_key in adata.obsm:
            embeddings = jnp.array(
                np.asarray(
                    adata.obsm[config.embedding_key], dtype=np.float32
                )
            )
        else:
            logger.info(
                "Embedding key '%s' not in obsm (%s). "
                "Computing PCA (50 components).",
                config.embedding_key,
                list(adata.obsm.keys()),
            )
            embeddings = self._compute_pca(counts, n_components=50)

        gene_names = list(adata.var_names)

        return {
            "counts": counts,
            "batch_labels": batch_labels,
            "cell_type_labels": cell_type_labels,
            "embeddings": embeddings,
            "gene_names": gene_names,
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
            "n_batches": int(len(np.unique(batch_labels))),
            "n_types": int(len(np.unique(cell_type_labels))),
        }

    @staticmethod
    def _compute_pca(
        counts: jnp.ndarray, n_components: int = 50
    ) -> jnp.ndarray:
        """Compute PCA embeddings from count matrix.

        Log-normalizes, then computes truncated SVD for PCA.

        Args:
            counts: Dense count matrix (n_cells, n_genes).
            n_components: Number of principal components.

        Returns:
            PCA embeddings of shape (n_cells, n_components).
        """
        # Log-normalize: log1p(counts / total * 10000)
        totals = jnp.sum(counts, axis=1, keepdims=True)
        totals = jnp.maximum(totals, 1.0)
        normalized = jnp.log1p(counts / totals * 10000.0)

        # Center
        mean = jnp.mean(normalized, axis=0)
        centered = normalized - mean

        # Truncated SVD via numpy (JAX full SVD on 33K x 12K is expensive)
        centered_np = np.asarray(centered)
        from sklearn.decomposition import TruncatedSVD  # noqa: PLC0415

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        embeddings = svd.fit_transform(centered_np)
        return jnp.array(embeddings.astype(np.float32))

    def load(self) -> dict[str, Any]:
        """Return the full dataset as a dictionary.

        Returns:
            Dict with keys: counts, batch_labels, cell_type_labels,
            embeddings, gene_names, n_cells, n_genes, n_batches, n_types.
        """
        return self.data

    def __len__(self) -> int:
        """Return the number of cells in the dataset."""
        return self.data["n_cells"]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over individual cells."""
        for i in range(len(self)):
            yield {
                k: v[i] if hasattr(v, "__getitem__") and k != "gene_names"
                else v
                for k, v in self.data.items()
            }
