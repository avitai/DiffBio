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
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from flax import nnx

from diffbio.sources._utils import to_dense_float32 as _to_dense

logger = logging.getLogger(__name__)

_FILENAME = "seqfish_cortex.h5ad"


def _require_anndata() -> Any:
    """Import anndata with a clear error message."""
    try:
        import anndata  # noqa: PLC0415

        return anndata
    except ImportError as err:
        raise ImportError(
            "anndata is required for SeqFISHSource. "
            "Install with: uv pip install anndata"
        ) from err


@dataclass(frozen=True)
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
                f"Dataset not found: {path}. "
                f"Download via squidpy: "
                f"squidpy.datasets.seqfish()"
            )


class SeqFISHSource(DataSourceModule):
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

    data: dict[str, Any] = nnx.data()

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
        super().__init__(
            config, rngs=rngs, name=name or "SeqFISHSource"
        )
        self.data = self._load(config)
        logger.info(
            "Loaded seqfish_cortex: %d cells, %d genes, %d types",
            self.data["n_cells"],
            self.data["n_genes"],
            self.data["n_types"],
        )

    def _load(self, config: SeqFISHConfig) -> dict[str, Any]:
        """Load and preprocess the h5ad file."""
        anndata_mod = _require_anndata()
        path = Path(config.data_dir) / _FILENAME
        adata = anndata_mod.read_h5ad(path)

        if (
            config.subsample is not None
            and config.subsample < adata.n_obs
        ):
            rng = np.random.default_rng(42)
            indices = rng.choice(
                adata.n_obs,
                size=config.subsample,
                replace=False,
            )
            indices.sort()
            adata = adata[indices].copy()

        counts = jnp.array(_to_dense(adata.X))

        # Encode cell type labels as integer codes
        label_col = adata.obs[config.label_key]
        if hasattr(label_col, "cat"):
            cell_type_labels = np.asarray(
                label_col.cat.codes, dtype=np.int32
            )
            cell_type_names = list(label_col.cat.categories)
        else:
            unique_labels, cell_type_labels = np.unique(
                np.asarray(label_col), return_inverse=True
            )
            cell_type_labels = cell_type_labels.astype(np.int32)
            cell_type_names = list(unique_labels)

        # Spatial coordinates
        spatial_coords = jnp.array(
            np.asarray(
                adata.obsm[config.spatial_key], dtype=np.float32
            )
        )

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

    def load(self) -> dict[str, Any]:
        """Return the full dataset as a dictionary.

        Returns:
            Dict with keys: counts, cell_type_labels,
            cell_type_names, spatial_coords, gene_names,
            n_cells, n_genes, n_types.
        """
        return self.data

    def __len__(self) -> int:
        """Return the number of cells in the dataset."""
        return self.data["n_cells"]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over individual cells."""
        for i in range(len(self)):
            yield {
                k: (
                    v[i]
                    if hasattr(v, "__getitem__")
                    and k not in ("gene_names", "cell_type_names")
                    else v
                )
                for k, v in self.data.items()
            }
