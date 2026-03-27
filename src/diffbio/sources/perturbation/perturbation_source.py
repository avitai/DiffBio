"""Perturbation-aware AnnData source for single-cell experiments.

Extends AnnDataSource with perturbation metadata extraction, integer encoding,
control cell identification, output space selection, and one-hot perturbation
maps. Follows the eager-loading pattern from datarax.

References:
    - cell-load/src/cell_load/dataset/_perturbation.py (PerturbationDataset)
    - diffbio/sources/anndata_source.py (AnnDataSource)
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.sources._eager_source_ops import eager_get_batch, eager_iter

from diffbio.sources.anndata_source import (
    AnnDataSource,
    AnnDataSourceConfig,
    _load_obsm,
    _require_anndata,
    _to_dense_array,
)
from diffbio.sources.perturbation._types import OutputSpaceMode
from diffbio.sources.perturbation.output_space import select_output_counts

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerturbationSourceConfig(AnnDataSourceConfig):
    """Configuration for PerturbationAnnDataSource.

    Extends AnnDataSourceConfig with perturbation-specific settings.

    Attributes:
        pert_col: Obs column name for perturbation identity.
        cell_type_col: Obs column name for cell type.
        batch_col: Obs column name for batch/plate.
        control_pert: Label identifying control cells.
        output_space: Output representation mode (``"gene"``, ``"all"``,
            or ``"embedding"``).
        embedding_key: Key in obsm for pre-computed embeddings.
        hvg_col: Column in var marking highly variable genes.
        include_barcodes: Whether to include cell barcodes in output.
        additional_obs: Extra obs columns to pass through in elements.
        should_yield_controls: Whether to include control cells in iteration.
        perturbation_features_file: Path to external perturbation embeddings
            (``.npy`` or ``.npz``). If provided, used instead of one-hot.
    """

    pert_col: str = "perturbation"
    cell_type_col: str = "cell_type"
    batch_col: str = "batch"
    control_pert: str = "non-targeting"
    output_space: str = "gene"
    embedding_key: str | None = None
    hvg_col: str | None = None
    include_barcodes: bool = False
    additional_obs: tuple[str, ...] = ()
    should_yield_controls: bool = True
    perturbation_features_file: str | None = None


class PerturbationAnnDataSource(AnnDataSource):
    """Perturbation-aware eager-loading AnnData source.

    Extends :class:`AnnDataSource` with:

    - Perturbation / cell type / batch metadata extraction and integer encoding
    - Control cell identification via a boolean mask
    - HVG subsetting and output space selection
    - One-hot perturbation maps (or external embeddings from file)
    - Cell barcode tracking
    - Additional obs column passthrough

    Output dictionary keys (in addition to AnnDataSource keys):

    - ``pert_code``: Integer-encoded perturbation.
    - ``cell_type_code``: Integer-encoded cell type.
    - ``batch_code``: Integer-encoded batch.
    - ``is_control``: Boolean flag.
    - ``pert_emb``: One-hot or external perturbation embedding.
    - ``cell_type_onehot``: Cell type one-hot vector.
    - ``batch_onehot``: Batch one-hot vector.
    - ``pert_name``: Perturbation label string.
    - ``cell_type_name``: Cell type label string.
    - ``batch_name``: Batch label string.
    - ``barcode``: Cell barcode (if ``include_barcodes=True``).
    """

    def __init__(
        self,
        config: PerturbationSourceConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize PerturbationAnnDataSource.

        Loads data eagerly and extracts perturbation metadata.

        Args:
            config: Source configuration.
            rngs: Optional RNG state for shuffling.
            name: Optional module name.
        """
        # Skip AnnDataSource.__init__ — we need to customize loading
        # Call DataSourceModule.__init__ directly
        if name is None:
            name = f"PerturbationAnnDataSource({config.file_path})"

        from datarax.core.data_source import DataSourceModule  # noqa: PLC0415

        DataSourceModule.__init__(self, config, rngs=rngs, name=name)

        anndata_mod = _require_anndata()

        file_path = Path(str(config.file_path))
        if not file_path.exists():
            raise FileNotFoundError(f"AnnData file not found: {file_path}")

        adata = anndata_mod.read_h5ad(
            file_path, backed="r" if config.backed else None
        )

        # -- Count matrix --
        full_counts = jnp.array(_to_dense_array(adata.X))

        # -- HVG indices --
        hvg_indices: np.ndarray | None = None
        if config.hvg_col is not None and config.hvg_col in adata.var.columns:
            hvg_indices = np.where(np.asarray(adata.var[config.hvg_col]))[0]

        # -- Apply output space selection --
        counts = select_output_counts(
            full_counts, hvg_indices, OutputSpaceMode(config.output_space)
        )

        # -- Obs metadata --
        obs: dict[str, Any] = {
            col: np.asarray(adata.obs[col]) for col in adata.obs.columns
        }

        # -- Var metadata --
        var: dict[str, Any] = {
            col: np.asarray(adata.var[col]) for col in adata.var.columns
        }

        # -- Obsm embeddings --
        obsm = _load_obsm(adata)

        # -- Perturbation metadata --
        pert_labels = np.asarray(adata.obs[config.pert_col])
        cell_type_labels = np.asarray(adata.obs[config.cell_type_col])
        batch_labels = np.asarray(adata.obs[config.batch_col])

        # Build category -> code mappings
        pert_cats = np.array(sorted(set(pert_labels)))
        ct_cats = np.array(sorted(set(cell_type_labels)))
        batch_cats = np.array(sorted(set(batch_labels)))

        pert_to_code = {p: i for i, p in enumerate(pert_cats)}
        ct_to_code = {c: i for i, c in enumerate(ct_cats)}
        batch_to_code = {b: i for i, b in enumerate(batch_cats)}

        pert_codes = np.array(
            [pert_to_code[p] for p in pert_labels], dtype=np.int32
        )
        ct_codes = np.array(
            [ct_to_code[c] for c in cell_type_labels], dtype=np.int32
        )
        batch_codes = np.array(
            [batch_to_code[b] for b in batch_labels], dtype=np.int32
        )

        # Control mask
        control_mask = pert_labels == config.control_pert

        # Group codes: ravel_multi_index for fast (celltype, pert) grouping
        group_codes = np.ravel_multi_index(
            (ct_codes, pert_codes), (len(ct_cats), len(pert_cats))
        ).astype(np.int32)

        # One-hot matrices (stacked identity, indexed by code; plain numpy)
        pert_onehot_matrix = np.eye(len(pert_cats), dtype=np.float32)
        ct_onehot_matrix = np.eye(len(ct_cats), dtype=np.float32)
        batch_onehot_matrix = np.eye(len(batch_cats), dtype=np.float32)

        # External perturbation embeddings (overrides one-hot)
        pert_embeddings_matrix: np.ndarray | None = None
        if config.perturbation_features_file is not None:
            from diffbio.sources.perturbation.output_space import (  # noqa: PLC0415
                load_external_embeddings,
            )

            ext_emb = load_external_embeddings(
                Path(config.perturbation_features_file)
            )
            pert_embeddings_matrix = np.asarray(ext_emb)

        # Barcodes
        barcodes: np.ndarray | None = None
        if config.include_barcodes and "barcode" in adata.obs.columns:
            barcodes = np.asarray(adata.obs["barcode"])

        # -- Visible index mapping for should_yield_controls --
        if config.should_yield_controls:
            visible_indices = np.arange(adata.n_obs, dtype=np.int64)
        else:
            visible_indices = np.where(~control_mask)[0].astype(np.int64)

        # -- Store all data --
        self.data = {
            "counts": counts,
            "obs": obs,
            "var": var,
            "obsm": obsm,
        }
        self.length: int = len(visible_indices)
        self._full_length: int = adata.n_obs
        self._visible_indices = visible_indices
        self.index = nnx.Variable(0)
        self.epoch = nnx.Variable(0)
        self._seed: int = config.seed
        self.shuffle: bool = config.shuffle
        self.dataset_name: str | None = str(config.file_path)
        self.split_name: str | None = config.split
        self._dataset_info: dict[str, int] = {
            "n_genes": adata.n_vars,
            "n_cells": adata.n_obs,
        }

        # Perturbation-specific attributes (all plain Python / numpy scalars)
        # Stored as plain types that NNX pytree walker won't inspect
        self._pert_codes = pert_codes
        self._ct_codes = ct_codes
        self._batch_codes = batch_codes
        self._pert_cats = tuple(pert_cats)
        self._ct_cats = tuple(ct_cats)
        self._batch_cats = tuple(batch_cats)
        self._control_mask = control_mask
        self._group_codes = group_codes
        self._pert_onehot_matrix = pert_onehot_matrix
        self._ct_onehot_matrix = ct_onehot_matrix
        self._batch_onehot_matrix = batch_onehot_matrix
        self._pert_embeddings_matrix = pert_embeddings_matrix
        self._barcodes = barcodes
        self._pert_labels = pert_labels
        self._ct_labels = cell_type_labels
        self._batch_labels = batch_labels
        self._hvg_indices = hvg_indices
        self._gene_names = tuple(adata.var_names)
        self._n_pert_cats = len(pert_cats)
        self._pert_to_code = {p: i for i, p in enumerate(pert_cats)}

    # =================================================================
    # DataSourceModule protocol overrides
    # =================================================================

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get data for a single cell by index, including perturbation metadata.

        When ``should_yield_controls=False``, indices are remapped to skip
        control cells transparently.

        Args:
            idx: Cell index (supports negative indexing).

        Returns:
            Dict with counts, perturbation metadata, and embeddings.
        """
        if idx < 0:
            idx = self.length + idx
        if idx < 0 or idx >= self.length:
            raise IndexError(
                f"Cell index {idx} out of range for dataset "
                f"with {self.length} cells"
            )
        internal_idx = int(self._visible_indices[idx])
        return self._build_pert_element(internal_idx)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over visible cells with optional shuffling."""
        return eager_iter(
            self.data,
            self.length,
            self.index,
            self.epoch,
            self.shuffle,
            self._seed,
            self._build_visible_element,
        )

    def get_batch(
        self, batch_size: int, key: jax.Array | None = None
    ) -> dict[str, Any]:
        """Get a batch of cells with perturbation metadata.

        Args:
            batch_size: Number of cells.
            key: Optional RNG key for stateless random sampling.

        Returns:
            Batched dictionary.
        """

        def _gather(data: dict[str, Any], indices: jax.Array) -> dict[str, Any]:
            np_indices = np.array(indices)
            return self._build_batch_element(np_indices)

        return eager_get_batch(
            self.data,
            self.length,
            self.index,
            self.epoch,
            self.shuffle,
            self._seed,
            batch_size,
            key,
            _gather,
        )

    # =================================================================
    # Perturbation-specific public API
    # =================================================================

    def get_control_mask(self) -> np.ndarray:
        """Return boolean mask where True indicates a control cell."""
        return self._control_mask

    def get_pert_codes(self) -> np.ndarray:
        """Return per-cell integer perturbation codes."""
        return self._pert_codes

    def get_cell_type_codes(self) -> np.ndarray:
        """Return per-cell integer cell type codes."""
        return self._ct_codes

    def get_batch_codes(self) -> np.ndarray:
        """Return per-cell integer batch codes."""
        return self._batch_codes

    def get_pert_categories(self) -> np.ndarray:
        """Return sorted array of unique perturbation labels."""
        return np.array(self._pert_cats)

    def get_cell_type_categories(self) -> np.ndarray:
        """Return sorted array of unique cell type labels."""
        return np.array(self._ct_cats)

    def get_batch_categories(self) -> np.ndarray:
        """Return sorted array of unique batch labels."""
        return np.array(self._batch_cats)

    def get_group_codes(self) -> np.ndarray:
        """Return per-cell group codes for (cell_type, perturbation) grouping."""
        return self._group_codes

    def get_onehot_map(self) -> dict[str, jnp.ndarray]:
        """Return perturbation one-hot encoding map (JAX arrays)."""
        return {
            p: jnp.array(self._pert_onehot_matrix[i])
            for i, p in enumerate(self._pert_cats)
        }

    def get_gene_names(self, output_space: str = "all") -> list[str]:
        """Return gene names, optionally filtered by output space.

        Args:
            output_space: ``"all"`` for all genes, ``"gene"`` for HVG subset.

        Returns:
            List of gene name strings.
        """
        if output_space == "gene" and self._hvg_indices is not None:
            return [self._gene_names[i] for i in self._hvg_indices]
        return list(self._gene_names)

    def get_n_genes(self) -> int:
        """Return total number of genes."""
        return len(self._gene_names)

    def get_var_dims(self) -> dict[str, int]:
        """Return dimensionality info for the dataset.

        Returns:
            Dict with ``n_genes``, ``n_cells``, ``n_perts``,
            ``n_cell_types``, ``n_batches``.
        """
        return {
            "n_genes": len(self._gene_names),
            "n_cells": self.length,
            "n_perts": len(self._pert_cats),
            "n_cell_types": len(self._ct_cats),
            "n_batches": len(self._batch_cats),
        }

    # =================================================================
    # Internal helpers
    # =================================================================

    def _build_pert_element(self, idx: int) -> dict[str, Any]:
        """Build a per-cell dictionary with perturbation metadata."""
        cell_counts = self.data["counts"][idx]
        cell_obs = {col: arr[idx] for col, arr in self.data["obs"].items()}
        cell_obsm: dict[str, jnp.ndarray] = {}
        for emb_name, emb_arr in self.data["obsm"].items():
            cell_obsm[emb_name] = emb_arr[idx]

        # Perturbation embedding (indexed by code, converted to JAX)
        pc = int(self._pert_codes[idx])
        cc = int(self._ct_codes[idx])
        bc = int(self._batch_codes[idx])

        if self._pert_embeddings_matrix is not None:
            pert_emb = jnp.array(self._pert_embeddings_matrix[pc])
        else:
            pert_emb = jnp.array(self._pert_onehot_matrix[pc])

        pert_name = str(self._pert_labels[idx])
        ct_name = str(self._ct_labels[idx])
        batch_name = str(self._batch_labels[idx])

        element: dict[str, Any] = {
            "counts": cell_counts,
            "obs": cell_obs,
            "obsm": cell_obsm,
            "pert_code": pc,
            "cell_type_code": cc,
            "batch_code": bc,
            "is_control": bool(self._control_mask[idx]),
            "pert_emb": pert_emb,
            "cell_type_onehot": jnp.array(self._ct_onehot_matrix[cc]),
            "batch_onehot": jnp.array(self._batch_onehot_matrix[bc]),
            "pert_name": pert_name,
            "cell_type_name": ct_name,
            "batch_name": batch_name,
        }

        if self._barcodes is not None:
            element["barcode"] = str(self._barcodes[idx])

        return element

    def _build_pert_element_from_data(
        self, data: dict[str, Any], idx: int
    ) -> dict[str, Any]:
        """Build element from data dict (used by eager_iter callback)."""
        return self._build_pert_element(idx)

    def _build_visible_element(
        self, data: dict[str, Any], idx: int
    ) -> dict[str, Any]:
        """Build element with visible index remapping (used by __iter__)."""
        internal_idx = int(self._visible_indices[idx])
        return self._build_pert_element(internal_idx)

    def _build_batch_element(self, indices: np.ndarray) -> dict[str, Any]:
        """Build a batched dictionary for multiple cells."""
        counts = self.data["counts"][indices]
        obs = {
            col: np.asarray(arr)[indices]
            for col, arr in self.data["obs"].items()
        }
        obsm: dict[str, jnp.ndarray] = {}
        for emb_name, emb_arr in self.data["obsm"].items():
            obsm[emb_name] = emb_arr[indices]

        pert_codes = self._pert_codes[indices]
        ct_codes = self._ct_codes[indices]
        batch_codes = self._batch_codes[indices]

        # Build batched embeddings (indexed by codes, converted to JAX)
        if self._pert_embeddings_matrix is not None:
            pert_embs = jnp.array(self._pert_embeddings_matrix[pert_codes])
        else:
            pert_embs = jnp.array(self._pert_onehot_matrix[pert_codes])

        ct_embs = jnp.array(self._ct_onehot_matrix[ct_codes])
        batch_embs = jnp.array(self._batch_onehot_matrix[batch_codes])

        return {
            "counts": counts,
            "obs": obs,
            "obsm": obsm,
            "pert_code": jnp.array(pert_codes),
            "cell_type_code": jnp.array(ct_codes),
            "batch_code": jnp.array(batch_codes),
            "is_control": jnp.array(self._control_mask[indices]),
            "pert_emb": pert_embs,
            "cell_type_onehot": ct_embs,
            "batch_onehot": batch_embs,
            "pert_name": [str(self._pert_labels[i]) for i in indices],
            "cell_type_name": [str(self._ct_labels[i]) for i in indices],
            "batch_name": [str(self._batch_labels[i]) for i in indices],
        }
