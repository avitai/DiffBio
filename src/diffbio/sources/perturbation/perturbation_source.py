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

from diffbio.sources._anndata_shared import (
    build_anndata_data,
    extract_anndata_annotations,
    read_h5ad,
    to_dense_array,
)
from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig
from diffbio.sources.perturbation._types import OutputSpaceMode
from diffbio.sources.perturbation.output_space import select_output_counts

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _EncodedObsColumn:
    """Categorical obs column with codes, labels, and one-hot lookup."""

    codes: np.ndarray
    categories: np.ndarray
    labels: np.ndarray
    onehot_matrix: np.ndarray


@dataclass(frozen=True, slots=True)
class _CategoricalMetadataState:
    """Categorical perturbation metadata and lookup tables."""

    perturbation: _EncodedObsColumn
    cell_type: _EncodedObsColumn
    batch: _EncodedObsColumn
    control_mask: np.ndarray
    group_codes: np.ndarray


@dataclass(frozen=True, slots=True)
class _FeatureViewState:
    """Feature-view metadata used to build source elements."""

    perturbation_embeddings_matrix: np.ndarray | None
    barcodes: np.ndarray | None
    additional_obs: tuple[str, ...]
    hvg_indices: np.ndarray | None
    gene_names: tuple[str, ...]


def _encode_obs_column(labels: np.ndarray) -> _EncodedObsColumn:
    """Encode one categorical obs column into reusable lookup tables."""
    categories = np.array(sorted(set(labels)))
    label_to_code = {label: idx for idx, label in enumerate(categories)}
    codes = np.array([label_to_code[label] for label in labels], dtype=np.int32)
    onehot_matrix = np.eye(len(categories), dtype=np.float32)
    return _EncodedObsColumn(
        codes=codes,
        categories=categories,
        labels=labels,
        onehot_matrix=onehot_matrix,
    )


@dataclass(frozen=True)
class _PerturbationMetadataConfig:
    """Perturbation metadata column and passthrough configuration."""

    pert_col: str = "perturbation"
    cell_type_col: str = "cell_type"
    batch_col: str = "batch"
    control_pert: str = "non-targeting"
    include_barcodes: bool = False
    additional_obs: tuple[str, ...] = ()


@dataclass(frozen=True)
class _PerturbationViewConfig:
    """Perturbation output-view and feature configuration."""

    output_space: str = "gene"
    embedding_key: str | None = None
    hvg_col: str | None = None
    should_yield_controls: bool = True
    perturbation_features_file: str | None = None


@dataclass(frozen=True)
class PerturbationSourceConfig(
    _PerturbationMetadataConfig,
    _PerturbationViewConfig,
    AnnDataSourceConfig,
):
    """Configuration for PerturbationAnnDataSource."""

    def __post_init__(self) -> None:
        """Validate perturbation-specific configuration."""
        super().__post_init__()

        OutputSpaceMode(self.output_space)
        if self.output_space == OutputSpaceMode.EMBEDDING.value and self.embedding_key is None:
            raise ValueError("embedding_key is required when output_space='embedding'")

        if len(set(self.additional_obs)) != len(self.additional_obs):
            raise ValueError("additional_obs must not contain duplicates")


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

        adata = read_h5ad(config)

        # -- Count matrix --
        full_counts = jnp.array(to_dense_array(adata.X))

        # -- HVG indices --
        hvg_indices: np.ndarray | None = None
        if config.hvg_col is not None and config.hvg_col in adata.var.columns:
            hvg_indices = np.where(np.asarray(adata.var[config.hvg_col]))[0]

        # -- Apply output space selection --
        counts = select_output_counts(
            full_counts, hvg_indices, OutputSpaceMode(config.output_space)
        )

        obs, var, obsm = extract_anndata_annotations(adata)

        required_obs_cols = (
            config.pert_col,
            config.cell_type_col,
            config.batch_col,
            *config.additional_obs,
        )
        missing_obs_cols = tuple(column for column in required_obs_cols if column not in obs)
        if missing_obs_cols:
            missing = ", ".join(missing_obs_cols)
            raise ValueError(f"additional_obs/required obs columns missing from AnnData: {missing}")

        if config.embedding_key is not None:
            if config.embedding_key not in obsm:
                raise ValueError(
                    f"embedding_key '{config.embedding_key}' not found in AnnData.obsm"
                )
            obsm = {config.embedding_key: obsm[config.embedding_key]}

        # -- Perturbation metadata --
        perturbation = _encode_obs_column(np.asarray(obs[config.pert_col]))
        cell_type = _encode_obs_column(np.asarray(obs[config.cell_type_col]))
        batch = _encode_obs_column(np.asarray(obs[config.batch_col]))

        # Control mask
        control_mask = perturbation.labels == config.control_pert

        # Group codes: ravel_multi_index for fast (celltype, pert) grouping
        group_codes = np.ravel_multi_index(
            (cell_type.codes, perturbation.codes),
            (len(cell_type.categories), len(perturbation.categories)),
        ).astype(np.int32)

        # External perturbation embeddings (overrides one-hot)
        pert_embeddings_matrix: np.ndarray | None = None
        if config.perturbation_features_file is not None:
            from diffbio.sources.embeddings import (  # noqa: PLC0415
                EmbeddingArtifactSource,
                EmbeddingArtifactSourceConfig,
            )

            ext_emb = EmbeddingArtifactSource(
                EmbeddingArtifactSourceConfig(
                    file_path=str(Path(config.perturbation_features_file))
                )
            ).embeddings
            pert_embeddings_matrix = np.asarray(ext_emb)
            if pert_embeddings_matrix.shape[0] != len(perturbation.categories):
                raise ValueError(
                    "perturbation_features_file must have one row per perturbation category; "
                    f"expected {len(perturbation.categories)}, "
                    f"got {pert_embeddings_matrix.shape[0]}"
                )

        # Barcodes
        barcodes: np.ndarray | None = None
        if config.include_barcodes:
            if "barcode" not in obs:
                raise ValueError("barcode column is required when include_barcodes=True")
            barcodes = np.asarray(obs["barcode"])

        # -- Visible index mapping for should_yield_controls --
        if config.should_yield_controls:
            visible_indices = np.arange(adata.n_obs, dtype=np.int64)
        else:
            visible_indices = np.where(~control_mask)[0].astype(np.int64)

        # -- Store all data --
        self._initialize_loaded_source(
            config=config,
            adata=adata,
            data=build_anndata_data(counts=counts, obs=obs, var=var, obsm=obsm),
            length=len(visible_indices),
        )
        self._visible_indices = visible_indices

        self._categorical_state = nnx.static(
            _CategoricalMetadataState(
                perturbation=perturbation,
                cell_type=cell_type,
                batch=batch,
                control_mask=control_mask,
                group_codes=group_codes,
            )
        )
        self._feature_state = nnx.static(
            _FeatureViewState(
                perturbation_embeddings_matrix=pert_embeddings_matrix,
                barcodes=barcodes,
                additional_obs=config.additional_obs,
                hvg_indices=hvg_indices,
                gene_names=tuple(adata.var_names),
            )
        )

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
            raise IndexError(f"Cell index {idx} out of range for dataset with {self.length} cells")
        internal_idx = int(self._visible_indices[idx])
        return self._build_pert_element(internal_idx)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over visible cells with optional shuffling."""
        return self._iter_with_builder(self._build_visible_element)

    def get_batch(self, batch_size: int, key: jax.Array | None = None) -> dict[str, Any]:
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

        return self._get_batch_with_gather(batch_size, key, _gather)

    # =================================================================
    # Perturbation-specific public API
    # =================================================================

    def get_control_mask(self) -> np.ndarray:
        """Return boolean mask where True indicates a control cell."""
        return self._categorical_state.control_mask

    def get_pert_codes(self) -> np.ndarray:
        """Return per-cell integer perturbation codes."""
        return self._categorical_state.perturbation.codes

    def get_cell_type_codes(self) -> np.ndarray:
        """Return per-cell integer cell type codes."""
        return self._categorical_state.cell_type.codes

    def get_batch_codes(self) -> np.ndarray:
        """Return per-cell integer batch codes."""
        return self._categorical_state.batch.codes

    def get_pert_categories(self) -> np.ndarray:
        """Return sorted array of unique perturbation labels."""
        return np.array(self._categorical_state.perturbation.categories)

    def get_cell_type_categories(self) -> np.ndarray:
        """Return sorted array of unique cell type labels."""
        return np.array(self._categorical_state.cell_type.categories)

    def get_batch_categories(self) -> np.ndarray:
        """Return sorted array of unique batch labels."""
        return np.array(self._categorical_state.batch.categories)

    def get_group_codes(self) -> np.ndarray:
        """Return per-cell group codes for (cell_type, perturbation) grouping."""
        return self._categorical_state.group_codes

    def get_onehot_map(self) -> dict[str, jnp.ndarray]:
        """Return perturbation one-hot encoding map (JAX arrays)."""
        categories = self._categorical_state.perturbation.categories
        matrix = self._categorical_state.perturbation.onehot_matrix
        return {str(category): jnp.array(matrix[i]) for i, category in enumerate(categories)}

    def get_gene_names(self, output_space: str = "all") -> list[str]:
        """Return gene names, optionally filtered by output space.

        Args:
            output_space: ``"all"`` for all genes, ``"gene"`` for HVG subset.

        Returns:
            List of gene name strings.
        """
        if output_space == "gene" and self._feature_state.hvg_indices is not None:
            return [self._feature_state.gene_names[i] for i in self._feature_state.hvg_indices]
        return list(self._feature_state.gene_names)

    def get_n_genes(self) -> int:
        """Return total number of genes."""
        return len(self._feature_state.gene_names)

    def get_var_dims(self) -> dict[str, int]:
        """Return dimensionality info for the dataset.

        Returns:
            Dict with ``n_genes``, ``n_cells``, ``n_perts``,
            ``n_cell_types``, ``n_batches``.
        """
        return {
            "n_genes": len(self._feature_state.gene_names),
            "n_cells": self.length,
            "n_perts": len(self._categorical_state.perturbation.categories),
            "n_cell_types": len(self._categorical_state.cell_type.categories),
            "n_batches": len(self._categorical_state.batch.categories),
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
        categorical_state = self._categorical_state
        feature_state = self._feature_state
        pc = int(categorical_state.perturbation.codes[idx])
        cc = int(categorical_state.cell_type.codes[idx])
        bc = int(categorical_state.batch.codes[idx])

        if feature_state.perturbation_embeddings_matrix is not None:
            pert_emb = jnp.array(feature_state.perturbation_embeddings_matrix[pc])
        else:
            pert_emb = jnp.array(categorical_state.perturbation.onehot_matrix[pc])

        pert_name = str(categorical_state.perturbation.labels[idx])
        ct_name = str(categorical_state.cell_type.labels[idx])
        batch_name = str(categorical_state.batch.labels[idx])

        element: dict[str, Any] = {
            "counts": cell_counts,
            "obs": cell_obs,
            "obsm": cell_obsm,
            "pert_code": pc,
            "cell_type_code": cc,
            "batch_code": bc,
            "is_control": bool(categorical_state.control_mask[idx]),
            "pert_emb": pert_emb,
            "cell_type_onehot": jnp.array(categorical_state.cell_type.onehot_matrix[cc]),
            "batch_onehot": jnp.array(categorical_state.batch.onehot_matrix[bc]),
            "pert_name": pert_name,
            "cell_type_name": ct_name,
            "batch_name": batch_name,
        }

        for column in feature_state.additional_obs:
            element[column] = cell_obs[column]

        if feature_state.barcodes is not None:
            element["barcode"] = str(feature_state.barcodes[idx])

        return element

    def _build_pert_element_from_data(self, data: dict[str, Any], idx: int) -> dict[str, Any]:
        """Build element from data dict (used by eager_iter callback)."""
        return self._build_pert_element(idx)

    def _build_visible_element(self, data: dict[str, Any], idx: int) -> dict[str, Any]:
        """Build element with visible index remapping (used by __iter__)."""
        internal_idx = int(self._visible_indices[idx])
        return self._build_pert_element(internal_idx)

    def _build_batch_element(self, indices: np.ndarray) -> dict[str, Any]:
        """Build a batched dictionary for multiple cells."""
        counts = self.data["counts"][indices]
        obs = {col: np.asarray(arr)[indices] for col, arr in self.data["obs"].items()}
        obsm: dict[str, jnp.ndarray] = {}
        for emb_name, emb_arr in self.data["obsm"].items():
            obsm[emb_name] = emb_arr[indices]

        categorical_state = self._categorical_state
        feature_state = self._feature_state
        pert_codes = categorical_state.perturbation.codes[indices]
        ct_codes = categorical_state.cell_type.codes[indices]
        batch_codes = categorical_state.batch.codes[indices]

        # Build batched embeddings (indexed by codes, converted to JAX)
        if feature_state.perturbation_embeddings_matrix is not None:
            pert_embs = jnp.array(feature_state.perturbation_embeddings_matrix[pert_codes])
        else:
            pert_embs = jnp.array(categorical_state.perturbation.onehot_matrix[pert_codes])

        transformed_data = {
            "counts": counts,
            "obs": obs,
            "obsm": obsm,
            "pert_code": jnp.array(pert_codes),
            "cell_type_code": jnp.array(ct_codes),
            "batch_code": jnp.array(batch_codes),
            "is_control": jnp.array(categorical_state.control_mask[indices]),
            "pert_emb": pert_embs,
            "cell_type_onehot": jnp.array(categorical_state.cell_type.onehot_matrix[ct_codes]),
            "batch_onehot": jnp.array(categorical_state.batch.onehot_matrix[batch_codes]),
            "pert_name": [str(categorical_state.perturbation.labels[i]) for i in indices],
            "cell_type_name": [str(categorical_state.cell_type.labels[i]) for i in indices],
            "batch_name": [str(categorical_state.batch.labels[i]) for i in indices],
        }

        for column in feature_state.additional_obs:
            transformed_data[column] = obs[column]

        if feature_state.barcodes is not None:
            transformed_data["barcode"] = (
                np.asarray(feature_state.barcodes)[indices].astype(str).tolist()
            )

        return transformed_data
