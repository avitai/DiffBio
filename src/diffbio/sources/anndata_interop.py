"""AnnData interop layer for DiffBio data dictionaries.

Provides bidirectional conversion between DiffBio's standard data dict format
(with keys ``counts``, ``obs``, ``var``, ``obsm``) and AnnData objects.

This enables integration with the broader single-cell ecosystem (scanpy,
scvi-tools, etc.) while keeping DiffBio's internal representation as
JAX-native dictionaries suitable for differentiable pipelines.

Also provides utilities for benchmark evaluation:

- ``from_anndata_to_operator_input``: Convert AnnData to operator-specific dicts.
- ``to_grader_answer``: Convert operator output dicts to grader-expected formats.

Both ``anndata`` and ``pandas`` are optional dependencies. Functions raise
``ImportError`` with installation instructions if they are not available.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from diffbio.sources._utils import _require_anndata

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import anndata


def _require_pandas() -> Any:
    """Import pandas, raising a clear error if not installed.

    Returns:
        The pandas module.

    Raises:
        ImportError: If pandas is not installed.
    """
    try:
        import pandas as pd  # noqa: PLC0415

        return pd
    except ImportError as err:
        raise ImportError(
            "pandas is required for AnnData interop. Install with: uv pip install pandas"
        ) from err


def _to_dense_numpy(matrix: Any) -> np.ndarray:
    """Convert a matrix (dense, JAX, or sparse) to a dense numpy array.

    Args:
        matrix: A numpy array, JAX array, or scipy sparse matrix.

    Returns:
        Dense numpy array with float32 dtype.
    """
    try:
        import scipy.sparse  # noqa: PLC0415

        if scipy.sparse.issparse(matrix):
            return np.asarray(matrix.toarray(), dtype=np.float32)
    except ImportError:
        pass

    return np.asarray(matrix, dtype=np.float32)


def to_anndata(data_dict: dict[str, Any]) -> anndata.AnnData:
    """Convert a DiffBio data dict to an AnnData object.

    Translates the standard DiffBio dictionary format (as produced by
    ``AnnDataSource.load()``) into an ``anndata.AnnData`` object for use
    with scanpy, scvi-tools, and other AnnData-based tools.

    JAX arrays in ``counts`` and ``obsm`` are converted to numpy via
    ``np.asarray()``.  The ``obs`` and ``var`` dicts become pandas
    DataFrames.

    Args:
        data_dict: Dictionary with keys:
            - ``counts``: JAX or numpy array of shape (n_cells, n_genes).
            - ``obs``: Dict mapping column names to per-cell arrays.
            - ``var``: Dict mapping column names to per-gene arrays.
            - ``obsm`` (optional): Dict mapping embedding names to arrays.

    Returns:
        AnnData object with ``.X``, ``.obs``, ``.var``, and ``.obsm``
        populated from the input dictionary.

    Raises:
        ImportError: If anndata or pandas is not installed.
    """
    ad = _require_anndata()
    pd = _require_pandas()

    counts_np = _to_dense_numpy(data_dict["counts"])

    obs_df = pd.DataFrame(data_dict.get("obs", {}))
    var_df = pd.DataFrame(data_dict.get("var", {}))

    adata = ad.AnnData(X=counts_np, obs=obs_df, var=var_df)

    obsm = data_dict.get("obsm", {})
    for key, value in obsm.items():
        adata.obsm[key] = np.asarray(value, dtype=np.float32)

    return adata


def from_anndata(adata: anndata.AnnData) -> dict[str, Any]:
    """Convert an AnnData object to a DiffBio data dict.

    Translates an ``anndata.AnnData`` object into the standard DiffBio
    dictionary format compatible with ``AnnDataSource.load()`` output.

    Sparse ``.X`` matrices are converted to dense before wrapping in a
    JAX array. ``.obs`` and ``.var`` DataFrames become plain dicts of
    numpy arrays. ``.obsm`` entries become JAX arrays.

    Args:
        adata: AnnData object to convert.

    Returns:
        Dictionary with keys:
            - ``counts``: Dense JAX array of shape (n_cells, n_genes).
            - ``obs``: Dict mapping column names to numpy arrays.
            - ``var``: Dict mapping column names to numpy arrays.
            - ``obsm``: Dict mapping embedding names to JAX arrays.
    """
    counts = jnp.array(_to_dense_numpy(adata.X))

    obs: dict[str, Any] = {col: np.asarray(adata.obs[col]) for col in adata.obs.columns}
    var: dict[str, Any] = {col: np.asarray(adata.var[col]) for col in adata.var.columns}

    obsm: dict[str, jnp.ndarray] = {}
    if adata.obsm is not None and len(adata.obsm) > 0:
        for key in adata.obsm.keys():
            obsm[key] = jnp.array(np.asarray(adata.obsm[key], dtype=np.float32))

    return {
        "counts": counts,
        "obs": obs,
        "var": var,
        "obsm": obsm,
    }


# ---------------------------------------------------------------------------
# Benchmark evaluation utilities
# ---------------------------------------------------------------------------

# Mapping from task_type to the conversion strategy for building operator inputs.
_TASK_TYPE_BUILDERS: dict[str, str] = {
    "qc_filtering": "counts",
    "clustering": "embeddings",
    "batch_correction": "embeddings_with_batch",
    "differential_expression": "counts_with_design",
    "trajectory": "embeddings",
    "normalization": "counts_with_library",
    "spatial_analysis": "counts_with_spatial",
    "cell_annotation": "embeddings_with_batch",
}


def from_anndata_to_operator_input(
    adata: anndata.AnnData,
    task_type: str,
) -> dict[str, Any]:
    """Convert an AnnData object to a DiffBio operator input dict.

    Produces the specific data dict keys expected by the operator
    associated with ``task_type``. For example, clustering operators
    expect an ``"embeddings"`` key from PCA, while batch correction
    additionally requires ``"batch_labels"``.

    Args:
        adata: AnnData object with counts, obs metadata, and embeddings.
        task_type: Category of the benchmark task. Supported values:
            ``"qc_filtering"``, ``"clustering"``, ``"batch_correction"``,
            ``"differential_expression"``, ``"trajectory"``,
            ``"normalization"``, ``"spatial_analysis"``,
            ``"cell_annotation"``.

    Returns:
        Dictionary with keys appropriate for the target operator.

    Raises:
        ValueError: If task_type is not recognised.
        KeyError: If required data is missing from adata.
    """
    strategy = _TASK_TYPE_BUILDERS.get(task_type)
    if strategy is None:
        raise ValueError(
            f"Unknown task_type {task_type!r}. Supported: {sorted(_TASK_TYPE_BUILDERS)}"
        )

    counts = jnp.array(_to_dense_numpy(adata.X))

    if strategy == "counts":
        library_size = jnp.sum(counts, axis=1, keepdims=True)
        return {"counts": counts, "library_size": library_size}

    if strategy == "counts_with_design":
        n_cells = adata.n_obs
        if "batch" in adata.obs.columns:
            batch_arr = np.asarray(adata.obs["batch"])
            unique_batches = np.unique(batch_arr)
            design = np.zeros((n_cells, len(unique_batches)), dtype=np.float32)
            for i, b in enumerate(unique_batches):
                design[batch_arr == b, i] = 1.0
        else:
            design = np.ones((n_cells, 1), dtype=np.float32)
        return {"counts": counts, "design": jnp.array(design)}

    if strategy == "counts_with_library":
        library_size = jnp.sum(counts, axis=1, keepdims=True)
        return {"counts": counts, "library_size": library_size}

    if strategy == "counts_with_spatial":
        if "spatial" not in (adata.obsm or {}):
            raise KeyError("AnnData missing obsm['spatial'] required for spatial_analysis tasks")
        spatial = jnp.array(np.asarray(adata.obsm["spatial"], dtype=np.float32))
        return {"counts": counts, "spatial_coords": spatial}

    # embeddings-based strategies
    embedding_key = _pick_embedding_key(adata)
    embeddings = jnp.array(np.asarray(adata.obsm[embedding_key], dtype=np.float32))

    if strategy == "embeddings":
        return {"embeddings": embeddings}

    # embeddings_with_batch
    if "batch" in adata.obs.columns:
        batch_arr = np.asarray(adata.obs["batch"])
        unique_batches = np.unique(batch_arr)
        batch_indices = np.searchsorted(unique_batches, batch_arr)
        batch_labels = jnp.array(batch_indices, dtype=jnp.int32)
    else:
        batch_labels = jnp.zeros(adata.n_obs, dtype=jnp.int32)

    return {"embeddings": embeddings, "batch_labels": batch_labels}


def _pick_embedding_key(adata: anndata.AnnData) -> str:
    """Select the best available embedding key from adata.obsm.

    Prefers ``X_pca`` > ``X_scvi`` > first available key.

    Args:
        adata: AnnData object.

    Returns:
        Key name from adata.obsm.

    Raises:
        KeyError: If no embeddings are available.
    """
    if adata.obsm is None or len(adata.obsm) == 0:
        raise KeyError("AnnData has no obsm embeddings for operator input")

    for preferred in ("X_pca", "X_scvi", "X_umap"):
        if preferred in adata.obsm:
            return preferred

    return next(iter(adata.obsm.keys()))


def to_grader_answer(
    operator_output: dict[str, Any],
    task_type: str,
) -> Any:
    """Convert DiffBio operator output to grader-expected answer format.

    Extracts the relevant result from an operator's output dict and
    converts it to the primitive type expected by the benchmark grader
    (float, str, list, dict, or set).

    Args:
        operator_output: Dictionary returned by an operator's ``apply()``
            method (the data dict component).
        task_type: Category of the benchmark task, determining which
            output key to extract and how to format it.

    Returns:
        The answer in the format expected by the corresponding grader:
            - ``"qc_filtering"`` -> ``float`` (number of cells passing)
            - ``"clustering"`` -> ``dict[str, float]`` (cluster distribution)
              or ``set[str]`` (cluster label set)
            - ``"differential_expression"`` -> ``list[str]`` (top DE genes)
            - ``"batch_correction"`` -> ``float`` (batch mixing metric)
            - ``"normalization"`` -> ``float`` (reconstruction metric)
            - ``"trajectory"`` -> ``float`` (pseudotime correlation)
            - ``"spatial_analysis"`` -> ``set[str]`` (domain labels)
            - ``"cell_annotation"`` -> ``str`` (predicted cell type)

    Raises:
        ValueError: If task_type is not recognised.
    """
    converters: dict[str, Any] = {
        "qc_filtering": _answer_qc_filtering,
        "clustering": _answer_clustering,
        "differential_expression": _answer_de,
        "batch_correction": _answer_batch_correction,
        "normalization": _answer_normalization,
        "trajectory": _answer_trajectory,
        "spatial_analysis": _answer_spatial,
        "cell_annotation": _answer_cell_annotation,
    }
    converter = converters.get(task_type)
    if converter is None:
        raise ValueError(f"Unknown task_type {task_type!r}. Supported: {sorted(converters)}")
    return converter(operator_output)


def _answer_qc_filtering(output: dict[str, Any]) -> float:
    """Extract cell count after quality filtering."""
    if "retention_weights" in output:
        weights = np.asarray(output["retention_weights"])
        return float(np.sum(weights > 0.5))
    if "counts" in output:
        return float(np.asarray(output["counts"]).shape[0])
    return 0.0


def _answer_clustering(output: dict[str, Any]) -> dict[str, float]:
    """Extract cluster label distribution from clustering output."""
    if "cluster_labels" in output:
        labels = np.asarray(output["cluster_labels"])
    elif "cluster_assignments" in output:
        labels = np.asarray(output["cluster_assignments"])
        if labels.ndim == 2:
            labels = np.argmax(labels, axis=-1)
    else:
        return {}

    unique, counts = np.unique(labels, return_counts=True)
    total = float(counts.sum())
    return {str(label): float(count / total) for label, count in zip(unique, counts)}


def _answer_de(output: dict[str, Any]) -> list[str]:
    """Extract top differentially expressed gene names."""
    if "top_genes" in output:
        return [str(g) for g in output["top_genes"]]
    if "significant" in output and "gene_names" in output:
        sig = np.asarray(output["significant"])
        genes = np.asarray(output["gene_names"])
        return [str(g) for g in genes[sig > 0.5]]
    if "log_fold_change" in output and "gene_names" in output:
        lfc = np.abs(np.asarray(output["log_fold_change"]))
        genes = np.asarray(output["gene_names"])
        top_idx = np.argsort(lfc)[::-1][:20]
        return [str(genes[i]) for i in top_idx]
    return []


def _answer_batch_correction(output: dict[str, Any]) -> float:
    """Extract batch mixing metric from correction output."""
    if "batch_mixing_score" in output:
        return float(output["batch_mixing_score"])
    if "corrected_embeddings" in output and "batch_labels" in output:
        corrected = np.asarray(output["corrected_embeddings"])
        return float(np.std(corrected))
    return 0.0


def _answer_normalization(output: dict[str, Any]) -> float:
    """Extract reconstruction quality from normalizer output."""
    if "reconstruction_loss" in output:
        return float(output["reconstruction_loss"])
    if "normalized" in output:
        return float(np.mean(np.asarray(output["normalized"])))
    return 0.0


def _answer_trajectory(output: dict[str, Any]) -> float:
    """Extract pseudotime summary from trajectory output."""
    if "pseudotime" in output:
        return float(np.max(np.asarray(output["pseudotime"])))
    return 0.0


def _answer_spatial(output: dict[str, Any]) -> set[str]:
    """Extract spatial domain label set."""
    if "domain_assignments" in output:
        assignments = np.asarray(output["domain_assignments"])
        if assignments.ndim == 2:
            assignments = np.argmax(assignments, axis=-1)
        return {str(d) for d in np.unique(assignments)}
    return set()


def _answer_cell_annotation(output: dict[str, Any]) -> str:
    """Extract dominant cell type annotation."""
    if "cell_type" in output:
        return str(output["cell_type"])
    if "cluster_labels" in output:
        labels = np.asarray(output["cluster_labels"])
        unique, counts = np.unique(labels, return_counts=True)
        return str(unique[np.argmax(counts)])
    return ""
