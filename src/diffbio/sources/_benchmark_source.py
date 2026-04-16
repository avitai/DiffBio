"""Shared substrate for benchmark-oriented eager data sources."""

from __future__ import annotations

from collections.abc import Iterator
import logging
from pathlib import Path
from typing import Any, Callable, Literal, overload

import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.core.data_source import DataSourceModule

from diffbio.sources._utils import _require_anndata


def load_benchmark_adata(
    *,
    data_dir: str,
    filename: str,
    subsample: int | None = None,
    seed: int = 42,
) -> Any:
    """Load a benchmark h5ad file with optional deterministic subsampling."""
    anndata_mod = _require_anndata()
    path = Path(data_dir) / filename
    adata = anndata_mod.read_h5ad(path)

    if subsample is not None and subsample < adata.n_obs:
        rng = np.random.default_rng(seed)
        indices = rng.choice(adata.n_obs, size=subsample, replace=False)
        indices.sort()
        adata = adata[indices].copy()

    return adata


def load_benchmark_counts(
    *,
    data_dir: str,
    filename: str,
    to_dense: Callable[[Any], np.ndarray],
    subsample: int | None = None,
    seed: int = 42,
) -> tuple[Any, jnp.ndarray]:
    """Load a benchmark h5ad file and convert its count matrix to a JAX array."""
    adata = load_benchmark_adata(
        data_dir=data_dir,
        filename=filename,
        subsample=subsample,
        seed=seed,
    )
    counts = jnp.array(to_dense(adata.X))
    return adata, counts


@overload
def encode_label_column(
    column: Any,
    *,
    include_names: Literal[False] = False,
) -> np.ndarray: ...


@overload
def encode_label_column(
    column: Any,
    *,
    include_names: Literal[True],
) -> tuple[np.ndarray, list[str]]: ...


def encode_label_column(
    column: Any,
    *,
    include_names: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[str]]:
    """Encode an observation column into stable int32 label codes."""
    if hasattr(column, "cat"):
        codes = np.asarray(column.cat.codes, dtype=np.int32)
        names = [str(value) for value in column.cat.categories]
    else:
        unique_labels, codes = np.unique(np.asarray(column), return_inverse=True)
        codes = codes.astype(np.int32)
        names = [str(value) for value in unique_labels]

    if include_names:
        return codes, names
    return codes


def iter_loaded_rows(
    data: dict[str, Any],
    *,
    static_keys: tuple[str, ...] = ("gene_names",),
) -> Iterator[dict[str, Any]]:
    """Iterate row-wise over an eager benchmark payload."""
    static_key_set = set(static_keys)
    for i in range(int(data["n_cells"])):
        yield {
            key: value[i] if hasattr(value, "__getitem__") and key not in static_key_set else value
            for key, value in data.items()
        }


class BenchmarkDataSource(DataSourceModule):
    """Shared base class for eager benchmark data sources backed by a data dict."""

    data: dict[str, Any] = nnx.data()
    iter_static_keys: tuple[str, ...] = ("gene_names",)

    def load(self) -> dict[str, Any]:
        """Return the eagerly loaded dataset payload."""
        return self.data

    def __len__(self) -> int:
        """Return the number of rows in the eagerly loaded dataset."""
        return int(self.data["n_cells"])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate row-wise through the loaded dataset payload."""
        return iter_loaded_rows(self.data, static_keys=self.iter_static_keys)

    def _load_benchmark_counts(
        self,
        config: Any,
        filename: str,
        to_dense: Callable[[Any], np.ndarray],
    ) -> tuple[Any, jnp.ndarray]:
        """Load benchmark AnnData and convert its count matrix to a JAX array."""
        return load_benchmark_counts(
            data_dir=config.data_dir,
            filename=filename,
            to_dense=to_dense,
            subsample=getattr(config, "subsample", None),
        )

    def _log_loaded_summary(
        self,
        logger: logging.Logger,
        dataset_name: str,
        metric_keys: tuple[str, ...],
    ) -> None:
        """Log a standard loaded-dataset summary using the requested metric keys."""
        labels = [key[2:] if key.startswith("n_") else key for key in metric_keys]
        metric_summary = ", ".join(f"%d {label}" for label in labels)
        logger.info(
            f"Loaded {dataset_name}: {metric_summary}",
            *(int(self.data[key]) for key in metric_keys),
        )
