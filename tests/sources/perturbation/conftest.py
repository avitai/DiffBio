"""Shared test fixtures for perturbation sub-package tests.

Creates synthetic H5AD files with perturbation metadata matching the structure
expected by cell-load and the DiffBio perturbation source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CELL_TYPES = ["TypeA", "TypeB", "TypeC"]
PERTURBATIONS = ["GeneX", "GeneY", "GeneZ", "GeneW", "GeneV"]
CONTROL_PERT = "non-targeting"
ALL_PERTS = [CONTROL_PERT, *PERTURBATIONS]
BATCHES = ["batch1", "batch2"]

N_CELLS_PER_GROUP = 50  # per (cell_type, perturbation) combo
N_GENES = 100
N_HVG = 20
EMBED_DIM = 15

# Total cells: 3 cell types * 6 perts (5 + control) * 50 = 900
N_TOTAL_CELLS = len(CELL_TYPES) * len(ALL_PERTS) * N_CELLS_PER_GROUP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_anndata() -> Any:
    """Import anndata, skip test if unavailable."""
    anndata = pytest.importorskip("anndata")
    return anndata


def _build_synthetic_adata(
    rng: np.random.Generator,
    *,
    n_genes: int = N_GENES,
    include_hvg: bool = True,
    include_embedding: bool = True,
    include_barcodes: bool = True,
) -> Any:
    """Build a synthetic AnnData object with perturbation metadata.

    Returns:
        AnnData with obs columns: perturbation, cell_type, batch, barcode
        and var columns: gene_name, is_hvg (if include_hvg).
    """
    ad = _require_anndata()

    obs_records: list[dict[str, str]] = []
    for ct in CELL_TYPES:
        for pert in ALL_PERTS:
            for i in range(N_CELLS_PER_GROUP):
                obs_records.append(
                    {
                        "perturbation": pert,
                        "cell_type": ct,
                        "batch": BATCHES[i % len(BATCHES)],
                    }
                )

    n_cells = len(obs_records)

    # Count matrix: random Poisson counts
    counts = rng.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float32)

    # Inject knockdown signal for known perturbations:
    # For each perturbation gene, reduce expression in perturbed cells
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    # Map first 5 gene names to perturbation names for knockdown tests
    for idx, pert in enumerate(PERTURBATIONS):
        gene_names[idx] = pert

    for cell_idx, rec in enumerate(obs_records):
        pert = rec["perturbation"]
        if pert in PERTURBATIONS:
            gene_idx = PERTURBATIONS.index(pert)
            # Strong knockdown: reduce to ~10% of normal
            counts[cell_idx, gene_idx] *= 0.1

    import pandas as pd  # noqa: PLC0415

    obs = pd.DataFrame(obs_records)
    obs.index = [f"cell_{i}" for i in range(n_cells)]

    # Make obs columns categorical (matching real H5AD files)
    for col in ["perturbation", "cell_type", "batch"]:
        obs[col] = pd.Categorical(obs[col])

    if include_barcodes:
        obs["barcode"] = [f"ACGT{i:04d}" for i in range(n_cells)]

    var = pd.DataFrame({"gene_name": gene_names}, index=gene_names)
    if include_hvg:
        is_hvg = np.zeros(n_genes, dtype=bool)
        is_hvg[:n_genes // 5] = True  # first 20 genes are HVG
        var["is_hvg"] = is_hvg

    adata = ad.AnnData(X=counts, obs=obs, var=var)

    if include_embedding:
        adata.obsm["X_hvg"] = rng.standard_normal((n_cells, EMBED_DIM)).astype(
            np.float32
        )

    return adata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_adata():
    """Create a synthetic AnnData object."""
    rng = np.random.default_rng(42)
    return _build_synthetic_adata(rng)


@pytest.fixture()
def synthetic_h5ad_path(tmp_path: Path, synthetic_adata) -> Path:
    """Write synthetic AnnData to a temp .h5ad file and return the path."""
    path = tmp_path / "test_dataset.h5ad"
    synthetic_adata.write_h5ad(path)
    return path


@pytest.fixture()
def synthetic_h5ad_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Create two synthetic H5AD files for multi-dataset tests."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(123)
    adata1 = _build_synthetic_adata(rng1)
    adata2 = _build_synthetic_adata(rng2)

    path1 = tmp_path / "dataset1.h5ad"
    path2 = tmp_path / "dataset2.h5ad"
    adata1.write_h5ad(path1)
    adata2.write_h5ad(path2)
    return path1, path2
