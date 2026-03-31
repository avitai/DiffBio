"""BenGRN ground truth DataSource for GRN inference benchmarking.

Loads mESC expression data and ChIP+Perturb ground truth edges from
the benGRN repository (Stone & Sroy gold standards).

Data source: /media/mahdi/ssd23/Works/benGRN/data/GroundTruth/

References:
    - benGRN: https://github.com/your-org/benGRN
    - Stone & Sroy gold standards
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

_BASE_DIR = Path(
    "/media/mahdi/ssd23/Works/benGRN/data/GroundTruth/stone_and_sroy"
)


@dataclass(frozen=True)
class BenGRNConfig(StructuralConfig):
    """Configuration for BenGRNSource.

    Attributes:
        data_dir: Root directory of benGRN ground truth data.
        species: Species to load ('mouse' or 'human').
        expression_dataset: Which expression dataset to use.
        ground_truth: Which ground truth network to use.
        max_genes: Maximum number of genes to include (for speed).
    """

    data_dir: str = str(_BASE_DIR)
    species: str = "mouse"
    expression_dataset: str = "duren"
    ground_truth: str = "chipunion_KDUnion_intersect"
    max_genes: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        base = Path(self.data_dir)
        if not base.exists():
            raise FileNotFoundError(
                f"benGRN data not found: {base}. "
                f"Clone benGRN repo to ../benGRN/"
            )


class BenGRNSource(DataSourceModule):
    """DataSource for benGRN ground truth GRN data.

    Loads expression matrix and ground truth regulatory edges for
    GRN inference benchmarking.
    """

    data: dict[str, Any] = nnx.data()

    def __init__(
        self,
        config: BenGRNConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Load the benGRN ground truth data."""
        super().__init__(
            config, rngs=rngs, name=name or "BenGRNSource"
        )
        self.data = self._load(config)
        logger.info(
            "Loaded benGRN: %d cells, %d genes, %d TFs, %d GT edges",
            self.data["n_cells"],
            self.data["n_genes"],
            self.data["n_tfs"],
            self.data["n_edges"],
        )

    def _load(self, config: BenGRNConfig) -> dict[str, Any]:
        """Load expression data and ground truth edges."""
        import gzip  # noqa: PLC0415

        base = Path(config.data_dir)
        prefix = "mESC" if config.species == "mouse" else "hESC"

        # Load expression data
        expr_file = (
            base / "scRNA"
            / f"{config.expression_dataset}_rna_filtered_log2.tsv.gz"
        )
        if not expr_file.exists():
            raise FileNotFoundError(
                f"Expression data not found: {expr_file}"
            )

        with gzip.open(expr_file, "rt") as f:
            lines = f.readlines()

        # Parse TSV: row 0 = cell/sample names (header)
        # Rows 1+: gene_name \t val1 \t val2 \t ...
        # Transpose: genes are rows, cells are columns
        _cell_names = lines[0].strip().split("\t")  # noqa: F841
        gene_names: list[str] = []
        expr_cols: list[list[float]] = []
        for line in lines[1:]:
            parts = line.strip().split("\t")
            gene_names.append(parts[0])
            expr_cols.append([float(v) for v in parts[1:]])

        # expression is (n_genes, n_cells), transpose to (n_cells, n_genes)
        expression = np.array(expr_cols, dtype=np.float32).T
        n_cells = expression.shape[0]

        # Load TF list
        tf_file = (
            base
            / f"{prefix}_{config.expression_dataset.capitalize()}_TFs.tsv"
        )
        if not tf_file.exists():
            # Try alternative naming
            tf_file = base / f"{prefix}_Duren_TFs.tsv"

        if tf_file.exists():
            tf_names = [
                l.strip() for l in tf_file.read_text().splitlines()
                if l.strip()
            ]
        else:
            logger.warning("TF file not found, using first 50 genes")
            tf_names = gene_names[:50]

        # Load ground truth edges
        gt_file = (
            base / "gold_standards" / prefix
            / f"{prefix}_{config.ground_truth}.txt"
        )
        if not gt_file.exists():
            raise FileNotFoundError(
                f"Ground truth not found: {gt_file}"
            )

        gt_edges: list[tuple[str, str]] = []
        for line in gt_file.read_text().splitlines():
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                gt_edges.append((parts[0], parts[1]))

        # Build gene index (intersection of expression + GT genes)
        expr_gene_set = set(gene_names)
        gt_genes = set()
        for src, tgt in gt_edges:
            gt_genes.add(src)
            gt_genes.add(tgt)
        common_genes = sorted(expr_gene_set & gt_genes)

        if config.max_genes is not None:
            common_genes = common_genes[: config.max_genes]

        gene_to_idx = {g: i for i, g in enumerate(common_genes)}
        n_genes = len(common_genes)

        # Build expression submatrix for common genes
        expr_col_idx = [
            gene_names.index(g) for g in common_genes
            if g in gene_names
        ]
        counts = jnp.array(expression[:, expr_col_idx])

        # Build ground truth binary matrix
        gt_matrix = np.zeros((n_genes, n_genes), dtype=np.float32)
        n_edges = 0
        for src, tgt in gt_edges:
            if src in gene_to_idx and tgt in gene_to_idx:
                gt_matrix[gene_to_idx[src], gene_to_idx[tgt]] = 1.0
                n_edges += 1

        # TF indices
        tf_indices = np.array(
            [gene_to_idx[tf] for tf in tf_names if tf in gene_to_idx],
            dtype=np.int32,
        )

        return {
            "counts": counts,
            "gene_names": common_genes,
            "tf_names": [
                tf for tf in tf_names if tf in gene_to_idx
            ],
            "tf_indices": tf_indices,
            "ground_truth_matrix": gt_matrix,
            "ground_truth_edges": gt_edges,
            "n_cells": n_cells,
            "n_genes": n_genes,
            "n_tfs": len(tf_indices),
            "n_edges": n_edges,
        }

    def load(self) -> dict[str, Any]:
        """Return the full dataset."""
        return self.data

    def __len__(self) -> int:
        """Return number of cells."""
        return self.data["n_cells"]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over cells."""
        for i in range(len(self)):
            yield {
                k: v[i] if hasattr(v, "__getitem__")
                and k not in ("gene_names", "tf_names",
                              "ground_truth_edges")
                else v
                for k, v in self.data.items()
            }
