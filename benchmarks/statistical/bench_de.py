#!/usr/bin/env python3
"""Differential expression benchmark: DifferentiableNBGLM on immune_human.

Evaluates DiffBio's DifferentiableNBGLM operator on the immune human
dataset (33,506 cells, 10 batches, 16 cell types). A two-condition
DE analysis is created by comparing the two most abundant cell types.

The NB GLM is fit to each sample via ``batch_log_likelihood`` and
per-gene DE significance is derived from the fitted coefficients.
Results are compared against a simple Welch t-test on log-normalized
counts, with concordance measured as Jaccard overlap of top-50 genes.

Usage:
    python benchmarks/statistical/bench_de.py
    python benchmarks/statistical/bench_de.py --quick
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from scipy import stats as sp_stats

from benchmarks._base import (
    DiffBioBenchmark,
    DiffBioBenchmarkConfig,
)
from benchmarks._baselines.de import DE_BASELINES
from diffbio.operators.statistical.nb_glm import (
    DifferentiableNBGLM,
    NBGLMConfig,
)
from diffbio.sources.immune_human import (
    ImmuneHumanConfig,
    ImmuneHumanSource,
)

logger = logging.getLogger(__name__)

_CONFIG = DiffBioBenchmarkConfig(
    name="statistical/de",
    domain="statistical",
    quick_subsample=2000,
)

_TOP_K = 50


def _select_two_conditions(
    cell_type_labels: np.ndarray,
) -> tuple[int, int, np.ndarray]:
    """Pick the two most abundant cell types and their mask.

    Args:
        cell_type_labels: Integer cell type codes (n_cells,).

    Returns:
        Tuple of (type_a, type_b, mask) where mask is a boolean
        array selecting cells belonging to either type.
    """
    unique, counts = np.unique(cell_type_labels, return_counts=True)
    top_two = unique[np.argsort(-counts)[:2]]
    type_a, type_b = int(top_two[0]), int(top_two[1])
    mask = np.isin(cell_type_labels, top_two)
    return type_a, type_b, mask


def _compute_size_factors(
    counts: jnp.ndarray,
) -> jnp.ndarray:
    """Compute library-size normalization factors per cell.

    Uses the total count per cell, normalized to a median of 1.

    Args:
        counts: Count matrix (n_cells, n_genes).

    Returns:
        Size factors (n_cells,).
    """
    totals = jnp.sum(counts, axis=1)
    median_total = jnp.median(totals)
    return totals / jnp.maximum(median_total, 1.0)


def _welch_ttest_ranking(
    counts_np: np.ndarray,
    condition: np.ndarray,
) -> np.ndarray:
    """Rank genes by Welch t-test p-value on log-normalized counts.

    Args:
        counts_np: Count matrix (n_cells, n_genes) as numpy.
        condition: Binary condition array (n_cells,).

    Returns:
        Gene indices sorted by ascending p-value.
    """
    # Log-normalize: log1p(counts / total * 10000)
    totals = counts_np.sum(axis=1, keepdims=True)
    totals = np.maximum(totals, 1.0)
    log_norm = np.log1p(counts_np / totals * 10000.0)

    group_a = log_norm[condition == 1]
    group_b = log_norm[condition == 0]

    n_genes = counts_np.shape[1]
    p_values = np.ones(n_genes)
    for g in range(n_genes):
        if group_a[:, g].std() > 0 or group_b[:, g].std() > 0:
            _, p_val = sp_stats.ttest_ind(
                group_a[:, g], group_b[:, g], equal_var=False,
            )
            p_values[g] = p_val

    return np.argsort(p_values)


def _jaccard(set_a: set[int], set_b: set[int]) -> float:
    """Compute Jaccard similarity between two sets.

    Args:
        set_a: First set of indices.
        set_b: Second set of indices.

    Returns:
        Jaccard similarity in [0, 1].
    """
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


class DEBenchmark(DiffBioBenchmark):
    """Evaluate DifferentiableNBGLM for differential expression."""

    def __init__(
        self,
        config: DiffBioBenchmarkConfig = _CONFIG,
        *,
        quick: bool = False,
        data_dir: str = "/media/mahdi/ssd23/Data/scib",
    ) -> None:
        super().__init__(config, quick=quick, data_dir=data_dir)

    def _run_core(self) -> dict[str, Any]:
        """Load immune_human, fit NB GLM, compute DE metrics."""
        subsample = (
            self.config.quick_subsample if self.quick else None
        )

        # 1. Load dataset
        logger.info("Loading immune_human dataset...")
        source = ImmuneHumanSource(
            ImmuneHumanConfig(
                data_dir=self.data_dir, subsample=subsample,
            )
        )
        data = source.load()
        counts = data["counts"]
        cell_type_labels = np.asarray(data["cell_type_labels"])

        # 2. Select two most abundant cell types
        type_a, type_b, mask = _select_two_conditions(
            cell_type_labels
        )
        counts_sub = counts[mask]
        labels_sub = cell_type_labels[mask]
        n_cells = int(counts_sub.shape[0])
        n_genes = int(counts_sub.shape[1])

        logger.info(
            "  DE comparison: type %d vs type %d (%d cells, "
            "%d genes)",
            type_a, type_b, n_cells, n_genes,
        )

        # 3. Build design matrix: intercept + condition indicator
        condition = (labels_sub == type_a).astype(np.float32)
        design = jnp.column_stack([
            jnp.ones(n_cells),
            jnp.array(condition),
        ])
        size_factors = _compute_size_factors(counts_sub)

        # 4. Create NB GLM operator
        op_config = NBGLMConfig(
            n_features=n_genes,
            n_covariates=2,
        )
        rngs = nnx.Rngs(42)
        operator = DifferentiableNBGLM(op_config, rngs=rngs)

        # 5. Fit: compute batch log-likelihood
        logger.info("Computing NB GLM log-likelihood...")
        total_ll = operator.batch_log_likelihood(
            counts_sub, design, size_factors,
        )
        logger.info("  Total log-likelihood: %.2f", float(total_ll))

        # 6. Rank genes by per-gene log-likelihood contribution
        # Compute per-sample, per-gene log probs via vmap
        beta = operator.get_coefficients()
        dispersion = operator.get_dispersion()

        def _per_sample_per_gene_ll(
            count_row: jnp.ndarray,
            design_row: jnp.ndarray,
            sf: jnp.ndarray,
        ) -> jnp.ndarray:
            """Per-gene log prob for one sample."""
            log_mu = jnp.dot(design_row, beta)
            log_mu = log_mu + jnp.log(sf + 1e-8)
            mu = jnp.exp(log_mu)
            r = dispersion
            k = count_row
            log_prob = (
                jax.scipy.special.gammaln(k + r)
                - jax.scipy.special.gammaln(k + 1)
                - jax.scipy.special.gammaln(r)
                + r * jnp.log(r / (r + mu + 1e-8))
                + k * jnp.log(mu / (r + mu + 1e-8) + 1e-8)
            )
            return log_prob  # (n_genes,)

        per_gene_ll = jax.vmap(
            _per_sample_per_gene_ll
        )(counts_sub, design, size_factors)
        # per_gene_ll shape: (n_cells, n_genes)

        # Sum across samples to get per-gene total LL
        gene_ll = jnp.sum(per_gene_ll, axis=0)  # (n_genes,)

        # Genes with most negative LL are poorest fit -> DE
        # Rank by absolute deviation from mean LL
        mean_ll = jnp.mean(gene_ll)
        gene_deviation = jnp.abs(gene_ll - mean_ll)
        nbglm_ranking = jnp.argsort(-gene_deviation)

        # 7. Compare against Welch t-test ranking
        logger.info("Computing t-test baseline ranking...")
        ttest_ranking = _welch_ttest_ranking(
            np.asarray(counts_sub), np.asarray(condition),
        )

        nbglm_top = set(
            int(x) for x in np.asarray(nbglm_ranking[:_TOP_K])
        )
        ttest_top = set(int(x) for x in ttest_ranking[:_TOP_K])

        concordance = _jaccard(nbglm_top, ttest_top)
        n_de_genes = int(
            jnp.sum(gene_deviation > jnp.median(gene_deviation))
        )

        logger.info(
            "  Concordance (Jaccard top-%d): %.4f",
            _TOP_K, concordance,
        )
        logger.info("  DE genes (above median dev): %d", n_de_genes)

        metrics = {
            "concordance_with_ttest": float(concordance),
            "n_de_genes": float(n_de_genes),
            "total_log_likelihood": float(total_ll),
        }

        # 8. Build gradient check inputs (single sample)
        single_input = {
            "counts": counts_sub[0],
            "design": design[0],
            "size_factor": size_factors[0],
        }

        def loss_fn(
            model: DifferentiableNBGLM, d: dict[str, Any],
        ) -> jnp.ndarray:
            res, _, _ = model.apply(d, {}, None)
            return res["log_likelihood"]

        baselines = DE_BASELINES

        return {
            "metrics": metrics,
            "operator": operator,
            "input_data": single_input,
            "loss_fn": loss_fn,
            "n_items": n_cells,
            "iterate_fn": lambda: operator.batch_log_likelihood(
                counts_sub, design, size_factors,
            ),
            "baselines": baselines,
            "dataset_info": {
                "name": "immune_human",
                "n_cells": n_cells,
                "n_genes": n_genes,
                "type_a": type_a,
                "type_b": type_b,
            },
            "operator_config": {
                "n_features": n_genes,
                "n_covariates": 2,
            },
            "operator_name": "DifferentiableNBGLM",
            "dataset_name": "immune_human",
        }


def main() -> None:
    """CLI entry point."""
    DiffBioBenchmark.cli_main(
        DEBenchmark,
        _CONFIG,
        data_dir="/media/mahdi/ssd23/Data/scib",
    )


if __name__ == "__main__":
    main()
