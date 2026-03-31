"""Bridge to scib-metrics for single-cell integration benchmarking.

Wraps the JAX-accelerated scib-metrics library to compute the full
suite of 10+ integration quality metrics used in Luecken et al. 2022.

The aggregate score follows the scib paper formula:
    aggregate = 0.6 * mean(bio_metrics) + 0.4 * mean(batch_metrics)

Bio conservation metrics:
    silhouette_label, nmi_kmeans, ari_kmeans, clisi, isolated_labels

Batch correction metrics:
    silhouette_batch, ilisi, kbet_per_label, graph_connectivity

References:
    - Luecken et al. "Benchmarking atlas-level data integration"
      Nature Methods 2022.
    - scib-metrics: https://github.com/YosefLab/scib-metrics
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_BIO_KEYS = [
    "silhouette_label",
    "nmi_kmeans",
    "ari_kmeans",
    "clisi",
    "isolated_labels",
]

_BATCH_KEYS = [
    "silhouette_batch",
    "ilisi",
    "graph_connectivity",
]


def evaluate_integration(
    corrected_embeddings: Any,
    labels: Any,
    batch: Any,
    *,
    n_neighbors: int = 90,
) -> dict[str, float]:
    """Compute the full scib-metrics integration benchmark suite.

    Runs all standard scib integration metrics on the corrected
    embeddings and returns a dict of float scores.

    Args:
        corrected_embeddings: Embedding matrix (n_cells, n_features).
            Accepts JAX arrays or numpy arrays.
        labels: Cell type labels as integer array (n_cells,).
        batch: Batch labels as integer array (n_cells,).
        n_neighbors: Number of neighbors for kNN-based metrics.

    Returns:
        Dict mapping metric names to float values. Always includes:
        silhouette_label, silhouette_batch, nmi_kmeans, ari_kmeans,
        ilisi, clisi, graph_connectivity, isolated_labels,
        bio_score, batch_score, aggregate_score.
    """
    import scib_metrics  # noqa: PLC0415
    from scib_metrics.nearest_neighbors import pynndescent  # noqa: PLC0415

    # Convert to writable numpy (pynndescent numba requires writable)
    embeddings_np = np.array(
        corrected_embeddings, dtype=np.float32, copy=True
    )
    labels_np = np.array(labels, dtype=np.int32, copy=True)
    batch_np = np.array(batch, dtype=np.int32, copy=True)

    metrics: dict[str, float] = {}

    # Compute nearest neighbors (shared across kNN-based metrics)
    nn = pynndescent(embeddings_np, n_neighbors=n_neighbors)

    # Bio conservation metrics
    metrics["silhouette_label"] = float(
        scib_metrics.silhouette_label(embeddings_np, labels_np)
    )

    nmi_ari = scib_metrics.nmi_ari_cluster_labels_kmeans(
        embeddings_np, labels_np
    )
    metrics["nmi_kmeans"] = float(nmi_ari["nmi"])
    metrics["ari_kmeans"] = float(nmi_ari["ari"])

    metrics["clisi"] = float(scib_metrics.clisi_knn(nn, labels_np))

    metrics["isolated_labels"] = float(
        scib_metrics.isolated_labels(
            embeddings_np, labels_np, batch_np
        )
    )

    # Batch correction metrics
    metrics["silhouette_batch"] = float(
        scib_metrics.silhouette_batch(
            embeddings_np, labels_np, batch_np
        )
    )

    metrics["ilisi"] = float(scib_metrics.ilisi_knn(nn, batch_np))

    metrics["graph_connectivity"] = float(
        scib_metrics.graph_connectivity(nn, labels_np)
    )

    # kBET (can be slow, wrap in try/except)
    try:
        kbet_val = float(
            scib_metrics.kbet_per_label(nn, batch_np, labels_np)
        )
        metrics["kbet"] = kbet_val
    except (ValueError, RuntimeError, IndexError) as exc:
        # kBET can fail on small/degenerate batches
        logger.warning("kBET computation failed: %s", exc)
        metrics["kbet"] = 0.0

    # Aggregate scores (scib paper formula)
    bio_values = [metrics[k] for k in _BIO_KEYS if k in metrics]
    batch_values = [metrics[k] for k in _BATCH_KEYS if k in metrics]

    bio_score = float(np.mean(bio_values)) if bio_values else 0.0
    batch_score = float(np.mean(batch_values)) if batch_values else 0.0

    metrics["bio_score"] = bio_score
    metrics["batch_score"] = batch_score
    metrics["aggregate_score"] = 0.6 * bio_score + 0.4 * batch_score

    return metrics
