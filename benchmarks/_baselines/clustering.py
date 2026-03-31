"""Published clustering baselines for comparison tables.

Approximate ARI/NMI from scanpy/sklearn benchmarks on immune_human.
"""

from __future__ import annotations

from calibrax.core.models import Metric, Point

_SOURCE = "scanpy/sklearn benchmark literature"

CLUSTERING_BASELINES: dict[str, Point] = {
    "Leiden (scanpy)": Point(
        name="Leiden (scanpy)",
        scenario="immune_human",
        tags={"framework": "scanpy", "source": _SOURCE},
        metrics={
            "ari_kmeans": Metric(value=0.65),
            "nmi_kmeans": Metric(value=0.75),
        },
    ),
    "Louvain (scanpy)": Point(
        name="Louvain (scanpy)",
        scenario="immune_human",
        tags={"framework": "scanpy", "source": _SOURCE},
        metrics={
            "ari_kmeans": Metric(value=0.60),
            "nmi_kmeans": Metric(value=0.72),
        },
    ),
    "k-means (sklearn)": Point(
        name="k-means (sklearn)",
        scenario="immune_human",
        tags={"framework": "sklearn", "source": _SOURCE},
        metrics={
            "ari_kmeans": Metric(value=0.45),
            "nmi_kmeans": Metric(value=0.60),
        },
    ),
}
