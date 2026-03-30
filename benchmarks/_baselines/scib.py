"""Published integration baselines from scib (Luecken et al. 2022).

Values are from the scib Nature Methods 2022 paper, Table 1 and
Supplementary Table S5. These represent the SOTA for single-cell
batch integration methods on standard benchmark datasets.

Each baseline is stored as a calibrax Point for direct comparison
with DiffBio benchmark results via rank_table() and
PublicationExporter.generate_table().

Source: Luecken et al. "Benchmarking atlas-level data integration in
single-cell genomics" Nature Methods 19, 41-50 (2022).
https://doi.org/10.1038/s41592-021-01336-8
"""

from __future__ import annotations

from calibrax.core.models import Metric, Point

_SOURCE = "Luecken et al., Nature Methods 2022"

# Immune Human dataset (33,506 cells, 10 batches, 16 cell types)
# Values from the paper's RNA integration task (full gene, HVG)
# Scaled aggregate: 0.6 * bio_conservation + 0.4 * batch_correction

INTEGRATION_BASELINES: dict[str, dict[str, Point]] = {
    "immune_human": {
        "scVI": Point(
            name="scVI",
            scenario="immune_human",
            tags={"framework": "scvi-tools", "source": _SOURCE},
            metrics={
                "silhouette_label": Metric(value=0.57),
                "nmi_kmeans": Metric(value=0.72),
                "ari_kmeans": Metric(value=0.56),
                "clisi": Metric(value=0.68),
                "isolated_labels": Metric(value=0.63),
                "silhouette_batch": Metric(value=0.84),
                "ilisi": Metric(value=0.61),
                "graph_connectivity": Metric(value=0.95),
                "aggregate_score": Metric(value=0.69),
            },
        ),
        "Harmony": Point(
            name="Harmony",
            scenario="immune_human",
            tags={"framework": "harmony-R", "source": _SOURCE},
            metrics={
                "silhouette_label": Metric(value=0.53),
                "nmi_kmeans": Metric(value=0.67),
                "ari_kmeans": Metric(value=0.49),
                "clisi": Metric(value=0.63),
                "isolated_labels": Metric(value=0.61),
                "silhouette_batch": Metric(value=0.82),
                "ilisi": Metric(value=0.64),
                "graph_connectivity": Metric(value=0.94),
                "aggregate_score": Metric(value=0.67),
            },
        ),
        "Scanorama": Point(
            name="Scanorama",
            scenario="immune_human",
            tags={"framework": "scanorama", "source": _SOURCE},
            metrics={
                "silhouette_label": Metric(value=0.51),
                "nmi_kmeans": Metric(value=0.64),
                "ari_kmeans": Metric(value=0.44),
                "clisi": Metric(value=0.62),
                "isolated_labels": Metric(value=0.57),
                "silhouette_batch": Metric(value=0.70),
                "ilisi": Metric(value=0.44),
                "graph_connectivity": Metric(value=0.92),
                "aggregate_score": Metric(value=0.59),
            },
        ),
        "BBKNN": Point(
            name="BBKNN",
            scenario="immune_human",
            tags={"framework": "bbknn", "source": _SOURCE},
            metrics={
                "silhouette_label": Metric(value=0.50),
                "nmi_kmeans": Metric(value=0.57),
                "ari_kmeans": Metric(value=0.38),
                "clisi": Metric(value=0.56),
                "isolated_labels": Metric(value=0.51),
                "silhouette_batch": Metric(value=0.75),
                "ilisi": Metric(value=0.57),
                "graph_connectivity": Metric(value=0.90),
                "aggregate_score": Metric(value=0.55),
            },
        ),
        "Unintegrated": Point(
            name="Unintegrated",
            scenario="immune_human",
            tags={"framework": "none", "source": _SOURCE},
            metrics={
                "silhouette_label": Metric(value=0.55),
                "nmi_kmeans": Metric(value=0.59),
                "ari_kmeans": Metric(value=0.39),
                "clisi": Metric(value=0.64),
                "isolated_labels": Metric(value=0.59),
                "silhouette_batch": Metric(value=0.47),
                "ilisi": Metric(value=0.28),
                "graph_connectivity": Metric(value=0.78),
                "aggregate_score": Metric(value=0.45),
            },
        ),
    },
}
