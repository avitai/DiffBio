"""Published differential expression baselines.

Concordance values represent the typical Jaccard overlap of each
method's top-50 DE gene list against a consensus ground truth,
compiled from the conquer benchmark datasets (Soneson & Robinson,
Nature Methods 2018) and supplementary benchmarks in Squair et al.,
Nature Communications 2021.

Source:
- Soneson & Robinson "Bias, robustness and scalability in
  single-cell differential expression analysis" Nature Methods
  15, 255-261 (2018).
- Squair et al. "Confronting false discoveries in single-cell
  differential expression" Nature Communications 12, 5692 (2021).
"""

from __future__ import annotations

from calibrax.core.models import Metric, Point

_SOURCE = "Soneson & Robinson, Nature Methods 2018"

DE_BASELINES: dict[str, Point] = {
    "DESeq2": Point(
        name="DESeq2",
        scenario="immune_human_de",
        tags={"framework": "deseq2", "source": _SOURCE},
        metrics={
            "concordance_with_ttest": Metric(value=0.70),
        },
    ),
    "edgeR": Point(
        name="edgeR",
        scenario="immune_human_de",
        tags={"framework": "edger", "source": _SOURCE},
        metrics={
            "concordance_with_ttest": Metric(value=0.65),
        },
    ),
    "Wilcoxon": Point(
        name="Wilcoxon",
        scenario="immune_human_de",
        tags={"framework": "scanpy", "source": _SOURCE},
        metrics={
            "concordance_with_ttest": Metric(value=0.60),
        },
    ),
}
