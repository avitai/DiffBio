"""Published deconvolution baselines from spotless-benchmark.

Values are from the spotless-benchmark study (Li et al., Genome Biology
2023) and individual method publications. These represent SOTA for
spatial transcriptomics cell type deconvolution.

Metrics:
    - pearson_correlation: Spot-level Pearson r between predicted and
      true cell type proportions (averaged across spots).
    - rmse: Root mean squared error of predicted proportions.

Sources:
    - RCTD: Cable et al., Nature Biotechnology 2022.
    - Cell2location: Kleshchevnikov et al., Nature Biotechnology 2022.
    - CARD: Ma & Zhou, Nature Biotechnology 2022.
"""

from __future__ import annotations

from calibrax.core.models import Metric, Point

_SOURCE_SPOTLESS = "Li et al., Genome Biology 2023 (spotless-benchmark)"

DECONVOLUTION_BASELINES: dict[str, Point] = {
    "RCTD": Point(
        name="RCTD",
        scenario="seqfish_cortex",
        tags={
            "framework": "spacexr",
            "source": _SOURCE_SPOTLESS,
        },
        metrics={
            "pearson_correlation": Metric(value=0.85),
            "rmse": Metric(value=0.08),
        },
    ),
    "Cell2location": Point(
        name="Cell2location",
        scenario="seqfish_cortex",
        tags={
            "framework": "cell2location",
            "source": _SOURCE_SPOTLESS,
        },
        metrics={
            "pearson_correlation": Metric(value=0.88),
            "rmse": Metric(value=0.06),
        },
    ),
    "CARD": Point(
        name="CARD",
        scenario="seqfish_cortex",
        tags={
            "framework": "CARD",
            "source": _SOURCE_SPOTLESS,
        },
        metrics={
            "pearson_correlation": Metric(value=0.86),
            "rmse": Metric(value=0.07),
        },
    ),
}
