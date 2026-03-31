"""Published trajectory inference baselines.

Approximate Spearman correlations from scVelo/Monocle/DPT benchmarks.
"""

from __future__ import annotations

from calibrax.core.models import Metric, Point

_SOURCE = "scVelo/Monocle/scanpy benchmark literature"

TRAJECTORY_BASELINES: dict[str, Point] = {
    "scVelo (dynamic)": Point(
        name="scVelo (dynamic)",
        scenario="pancreas",
        tags={"framework": "scvelo", "source": _SOURCE},
        metrics={"spearman_pancreas": Metric(value=0.85)},
    ),
    "DPT (scanpy)": Point(
        name="DPT (scanpy)",
        scenario="pancreas",
        tags={"framework": "scanpy", "source": _SOURCE},
        metrics={"spearman_pancreas": Metric(value=0.75)},
    ),
    "Monocle3": Point(
        name="Monocle3",
        scenario="pancreas",
        tags={"framework": "monocle3", "source": _SOURCE},
        metrics={"spearman_pancreas": Metric(value=0.80)},
    ),
}
