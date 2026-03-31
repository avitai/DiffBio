"""Published GRN inference baselines.

Approximate AUPRC from benGRN/Beeline benchmark papers.
"""

from __future__ import annotations

from calibrax.core.models import Metric, Point

_SOURCE = "benGRN/Beeline benchmark literature"

GRN_BASELINES: dict[str, Point] = {
    "GENIE3": Point(
        name="GENIE3",
        scenario="mesc_chip_perturb",
        tags={"framework": "genie3", "source": _SOURCE},
        metrics={"auprc": Metric(value=0.08)},
    ),
    "GRNBoost2": Point(
        name="GRNBoost2",
        scenario="mesc_chip_perturb",
        tags={"framework": "grnboost2", "source": _SOURCE},
        metrics={"auprc": Metric(value=0.07)},
    ),
    "pySCENIC": Point(
        name="pySCENIC",
        scenario="mesc_chip_perturb",
        tags={"framework": "pyscenic", "source": _SOURCE},
        metrics={"auprc": Metric(value=0.06)},
    ),
    "Random": Point(
        name="Random",
        scenario="mesc_chip_perturb",
        tags={"framework": "random", "source": _SOURCE},
        metrics={"auprc": Metric(value=0.02)},
    ),
}
