"""Published RNA secondary structure prediction baselines.

Values represent approximate F1 scores on the ArchiveII benchmark
dataset, drawn from multiple comparative studies:

- Mathews (2004): Comparison of RNA folding algorithms
- Lorenz et al. (2011): ViennaRNA Package 2.0
- Huang et al. (2019): LinearFold
- Do et al. (2006): CONTRAfold
- Wayment-Steele et al. (2022): EternaFold

These baselines measure structure prediction accuracy using base pair
F1 score (harmonic mean of sensitivity and PPV) on ArchiveII families.

Note: Exact numbers vary by study, RNA family, and evaluation protocol.
The values below are representative averages across published results.
"""

from __future__ import annotations

from calibrax.core.models import Metric, Point

_LORENZ_2011 = "Lorenz et al., Algorithms Mol Biol 2011"
_HUANG_2019 = "Huang et al., NeurIPS 2019"
_WAYMENT_2022 = "Wayment-Steele et al., bioRxiv 2022"
_DO_2006 = "Do et al., Bioinformatics 2006"

RNA_FOLD_BASELINES: dict[str, Point] = {
    "ViennaRNA": Point(
        name="ViennaRNA",
        scenario="archiveII",
        tags={
            "framework": "ViennaRNA",
            "algorithm": "MFE + partition function",
            "source": _LORENZ_2011,
        },
        metrics={
            "f1": Metric(value=0.72),
            "sensitivity": Metric(value=0.71),
            "ppv": Metric(value=0.73),
        },
    ),
    "LinearFold": Point(
        name="LinearFold",
        scenario="archiveII",
        tags={
            "framework": "LinearFold",
            "algorithm": "O(n) beam search CKY",
            "source": _HUANG_2019,
        },
        metrics={
            "f1": Metric(value=0.73),
            "sensitivity": Metric(value=0.72),
            "ppv": Metric(value=0.74),
        },
    ),
    "EternaFold": Point(
        name="EternaFold",
        scenario="archiveII",
        tags={
            "framework": "EternaFold",
            "algorithm": "Learned energy parameters",
            "source": _WAYMENT_2022,
        },
        metrics={
            "f1": Metric(value=0.75),
            "sensitivity": Metric(value=0.74),
            "ppv": Metric(value=0.76),
        },
    ),
    "CONTRAfold": Point(
        name="CONTRAfold",
        scenario="archiveII",
        tags={
            "framework": "CONTRAfold",
            "algorithm": "Conditional log-linear model",
            "source": _DO_2006,
        },
        metrics={
            "f1": Metric(value=0.70),
            "sensitivity": Metric(value=0.69),
            "ppv": Metric(value=0.71),
        },
    ),
}
