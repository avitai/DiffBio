"""Published MSA baselines from BAliBASE benchmarks.

SP and TC scores for established multiple sequence alignment tools
on BAliBASE reference alignments. Values are representative averages
across BAliBASE families from published benchmarking studies.

Sources:
- Thompson et al. "BAliBASE 3.0: Latest developments of the
  multiple sequence alignment benchmark" Proteins (2005).
- Katoh & Standley "MAFFT Multiple Sequence Alignment Software
  Version 7" MBE (2013).
- Edgar "MUSCLE v5 enables improved estimates of phylogenetic
  tree confidence" Nature Methods (2022).
"""

from __future__ import annotations

from calibrax.core.models import Metric, Point

_SOURCE_BALIBASE = "BAliBASE benchmark literature"

MSA_BASELINES: dict[str, Point] = {
    "MAFFT": Point(
        name="MAFFT",
        scenario="balifam100",
        tags={"framework": "mafft", "source": _SOURCE_BALIBASE},
        metrics={
            "sp_score": Metric(value=0.85),
            "tc_score": Metric(value=0.56),
        },
    ),
    "ClustalW": Point(
        name="ClustalW",
        scenario="balifam100",
        tags={"framework": "clustalw", "source": _SOURCE_BALIBASE},
        metrics={
            "sp_score": Metric(value=0.75),
            "tc_score": Metric(value=0.42),
        },
    ),
    "MUSCLE": Point(
        name="MUSCLE",
        scenario="balifam100",
        tags={"framework": "muscle", "source": _SOURCE_BALIBASE},
        metrics={
            "sp_score": Metric(value=0.82),
            "tc_score": Metric(value=0.52),
        },
    ),
    "T-Coffee": Point(
        name="T-Coffee",
        scenario="balifam100",
        tags={"framework": "tcoffee", "source": _SOURCE_BALIBASE},
        metrics={
            "sp_score": Metric(value=0.84),
            "tc_score": Metric(value=0.55),
        },
    ),
}
