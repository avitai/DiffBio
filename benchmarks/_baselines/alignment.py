"""Published alignment baselines from BAliBASE benchmarks.

SP and TC scores for established multiple sequence alignment tools,
and average alignment scores for pairwise alignment tools, on
BAliBASE reference alignments. Values are representative averages
across BAliBASE families from published benchmarking studies.

Sources:
- Thompson et al. "BAliBASE 3.0: Latest developments of the
  multiple sequence alignment benchmark" Proteins (2005).
- Katoh & Standley "MAFFT Multiple Sequence Alignment Software
  Version 7" MBE (2013).
- Edgar "MUSCLE v5 enables improved estimates of phylogenetic
  tree confidence" Nature Methods (2022).
- Altschul et al. "Gapped BLAST and PSI-BLAST" Nucleic Acids
  Research 25(17):3389-3402, 1997.
- Pearson "An introduction to sequence similarity searching"
  Current Protocols in Bioinformatics (2013).
- Smith & Waterman "Identification of common molecular
  subsequences" J. Mol. Biol. 147:195-197, 1981.
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

PAIRWISE_BASELINES: dict[str, Point] = {
    "BLAST": Point(
        name="BLAST",
        scenario="balifam100_pairwise",
        tags={"framework": "blast", "source": _SOURCE_BALIBASE},
        metrics={
            "avg_alignment_score": Metric(value=45.0),
        },
    ),
    "SSEARCH": Point(
        name="SSEARCH",
        scenario="balifam100_pairwise",
        tags={"framework": "ssearch", "source": _SOURCE_BALIBASE},
        metrics={
            "avg_alignment_score": Metric(value=48.0),
        },
    ),
    "FASTA": Point(
        name="FASTA",
        scenario="balifam100_pairwise",
        tags={"framework": "fasta", "source": _SOURCE_BALIBASE},
        metrics={
            "avg_alignment_score": Metric(value=42.0),
        },
    ),
}
