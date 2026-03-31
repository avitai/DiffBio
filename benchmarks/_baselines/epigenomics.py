"""Published ChIP-seq peak calling baselines.

Values represent approximate F1 scores on ENCODE ChIP-seq benchmarks,
drawn from comparative peak calling evaluations:

- Zhang et al. (2008): MACS (Model-based Analysis of ChIP-Seq)
- Heinz et al. (2010): HOMER (Hypergeometric Optimization of Motif
  EnRichment)
- Gaspar (2018): Genrich peak caller

These baselines reflect peak-level F1 scores when evaluated against
ENCODE IDR-filtered gold standard peaks. Exact numbers vary by
cell line, antibody, and evaluation protocol. Values below are
representative averages from published benchmarking studies.

References:
    Zhang et al. "Model-based Analysis of ChIP-Seq (MACS)"
    Genome Biol 9, R137 (2008).

    Heinz et al. "Simple Combinations of Lineage-Determining
    Transcription Factors Prime cis-Regulatory Elements Required
    for Macrophage and B Cell Identities." Mol Cell 38, 576-589
    (2010).

    Gaspar. "Genrich: detecting sites of genomic enrichment."
    https://github.com/jsh58/Genrich (2018).
"""

from __future__ import annotations

from calibrax.core.models import Metric, Point

_ZHANG_2008 = "Zhang et al., Genome Biol 2008"
_HEINZ_2010 = "Heinz et al., Mol Cell 2010"
_GASPAR_2018 = "Gaspar, Genrich 2018"

PEAK_CALLING_BASELINES: dict[str, Point] = {
    "MACS2": Point(
        name="MACS2",
        scenario="ENCODE_CTCF_K562",
        tags={
            "framework": "MACS2",
            "algorithm": "Shifting model + Poisson test",
            "source": _ZHANG_2008,
        },
        metrics={
            "f1": Metric(value=0.87),
            "precision": Metric(value=0.88),
            "recall": Metric(value=0.86),
            "jaccard": Metric(value=0.77),
        },
    ),
    "HOMER": Point(
        name="HOMER",
        scenario="ENCODE_CTCF_K562",
        tags={
            "framework": "HOMER",
            "algorithm": "Tag clustering + Poisson test",
            "source": _HEINZ_2010,
        },
        metrics={
            "f1": Metric(value=0.82),
            "precision": Metric(value=0.84),
            "recall": Metric(value=0.80),
            "jaccard": Metric(value=0.70),
        },
    ),
    "Genrich": Point(
        name="Genrich",
        scenario="ENCODE_CTCF_K562",
        tags={
            "framework": "Genrich",
            "algorithm": "ATAC-seq / ChIP-seq enrichment",
            "source": _GASPAR_2018,
        },
        metrics={
            "f1": Metric(value=0.89),
            "precision": Metric(value=0.90),
            "recall": Metric(value=0.88),
            "jaccard": Metric(value=0.80),
        },
    ),
}
