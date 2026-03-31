"""Published protein secondary structure prediction baselines.

Values represent approximate Q3 accuracy on curated benchmark datasets,
drawn from the literature on DSSP and secondary structure assignment:

- Kabsch & Sander (1983): DSSP algorithm (exact, non-differentiable)
- Frishman & Argos (1995): STRIDE algorithm
- Martin et al. (2005): KAKSI consensus assignment
- Minami (2023): PyDSSP (simplified DSSP reimplementation)

These baselines measure secondary structure assignment accuracy using
Q3 (three-state accuracy: helix, strand, coil) on ideal backbone
geometries where the correct assignment is known a priori.

Note: On ideal structures, exact DSSP achieves near-perfect accuracy.
Differentiable approximations trade some accuracy for gradient flow.
"""

from __future__ import annotations

from calibrax.core.models import Metric, Point

_KABSCH_1983 = "Kabsch & Sander, Biopolymers 1983"
_FRISHMAN_1995 = "Frishman & Argos, Proteins 1995"
_MARTIN_2005 = "Martin et al., BMC Struct Biol 2005"
_MINAMI_2023 = "Minami, PyDSSP 2023"

SS_BASELINES: dict[str, Point] = {
    "DSSP (exact)": Point(
        name="DSSP (exact)",
        scenario="ideal_structures",
        tags={
            "framework": "DSSP",
            "algorithm": "Kabsch-Sander H-bond energy",
            "source": _KABSCH_1983,
        },
        metrics={
            "q3_overall": Metric(value=0.99),
            "q3_helix": Metric(value=0.99),
            "q3_strand": Metric(value=0.98),
            "q3_coil": Metric(value=0.99),
        },
    ),
    "STRIDE": Point(
        name="STRIDE",
        scenario="ideal_structures",
        tags={
            "framework": "STRIDE",
            "algorithm": "H-bond + dihedral angles",
            "source": _FRISHMAN_1995,
        },
        metrics={
            "q3_overall": Metric(value=0.97),
            "q3_helix": Metric(value=0.98),
            "q3_strand": Metric(value=0.96),
            "q3_coil": Metric(value=0.97),
        },
    ),
    "KAKSI": Point(
        name="KAKSI",
        scenario="ideal_structures",
        tags={
            "framework": "KAKSI",
            "algorithm": "Consensus of DSSP/STRIDE/DEFINE",
            "source": _MARTIN_2005,
        },
        metrics={
            "q3_overall": Metric(value=0.96),
            "q3_helix": Metric(value=0.97),
            "q3_strand": Metric(value=0.95),
            "q3_coil": Metric(value=0.96),
        },
    ),
    "PyDSSP": Point(
        name="PyDSSP",
        scenario="ideal_structures",
        tags={
            "framework": "PyDSSP",
            "algorithm": "Simplified DSSP (PyTorch)",
            "source": _MINAMI_2023,
        },
        metrics={
            "q3_overall": Metric(value=0.95),
            "q3_helix": Metric(value=0.96),
            "q3_strand": Metric(value=0.93),
            "q3_coil": Metric(value=0.95),
        },
    ),
}
