"""Published MoleculeNet baselines for molecular property prediction.

Values are from representative publications on the MoleculeNet benchmark
suite (Wu et al. 2018). BBBP ROC-AUC scores come from:

- GCN: Kipf & Welling, ICLR 2017 (reported on MoleculeNet by Wu et al.)
- AttentiveFP: Xiong et al., J. Med. Chem. 2020
- D-MPNN: Yang et al., J. Chem. Inf. Model. 2019

Each baseline is a calibrax Point for comparison via rank_table().
"""

from __future__ import annotations

from calibrax.core.models import Metric, Point

_SOURCE_GCN = "Wu et al., Chemical Science 2018"
_SOURCE_AFP = "Xiong et al., J. Med. Chem. 2020"
_SOURCE_DMPNN = "Yang et al., J. Chem. Inf. Model. 2019"

MOLNET_BASELINES: dict[str, dict[str, Point]] = {
    "bbbp": {
        "GCN": Point(
            name="GCN",
            scenario="bbbp",
            tags={"framework": "pytorch-geometric", "source": _SOURCE_GCN},
            metrics={
                "test_roc_auc": Metric(value=0.877),
            },
        ),
        "AttentiveFP": Point(
            name="AttentiveFP",
            scenario="bbbp",
            tags={"framework": "dgl", "source": _SOURCE_AFP},
            metrics={
                "test_roc_auc": Metric(value=0.858),
            },
        ),
        "D-MPNN": Point(
            name="D-MPNN",
            scenario="bbbp",
            tags={"framework": "chemprop", "source": _SOURCE_DMPNN},
            metrics={
                "test_roc_auc": Metric(value=0.910),
            },
        ),
    },
}
