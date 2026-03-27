"""DiffBio splitters module.

This module provides dataset splitting utilities extending Datarax's StructuralModule
for train/validation/test splitting in bioinformatics and drug discovery applications.

Splitters:
    SplitterModule: Base class for all splitters
    SplitResult: NamedTuple containing split indices
    RandomSplitter: Simple random splitting
    StratifiedSplitter: Stratified splitting preserving class distribution
    ScaffoldSplitter: Molecular scaffold-based splitting for drug discovery
    TanimotoClusterSplitter: Fingerprint similarity clustering for drug discovery
    SequenceIdentitySplitter: Sequence identity clustering for bioinformatics
"""

from diffbio.splitters.base import SplitResult, SplitterConfig, SplitterModule
from diffbio.splitters.molecular import (
    ScaffoldSplitter,
    ScaffoldSplitterConfig,
    TanimotoClusterSplitter,
    TanimotoClusterSplitterConfig,
)
from diffbio.splitters.random import (
    RandomSplitter,
    RandomSplitterConfig,
    StratifiedSplitter,
    StratifiedSplitterConfig,
)
from diffbio.splitters.perturbation import (
    FewShotSplitter,
    FewShotSplitterConfig,
    ZeroShotSplitter,
    ZeroShotSplitterConfig,
)
from diffbio.splitters.sequence import (
    SequenceIdentitySplitter,
    SequenceIdentitySplitterConfig,
)

__all__ = [
    # Base classes
    "SplitterModule",
    "SplitterConfig",
    "SplitResult",
    # Random splitters
    "RandomSplitter",
    "RandomSplitterConfig",
    "StratifiedSplitter",
    "StratifiedSplitterConfig",
    # Molecular splitters
    "ScaffoldSplitter",
    "ScaffoldSplitterConfig",
    "TanimotoClusterSplitter",
    "TanimotoClusterSplitterConfig",
    # Sequence splitters
    "SequenceIdentitySplitter",
    "SequenceIdentitySplitterConfig",
    # Perturbation splitters
    "FewShotSplitter",
    "FewShotSplitterConfig",
    "ZeroShotSplitter",
    "ZeroShotSplitterConfig",
]
