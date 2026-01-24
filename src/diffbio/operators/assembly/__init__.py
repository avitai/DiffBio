"""Assembly operators for differentiable genome assembly.

This module provides graph neural network-based approaches to
assembly graph traversal that enable gradient flow through the
assembly process.

- GNNAssemblyNavigator: Message passing GNN for soft edge selection
- DifferentiableMetagenomicBinner: VAMB-style VAE for metagenomic binning
"""

from diffbio.operators.assembly.gnn_assembly import (
    GNNAssemblyNavigator,
    GNNAssemblyNavigatorConfig,
)
from diffbio.operators.assembly.metagenomic_binning import (
    DifferentiableMetagenomicBinner,
    MetagenomicBinnerConfig,
    create_metagenomic_binner,
)

__all__ = [
    "GNNAssemblyNavigator",
    "GNNAssemblyNavigatorConfig",
    "DifferentiableMetagenomicBinner",
    "MetagenomicBinnerConfig",
    "create_metagenomic_binner",
]
