"""Assembly operators for differentiable genome assembly.

This module provides graph neural network-based approaches to
assembly graph traversal that enable gradient flow through the
assembly process.

- GNNAssemblyNavigator: Message passing GNN for soft edge selection
"""

from diffbio.operators.assembly.gnn_assembly import (
    GNNAssemblyNavigator,
    GNNAssemblyNavigatorConfig,
)

__all__ = [
    "GNNAssemblyNavigator",
    "GNNAssemblyNavigatorConfig",
]
