"""Shared type aliases and enums for perturbation data loading.

Provides canonical string enums and type aliases used across the perturbation
sub-package: output space modes, mapping strategies, and cell/perturbation
label types.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TypeAlias

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

CellIndex: TypeAlias = int
PerturbationLabel: TypeAlias = str
CellTypeLabel: TypeAlias = str
BatchLabel: TypeAlias = str


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OutputSpaceMode(StrEnum):
    """Output representation mode for perturbation data.

    Attributes:
        GENE: Highly variable gene (HVG) subset of the count matrix.
        ALL: Full gene expression matrix.
        EMBEDDING: Pre-computed embedding only (no raw counts).
    """

    GENE = "gene"
    ALL = "all"
    EMBEDDING = "embedding"


class MappingStrategy(StrEnum):
    """Strategy for mapping perturbed cells to control cells.

    Attributes:
        BATCH: Map within same batch and cell type.
        RANDOM: Map to random control of same cell type.
    """

    BATCH = "batch"
    RANDOM = "random"
