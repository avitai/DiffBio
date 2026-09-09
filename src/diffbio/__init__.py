"""DiffBio: End-to-end differentiable bioinformatics pipelines built on the wider JAX/NNX ecosystem.

This package provides differentiable bioinformatics pipeline components that
integrate with Datarax, Artifex, Opifex, Calibrax and Substrax for gradient-based
optimization of genomics workflows.

Key components:
- sequences: Biological sequence data types (DNA, RNA, Protein)
- operators: Differentiable bioinformatics operators (alignment, quality filtering)
- losses: Loss functions and biological regularization
- pipelines: Pre-built differentiable pipeline templates
- configs: Base configuration classes for operators
- constants: Centralized constants for the library
"""

import importlib.metadata

from diffbio import (
    configs,
    constants,
    evaluation,
    losses,
    operators,
    pipelines,
    sequences,
    utils,
)

__version__ = importlib.metadata.version("diffbio")

__all__ = [
    "__version__",
    "configs",
    "constants",
    "evaluation",
    "losses",
    "operators",
    "pipelines",
    "sequences",
    "utils",
]
