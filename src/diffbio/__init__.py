"""DiffBio: End-to-end differentiable bioinformatics pipelines built on Datarax.

This package provides differentiable bioinformatics pipeline components that
integrate with the Datarax framework for gradient-based optimization of
genomics workflows.

Key components:
- sequences: Biological sequence data types (DNA, RNA, Protein)
- operators: Differentiable bioinformatics operators (alignment, quality filtering)
- losses: Loss functions and biological regularization
- pipelines: Pre-built differentiable pipeline templates
- configs: Base configuration classes for operators
- constants: Centralized constants for the library
"""

from diffbio import configs, constants, losses, operators, pipelines, sequences, utils

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "configs",
    "constants",
    "losses",
    "operators",
    "pipelines",
    "sequences",
    "utils",
]
