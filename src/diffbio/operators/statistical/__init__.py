"""Differentiable statistical model operators.

This module provides operators for:
- Hidden Markov Models with differentiable forward algorithm
- Negative Binomial GLM for differential expression
- Unrolled EM for transcript quantification

All operators maintain gradient flow for end-to-end training.
"""

from diffbio.operators.statistical.em_quantification import (
    DifferentiableEMQuantifier,
    EMQuantifierConfig,
)
from diffbio.operators.statistical.hmm import (
    DifferentiableHMM,
    HMMConfig,
)
from diffbio.operators.statistical.nb_glm import (
    DifferentiableNBGLM,
    NBGLMConfig,
)

__all__ = [
    # HMM
    "HMMConfig",
    "DifferentiableHMM",
    # Negative Binomial GLM
    "NBGLMConfig",
    "DifferentiableNBGLM",
    # EM Quantification
    "EMQuantifierConfig",
    "DifferentiableEMQuantifier",
]
