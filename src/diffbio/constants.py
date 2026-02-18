"""Centralized constants for DiffBio.

This module provides centralized constants used across operators, pipelines,
and utilities. Using these constants ensures consistency and makes maintenance
easier.
"""

from enum import StrEnum

# =============================================================================
# Neural Network Architecture Defaults
# =============================================================================

DEFAULT_HIDDEN_DIM = 64
"""Default hidden layer dimension for MLP architectures."""

DEFAULT_HIDDEN_DIM_LARGE = 128
"""Larger hidden dimension for complex models."""

DEFAULT_EMBEDDING_DIM = 64
"""Default embedding dimension for sequence embeddings."""

DEFAULT_DROPOUT_RATE = 0.1
"""Default dropout rate for regularization."""

DEFAULT_NUM_LAYERS = 2
"""Default number of hidden layers in MLPs."""

DEFAULT_TEMPERATURE = 1.0
"""Default temperature for logsumexp smoothing and softmax."""


# =============================================================================
# Quality Score Constants (Phred Scale)
# =============================================================================

PHRED_QUALITY_MAX = 40.0
"""Maximum Phred quality score (99.99% accuracy)."""

PHRED_QUALITY_THRESHOLD = 20.0
"""Default Phred quality threshold (99% accuracy, 1% error rate)."""

PHRED_QUALITY_MIN = 0.0
"""Minimum Phred quality score."""


# =============================================================================
# Alignment Constants
# =============================================================================

DEFAULT_GAP_OPEN = -10.0
"""Default gap opening penalty for alignment."""

DEFAULT_GAP_EXTEND = -1.0
"""Default gap extension penalty for alignment."""

DNA_MATCH_SCORE = 2.0
"""Default match score for DNA alignment."""

DNA_MISMATCH_SCORE = -1.0
"""Default mismatch penalty for DNA alignment."""


# =============================================================================
# Numerical Stability Constants
# =============================================================================

EPSILON = 1e-8
"""Small value for numerical stability in divisions and log operations."""

EPSILON_LOG = 1e-10
"""Smaller epsilon specifically for log operations to prevent -inf."""


# =============================================================================
# Pileup and Coverage Constants
# =============================================================================

DEFAULT_PILEUP_WINDOW_SIZE = 21
"""Default window size for pileup context."""

DEFAULT_MIN_COVERAGE = 1
"""Default minimum coverage threshold."""

DEFAULT_MAX_COVERAGE = 100
"""Default maximum coverage for normalization."""


# =============================================================================
# Variant Calling Constants
# =============================================================================

DEFAULT_NUM_CLASSES = 3
"""Default number of variant classes (REF, SNV, INDEL)."""


class ClassifierType(StrEnum):
    """Classifier type for variant calling pipelines."""

    MLP = "mlp"
    """Multi-layer perceptron classifier."""

    CNN = "cnn"
    """Convolutional neural network classifier."""


# =============================================================================
# DNA/RNA Constants
# =============================================================================

DNA_ALPHABET_SIZE = 4
"""Size of DNA alphabet (A, C, G, T)."""

RNA_ALPHABET_SIZE = 4
"""Size of RNA alphabet (A, C, G, U)."""

PROTEIN_ALPHABET_SIZE = 20
"""Size of standard protein alphabet."""


# =============================================================================
# VAE / Latent Space Constants
# =============================================================================

DEFAULT_LATENT_DIM = 10
"""Default latent dimension for VAE models."""

DEFAULT_BETA_VAE = 1.0
"""Default beta parameter for beta-VAE (KL weight)."""


# =============================================================================
# Graph Neural Network Constants
# =============================================================================

DEFAULT_NODE_FEATURES = 32
"""Default node feature dimension for GNN."""

DEFAULT_EDGE_FEATURES = 8
"""Default edge feature dimension for GNN."""

DEFAULT_NUM_HEADS = 4
"""Default number of attention heads."""


# =============================================================================
# HMM Constants
# =============================================================================

DEFAULT_HMM_STATES = 3
"""Default number of hidden states for HMM."""

DEFAULT_HMM_EMISSIONS = 4
"""Default number of emissions for HMM (matches DNA alphabet)."""


# =============================================================================
# Sequence Length Limits
# =============================================================================

DEFAULT_MAX_SEQ_LENGTH = 1000
"""Default maximum sequence length."""

DEFAULT_MAX_ALIGNMENT_LENGTH = 500
"""Default maximum alignment length."""


# =============================================================================
# Data Dictionary Keys
# =============================================================================
# These constants ensure consistent key naming across operators


class DataKeys:
    """Standard keys for data dictionaries in DiffBio operators."""

    # Sequence data
    SEQUENCE = "sequence"
    SEQ1 = "seq1"
    SEQ2 = "seq2"
    READS = "reads"

    # Quality and positions
    QUALITY = "quality"
    QUALITY_SCORES = "quality_scores"
    POSITIONS = "positions"

    # Pileup related
    PILEUP = "pileup"
    PILEUP_WINDOW = "pileup_window"
    COVERAGE = "coverage"
    MEAN_QUALITY = "mean_quality"

    # Classification outputs
    LOGITS = "logits"
    PROBABILITIES = "probabilities"

    # Alignment outputs
    SCORE = "score"
    ALIGNMENT_MATRIX = "alignment_matrix"
    SOFT_ALIGNMENT = "soft_alignment"
