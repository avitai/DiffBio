"""Biological sequence data types for DiffBio.

This module provides JAX-compatible data types for representing biological
sequences (DNA, RNA, Protein) that integrate with the Datarax Element system.
"""

from diffbio.sequences.dna import (
    DNA_ALPHABET,
    DNA_ALPHABET_SIZE,
    complement_dna,
    create_dna_element_data,
    decode_dna_onehot,
    encode_dna_string,
    gc_content,
    phred_to_probability,
    probability_to_phred,
    reverse_complement_dna,
    soft_encode_dna,
)


__all__ = [
    "DNA_ALPHABET",
    "DNA_ALPHABET_SIZE",
    "complement_dna",
    "create_dna_element_data",
    "decode_dna_onehot",
    "encode_dna_string",
    "gc_content",
    "phred_to_probability",
    "probability_to_phred",
    "reverse_complement_dna",
    "soft_encode_dna",
]
