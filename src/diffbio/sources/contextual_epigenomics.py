"""Shared source validation for contextual epigenomics workloads."""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from diffbio.sequences.dna import encode_dna_string

CONTEXTUAL_EPIGENOMICS_DATASET_CONTRACT_KEYS = (
    "sequence",
    "tf_context",
    "chromatin_contacts",
    "targets",
)
CONTEXTUAL_TARGET_SEMANTICS = (
    "binary_peak_mask",
    "chromatin_state_id",
)

_GENERIC_MOTIF = "ACGTAC"


def validate_contextual_epigenomics_dataset(data: dict[str, Any]) -> None:
    """Validate the shared contextual epigenomics benchmark contract."""
    missing_keys = [key for key in CONTEXTUAL_EPIGENOMICS_DATASET_CONTRACT_KEYS if key not in data]
    if missing_keys:
        raise ValueError(f"Contextual epigenomics data is missing required keys: {missing_keys}")

    sequence = jnp.asarray(data["sequence"], dtype=jnp.float32)
    tf_context = jnp.asarray(data["tf_context"], dtype=jnp.float32)
    chromatin_contacts = jnp.asarray(data["chromatin_contacts"], dtype=jnp.float32)
    targets = np.asarray(data["targets"])

    if sequence.ndim != 3 or sequence.shape[-1] != 4:
        raise ValueError("sequence must have shape (n_examples, sequence_length, 4).")
    if tf_context.ndim != 2:
        raise ValueError("tf_context must have shape (n_examples, n_tf_features).")
    if chromatin_contacts.ndim != 3:
        raise ValueError(
            "chromatin_contacts must have shape (n_examples, sequence_length, sequence_length)."
        )
    if targets.ndim != 2:
        raise ValueError("targets must have shape (n_examples, sequence_length).")

    n_examples = int(sequence.shape[0])
    sequence_length = int(sequence.shape[1])
    if (
        tf_context.shape[0] != n_examples
        or chromatin_contacts.shape[0] != n_examples
        or targets.shape[0] != n_examples
    ):
        raise ValueError(
            "Contextual epigenomics data keys must all share the same leading dimension."
        )

    if (
        chromatin_contacts.shape[1] != sequence_length
        or chromatin_contacts.shape[2] != sequence_length
    ):
        raise ValueError(
            "chromatin_contacts must align with sequence length and provide a square contact map."
        )
    if targets.shape[1] != sequence_length:
        raise ValueError("targets must have shape (n_examples, sequence_length).")

    sequence_mass = np.asarray(sequence.sum(axis=-1))
    if not np.allclose(sequence_mass, 1.0, atol=1e-5):
        raise ValueError(
            "sequence must be one-hot or probability-normalized along the alphabet axis."
        )

    contacts = np.asarray(chromatin_contacts)
    if not np.allclose(contacts, np.swapaxes(contacts, -1, -2), atol=1e-5):
        raise ValueError("chromatin_contacts must be symmetric.")


def build_synthetic_contextual_epigenomics_dataset(
    *,
    n_examples: int,
    sequence_length: int,
    num_tf_features: int,
    target_semantics: Literal["binary_peak_mask", "chromatin_state_id"],
    num_output_classes: int,
) -> dict[str, jnp.ndarray]:
    """Build a deterministic contextual epigenomics dataset."""
    if target_semantics not in CONTEXTUAL_TARGET_SEMANTICS:
        raise ValueError(
            "target_semantics must be one of "
            f"{CONTEXTUAL_TARGET_SEMANTICS}, got {target_semantics!r}."
        )

    sequences: list[str] = []
    tf_context_rows: list[np.ndarray] = []
    chromatin_contacts: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    motif_length = len(_GENERIC_MOTIF)
    peak_width = max(motif_length, 4)
    candidate_starts = _build_candidate_starts(
        sequence_length=sequence_length,
        num_tf_features=num_tf_features,
        peak_width=peak_width,
    )
    template_sequence = _build_sequence_template(
        sequence_length=sequence_length,
        candidate_starts=candidate_starts,
        motif=_GENERIC_MOTIF,
    )

    for example_index in range(n_examples):
        tf_index = example_index % num_tf_features
        region_start = candidate_starts[tf_index]
        region_end = min(region_start + peak_width, sequence_length)
        sequences.append(template_sequence)

        tf_row = np.zeros(num_tf_features, dtype=np.float32)
        tf_row[tf_index] = 1.0
        tf_context_rows.append(tf_row)

        contact_map = np.eye(sequence_length, dtype=np.float32) * 0.1
        contact_map[region_start:region_end, region_start:region_end] = 1.0
        flank_start = max(region_start - 1, 0)
        flank_end = min(region_end + 1, sequence_length)
        contact_map[flank_start:flank_end, flank_start:flank_end] = np.maximum(
            contact_map[flank_start:flank_end, flank_start:flank_end],
            0.5,
        )
        contact_map = np.maximum(contact_map, contact_map.T)
        chromatin_contacts.append(contact_map)

        target = np.zeros(sequence_length, dtype=np.int32)
        if target_semantics == "binary_peak_mask":
            target[region_start:region_end] = 1
        else:
            target[region_start:region_end] = 1
            right_flank_end = min(region_end + peak_width // 2, sequence_length)
            target[region_end:right_flank_end] = 2 % num_output_classes
        targets.append(target)

    one_hot_sequences = jnp.asarray(
        np.stack(
            [np.asarray(encode_dna_string(sequence), dtype=np.float32) for sequence in sequences]
        ),
        dtype=jnp.float32,
    )
    dataset = {
        "sequence": one_hot_sequences,
        "tf_context": jnp.asarray(np.stack(tf_context_rows), dtype=jnp.float32),
        "chromatin_contacts": jnp.asarray(np.stack(chromatin_contacts), dtype=jnp.float32),
        "targets": jnp.asarray(np.stack(targets)),
    }
    validate_contextual_epigenomics_dataset(dataset)
    return dataset


def _build_candidate_starts(
    *,
    sequence_length: int,
    num_tf_features: int,
    peak_width: int,
) -> list[int]:
    """Build deterministic candidate windows selected by TF context."""
    if sequence_length < peak_width + 2:
        return [0 for _ in range(num_tf_features)]

    max_start = max(sequence_length - peak_width - 1, 1)
    raw_starts = np.linspace(
        1,
        max_start,
        num=num_tf_features,
        dtype=np.int32,
    )
    return [int(start) for start in raw_starts]


def _build_sequence_template(
    *,
    sequence_length: int,
    candidate_starts: list[int],
    motif: str,
) -> str:
    """Build one shared sequence carrying several candidate regulatory windows."""
    sequence = ["A"] * sequence_length
    motif_chars = list(motif)
    motif_length = len(motif_chars)

    for start in candidate_starts:
        end = min(start + motif_length, sequence_length)
        sequence[start:end] = motif_chars[: end - start]

    return "".join(sequence)
