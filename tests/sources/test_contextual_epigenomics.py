"""Tests for the shared contextual epigenomics data contract."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from diffbio.sources.contextual_epigenomics import (
    CONTEXTUAL_EPIGENOMICS_DATASET_CONTRACT_KEYS,
    validate_contextual_epigenomics_dataset,
)


def _make_valid_dataset() -> dict[str, jnp.ndarray]:
    sequence = jnp.asarray(
        np.array(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
            ],
            dtype=np.float32,
        )
    )
    tf_context = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    chromatin_contacts = jnp.asarray(
        np.array(
            [
                [
                    [1.0, 0.4, 0.1, 0.0],
                    [0.4, 1.0, 0.3, 0.1],
                    [0.1, 0.3, 1.0, 0.2],
                    [0.0, 0.1, 0.2, 1.0],
                ],
                [
                    [1.0, 0.2, 0.1, 0.0],
                    [0.2, 1.0, 0.4, 0.2],
                    [0.1, 0.4, 1.0, 0.5],
                    [0.0, 0.2, 0.5, 1.0],
                ],
            ],
            dtype=np.float32,
        )
    )
    targets = jnp.asarray([[0, 1, 1, 0], [2, 2, 1, 0]], dtype=jnp.int32)
    return {
        "sequence": sequence,
        "tf_context": tf_context,
        "chromatin_contacts": chromatin_contacts,
        "targets": targets,
    }


class TestValidateContextualEpigenomicsDataset:
    """Tests for the shared contextual epigenomics validation helper."""

    def test_contract_keys_are_stable(self) -> None:
        assert CONTEXTUAL_EPIGENOMICS_DATASET_CONTRACT_KEYS == (
            "sequence",
            "tf_context",
            "chromatin_contacts",
            "targets",
        )

    def test_accepts_valid_payload(self) -> None:
        validate_contextual_epigenomics_dataset(_make_valid_dataset())

    def test_rejects_non_binary_peak_targets_when_semantics_are_binary(self) -> None:
        data = _make_valid_dataset()
        data["targets"] = jnp.asarray([[0, 1, 2, 0], [1, 0, 1, 0]], dtype=jnp.int32)

        with pytest.raises(ValueError, match="binary_peak_mask"):
            validate_contextual_epigenomics_dataset(
                data,
                target_semantics="binary_peak_mask",
                num_output_classes=1,
            )

    def test_rejects_chromatin_state_targets_outside_class_range(self) -> None:
        data = _make_valid_dataset()
        data["targets"] = jnp.asarray([[0, 1, 2, 3], [1, 0, 1, 0]], dtype=jnp.int32)

        with pytest.raises(ValueError, match="chromatin_state_id"):
            validate_contextual_epigenomics_dataset(
                data,
                target_semantics="chromatin_state_id",
                num_output_classes=3,
            )

    def test_rejects_missing_required_keys(self) -> None:
        data = _make_valid_dataset()
        del data["tf_context"]

        with pytest.raises(ValueError, match="missing required keys"):
            validate_contextual_epigenomics_dataset(data)

    def test_rejects_leading_dimension_mismatch(self) -> None:
        data = _make_valid_dataset()
        data["tf_context"] = data["tf_context"][:1]

        with pytest.raises(ValueError, match="same leading dimension"):
            validate_contextual_epigenomics_dataset(data)

    def test_rejects_non_one_hot_sequence_shape(self) -> None:
        data = _make_valid_dataset()
        data["sequence"] = jnp.ones((2, 4, 3), dtype=jnp.float32)

        with pytest.raises(ValueError, match="shape \\(n_examples, sequence_length, 4\\)"):
            validate_contextual_epigenomics_dataset(data)

    def test_rejects_non_square_contact_maps(self) -> None:
        data = _make_valid_dataset()
        data["chromatin_contacts"] = jnp.ones((2, 4, 3), dtype=jnp.float32)

        with pytest.raises(ValueError, match="square contact map"):
            validate_contextual_epigenomics_dataset(data)

    def test_rejects_non_symmetric_contact_maps(self) -> None:
        data = _make_valid_dataset()
        contacts = np.asarray(data["chromatin_contacts"]).copy()
        contacts[0, 0, 1] = 0.9
        contacts[0, 1, 0] = 0.1
        data["chromatin_contacts"] = jnp.asarray(contacts, dtype=jnp.float32)

        with pytest.raises(ValueError, match="symmetric"):
            validate_contextual_epigenomics_dataset(data)

    def test_rejects_target_shape_mismatch(self) -> None:
        data = _make_valid_dataset()
        data["targets"] = jnp.ones((2, 3), dtype=jnp.int32)

        with pytest.raises(ValueError, match="targets must have shape"):
            validate_contextual_epigenomics_dataset(data)
