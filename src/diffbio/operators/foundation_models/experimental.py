"""Experimental foundation-model boundary policy.

This module is a policy namespace, not a stable implementation surface. It
keeps speculative foundation-model capabilities explicit until benchmarks,
provenance, regression guards, and docs promote them into stable support.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from diffbio.operators.foundation_models.contracts import _validate_ascii_text

EXPERIMENTAL_FOUNDATION_MODEL_NAMESPACE = "diffbio.operators.foundation_models.experimental"
FOUNDATION_EXPERIMENTAL_PROMOTION_CRITERIA = (
    "explicit experimental namespace",
    "canonical artifact provenance",
    "downstream benchmark suite",
    "Calibrax regression guard",
    "shared audit bundle",
    "stable documentation update",
)


@dataclass(frozen=True)
class ExperimentalFoundationCapability:
    """Fail-closed policy record for speculative foundation-model scope."""

    key: str
    display_name: str
    stable_exclusion: str
    namespace: str = EXPERIMENTAL_FOUNDATION_MODEL_NAMESPACE
    stable_support: bool = False
    benchmark_status: str = "benchmark_unverified"
    required_promotion_criteria: tuple[str, ...] = FOUNDATION_EXPERIMENTAL_PROMOTION_CRITERIA

    def __post_init__(self) -> None:
        """Validate the policy record cannot silently become stable."""
        for field_name, value in (
            ("key", self.key),
            ("display_name", self.display_name),
            ("stable_exclusion", self.stable_exclusion),
            ("namespace", self.namespace),
            ("benchmark_status", self.benchmark_status),
        ):
            _validate_ascii_text(value, field_name=field_name)
        if self.namespace != EXPERIMENTAL_FOUNDATION_MODEL_NAMESPACE:
            raise ValueError(
                "experimental foundation capabilities must use the experimental namespace."
            )
        if self.stable_support:
            raise ValueError("experimental foundation capabilities cannot be stable.")
        if self.required_promotion_criteria != FOUNDATION_EXPERIMENTAL_PROMOTION_CRITERIA:
            raise ValueError(
                "experimental foundation capabilities must use the shared promotion criteria."
            )


_FOUNDATION_EXPERIMENTAL_CAPABILITIES = {
    "long_context_sequence_models": ExperimentalFoundationCapability(
        key="long_context_sequence_models",
        display_name="Long-context sequence models",
        stable_exclusion=(
            "No stable long-context sequence model support until promotion criteria are satisfied."
        ),
    ),
    "hyena_style_sequence_models": ExperimentalFoundationCapability(
        key="hyena_style_sequence_models",
        display_name="Hyena-style sequence models",
        stable_exclusion=(
            "No stable Hyena-style runtime support until promotion criteria are satisfied."
        ),
    ),
    "external_native_trainable_checkpoint_import": ExperimentalFoundationCapability(
        key="external_native_trainable_checkpoint_import",
        display_name="External native_trainable checkpoint import",
        stable_exclusion=(
            "The stable native_trainable adapter mode only covers DiffBio-native "
            "operators, not external checkpoint conversion."
        ),
    ),
    "peft_finetuning": ExperimentalFoundationCapability(
        key="peft_finetuning",
        display_name="PEFT fine-tuning utilities",
        stable_exclusion=(
            "No stable PEFT utilities are shipped until promotion criteria are satisfied."
        ),
    ),
    "lora_adaptation": ExperimentalFoundationCapability(
        key="lora_adaptation",
        display_name="LoRA adaptation utilities",
        stable_exclusion=(
            "No stable LoRA adaptation utilities are shipped until promotion "
            "criteria are satisfied."
        ),
    ),
}
FOUNDATION_EXPERIMENTAL_CAPABILITIES: Mapping[
    str,
    ExperimentalFoundationCapability,
] = MappingProxyType(_FOUNDATION_EXPERIMENTAL_CAPABILITIES)


def is_experimental_foundation_capability(key: str) -> bool:
    """Return whether a capability is explicitly fenced as experimental."""
    return key in FOUNDATION_EXPERIMENTAL_CAPABILITIES


def get_experimental_foundation_capability(
    key: str,
) -> ExperimentalFoundationCapability:
    """Return the experimental policy record for a capability key."""
    try:
        return FOUNDATION_EXPERIMENTAL_CAPABILITIES[key]
    except KeyError as exc:
        raise KeyError(f"No experimental foundation capability registered for {key!r}.") from exc


__all__ = [
    "EXPERIMENTAL_FOUNDATION_MODEL_NAMESPACE",
    "FOUNDATION_EXPERIMENTAL_CAPABILITIES",
    "FOUNDATION_EXPERIMENTAL_PROMOTION_CRITERIA",
    "ExperimentalFoundationCapability",
    "get_experimental_foundation_capability",
    "is_experimental_foundation_capability",
]
