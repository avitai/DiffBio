"""Tests for experimental foundation-model boundary policy."""

from __future__ import annotations

import re
from pathlib import Path

from diffbio.operators.foundation_models import (
    EXPERIMENTAL_FOUNDATION_MODEL_NAMESPACE,
    FOUNDATION_EXPERIMENTAL_CAPABILITIES,
    FOUNDATION_EXPERIMENTAL_PROMOTION_CRITERIA,
    get_experimental_foundation_capability,
    is_experimental_foundation_capability,
)

ROOT = Path(__file__).resolve().parents[3]
FOUNDATION_DOC = ROOT / "docs/user-guide/operators/foundation-models.md"
EXPERIMENTAL_SOURCE = ROOT / "src/diffbio/operators/foundation_models/experimental.py"

EXPERIMENTAL_SCOPE_STATEMENT = (
    "Experimental foundation-model boundary: long-context sequence models, "
    "Hyena-style systems, external native_trainable checkpoint import, PEFT, "
    "and LoRA are not stable DiffBio support."
)
EXPECTED_EXPERIMENTAL_CAPABILITIES = {
    "long_context_sequence_models",
    "hyena_style_sequence_models",
    "external_native_trainable_checkpoint_import",
    "peft_finetuning",
    "lora_adaptation",
}
EXPECTED_PROMOTION_CRITERIA = (
    "explicit experimental namespace",
    "canonical artifact provenance",
    "downstream benchmark suite",
    "Calibrax regression guard",
    "shared audit bundle",
    "stable documentation update",
)
TERM_ALLOWED_PATHS = {
    "long-context": {
        FOUNDATION_DOC,
        EXPERIMENTAL_SOURCE,
    },
    "native_trainable": {
        FOUNDATION_DOC,
        EXPERIMENTAL_SOURCE,
        ROOT / "src/diffbio/operators/foundation_models/contracts.py",
        ROOT / "src/diffbio/operators/drug_discovery/dti.py",
    },
    "Hyena": {
        FOUNDATION_DOC,
        EXPERIMENTAL_SOURCE,
    },
    "PEFT": {
        FOUNDATION_DOC,
        EXPERIMENTAL_SOURCE,
    },
    "LoRA": {
        FOUNDATION_DOC,
        EXPERIMENTAL_SOURCE,
    },
}


def _public_text_paths() -> list[Path]:
    roots = (
        ROOT / "README.md",
        ROOT / "docs",
        ROOT / "src/diffbio",
    )
    paths: list[Path] = []
    for root in roots:
        if root.is_file():
            paths.append(root)
            continue
        paths.extend(
            path
            for path in root.rglob("*")
            if path.suffix in {".md", ".py"} and "docs/_build" not in path.as_posix()
        )
    return paths


def test_experimental_boundary_policy_is_public_and_fail_closed() -> None:
    """Speculative foundation capabilities should share one explicit policy."""
    assert (
        EXPERIMENTAL_FOUNDATION_MODEL_NAMESPACE
        == "diffbio.operators.foundation_models.experimental"
    )
    assert set(FOUNDATION_EXPERIMENTAL_CAPABILITIES) == EXPECTED_EXPERIMENTAL_CAPABILITIES
    assert FOUNDATION_EXPERIMENTAL_PROMOTION_CRITERIA == EXPECTED_PROMOTION_CRITERIA

    for key, capability in FOUNDATION_EXPERIMENTAL_CAPABILITIES.items():
        assert capability.key == key
        assert capability.namespace == EXPERIMENTAL_FOUNDATION_MODEL_NAMESPACE
        assert capability.stable_support is False
        assert capability.benchmark_status == "benchmark_unverified"
        assert capability.required_promotion_criteria == EXPECTED_PROMOTION_CRITERIA
        assert is_experimental_foundation_capability(key) is True
        assert get_experimental_foundation_capability(key) == capability

    assert is_experimental_foundation_capability("geneformer_precomputed") is False


def test_foundation_docs_define_experimental_boundary_and_promotion_path() -> None:
    """Docs should fence experimental claims with the shared policy."""
    doc = FOUNDATION_DOC.read_text(encoding="utf-8")
    normalized_doc = " ".join(doc.split())

    assert "## Experimental Foundation-Model Boundary" in doc
    assert EXPERIMENTAL_SCOPE_STATEMENT in normalized_doc
    for criterion in EXPECTED_PROMOTION_CRITERIA:
        assert criterion in normalized_doc
    for term in TERM_ALLOWED_PATHS:
        assert term in doc


def test_experimental_terms_are_only_used_in_fenced_policy_surfaces() -> None:
    """Experimental capability terms should not leak into unrelated stable docs."""
    for path in _public_text_paths():
        text = path.read_text(encoding="utf-8")
        for term, allowed_paths in TERM_ALLOWED_PATHS.items():
            term_pattern = rf"\b{re.escape(term)}\b"
            if re.search(term_pattern, text, flags=re.IGNORECASE):
                assert path in allowed_paths, path.relative_to(ROOT)
