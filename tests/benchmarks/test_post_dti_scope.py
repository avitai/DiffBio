"""Regression checks for post-DTI stable versus experimental scope."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POST_DTI_SCOPE_STATEMENT = (
    "Post-DTI stable boundary: benchmark-backed operator support is separate from imported "
    "foundation-model promotion."
)
POST_DTI_SCOPE_DOCS = (
    ROOT / "docs/user-guide/operators/foundation-models.md",
    ROOT / "docs/user-guide/operators/protein.md",
    ROOT / "docs/user-guide/operators/multiomics.md",
    ROOT / "docs/user-guide/operators/metabolomics.md",
)
UNSUPPORTED_CLAIM_PATTERNS = (
    r"\bgeneric biomedical LLM\b",
    r"\bclinical NLP\b",
    r"\bpublic-health\b",
    r"\bPEFT\b",
    r"\bLoRA\b",
)


def test_post_dti_docs_reuse_stable_boundary_statement() -> None:
    """Post-DTI docs should use one exact stable/experimental boundary."""
    for doc_path in POST_DTI_SCOPE_DOCS:
        doc = doc_path.read_text(encoding="utf-8")
        assert POST_DTI_SCOPE_STATEMENT in doc, doc_path


def test_foundation_docs_record_post_dti_scope_matrix() -> None:
    """Foundation docs should be the canonical post-DTI scope matrix."""
    doc = (ROOT / "docs/user-guide/operators/foundation-models.md").read_text(encoding="utf-8")

    assert "## Post-DTI Stable Boundary" in doc
    assert (
        "| Protein | secondary-structure scaffold context | "
        "excluded from stable imported protein-LM promotion |"
    ) in doc
    assert (
        "| Multi-omics | seqFISH spatial deconvolution | benchmark-backed operator benchmark |"
    ) in doc
    assert (
        "| Metabolomics | precomputed spectrum embedding alignment | not benchmark-promoted |"
    ) in doc


def test_docs_do_not_claim_generic_biomedical_llm_scope() -> None:
    """Docs and source should not drift into unsupported generic biomedical claims."""
    search_roots = (
        ROOT / "README.md",
        ROOT / "docs",
        ROOT / "src/diffbio",
    )
    combined_text = "\n".join(
        path.read_text(encoding="utf-8")
        for root in search_roots
        for path in (root.rglob("*") if root.is_dir() else (root,))
        if path.suffix in {".md", ".py"}
    )

    for pattern in UNSUPPORTED_CLAIM_PATTERNS:
        assert re.search(pattern, combined_text, flags=re.IGNORECASE) is None
