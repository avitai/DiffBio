"""Repo-wide ownership-boundary checks."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src" / "diffbio"
FORBIDDEN_MIGRATION_TODOS = (
    "# TODO: Migrate to artifex",
    "# TODO: Migrate to datarax",
    "# TODO: Migrate to opifex",
    "# TODO: Migrate to calibrax",
)


def test_source_ownership_notes_do_not_defer_to_vague_migration_todos() -> None:
    """Ownership decisions should be explicit, not left as broad migration TODOs."""
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(ROOT)
        for forbidden_todo in FORBIDDEN_MIGRATION_TODOS:
            assert forbidden_todo not in source, f"{relative_path} contains {forbidden_todo}"
