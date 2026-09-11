"""setup.sh names only the extras pyproject.toml declares."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = REPO_ROOT / "setup.sh"
# An extra setup.sh documents (`name` extra) or syncs (--extra name).
_EXTRA_REFERENCE = re.compile(r"`([a-z0-9][a-z0-9_-]*)` extra|--extra ([a-z0-9][a-z0-9_-]*)")


def test_setup_script_names_only_declared_extras() -> None:
    """Every extra setup.sh documents or syncs is declared in pyproject.toml."""
    named = {
        documented or synced
        for documented, synced in _EXTRA_REFERENCE.findall(SETUP_SCRIPT.read_text())
    }
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    declared = set(pyproject["project"]["optional-dependencies"])

    assert {"dev", "test", "cuda12"} <= named
    assert named - declared == set()
