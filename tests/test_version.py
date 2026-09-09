"""``diffbio.__version__`` is the installed distribution's version."""

from __future__ import annotations

import importlib.metadata

import diffbio


def test_version_matches_the_installed_distribution() -> None:
    assert diffbio.__version__ == importlib.metadata.version("diffbio")
    assert diffbio.__version__ != "0.1.0"
