"""Every workflow a push triggers cancels the run a newer push supersedes.

Without a concurrency group keyed on the ref, two pushes in a row queue two full runs, and
the older one holds runners (the organisation's few macOS runners above all) for work
nobody will read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _documents() -> dict[str, dict[str, Any]]:
    return {
        workflow.name: yaml.safe_load(workflow.read_text(encoding="utf-8"))
        for workflow in sorted(WORKFLOWS.glob("*.yml"))
    }


def _triggers(document: dict[str, Any]) -> dict[str, Any]:
    """The ``on`` mapping; PyYAML reads the bare key ``on`` as the boolean ``True``."""
    keys: dict[Any, Any] = document
    return keys.get("on") or keys.get(True) or {}


def test_every_workflow_a_push_triggers_cancels_the_run_it_supersedes() -> None:
    pushed = {name: doc for name, doc in _documents().items() if "push" in _triggers(doc)}
    assert pushed, "no workflow runs on push"
    for name, document in pushed.items():
        concurrency = document.get("concurrency")
        assert isinstance(concurrency, dict), f"{name} declares no concurrency group"
        assert "github.ref" in str(concurrency.get("group")), f"{name}'s group ignores the ref"
        assert concurrency.get("cancel-in-progress") is True, f"{name} keeps superseded runs"


def test_every_uv_cache_is_pruned_before_it_is_saved() -> None:
    """A saved uv cache holds only what uv built, not every wheel it downloaded.

    setup-uv prunes only when asked (``prune-cache`` defaults to false from v9); unpruned, the
    caches of the heavy extras grow to gigabytes each and evict the repository's other caches.
    """
    github = WORKFLOWS.parents[1] / ".github"
    documents = [
        *sorted(github.glob("workflows/*.yml")),
        *sorted(github.glob("actions/*/action.yml")),
    ]
    checked = 0
    unpruned: list[str] = []
    for path in documents:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        owners = {**document.get("jobs", {}), "runs": document.get("runs") or {}}
        for owner, body in owners.items():
            for step in body.get("steps", []):
                if not str(step.get("uses", "")).startswith("astral-sh/setup-uv@"):
                    continue
                checked += 1
                if (step.get("with") or {}).get("prune-cache") is not True:
                    unpruned.append(f"{path.relative_to(WORKFLOWS.parents[1])}:{owner}")

    assert checked, "no setup-uv step found; the contract is reading the wrong files"
    assert unpruned == [], f"setup-uv steps saving an unpruned cache: {unpruned}"
