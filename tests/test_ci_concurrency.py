"""CI spends runner time only on work somebody will read.

Every workflow a push triggers cancels the run a newer push supersedes: without a concurrency
group keyed on the ref, two pushes in a row queue two full runs, and the older one holds runners
(the organisation's few macOS runners above all) for work nobody will read.

A merge onto ``main`` does not repeat the CI jobs its pull request already ran over the same tree:
they consult substrax's already-tested action, pinned by commit, which compares only on a push.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml


WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
GATE_JOB = "already_tested"
GATE_CONDITION = f"needs.{GATE_JOB}.outputs.skip != 'true'"
GATE_ACTION = re.compile(r"^avitai/substrax/\.github/actions/already-tested@[0-9a-f]{40}$")
GATED_WORKFLOWS = ("ci.yml",)


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


def _workflow_jobs(name: str) -> dict[str, dict[str, Any]]:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))["jobs"]


@pytest.mark.parametrize("name", GATED_WORKFLOWS)
def test_the_gate_compares_only_on_a_push(name: str) -> None:
    """A manual run re-measures on purpose; only a merge repeats a pull request."""
    gate = _workflow_jobs(name)[GATE_JOB]
    steps = [step for step in gate["steps"] if "already-tested" in str(step.get("uses", ""))]

    assert [step.get("if") for step in steps] == ["github.event_name == 'push'"]
    assert steps[0]["id"] in gate["outputs"]["skip"]


@pytest.mark.parametrize("name", GATED_WORKFLOWS)
def test_the_gate_is_the_shared_action_pinned_to_a_commit(name: str) -> None:
    """The compare is substrax's already-tested action, pinned by a full commit SHA.

    The action finds the pull request a push merged (squash or rebase) and skips only when that
    pull request tested this tree and every one of its checks succeeded; its rules are tested in
    substrax. A full commit SHA pins exactly the code that runs.
    """
    steps = _workflow_jobs(name)[GATE_JOB]["steps"]
    compare = next(step for step in steps if step.get("id") == "compare")

    assert GATE_ACTION.match(compare.get("uses", "")), compare.get("uses")
    assert "run" not in compare, "the gate runs the shared action, not an inline script"


@pytest.mark.parametrize("name", GATED_WORKFLOWS)
def test_every_job_that_repeats_the_pull_request_consults_the_gate(name: str) -> None:
    jobs = _workflow_jobs(name)
    ungated = sorted(
        job_id
        for job_id, job in jobs.items()
        if job_id != GATE_JOB
        and (job.get("if") != GATE_CONDITION or GATE_JOB not in job.get("needs", []))
    )

    assert ungated == [], f"{name}: these repeat the pull request without the gate: {ungated}"


@pytest.mark.parametrize("name", GATED_WORKFLOWS)
def test_an_unanswered_gate_leaves_the_work_running(name: str) -> None:
    """An empty output (no compare, or a lookup that failed) reads as "test it"."""
    for job_id, job in _workflow_jobs(name).items():
        if job_id == GATE_JOB:
            continue
        text = yaml.safe_dump(job)
        assert "outputs.skip == " not in text, f"{job_id} tests the gate for equality"
        assert "outputs.skip != 'false'" not in text, f"{job_id} runs only on an explicit false"
