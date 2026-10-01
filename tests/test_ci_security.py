"""The Security workflow audits DiffBio's whole lockfile and fails on either scan.

``uv run --with pip-audit pip-audit --local`` audits the environment pip-audit itself runs in,
not DiffBio's lock, so it could report nothing about DiffBio at all. The audit is substrax's
``audit-lock`` action, pinned by commit: it exports every extra the lock resolves and audits each
export with a pinned pip-audit. Its ignore table lives in ``pyproject.toml`` as
``[tool.substrax.audit-lock.ignore]``; the action refuses an empty reason and an entry no
advisory matches, so those rules are tested in substrax, not here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


GITHUB = Path(__file__).resolve().parents[1] / ".github"
SECURITY_WORKFLOW = GITHUB / "workflows" / "security.yml"
SETUP_ACTION = "./.github/actions/setup-diffbio"
AUDIT_ACTION = re.compile(r"^avitai/substrax/\.github/actions/audit-lock@[0-9a-f]{40}$")


def _steps() -> list[dict[str, Any]]:
    jobs = yaml.safe_load(SECURITY_WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    return [step for job in jobs.values() for step in job["steps"]]


def _index(steps: list[dict[str, Any]], matches: Any) -> int:
    return next(i for i, step in enumerate(steps) if matches(step))


def test_the_audit_is_the_shared_action_pinned_to_a_commit() -> None:
    steps = _steps()
    audits = [step for step in steps if AUDIT_ACTION.match(str(step.get("uses", "")))]

    assert len(audits) == 1
    assert "continue-on-error" not in audits[0]


def test_uv_is_on_path_before_the_audit_runs() -> None:
    """The action runs ``uv export`` and ``uvx``; the setup action installs uv."""
    steps = _steps()
    setup = _index(steps, lambda step: step.get("uses") == SETUP_ACTION)
    audit = _index(steps, lambda step: AUDIT_ACTION.match(str(step.get("uses", ""))))
    setup_action = (GITHUB / "actions" / "setup-diffbio" / "action.yml").read_text(encoding="utf-8")

    assert setup < audit
    assert "astral-sh/setup-uv@" in setup_action


def test_no_step_runs_a_blind_pip_audit() -> None:
    commands = "\n".join(str(step.get("run", "")) for step in _steps())

    assert "pip-audit" not in commands
    assert "pip_audit" not in commands


def test_bandit_runs_even_after_a_failed_audit_and_its_failure_fails_the_job() -> None:
    steps = _steps()
    bandit = steps[_index(steps, lambda step: "bandit " in str(step.get("run", "")))]

    assert bandit.get("if") == "${{ !cancelled() }}"
    assert "continue-on-error" not in bandit
    assert "||" not in bandit["run"], "a captured exit status can hide bandit's failure"
