"""CI fails when coverage drops below the floor, and uploads coverage nowhere.

coverage.py is the only coverage gate: the unit shards collect data with the floor off,
and the combined Test Coverage job applies ``[tool.coverage.report] fail_under``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def coverage_cap_violations(workflow: dict, pyproject: dict) -> list[str]:
    """Return why CI would not fail below the coverage floor, if it would not."""
    triggers = workflow.get("on", workflow.get(True))
    job = workflow["jobs"]["coverage"]
    commands = "\n".join(str(step.get("run", "")) for step in job["steps"])
    floor = pyproject["tool"]["coverage"]["report"].get("fail_under")

    problems = []
    if floor is None or float(floor) < 80:
        problems.append(f"[tool.coverage.report] fail_under is {floor}, not at least 80")
    if not {"push", "pull_request"} <= set(triggers):
        problems.append(f"CI runs on {sorted(triggers)}, not on both push and pull_request")
    if "if" in job:
        problems.append(f"the combined coverage job only runs when {job['if']}")
    if "coverage report" not in commands or "--fail-under=0" in commands:
        problems.append("the combined coverage job does not run coverage report against fail_under")
    return problems


def test_ci_fails_below_the_coverage_floor_on_every_change() -> None:
    """The combined report runs for pushes and pull requests with the pyproject floor."""
    workflow = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

    assert coverage_cap_violations(workflow, pyproject) == []


def test_ci_uploads_no_coverage_to_codecov() -> None:
    """coverage.py in CI is the coverage gate; no workflow uploads to Codecov."""
    uses = [
        str(step.get("uses", ""))
        for path in sorted(WORKFLOWS.glob("*.yml"))
        for job in yaml.safe_load(path.read_text()).get("jobs", {}).values()
        for step in job.get("steps", [])
    ]

    assert any(action.startswith("actions/checkout@") for action in uses)
    assert [action for action in uses if action.startswith("codecov/")] == []
