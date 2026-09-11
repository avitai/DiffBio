"""Every test module runs in a CI job.

The unit-test shards in ``.github/workflows/ci.yml`` select test files by path. A
directory that no shard names never runs, and nothing reports it: the shards pass on
what they collect.
"""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
# Paths a job other than the unit shards runs on its own.
OTHER_JOB_PATHS = ("tests/integration",)


def _is_under(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root.rstrip('/')}/")


def _unit_shards() -> list[dict[str, str]]:
    workflow = yaml.safe_load(CI_WORKFLOW.read_text())
    return workflow["jobs"]["unit_tests"]["strategy"]["matrix"]["shard"]


def _shard_selects(shard: dict[str, str], path: str) -> bool:
    ignored = [
        argument.removeprefix("--ignore=")
        for argument in shard["extra"].split()
        if argument.startswith("--ignore=")
    ]
    return any(_is_under(path, root) for root in shard["paths"].split()) and not any(
        _is_under(path, root) for root in ignored
    )


def test_every_test_module_is_selected_by_a_ci_job() -> None:
    """Each ``test_*.py`` under ``tests/`` is collected by a unit shard or another job."""
    test_modules = sorted(
        path.relative_to(REPO_ROOT).as_posix() for path in (REPO_ROOT / "tests").rglob("test_*.py")
    )
    shards = _unit_shards()

    unselected = [
        module
        for module in test_modules
        if not any(_is_under(module, root) for root in OTHER_JOB_PATHS)
        and not any(_shard_selects(shard, module) for shard in shards)
    ]

    assert len(test_modules) > 200
    assert unselected == []
