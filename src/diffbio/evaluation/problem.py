"""Benchmark problem definition and JSON loading.

Defines the ``BenchmarkProblem`` dataclass representing a single evaluation
problem from scBench or SpatialBench, and provides utilities for loading
problem sets from JSON files.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkProblem:
    """A single benchmark evaluation problem.

    Attributes:
        problem_id: Unique identifier for the problem.
        task_type: Category of the task (e.g., ``"qc_filtering"``,
            ``"clustering"``, ``"differential_expression"``).
        grader_type: Which grading algorithm to use (e.g.,
            ``"numeric_tolerance"``, ``"multiple_choice"``).
        expected_answer: The ground-truth answer in the format expected
            by the corresponding grader.
        grader_config: Extra parameters for the grader (e.g.,
            ``{"tolerance": 0.05, "mode": "relative"}``).
        task_config: Parameters for the DiffBio operator invocation
            (e.g., ``{"n_clusters": 5, "temperature": 1.0}``).
        data_path: Optional path to the h5ad or other data file.
        description: Human-readable description of the problem.
        source: Origin benchmark suite (``"scbench"`` or ``"spatialbench"``).
    """

    problem_id: str
    task_type: str
    grader_type: str
    expected_answer: Any
    grader_config: dict[str, Any] = field(default_factory=dict)
    task_config: dict[str, Any] = field(default_factory=dict)
    data_path: str | None = None
    description: str = ""
    source: str = ""


def load_problems(path: Path | str) -> list[BenchmarkProblem]:
    """Load benchmark problems from a JSON file.

    The JSON file should contain a list of objects, each with fields matching
    ``BenchmarkProblem`` attributes.

    Args:
        path: Path to the JSON file.

    Returns:
        List of parsed BenchmarkProblem instances.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        KeyError: If a required field is missing from a problem entry.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark problems file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, dict) and "problems" in raw:
        raw = raw["problems"]

    problems: list[BenchmarkProblem] = []
    for entry in raw:
        problem = BenchmarkProblem(
            problem_id=entry["problem_id"],
            task_type=entry["task_type"],
            grader_type=entry["grader_type"],
            expected_answer=entry["expected_answer"],
            grader_config=entry.get("grader_config", {}),
            task_config=entry.get("task_config", {}),
            data_path=entry.get("data_path"),
            description=entry.get("description", ""),
            source=entry.get("source", ""),
        )
        problems.append(problem)

    logger.info("Loaded %d benchmark problems from %s", len(problems), path)
    return problems
