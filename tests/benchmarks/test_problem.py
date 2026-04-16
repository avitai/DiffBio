"""Tests for BenchmarkProblem dataclass and JSON loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from diffbio.evaluation.problem import BenchmarkProblem, load_problems


class TestBenchmarkProblem:
    """Tests for BenchmarkProblem dataclass."""

    def test_required_fields(self) -> None:
        """BenchmarkProblem stores all required fields."""
        problem = BenchmarkProblem(
            problem_id="test_001",
            task_type="clustering",
            grader_type="numeric_tolerance",
            expected_answer=42.0,
        )
        assert problem.problem_id == "test_001"
        assert problem.task_type == "clustering"
        assert problem.grader_type == "numeric_tolerance"
        assert problem.expected_answer == 42.0

    def test_default_fields(self) -> None:
        """Optional fields have sensible defaults."""
        problem = BenchmarkProblem(
            problem_id="t1",
            task_type="x",
            grader_type="y",
            expected_answer=0,
        )
        assert problem.grader_config == {}
        assert problem.task_config == {}
        assert problem.data_path is None
        assert problem.description == ""
        assert problem.source == ""

    def test_configs_are_copied_on_construction(self) -> None:
        """Task and grader configs should not alias caller-owned dicts."""
        grader_config = {"tolerance": 0.1}
        task_config = {"n_clusters": 5}

        problem = BenchmarkProblem(
            problem_id="t1",
            task_type="clustering",
            grader_type="numeric_tolerance",
            expected_answer=0,
            grader_config=grader_config,
            task_config=task_config,
        )

        grader_config["tolerance"] = 0.9
        task_config["n_clusters"] = 99

        assert problem.grader_config["tolerance"] == 0.1
        assert problem.task_config["n_clusters"] == 5

    def test_immutable(self) -> None:
        """BenchmarkProblem is frozen."""
        problem = BenchmarkProblem(
            problem_id="t1",
            task_type="x",
            grader_type="y",
            expected_answer=0,
        )
        with pytest.raises(AttributeError):
            problem.problem_id = "new"  # type: ignore[misc]


class TestLoadProblems:
    """Tests for loading problems from JSON."""

    def test_load_from_list(self, tmp_path: Path) -> None:
        """Load problems from a JSON array."""
        data = [
            {
                "problem_id": "p1",
                "task_type": "clustering",
                "grader_type": "numeric_tolerance",
                "expected_answer": 5.0,
            }
        ]
        path = tmp_path / "problems.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        problems = load_problems(path)
        assert len(problems) == 1
        assert problems[0].problem_id == "p1"

    def test_load_from_dict_with_problems_key(self, tmp_path: Path) -> None:
        """Load problems from a dict with 'problems' key."""
        data = {
            "problems": [
                {
                    "problem_id": "p1",
                    "task_type": "qc",
                    "grader_type": "numeric_tolerance",
                    "expected_answer": 42,
                }
            ]
        }
        path = tmp_path / "problems.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        problems = load_problems(path)
        assert len(problems) == 1

    def test_file_not_found(self) -> None:
        """Raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_problems(Path("/nonexistent/path.json"))

    def test_load_preserves_config(self, tmp_path: Path) -> None:
        """Grader and task configs are loaded correctly."""
        data = [
            {
                "problem_id": "p1",
                "task_type": "clustering",
                "grader_type": "numeric_tolerance",
                "expected_answer": 5.0,
                "grader_config": {"tolerance": 0.5, "mode": "relative"},
                "task_config": {"n_clusters": 10},
            }
        ]
        path = tmp_path / "problems.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        problems = load_problems(path)
        assert problems[0].grader_config["tolerance"] == 0.5
        assert problems[0].task_config["n_clusters"] == 10

    def test_load_from_fixture(self, problems_json_path: Path) -> None:
        """Load from conftest fixture path."""
        problems = load_problems(problems_json_path)
        assert len(problems) == 2
        assert problems[0].problem_id == "p1"
        assert problems[1].grader_type == "multiple_choice"
