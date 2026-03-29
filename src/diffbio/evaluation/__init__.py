"""Evaluation harness for scBench and SpatialBench benchmarks.

Provides grading algorithms, benchmark problem definitions, task adapters,
and a runner for evaluating DiffBio operators against real-world benchmark
problems from single-cell and spatial transcriptomics domains.
"""

from diffbio.evaluation.adapters import TaskAdapter, compute_quality_metrics
from diffbio.evaluation.graders import (
    GradeResult,
    grade_distribution_comparison,
    grade_label_set_jaccard,
    grade_marker_gene_precision_recall,
    grade_multiple_choice,
    grade_numeric_tolerance,
)
from diffbio.evaluation.problem import BenchmarkProblem, load_problems
from diffbio.evaluation.runner import (
    BenchmarkSummary,
    EvalResult,
    run_benchmark,
    run_problem,
    summarize,
)

__all__ = [
    "BenchmarkProblem",
    "BenchmarkSummary",
    "EvalResult",
    "GradeResult",
    "TaskAdapter",
    "compute_quality_metrics",
    "grade_distribution_comparison",
    "grade_label_set_jaccard",
    "grade_marker_gene_precision_recall",
    "grade_multiple_choice",
    "grade_numeric_tolerance",
    "load_problems",
    "run_benchmark",
    "run_problem",
    "summarize",
]
