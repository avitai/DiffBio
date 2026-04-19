"""Ownership-boundary checks for benchmark training utilities."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_ROOT = ROOT / "benchmarks"
OPTIFEX_OPTIMIZER_IMPORT = "from opifex.core.training.optimizers import"
DIRECT_OPTAX_OPTIMIZER_CALLS = (
    "optax.adam(",
    "optax.adamw(",
    "optax.sgd(",
)


def _benchmark_python_files() -> tuple[Path, ...]:
    return tuple(sorted(BENCHMARKS_ROOT.rglob("*.py")))


def test_benchmark_training_uses_one_opifex_optimizer_boundary() -> None:
    """Benchmarks should not construct optimizers outside the shared helper."""
    optimizer_helper = BENCHMARKS_ROOT / "_optimizers.py"

    for path in _benchmark_python_files():
        source = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(ROOT)
        if path != optimizer_helper:
            assert OPTIFEX_OPTIMIZER_IMPORT not in source, relative_path
        for direct_call in DIRECT_OPTAX_OPTIMIZER_CALLS:
            assert direct_call not in source, f"{relative_path} contains {direct_call}"
