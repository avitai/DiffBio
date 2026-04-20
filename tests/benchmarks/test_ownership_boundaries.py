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
CALIBRAX_STORAGE_IMPORT = "from calibrax.storage.store import Store"
CALIBRAX_CI_GUARD_IMPORT = "from calibrax.ci.guard import CIGuard"
CALIBRAX_PROFILING_IMPORT = "from calibrax.profiling"


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


def test_benchmark_storage_uses_one_calibrax_boundary() -> None:
    """Benchmarks should route Calibrax storage and guards through one helper."""
    calibrax_helper = BENCHMARKS_ROOT / "_calibrax.py"

    for path in _benchmark_python_files():
        if path == calibrax_helper:
            continue

        source = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(ROOT)
        assert CALIBRAX_STORAGE_IMPORT not in source, relative_path
        assert CALIBRAX_CI_GUARD_IMPORT not in source, relative_path


def test_benchmark_profiling_uses_one_calibrax_boundary() -> None:
    """Benchmarks should route Calibrax profiling through one helper."""
    calibrax_helper = BENCHMARKS_ROOT / "_calibrax.py"

    for path in _benchmark_python_files():
        if path == calibrax_helper:
            continue

        source = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(ROOT)
        assert CALIBRAX_PROFILING_IMPORT not in source, relative_path


def test_foundation_models_is_the_only_language_model_boundary() -> None:
    """The old language_models namespace should not reappear."""
    assert not (ROOT / "src/diffbio/operators/language_models").exists()
    assert (ROOT / "src/diffbio/operators/foundation_models/__init__.py").exists()


def test_public_positioning_names_diffbio_as_biology_specific_layer() -> None:
    """Public positioning should keep DiffBio scoped above sibling repos."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    package_doc = (ROOT / "src/diffbio/__init__.py").read_text(encoding="utf-8")

    assert "biology-specific differentiable operator layer" in readme
    for sibling_repo in ("Datarax", "Artifex", "Opifex", "Calibrax"):
        assert sibling_repo in readme
    assert "integrate with Datarax, Artifex, Opifex, and Calibrax" in package_doc
