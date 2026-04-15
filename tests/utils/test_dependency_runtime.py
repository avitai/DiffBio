"""Tests for dependency runtime inspection helpers."""

from types import ModuleType

import pytest

from diffbio.utils.dependency_runtime import (
    ECOSYSTEM_PACKAGES,
    DependencyRuntimeRecord,
    FNOConstructorContract,
    _resolve_module_file,
    collect_dependency_runtime,
    inspect_fno_constructor,
    verify_canonical_dependency_runtime,
)


def test_collect_dependency_runtime_covers_ecosystem_packages() -> None:
    """Runtime inspection should report every managed ecosystem package."""
    runtime = collect_dependency_runtime()

    assert set(runtime) == set(ECOSYSTEM_PACKAGES)
    assert all(record.module_file for record in runtime.values())


def test_collect_dependency_runtime_uses_installed_packages() -> None:
    """Canonical runtime should resolve dependencies from the installed environment."""
    runtime = collect_dependency_runtime()

    assert all(record.installed_from_site_packages for record in runtime.values())


def test_inspect_fno_constructor_reports_spatial_dims_support() -> None:
    """The live Opifex FNO constructor must expose the spatial-dims contract."""
    contract = inspect_fno_constructor()

    assert contract.import_path == "opifex.neural.operators"
    assert contract.supports_spatial_dims is True
    assert "spatial_dims" in contract.constructor_signature


def test_verify_canonical_dependency_runtime_succeeds_for_active_environment() -> None:
    """The active activated environment should satisfy the canonical runtime contract."""
    runtime, contract = verify_canonical_dependency_runtime()

    assert set(runtime) == set(ECOSYSTEM_PACKAGES)
    assert all(record.installed_from_site_packages for record in runtime.values())
    assert contract.supports_spatial_dims is True


def test_resolve_module_file_requires_file_attribute() -> None:
    """Modules without a concrete file path should fail fast."""
    module = ModuleType("synthetic_module")

    with pytest.raises(RuntimeError, match="does not expose __file__"):
        _resolve_module_file(module)


def test_verify_canonical_dependency_runtime_rejects_non_installed_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical verification should reject sibling-checkout style imports."""
    runtime = {
        package: DependencyRuntimeRecord(
            package=package,
            module_file=f"/tmp/{package}/__init__.py",
            installed_from_site_packages=False,
        )
        for package in ECOSYSTEM_PACKAGES
    }

    monkeypatch.setattr(
        "diffbio.utils.dependency_runtime.collect_dependency_runtime",
        lambda package_names=ECOSYSTEM_PACKAGES: runtime,
    )
    monkeypatch.setattr(
        "diffbio.utils.dependency_runtime.inspect_fno_constructor",
        lambda import_path="opifex.neural.operators": FNOConstructorContract(
            import_path=import_path,
            constructor_signature="(self, *, spatial_dims: int = 1)",
            supports_spatial_dims=True,
        ),
    )

    with pytest.raises(RuntimeError, match="site-packages"):
        verify_canonical_dependency_runtime()


def test_verify_canonical_dependency_runtime_rejects_missing_spatial_dims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical verification should fail if the live FNO contract regresses."""
    runtime = {
        package: DependencyRuntimeRecord(
            package=package,
            module_file=f"/venv/site-packages/{package}/__init__.py",
            installed_from_site_packages=True,
        )
        for package in ECOSYSTEM_PACKAGES
    }

    monkeypatch.setattr(
        "diffbio.utils.dependency_runtime.collect_dependency_runtime",
        lambda package_names=ECOSYSTEM_PACKAGES: runtime,
    )
    monkeypatch.setattr(
        "diffbio.utils.dependency_runtime.inspect_fno_constructor",
        lambda import_path="opifex.neural.operators": FNOConstructorContract(
            import_path=import_path,
            constructor_signature="(self)",
            supports_spatial_dims=False,
        ),
    )

    with pytest.raises(RuntimeError, match="spatial_dims"):
        verify_canonical_dependency_runtime()
