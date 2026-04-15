"""Helpers for verifying the installed ecosystem runtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import importlib
import inspect
from pathlib import Path
import site
from types import ModuleType

ECOSYSTEM_PACKAGES: tuple[str, ...] = ("datarax", "artifex", "opifex", "calibrax")


@dataclass(frozen=True)
class DependencyRuntimeRecord:
    """Resolved runtime provenance for one ecosystem package."""

    package: str
    module_file: str
    installed_from_site_packages: bool


@dataclass(frozen=True)
class FNOConstructorContract:
    """Observed constructor contract for the live Opifex FNO surface."""

    import_path: str
    constructor_signature: str
    supports_spatial_dims: bool


def _site_packages_roots() -> tuple[Path, ...]:
    """Return normalized site-packages roots for the active interpreter."""
    return tuple(Path(root).resolve() for root in site.getsitepackages())


def _resolve_module_file(module: ModuleType) -> Path:
    """Return the concrete module file for an imported package.

    Raises:
        RuntimeError: If the imported module does not expose a file path.
    """
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        msg = f"Imported module {module.__name__!r} does not expose __file__"
        raise RuntimeError(msg)
    return Path(module_file).resolve()


def _is_in_site_packages(module_file: Path, site_roots: Sequence[Path]) -> bool:
    """Return whether a module file resolves under one of the site-packages roots."""
    return any(module_file.is_relative_to(root) for root in site_roots)


def collect_dependency_runtime(
    package_names: Sequence[str] = ECOSYSTEM_PACKAGES,
) -> dict[str, DependencyRuntimeRecord]:
    """Collect import provenance for the configured ecosystem packages."""
    site_roots = _site_packages_roots()
    runtime: dict[str, DependencyRuntimeRecord] = {}

    for package_name in package_names:
        module = importlib.import_module(package_name)
        module_file = _resolve_module_file(module)
        runtime[package_name] = DependencyRuntimeRecord(
            package=package_name,
            module_file=str(module_file),
            installed_from_site_packages=_is_in_site_packages(module_file, site_roots),
        )

    return runtime


def inspect_fno_constructor(
    import_path: str = "opifex.neural.operators",
) -> FNOConstructorContract:
    """Inspect the live FourierNeuralOperator constructor contract."""
    module = importlib.import_module(import_path)
    constructor_signature = str(inspect.signature(module.FourierNeuralOperator.__init__))
    return FNOConstructorContract(
        import_path=import_path,
        constructor_signature=constructor_signature,
        supports_spatial_dims="spatial_dims" in constructor_signature,
    )


def verify_canonical_dependency_runtime(
    package_names: Sequence[str] = ECOSYSTEM_PACKAGES,
) -> tuple[dict[str, DependencyRuntimeRecord], FNOConstructorContract]:
    """Validate the canonical installed runtime contract for ecosystem dependencies.

    Raises:
        RuntimeError: If any ecosystem package resolves outside site-packages or
            if the live Opifex FNO constructor lacks ``spatial_dims`` support.
    """
    runtime = collect_dependency_runtime(package_names)
    non_installed_packages = sorted(
        package for package, record in runtime.items() if not record.installed_from_site_packages
    )
    if non_installed_packages:
        package_list = ", ".join(non_installed_packages)
        msg = (
            "Canonical runtime must resolve ecosystem dependencies from installed "
            f"site-packages; found non-installed imports for: {package_list}"
        )
        raise RuntimeError(msg)

    fno_contract = inspect_fno_constructor()
    if not fno_contract.supports_spatial_dims:
        msg = "Live Opifex FourierNeuralOperator constructor does not expose spatial_dims"
        raise RuntimeError(msg)

    return runtime, fno_contract
