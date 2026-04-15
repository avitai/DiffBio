"""Verify that DiffBio is running against the canonical installed ecosystem."""

from __future__ import annotations

import sys

from diffbio.utils.dependency_runtime import verify_canonical_dependency_runtime


def _write_line(line: str) -> None:
    """Write one line to stdout."""
    sys.stdout.write(f"{line}\n")


def main() -> int:
    """Validate dependency provenance and the live Opifex FNO contract."""
    runtime, fno_contract = verify_canonical_dependency_runtime()

    _write_line("Dependency runtime")
    for package_name, record in runtime.items():
        _write_line(
            f"- {package_name}: {record.module_file} "
            f"(installed={record.installed_from_site_packages})"
        )

    _write_line("")
    _write_line("FNO constructor contract")
    _write_line(f"- import path: {fno_contract.import_path}")
    _write_line(f"- signature: {fno_contract.constructor_signature}")
    _write_line(f"- supports spatial_dims: {fno_contract.supports_spatial_dims}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
