#!/usr/bin/env python3
"""Verify the active JAX backend without relying on system CUDA paths.

The device identity comes from ``substrax.devices.detect_devices``, the same
reading every Avitai package uses; the script adds the environment DiffBio's
``setup.sh`` configured around it.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True, kw_only=True)
class VerificationReport:
    """What the script reports about the interpreter and its JAX backend."""

    diffbio_backend: str | None
    jax_platforms: str | None
    host: str
    python: str
    jax_version: str | None
    platform: str | None
    kind: str | None
    device_count: int
    device_kinds: tuple[str, ...]
    error: str | None


def emit(message: str) -> None:
    """Write a single line to stdout."""
    sys.stdout.write(f"{message}\n")


@contextlib.contextmanager
def suppress_process_stderr() -> Iterator[None]:
    """Redirect the process's stderr to devnull while JAX initialises its plugins.

    A captured stderr (pytest's capsys) has no file descriptor; nothing is
    redirected then.
    """
    try:
        stderr_fd = sys.stderr.fileno()
    except io.UnsupportedOperation:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as null_stream:
        saved_stderr_fd = os.dup(stderr_fd)
        try:
            os.dup2(null_stream.fileno(), stderr_fd)
            yield
        finally:
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stderr_fd)


def collect_report() -> VerificationReport:
    """Collect backend information from the active environment."""
    environment: dict[str, Any] = {
        "diffbio_backend": os.environ.get("DIFFBIO_BACKEND"),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "host": platform.platform(),
        "python": sys.version.split()[0],
    }
    try:
        with suppress_process_stderr():
            import jax  # noqa: PLC0415
            from substrax.devices import detect_devices  # noqa: PLC0415

            info = detect_devices()
    except (ImportError, OSError, RuntimeError) as exc:
        return VerificationReport(
            **environment,
            jax_version=None,
            platform=None,
            kind=None,
            device_count=0,
            device_kinds=(),
            error=str(exc),
        )
    return VerificationReport(
        **environment,
        jax_version=jax.__version__,
        platform=info.platform,
        kind=info.kind.value,
        device_count=info.count,
        device_kinds=tuple(info.device_kinds),
        error=None,
    )


def render_human_report(report: VerificationReport) -> str:
    """Render the report for terminal output."""
    lines = [
        "DiffBio JAX backend verification",
        f"Host: {report.host}",
        f"Python: {report.python}",
        f"Configured DiffBio backend: {report.diffbio_backend or 'unset'}",
        f"JAX_PLATFORMS: {report.jax_platforms or 'unset'}",
    ]
    if report.error is not None:
        lines.append(f"JAX backend unavailable: {report.error}")
        return "\n".join(lines)
    lines.extend(
        [
            f"JAX version: {report.jax_version}",
            f"Platform: {report.platform}",
            f"Device kind: {report.kind}",
            f"Devices: {report.device_count}",
            *(f"  - {kind}" for kind in report.device_kinds),
            "Note: DiffBio leaves JAX_PLATFORMS unset by default so JAX can pick"
            " GPU when available and CPU otherwise.",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Exit non-zero unless the default backend is a GPU",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the verification report as JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the verification script."""
    args = parse_args(argv)
    report = collect_report()

    if args.json:
        emit(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        emit(render_human_report(report))

    if report.error is not None:
        return 1
    if args.require_gpu and report.kind != "gpu":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
