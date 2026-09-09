"""The backend verification script reports substrax's device identity."""

from __future__ import annotations

import json

import jax
import pytest
from substrax.devices import DeviceKind

from scripts.verify_gpu_setup import collect_report, main, render_human_report


def test_collect_report_reads_the_active_backend_through_substrax() -> None:
    report = collect_report()

    assert report.error is None
    assert report.jax_version == jax.__version__
    assert report.platform == jax.default_backend()
    assert report.kind in {kind.value for kind in DeviceKind}
    assert report.device_count == jax.device_count()
    assert len(report.device_kinds) >= 1


def test_human_report_names_platform_and_devices() -> None:
    text = render_human_report(collect_report())

    assert f"Platform: {jax.default_backend()}" in text
    assert f"Devices: {jax.device_count()}" in text


def test_json_report_round_trips(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["platform"] == jax.default_backend()
    assert payload["device_count"] == jax.device_count()
    assert payload["error"] is None


def test_require_gpu_follows_the_device_kind(capsys: pytest.CaptureFixture[str]) -> None:
    expected = 0 if collect_report().kind == DeviceKind.GPU.value else 1

    assert main(["--require-gpu"]) == expected
    assert "DiffBio JAX backend verification" in capsys.readouterr().out
