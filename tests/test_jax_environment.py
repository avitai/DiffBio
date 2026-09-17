"""The test session's JAX environment keeps what the caller exported.

``conftest.setup_jax_environment`` adds the flags the suite needs; a flag the caller set,
such as an emulated device count, must survive it.
"""

from __future__ import annotations

import os

import pytest
from conftest import setup_jax_environment


def test_setup_merges_xla_flags_instead_of_replacing_them(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

    setup_jax_environment()

    flags = os.environ["XLA_FLAGS"].split()
    assert "--xla_force_host_platform_device_count=2" in flags
    assert "--xla_gpu_strict_conv_algorithm_picker=false" in flags


def test_setup_refuses_a_conflicting_flag_value(monkeypatch: pytest.MonkeyPatch) -> None:
    from substrax.runtime import XlaFlagConflictError

    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_strict_conv_algorithm_picker=true")

    with pytest.raises(XlaFlagConflictError):
        setup_jax_environment()
