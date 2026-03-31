"""Tests for benchmarks.molecular_dynamics.bench_lj.

Validates the Lennard-Jones molecular dynamics benchmark and its helpers.
"""

from __future__ import annotations

import pytest
from calibrax.core.result import BenchmarkResult

from benchmarks.molecular_dynamics.bench_lj import (
    LJBenchmark,
    _generate_fcc_lattice,
    _generate_velocities,
)
from tests.benchmarks.conftest import assert_valid_benchmark_result


class TestFCCLattice:
    """Tests for FCC lattice generation."""

    def test_returns_correct_shape(self) -> None:
        """FCC lattice with 4 atoms/cell, nearest cube >= n."""
        positions, box_size = _generate_fcc_lattice(32)
        assert positions.ndim == 2
        assert positions.shape[1] == 3
        # 2^3 * 4 = 32 atoms for n_particles=32
        assert positions.shape[0] == 32

    def test_box_size_positive(self) -> None:
        positions, box_size = _generate_fcc_lattice(108)
        assert box_size > 0.0

    def test_particles_inside_box(self) -> None:
        positions, box_size = _generate_fcc_lattice(256)
        assert float(positions.max()) < box_size
        assert float(positions.min()) >= 0.0

    def test_scales_with_particle_count(self) -> None:
        _, box_small = _generate_fcc_lattice(32)
        _, box_large = _generate_fcc_lattice(4000)
        assert box_large > box_small


class TestVelocityGeneration:
    """Tests for Maxwell-Boltzmann velocity generation."""

    def test_shape(self) -> None:
        import jax  # noqa: PLC0415

        key = jax.random.PRNGKey(0)
        vel = _generate_velocities(100, 1.0, 1.0, key)
        assert vel.shape == (100, 3)

    def test_zero_com_velocity(self) -> None:
        import jax  # noqa: PLC0415
        import jax.numpy as jnp  # noqa: PLC0415

        key = jax.random.PRNGKey(0)
        vel = _generate_velocities(500, 1.0, 1.0, key)
        com = jnp.mean(vel, axis=0)
        assert float(jnp.max(jnp.abs(com))) < 1e-5


class TestLJBenchmark:
    """Tests for the full LJ benchmark (quick mode)."""

    @pytest.fixture(scope="class")
    def result(self) -> BenchmarkResult:
        """Run benchmark in quick mode."""
        bench = LJBenchmark(quick=True)
        return bench.run()

    def test_standard_contract(self, result: BenchmarkResult) -> None:
        """Verify the full standard benchmark result contract."""
        assert_valid_benchmark_result(
            result,
            expected_name="molecular_dynamics/lj",
            required_metric_keys=[
                "energy_drift",
                "jaxmd_steps_per_sec",
                "jaxmd_energy_drift",
                "initial_energy",
                "final_energy",
            ],
        )

    def test_has_operator_tag(self, result: BenchmarkResult) -> None:
        assert "ForceFieldOperator" in result.tags["operator"]

    def test_energy_drift_nonnegative(self, result: BenchmarkResult) -> None:
        """Energy conservation drift must be non-negative."""
        drift = result.metrics["energy_drift"].value
        assert drift >= 0.0

    def test_jaxmd_steps_per_sec_positive(self, result: BenchmarkResult) -> None:
        """jax-md baseline throughput must be positive."""
        sps = result.metrics["jaxmd_steps_per_sec"].value
        assert sps > 0.0

    def test_has_config(self, result: BenchmarkResult) -> None:
        assert "n_particles" in result.config
        assert "n_steps" in result.config
        assert "dt" in result.config
        assert "sigma" in result.config
        assert "epsilon" in result.config
        assert "cutoff" in result.config
        assert "box_size" in result.config

    def test_has_system_metadata(self, result: BenchmarkResult) -> None:
        info = result.metadata["dataset_info"]
        assert "n_particles" in info
        assert "dimension" in info
        assert "lattice" in info

    def test_energy_values_finite(self, result: BenchmarkResult) -> None:
        """Initial and final energies must be finite."""
        import math  # noqa: PLC0415

        e_init = result.metrics["initial_energy"].value
        e_final = result.metrics["final_energy"].value
        assert math.isfinite(e_init)
        assert math.isfinite(e_final)
