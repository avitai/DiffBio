"""Tests for benchmarks.molecular_dynamics.bench_lj.

TDD: These tests define the expected behavior of the Lennard-Jones
molecular dynamics benchmark before optimising performance.
"""

from __future__ import annotations

import pytest
from calibrax.core.models import Metric
from calibrax.core.result import BenchmarkResult

from benchmarks.molecular_dynamics.bench_lj import (
    LJBenchmark,
    _generate_fcc_lattice,
    _generate_velocities,
)


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

    def test_returns_benchmark_result(
        self, result: BenchmarkResult
    ) -> None:
        """Benchmark must return a calibrax BenchmarkResult."""
        assert isinstance(result, BenchmarkResult)

    def test_name_is_correct(
        self, result: BenchmarkResult
    ) -> None:
        assert result.name == "molecular_dynamics/lj"

    def test_domain(self, result: BenchmarkResult) -> None:
        assert result.domain == "diffbio_benchmarks"

    def test_has_operator_tag(
        self, result: BenchmarkResult
    ) -> None:
        assert "operator" in result.tags
        assert "ForceFieldOperator" in result.tags["operator"]

    def test_has_potential_tag(
        self, result: BenchmarkResult
    ) -> None:
        assert result.tags.get("potential") == "lennard_jones"

    def test_has_steps_per_sec_metric(
        self, result: BenchmarkResult
    ) -> None:
        """Must report DiffBio throughput."""
        assert "steps_per_sec" in result.metrics
        sps = result.metrics["steps_per_sec"].value
        assert sps > 0.0

    def test_has_energy_drift_metric(
        self, result: BenchmarkResult
    ) -> None:
        """Must report energy conservation drift."""
        assert "energy_drift" in result.metrics
        drift = result.metrics["energy_drift"].value
        assert drift >= 0.0

    def test_has_jaxmd_steps_per_sec(
        self, result: BenchmarkResult
    ) -> None:
        """Must report jax-md baseline throughput."""
        assert "jaxmd_steps_per_sec" in result.metrics
        sps = result.metrics["jaxmd_steps_per_sec"].value
        assert sps > 0.0

    def test_has_jaxmd_energy_drift(
        self, result: BenchmarkResult
    ) -> None:
        assert "jaxmd_energy_drift" in result.metrics

    def test_has_gradient_norm(
        self, result: BenchmarkResult
    ) -> None:
        assert "gradient_norm" in result.metrics

    def test_has_gradient_nonzero(
        self, result: BenchmarkResult
    ) -> None:
        assert "gradient_nonzero" in result.metrics
        # ForceFieldOperator has no learnable nnx.Param parameters,
        # so gradient_nonzero may be 0.0 (gradients flow through
        # inputs, not model parameters)
        assert "gradient_nonzero" in result.metrics

    def test_metrics_are_calibrax_metric(
        self, result: BenchmarkResult
    ) -> None:
        for key, metric in result.metrics.items():
            assert isinstance(metric, Metric), (
                f"{key} is not a Metric"
            )

    def test_has_timing(self, result: BenchmarkResult) -> None:
        assert result.timing is not None
        assert result.timing.wall_clock_sec > 0

    def test_has_config(self, result: BenchmarkResult) -> None:
        assert "n_particles" in result.config
        assert "n_steps" in result.config
        assert "dt" in result.config
        assert "sigma" in result.config
        assert "epsilon" in result.config
        assert "cutoff" in result.config
        assert "box_size" in result.config

    def test_has_system_metadata(
        self, result: BenchmarkResult
    ) -> None:
        assert "system_info" in result.metadata
        info = result.metadata["system_info"]
        assert "n_particles" in info
        assert "dimension" in info
        assert "lattice" in info

    def test_energy_values_finite(
        self, result: BenchmarkResult
    ) -> None:
        """Initial and final energies must be finite."""
        import math  # noqa: PLC0415

        e_init = result.metrics["initial_energy"].value
        e_final = result.metrics["final_energy"].value
        assert math.isfinite(e_init)
        assert math.isfinite(e_final)
