"""Tests for JAX-MD wrapper operators.

This module tests the differentiable molecular dynamics operators that wrap
JAX-MD functionality for integration with DiffBio pipelines.

These tests define the expected behavior - implementation must pass them
without modification (TDD principle).
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.molecular_dynamics import (
    ForceFieldConfig,
    ForceFieldOperator,
    MDIntegratorConfig,
    MDIntegratorOperator,
    PotentialType,
    create_force_field,
    create_integrator,
)
from diffbio.operators.molecular_dynamics.primitives import (
    create_displacement_fn,
    create_energy_fn,
    create_force_fn,
)


# =============================================================================
# Tests for Primitive Functions (DRY: shared utilities)
# =============================================================================


class TestPrimitives:
    """Tests for shared JAX-MD primitive creation functions."""

    def test_create_displacement_fn_periodic(self):
        """Test periodic displacement function creation."""
        box_size = 10.0
        disp_fn, shift_fn = create_displacement_fn(box_size=box_size)

        # Test displacement calculation
        r1 = jnp.array([1.0, 1.0, 1.0])
        r2 = jnp.array([9.0, 9.0, 9.0])
        disp = disp_fn(r1, r2)

        # Should wrap around periodic boundary (distance ~2.8, not ~13.9)
        assert jnp.linalg.norm(disp) < 5.0

    def test_create_displacement_fn_free(self):
        """Test free (non-periodic) displacement function creation."""
        disp_fn, shift_fn = create_displacement_fn(box_size=None)

        r1 = jnp.array([1.0, 1.0, 1.0])
        r2 = jnp.array([9.0, 9.0, 9.0])
        disp = disp_fn(r1, r2)

        # Should NOT wrap (distance ~13.9)
        assert jnp.linalg.norm(disp) > 10.0

    def test_create_energy_fn_lennard_jones(self):
        """Test Lennard-Jones energy function creation."""
        disp_fn, _ = create_displacement_fn(box_size=10.0)
        energy_fn = create_energy_fn(
            disp_fn, potential_type=PotentialType.LENNARD_JONES, sigma=1.0, epsilon=1.0
        )

        # Two particles at distance 2^(1/6) sigma should be at minimum
        r_min = 2 ** (1 / 6)
        positions = jnp.array([[0.0, 0.0, 0.0], [r_min, 0.0, 0.0]])
        energy = energy_fn(positions)

        assert energy < 0  # Attractive at minimum

    def test_create_energy_fn_soft_sphere(self):
        """Test soft sphere energy function creation."""
        disp_fn, _ = create_displacement_fn(box_size=10.0)
        energy_fn = create_energy_fn(
            disp_fn, potential_type=PotentialType.SOFT_SPHERE, sigma=1.0, epsilon=1.0
        )

        # Energy should be purely repulsive (positive for overlapping particles)
        positions = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        energy = energy_fn(positions)

        assert energy > 0

    def test_create_force_fn(self):
        """Test force function creation from energy function."""
        disp_fn, _ = create_displacement_fn(box_size=10.0)
        energy_fn = create_energy_fn(
            disp_fn, potential_type=PotentialType.LENNARD_JONES, sigma=1.0, epsilon=1.0
        )
        force_fn = create_force_fn(energy_fn)

        positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        forces = force_fn(positions)

        # Forces should be equal and opposite (Newton's third law)
        assert jnp.allclose(forces[0], -forces[1], atol=1e-5)

    def test_create_energy_fn_invalid_type(self):
        """Test that invalid potential type raises ValueError."""
        disp_fn, _ = create_displacement_fn(box_size=10.0)
        with pytest.raises(ValueError, match="Unknown potential type"):
            create_energy_fn(disp_fn, potential_type="invalid_potential")


# =============================================================================
# Tests for ForceFieldConfig
# =============================================================================


class TestForceFieldConfig:
    """Tests for ForceFieldConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ForceFieldConfig()

        assert config.potential_type == PotentialType.LENNARD_JONES
        assert config.sigma == 1.0
        assert config.epsilon == 1.0
        assert config.cutoff == 2.5
        assert config.box_size is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ForceFieldConfig(
            potential_type=PotentialType.SOFT_SPHERE,
            sigma=2.0,
            epsilon=0.5,
            cutoff=3.0,
            box_size=10.0,
        )

        assert config.potential_type == PotentialType.SOFT_SPHERE
        assert config.sigma == 2.0
        assert config.epsilon == 0.5
        assert config.cutoff == 3.0
        assert config.box_size == 10.0

    def test_config_with_string_potential_type(self):
        """Test config accepts string potential type for backward compatibility."""
        config = ForceFieldConfig(potential_type="lennard_jones")
        assert config.potential_type == "lennard_jones"


# =============================================================================
# Tests for ForceFieldOperator
# =============================================================================


class TestForceFieldOperator:
    """Tests for ForceFieldOperator."""

    @pytest.fixture
    def config(self):
        """Create default config for tests."""
        return ForceFieldConfig(
            potential_type=PotentialType.LENNARD_JONES,
            sigma=1.0,
            epsilon=1.0,
            box_size=10.0,
        )

    @pytest.fixture
    def operator(self, config):
        """Create operator for tests."""
        return ForceFieldOperator(config, rngs=nnx.Rngs(42))

    def test_initialization(self, operator, config):
        """Test operator initializes correctly."""
        assert operator.config.potential_type == config.potential_type
        assert operator.config.sigma == config.sigma
        assert operator.config.epsilon == config.epsilon

    def test_forward_pass_shape(self, operator):
        """Test forward pass produces correct output shapes."""
        n_particles = 10
        dim = 3
        positions = jax.random.uniform(
            jax.random.PRNGKey(0), (n_particles, dim), minval=0, maxval=10.0
        )

        data = {"positions": positions}
        result, state, metadata = operator.apply(data, {}, None)

        # Check output keys
        assert "positions" in result
        assert "energy" in result
        assert "forces" in result

        # Check shapes
        assert result["positions"].shape == (n_particles, dim)
        assert result["energy"].shape == ()
        assert result["forces"].shape == (n_particles, dim)

    def test_energy_is_scalar(self, operator):
        """Test that energy output is a scalar."""
        n_particles = 5
        positions = jax.random.uniform(
            jax.random.PRNGKey(0), (n_particles, 3), minval=0, maxval=10.0
        )

        data = {"positions": positions}
        result, _, _ = operator.apply(data, {}, None)

        assert result["energy"].ndim == 0

    def test_forces_are_negative_gradient(self, operator):
        """Test that forces are negative gradient of energy."""
        n_particles = 4
        positions = jax.random.uniform(
            jax.random.PRNGKey(0), (n_particles, 3), minval=2, maxval=8.0
        )

        data = {"positions": positions}
        result, _, _ = operator.apply(data, {}, None)

        # Compute gradient manually
        def energy_fn(pos):
            d = {"positions": pos}
            r, _, _ = operator.apply(d, {}, None)
            return r["energy"]

        grad_energy = jax.grad(energy_fn)(positions)
        expected_forces = -grad_energy

        # Forces should be -gradient of energy
        assert jnp.allclose(result["forces"], expected_forces, atol=1e-5)

    def test_batched_input(self, operator):
        """Test batched input processing."""
        batch_size = 4
        n_particles = 8
        dim = 3
        positions = jax.random.uniform(
            jax.random.PRNGKey(0), (batch_size, n_particles, dim), minval=0, maxval=10.0
        )

        data = {"positions": positions}
        result, _, _ = operator.apply(data, {}, None)

        assert result["energy"].shape == (batch_size,)
        assert result["forces"].shape == (batch_size, n_particles, dim)

    def test_gradient_flow(self, operator):
        """Test that gradients flow through the operator."""
        n_particles = 6
        positions = jax.random.uniform(
            jax.random.PRNGKey(0), (n_particles, 3), minval=2, maxval=8.0
        )

        def loss_fn(pos):
            data = {"positions": pos}
            result, _, _ = operator.apply(data, {}, None)
            return result["energy"]

        grads = jax.grad(loss_fn)(positions)

        # Gradients should exist and be non-zero
        assert grads is not None
        assert grads.shape == positions.shape
        assert jnp.abs(grads).max() > 1e-10

    def test_jit_compatibility(self, operator):
        """Test JIT compilation works."""
        n_particles = 8
        positions = jax.random.uniform(
            jax.random.PRNGKey(0), (n_particles, 3), minval=0, maxval=10.0
        )

        @jax.jit
        def compute(pos):
            data = {"positions": pos}
            result, _, _ = operator.apply(data, {}, None)
            return result["energy"], result["forces"]

        # Should compile and run without errors
        energy, forces = compute(positions)
        assert energy.shape == ()
        assert forces.shape == (n_particles, 3)

        # Second call should produce same result
        energy2, forces2 = compute(positions)
        assert jnp.allclose(energy, energy2)
        assert jnp.allclose(forces, forces2)

    def test_periodic_boundary_conditions(self):
        """Test periodic boundary conditions work."""
        config = ForceFieldConfig(box_size=5.0)
        operator = ForceFieldOperator(config, rngs=nnx.Rngs(42))

        # Particles near box edges
        positions = jnp.array([[0.5, 0.5, 0.5], [4.5, 4.5, 4.5]])

        data = {"positions": positions}
        result, _, _ = operator.apply(data, {}, None)

        # Should compute interaction across periodic boundary
        assert jnp.isfinite(result["energy"])
        assert jnp.all(jnp.isfinite(result["forces"]))

    def test_preserves_extra_data_keys(self, operator):
        """Test that extra keys in data are preserved."""
        positions = jax.random.uniform(jax.random.PRNGKey(0), (5, 3), minval=0, maxval=10.0)
        data = {"positions": positions, "species": jnp.array([0, 0, 1, 1, 0]), "metadata": "test"}

        result, _, _ = operator.apply(data, {}, None)

        assert "species" in result
        assert "metadata" in result
        assert result["metadata"] == "test"


# =============================================================================
# Tests for MDIntegratorConfig
# =============================================================================


class TestMDIntegratorConfig:
    """Tests for MDIntegratorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MDIntegratorConfig()

        assert config.integrator_type == "velocity_verlet"
        assert config.dt == 0.001
        assert config.n_steps == 100
        assert config.mass == 1.0
        assert config.kT == 1.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MDIntegratorConfig(
            integrator_type="nvt_langevin",
            dt=0.002,
            n_steps=500,
            mass=2.0,
            kT=0.5,
            gamma=2.0,
        )

        assert config.integrator_type == "nvt_langevin"
        assert config.dt == 0.002
        assert config.n_steps == 500
        assert config.mass == 2.0
        assert config.kT == 0.5
        assert config.gamma == 2.0


# =============================================================================
# Tests for MDIntegratorOperator
# =============================================================================


class TestMDIntegratorOperator:
    """Tests for MDIntegratorOperator."""

    @pytest.fixture
    def config(self):
        """Create default config for tests."""
        return MDIntegratorConfig(
            integrator_type="velocity_verlet",
            dt=0.001,
            n_steps=10,
            box_size=10.0,
        )

    @pytest.fixture
    def operator(self, config):
        """Create operator for tests."""
        return MDIntegratorOperator(config, rngs=nnx.Rngs(42))

    def test_initialization(self, operator, config):
        """Test operator initializes correctly."""
        assert operator.config.integrator_type == config.integrator_type
        assert operator.config.dt == config.dt
        assert operator.config.n_steps == config.n_steps

    def test_forward_pass_shape(self, operator):
        """Test forward pass produces correct output shapes."""
        n_particles = 10
        dim = 3
        positions = jax.random.uniform(
            jax.random.PRNGKey(0), (n_particles, dim), minval=2, maxval=8.0
        )
        velocities = jax.random.normal(jax.random.PRNGKey(1), (n_particles, dim)) * 0.1

        data = {"positions": positions, "velocities": velocities}
        result, state, metadata = operator.apply(data, {}, None)

        # Check output keys
        assert "positions" in result
        assert "velocities" in result
        assert "trajectory" in result

        # Check final state shapes
        assert result["positions"].shape == (n_particles, dim)
        assert result["velocities"].shape == (n_particles, dim)

        # Check trajectory shape (n_steps+1 frames including initial)
        assert result["trajectory"].shape[0] == operator.config.n_steps + 1
        assert result["trajectory"].shape[1:] == (n_particles, dim)

    def test_positions_change(self, operator):
        """Test that positions actually change during simulation."""
        n_particles = 5
        positions = jax.random.uniform(
            jax.random.PRNGKey(0), (n_particles, 3), minval=2, maxval=8.0
        )
        velocities = jax.random.normal(jax.random.PRNGKey(1), (n_particles, 3)) * 0.5

        data = {"positions": positions, "velocities": velocities}
        result, _, _ = operator.apply(data, {}, None)

        # Final positions should differ from initial
        assert not jnp.allclose(result["positions"], positions)

    def test_gradient_flow(self, operator):
        """Test that gradients flow through the integrator."""
        n_particles = 4
        positions = jax.random.uniform(
            jax.random.PRNGKey(0), (n_particles, 3), minval=2, maxval=8.0
        )
        velocities = jax.random.normal(jax.random.PRNGKey(1), (n_particles, 3)) * 0.1

        def loss_fn(pos):
            data = {"positions": pos, "velocities": velocities}
            result, _, _ = operator.apply(data, {}, None)
            return result["positions"].sum()

        grads = jax.grad(loss_fn)(positions)

        # Gradients should exist
        assert grads is not None
        assert grads.shape == positions.shape

    def test_jit_compatibility(self, operator):
        """Test JIT compilation works for MDIntegratorOperator."""
        n_particles = 6
        positions = jax.random.uniform(
            jax.random.PRNGKey(0), (n_particles, 3), minval=2, maxval=8.0
        )
        velocities = jax.random.normal(jax.random.PRNGKey(1), (n_particles, 3)) * 0.1

        @jax.jit
        def compute(pos, vel):
            data = {"positions": pos, "velocities": vel}
            result, _, _ = operator.apply(data, {}, None)
            return result["positions"], result["trajectory"]

        final_pos, trajectory = compute(positions, velocities)
        assert final_pos.shape == (n_particles, 3)
        assert jnp.all(jnp.isfinite(final_pos))

        # Second call should produce same result
        final_pos2, _ = compute(positions, velocities)
        assert jnp.allclose(final_pos, final_pos2)

    def test_invalid_integrator_type(self):
        """Test that invalid integrator type raises ValueError at init time."""
        config = MDIntegratorConfig(integrator_type="invalid_type", box_size=10.0)

        # Error should be raised at initialization time (fail fast)
        with pytest.raises(ValueError, match="Unknown integrator type"):
            MDIntegratorOperator(config, rngs=nnx.Rngs(42))

    def test_trajectory_first_frame_is_initial(self, operator):
        """Test that first frame of trajectory is initial positions."""
        n_particles = 5
        positions = jax.random.uniform(
            jax.random.PRNGKey(0), (n_particles, 3), minval=2, maxval=8.0
        )
        velocities = jax.random.normal(jax.random.PRNGKey(1), (n_particles, 3)) * 0.1

        data = {"positions": positions, "velocities": velocities}
        result, _, _ = operator.apply(data, {}, None)

        # First frame should be initial positions
        assert jnp.allclose(result["trajectory"][0], positions)


# =============================================================================
# Tests for Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_force_field_default(self):
        """Test default force field creation."""
        operator = create_force_field()
        assert isinstance(operator, ForceFieldOperator)
        assert operator.config.potential_type == PotentialType.LENNARD_JONES

    def test_create_force_field_custom(self):
        """Test custom force field creation."""
        operator = create_force_field(
            potential_type=PotentialType.SOFT_SPHERE,
            sigma=1.5,
            epsilon=2.0,
            box_size=15.0,
        )
        assert isinstance(operator, ForceFieldOperator)
        assert operator.config.potential_type == PotentialType.SOFT_SPHERE
        assert operator.config.sigma == 1.5
        assert operator.config.epsilon == 2.0

    def test_create_integrator_default(self):
        """Test default integrator creation."""
        operator = create_integrator()
        assert isinstance(operator, MDIntegratorOperator)
        assert operator.config.integrator_type == "velocity_verlet"

    def test_create_integrator_custom(self):
        """Test custom integrator creation."""
        operator = create_integrator(
            integrator_type="nvt_langevin",
            dt=0.002,
            n_steps=100,
            box_size=20.0,
            kT=0.5,
        )
        assert isinstance(operator, MDIntegratorOperator)
        assert operator.config.integrator_type == "nvt_langevin"
        assert operator.config.dt == 0.002
        assert operator.config.n_steps == 100


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_two_particles(self):
        """Test with minimum number of particles."""
        config = ForceFieldConfig(box_size=10.0)
        operator = ForceFieldOperator(config, rngs=nnx.Rngs(42))

        positions = jnp.array([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]])
        data = {"positions": positions}
        result, _, _ = operator.apply(data, {}, None)

        assert jnp.isfinite(result["energy"])
        assert jnp.all(jnp.isfinite(result["forces"]))

    def test_2d_system(self):
        """Test 2D particle system."""
        config = ForceFieldConfig(box_size=10.0)
        operator = ForceFieldOperator(config, rngs=nnx.Rngs(42))

        n_particles = 6
        positions = jax.random.uniform(
            jax.random.PRNGKey(0), (n_particles, 2), minval=0, maxval=10.0
        )

        data = {"positions": positions}
        result, _, _ = operator.apply(data, {}, None)

        assert result["forces"].shape == (n_particles, 2)

    def test_non_periodic_boundary(self):
        """Test non-periodic (free) boundary conditions."""
        config = ForceFieldConfig(box_size=None)  # No periodic boundaries
        operator = ForceFieldOperator(config, rngs=nnx.Rngs(42))

        positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        data = {"positions": positions}
        result, _, _ = operator.apply(data, {}, None)

        assert jnp.isfinite(result["energy"])

    def test_single_step_integrator(self):
        """Test integrator with single step."""
        config = MDIntegratorConfig(n_steps=1, box_size=10.0)
        operator = MDIntegratorOperator(config, rngs=nnx.Rngs(42))

        positions = jax.random.uniform(jax.random.PRNGKey(0), (5, 3), minval=2, maxval=8.0)
        velocities = jax.random.normal(jax.random.PRNGKey(1), (5, 3)) * 0.1

        data = {"positions": positions, "velocities": velocities}
        result, _, _ = operator.apply(data, {}, None)

        # Trajectory should have 2 frames (initial + 1 step)
        assert result["trajectory"].shape[0] == 2

    def test_zero_velocity_initialization(self):
        """Test integrator with zero initial velocities."""
        config = MDIntegratorConfig(n_steps=5, box_size=10.0)
        operator = MDIntegratorOperator(config, rngs=nnx.Rngs(42))

        positions = jax.random.uniform(jax.random.PRNGKey(0), (5, 3), minval=2, maxval=8.0)
        velocities = jnp.zeros((5, 3))

        data = {"positions": positions, "velocities": velocities}
        result, _, _ = operator.apply(data, {}, None)

        # Should still run (forces will accelerate particles)
        assert jnp.all(jnp.isfinite(result["positions"]))


# =============================================================================
# Tests for Technical Verification
# =============================================================================


class TestTechnicalVerification:
    """Tests for physics and numerical verification."""

    def test_morse_potential(self):
        """Test Morse potential energy function."""
        config = ForceFieldConfig(
            potential_type=PotentialType.MORSE,
            sigma=1.0,
            epsilon=1.0,
            alpha=5.0,
            box_size=10.0,
        )
        operator = ForceFieldOperator(config, rngs=nnx.Rngs(42))

        # Particles at equilibrium distance should have negative energy
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # At sigma
        data = {"positions": positions}
        result, _, _ = operator.apply(data, {}, None)

        assert jnp.isfinite(result["energy"])
        # Morse at r=sigma has well depth approximately -epsilon
        assert result["energy"] < 0

    def test_langevin_integrator_runs(self):
        """Test that Langevin (NVT) integrator runs without errors."""
        config = MDIntegratorConfig(
            integrator_type="nvt_langevin",
            dt=0.001,
            n_steps=10,
            box_size=10.0,
            kT=1.0,
            gamma=1.0,
        )
        operator = MDIntegratorOperator(config, rngs=nnx.Rngs(42))

        positions = jax.random.uniform(jax.random.PRNGKey(0), (5, 3), minval=2, maxval=8.0)
        velocities = jax.random.normal(jax.random.PRNGKey(1), (5, 3)) * 0.1

        data = {"positions": positions, "velocities": velocities}
        result, _, _ = operator.apply(data, {}, None)

        # Should produce valid output
        assert jnp.all(jnp.isfinite(result["positions"]))
        assert jnp.all(jnp.isfinite(result["velocities"]))
        assert jnp.all(jnp.isfinite(result["trajectory"]))

    def test_cutoff_excludes_distant_particles(self):
        """Test that cutoff affects energy calculation at appropriate distances."""
        config_cutoff = ForceFieldConfig(
            potential_type=PotentialType.LENNARD_JONES,
            sigma=1.0,
            epsilon=1.0,
            cutoff=2.5,  # Cutoff at 2.5 sigma
            box_size=20.0,
        )
        op_cutoff = ForceFieldOperator(config_cutoff, rngs=nnx.Rngs(42))

        # Particles at distance 2.0 (within cutoff)
        positions_within = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        result_within, _, _ = op_cutoff.apply({"positions": positions_within}, {}, None)

        # Particles at distance 3.0 (beyond cutoff of 2.5)
        positions_beyond = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        result_beyond, _, _ = op_cutoff.apply({"positions": positions_beyond}, {}, None)

        # Within cutoff should have non-zero energy
        assert jnp.abs(result_within["energy"]) > 0

        # Beyond cutoff should have zero energy
        assert jnp.abs(result_beyond["energy"]) < 1e-6

    def test_force_magnitude_decreases_with_distance(self):
        """Test that force magnitude decreases as particles separate."""
        config = ForceFieldConfig(box_size=50.0)
        operator = ForceFieldOperator(config, rngs=nnx.Rngs(42))

        # At r = 2 sigma (attractive region)
        positions_close = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        result_close, _, _ = operator.apply({"positions": positions_close}, {}, None)
        force_close = jnp.linalg.norm(result_close["forces"][0])

        # At r = 4 sigma
        positions_far = jnp.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        result_far, _, _ = operator.apply({"positions": positions_far}, {}, None)
        force_far = jnp.linalg.norm(result_far["forces"][0])

        # Force should be stronger at shorter distance
        assert force_close > force_far

    def test_numerical_stability_close_particles(self):
        """Test numerical stability with close (but not overlapping) particles."""
        config = ForceFieldConfig(box_size=10.0)
        operator = ForceFieldOperator(config, rngs=nnx.Rngs(42))

        # Particles at sigma distance (minimum of LJ potential)
        r_min = 2 ** (1 / 6)  # ~1.12
        positions = jnp.array([[0.0, 0.0, 0.0], [r_min, 0.0, 0.0]])

        data = {"positions": positions}
        result, _, _ = operator.apply(data, {}, None)

        # Should be numerically stable
        assert jnp.all(jnp.isfinite(result["energy"]))
        assert jnp.all(jnp.isfinite(result["forces"]))

    def test_many_particles_performance(self):
        """Test with larger particle system."""
        config = ForceFieldConfig(box_size=20.0)
        operator = ForceFieldOperator(config, rngs=nnx.Rngs(42))

        # 100 particles
        n_particles = 100
        positions = jax.random.uniform(
            jax.random.PRNGKey(0), (n_particles, 3), minval=0, maxval=20.0
        )

        data = {"positions": positions}
        result, _, _ = operator.apply(data, {}, None)

        # Should handle many particles
        assert result["forces"].shape == (n_particles, 3)
        assert jnp.all(jnp.isfinite(result["energy"]))

    def test_potential_type_enum_and_string_equivalence(self):
        """Test that enum and string potential types produce same results."""
        positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

        config_enum = ForceFieldConfig(potential_type=PotentialType.LENNARD_JONES, box_size=10.0)
        config_str = ForceFieldConfig(potential_type="lennard_jones", box_size=10.0)

        op_enum = ForceFieldOperator(config_enum, rngs=nnx.Rngs(42))
        op_str = ForceFieldOperator(config_str, rngs=nnx.Rngs(42))

        result_enum, _, _ = op_enum.apply({"positions": positions}, {}, None)
        result_str, _, _ = op_str.apply({"positions": positions}, {}, None)

        # Should produce identical results
        assert jnp.allclose(result_enum["energy"], result_str["energy"])
        assert jnp.allclose(result_enum["forces"], result_str["forces"])
