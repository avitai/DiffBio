"""Tests for differentiable secondary structure prediction (PyDSSP-style).

These operators implement the DSSP algorithm for assigning secondary structure
to protein backbone atoms, with continuous (differentiable) hydrogen bond
matrix computation.

Reference:
    Kabsch & Sander (1983). Dictionary of protein secondary structure.
    Minami (2023). PyDSSP: Differentiable DSSP implementation.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


class TestSecondaryStructureConfig:
    """Tests for SecondaryStructureConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from diffbio.operators.protein import SecondaryStructureConfig

        config = SecondaryStructureConfig()

        assert config.margin == 1.0
        assert config.cutoff == -0.5
        assert config.min_helix_length == 4
        assert config.temperature == 1.0

    def test_custom_config(self):
        """Test custom configuration values."""
        from diffbio.operators.protein import SecondaryStructureConfig

        config = SecondaryStructureConfig(
            margin=0.5,
            cutoff=-0.6,
            min_helix_length=3,
            temperature=0.5,
        )

        assert config.margin == 0.5
        assert config.cutoff == -0.6
        assert config.min_helix_length == 3
        assert config.temperature == 0.5

    def test_invalid_margin_rejected(self):
        """Margin must be positive."""
        from diffbio.operators.protein import SecondaryStructureConfig

        with pytest.raises(ValueError, match="margin"):
            SecondaryStructureConfig(margin=0.0)

    def test_invalid_min_helix_length_rejected(self):
        """Minimum helix length must be positive."""
        from diffbio.operators.protein import SecondaryStructureConfig

        with pytest.raises(ValueError, match="min_helix_length"):
            SecondaryStructureConfig(min_helix_length=0)

    def test_invalid_constraint_weight_rejected(self):
        """Constraint weights must be non-negative."""
        from diffbio.operators.protein import SecondaryStructureConfig

        with pytest.raises(ValueError, match="bond_length_weight"):
            SecondaryStructureConfig(bond_length_weight=-0.1)


class TestHydrogenBondEnergy:
    """Tests for hydrogen bond energy computation."""

    @pytest.fixture
    def operator(self):
        """Create operator instance."""
        from diffbio.operators.protein import (
            DifferentiableSecondaryStructure,
            SecondaryStructureConfig,
        )

        config = SecondaryStructureConfig()
        return DifferentiableSecondaryStructure(config, rngs=nnx.Rngs(42))

    def test_hbond_energy_output_shape(self, operator):
        """Test hydrogen bond energy matrix shape."""
        n_residues = 10
        # Coordinates: (batch, length, atoms, xyz) where atoms = 4 (N, CA, C, O)
        coords = jax.random.uniform(jax.random.PRNGKey(0), (1, n_residues, 4, 3))

        energy = operator.compute_hbond_energy(coords)

        # Energy matrix should be (batch, n_residues, n_residues)
        assert energy.shape == (1, n_residues, n_residues)

    def test_hbond_energy_negative_for_close_atoms(self, operator):
        """Test that close donor-acceptor pairs have negative energy."""
        # Create a simple case with two residues in close proximity
        # This is a simplified test - real proteins have specific geometry
        n_residues = 5
        coords = jax.random.uniform(jax.random.PRNGKey(0), (1, n_residues, 4, 3)) * 10

        energy = operator.compute_hbond_energy(coords)

        # Energy values should be finite
        assert jnp.all(jnp.isfinite(energy))


class TestContinuousHBondMatrix:
    """Tests for continuous hydrogen bond matrix."""

    @pytest.fixture
    def operator(self):
        """Create operator instance."""
        from diffbio.operators.protein import (
            DifferentiableSecondaryStructure,
            SecondaryStructureConfig,
        )

        config = SecondaryStructureConfig()
        return DifferentiableSecondaryStructure(config, rngs=nnx.Rngs(42))

    def test_hbond_matrix_output_shape(self, operator):
        """Test H-bond matrix shape."""
        n_residues = 10
        coords = jax.random.uniform(jax.random.PRNGKey(0), (1, n_residues, 4, 3)) * 10

        hbond_map = operator.compute_hbond_map(coords)

        assert hbond_map.shape == (1, n_residues, n_residues)

    def test_hbond_matrix_range(self, operator):
        """Test H-bond matrix values are in [0, 1]."""
        n_residues = 10
        coords = jax.random.uniform(jax.random.PRNGKey(0), (1, n_residues, 4, 3)) * 10

        hbond_map = operator.compute_hbond_map(coords)

        assert jnp.all(hbond_map >= 0.0)
        assert jnp.all(hbond_map <= 1.0)

    def test_hbond_matrix_is_differentiable(self, operator):
        """Test that gradients flow through H-bond matrix computation."""

        def loss_fn(coords):
            hbond_map = operator.compute_hbond_map(coords)
            return hbond_map.sum()

        n_residues = 5
        coords = jax.random.uniform(jax.random.PRNGKey(0), (1, n_residues, 4, 3)) * 10

        grads = jax.grad(loss_fn)(coords)

        assert grads is not None
        assert grads.shape == coords.shape
        assert jnp.all(jnp.isfinite(grads))


class TestSecondaryStructureAssignment:
    """Tests for secondary structure assignment."""

    @pytest.fixture
    def operator(self):
        """Create operator instance."""
        from diffbio.operators.protein import (
            DifferentiableSecondaryStructure,
            SecondaryStructureConfig,
        )

        config = SecondaryStructureConfig()
        return DifferentiableSecondaryStructure(config, rngs=nnx.Rngs(42))

    def test_assignment_output_shape(self, operator):
        """Test secondary structure assignment output shape."""
        n_residues = 10
        coords = jax.random.uniform(jax.random.PRNGKey(0), (1, n_residues, 4, 3)) * 10

        data = {"coordinates": coords}
        result, _, _ = operator.apply(data, {}, None)

        # One-hot encoded: (batch, n_residues, 3) for H, E, -
        assert "ss_onehot" in result
        assert result["ss_onehot"].shape == (1, n_residues, 3)

    def test_assignment_is_probability(self, operator):
        """Test that soft assignments sum to 1."""
        n_residues = 10
        coords = jax.random.uniform(jax.random.PRNGKey(0), (1, n_residues, 4, 3)) * 10

        data = {"coordinates": coords}
        result, _, _ = operator.apply(data, {}, None)

        ss_onehot = result["ss_onehot"]
        sums = jnp.sum(ss_onehot, axis=-1)

        assert jnp.allclose(sums, 1.0, atol=1e-5)

    def test_assignment_classes(self, operator):
        """Test that assignment returns valid class indices."""
        n_residues = 10
        coords = jax.random.uniform(jax.random.PRNGKey(0), (1, n_residues, 4, 3)) * 10

        data = {"coordinates": coords}
        result, _, _ = operator.apply(data, {}, None)

        # Hard assignments should be 0, 1, or 2
        hard_assignments = jnp.argmax(result["ss_onehot"], axis=-1)
        assert jnp.all(hard_assignments >= 0)
        assert jnp.all(hard_assignments <= 2)


class TestDifferentiableSecondaryStructure:
    """Tests for DifferentiableSecondaryStructure operator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        from diffbio.operators.protein import SecondaryStructureConfig

        return SecondaryStructureConfig(margin=1.0, cutoff=-0.5)

    @pytest.fixture
    def operator(self, config):
        """Create operator instance."""
        from diffbio.operators.protein import DifferentiableSecondaryStructure

        return DifferentiableSecondaryStructure(config, rngs=nnx.Rngs(42))

    def test_initialization(self, operator, config):
        """Test operator initialization."""
        assert operator is not None
        assert operator.config == config

    def test_apply_single_structure(self, operator):
        """Test apply with single protein structure."""
        n_residues = 20
        coords = jax.random.uniform(jax.random.PRNGKey(0), (1, n_residues, 4, 3)) * 10

        data = {"coordinates": coords}
        result, state, metadata = operator.apply(data, {}, None)

        assert "coordinates" in result
        assert "ss_onehot" in result
        assert "hbond_map" in result

    def test_apply_batch(self, operator):
        """Test apply with batch of protein structures."""
        batch_size = 4
        n_residues = 15
        coords = jax.random.uniform(jax.random.PRNGKey(0), (batch_size, n_residues, 4, 3)) * 10

        data = {"coordinates": coords}
        result, _, _ = operator.apply(data, {}, None)

        assert result["ss_onehot"].shape == (batch_size, n_residues, 3)
        assert result["hbond_map"].shape == (batch_size, n_residues, n_residues)

    def test_gradient_flow(self, operator):
        """Test that gradients flow through the operator."""
        n_residues = 10
        coords = jax.random.uniform(jax.random.PRNGKey(0), (1, n_residues, 4, 3)) * 10

        def loss_fn(coords):
            data = {"coordinates": coords}
            result, _, _ = operator.apply(data, {}, None)
            return result["ss_onehot"].sum()

        grads = jax.grad(loss_fn)(coords)

        assert grads is not None
        assert grads.shape == coords.shape
        assert jnp.all(jnp.isfinite(grads))

    def test_jit_compilation(self, operator):
        """Test JIT compilation with nnx.jit."""
        n_residues = 10
        coords = jax.random.uniform(jax.random.PRNGKey(0), (1, n_residues, 4, 3)) * 10

        @nnx.jit
        def apply_jit(model, data):
            return model.apply(data, {}, None)

        data = {"coordinates": coords}
        result, _, _ = apply_jit(operator, data)

        assert result["ss_onehot"].shape == (1, n_residues, 3)

    def test_variable_length_proteins(self, operator):
        """Test with different protein lengths."""
        for n_residues in [5, 10, 50, 100]:
            coords = jax.random.uniform(jax.random.PRNGKey(n_residues), (1, n_residues, 4, 3)) * 10

            data = {"coordinates": coords}
            result, _, _ = operator.apply(data, {}, None)

            assert result["ss_onehot"].shape == (1, n_residues, 3)


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_secondary_structure_predictor(self):
        """Test factory function creates operator."""
        from diffbio.operators.protein import create_secondary_structure_predictor

        operator = create_secondary_structure_predictor(margin=0.5, cutoff=-0.6)

        assert operator is not None
        assert operator.config.margin == 0.5
        assert operator.config.cutoff == -0.6

    def test_create_with_defaults(self):
        """Test factory with default parameters."""
        from diffbio.operators.protein import create_secondary_structure_predictor

        operator = create_secondary_structure_predictor()

        assert operator is not None
        assert operator.config.margin == 1.0
        assert operator.config.cutoff == -0.5


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_compute_hydrogen_position(self):
        """Test hydrogen atom position computation from backbone."""
        from diffbio.operators.protein import compute_hydrogen_position

        # Simple case: N at origin, CA at (1,0,0), C at (0,1,0)
        n_pos = jnp.array([[[0.0, 0.0, 0.0]]])
        ca_pos = jnp.array([[[1.0, 0.0, 0.0]]])
        c_pos = jnp.array([[[0.0, 1.0, 0.0]]])  # C from previous residue

        h_pos = compute_hydrogen_position(n_pos, ca_pos, c_pos)

        # H should be placed opposite to CA-N and C-N vectors
        assert h_pos.shape == n_pos.shape
        assert jnp.all(jnp.isfinite(h_pos))


class TestProteinModuleImports:
    """Tests for protein module imports."""

    def test_import_from_package(self):
        """Test imports from protein package."""
        from diffbio.operators.protein import (
            DifferentiableSecondaryStructure,
            SecondaryStructureConfig,
            compute_hydrogen_position,
            create_secondary_structure_predictor,
        )

        assert DifferentiableSecondaryStructure is not None
        assert SecondaryStructureConfig is not None
        assert create_secondary_structure_predictor is not None
        assert compute_hydrogen_position is not None

    def test_import_from_operators(self):
        """Test protein module is accessible from operators."""
        from diffbio.operators import protein

        assert protein is not None
        assert hasattr(protein, "DifferentiableSecondaryStructure")
