"""Tests for diffbio.operators.singlecell.trajectory module.

These tests define the expected behavior of the DifferentiablePseudotime and
DifferentiableFateProbability operators for trajectory inference in single-cell
analysis.
"""

import jax
import jax.numpy as jnp
import pytest

from diffbio.operators.singlecell.trajectory import (
    DifferentiableFateProbability,
    DifferentiablePseudotime,
    FateProbabilityConfig,
    PseudotimeConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linear_trajectory(
    n_cells: int = 30,
    n_features: int = 10,
    seed: int = 0,
) -> jax.Array:
    """Create embeddings along a linear trajectory with small noise.

    Cell i lives at roughly position ``i / n_cells`` along the first axis,
    with Gaussian noise in all dimensions so that k-NN still recovers the
    chain ordering.
    """
    key = jax.random.key(seed)
    t = jnp.linspace(0.0, 1.0, n_cells)
    noise = jax.random.normal(key, (n_cells, n_features)) * 0.01
    embeddings = noise.at[:, 0].add(t)
    return embeddings


def _make_branching_trajectory(
    n_cells_per_branch: int = 15,
    n_features: int = 10,
    seed: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Create a Y-shaped branching trajectory.

    Returns embeddings and terminal-state indices (last cell of each branch).
    """
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    # Trunk: along axis 0
    n_trunk = n_cells_per_branch
    t_trunk = jnp.linspace(0.0, 0.5, n_trunk)
    trunk = jax.random.normal(k1, (n_trunk, n_features)) * 0.01
    trunk = trunk.at[:, 0].add(t_trunk)

    # Branch A: continues axis 0, positive axis 1
    n_a = n_cells_per_branch
    t_a = jnp.linspace(0.5, 1.0, n_a)
    branch_a = jax.random.normal(k2, (n_a, n_features)) * 0.01
    branch_a = branch_a.at[:, 0].add(t_a)
    branch_a = branch_a.at[:, 1].add(jnp.linspace(0.0, 0.5, n_a))

    # Branch B: continues axis 0, negative axis 1
    n_b = n_cells_per_branch
    t_b = jnp.linspace(0.5, 1.0, n_b)
    branch_b = jax.random.normal(k3, (n_b, n_features)) * 0.01
    branch_b = branch_b.at[:, 0].add(t_b)
    branch_b = branch_b.at[:, 1].add(jnp.linspace(0.0, -0.5, n_b))

    embeddings = jnp.concatenate([trunk, branch_a, branch_b], axis=0)

    # Terminal states: last cell of branch A and last cell of branch B
    terminal_a = n_trunk + n_a - 1
    terminal_b = n_trunk + n_a + n_b - 1
    terminal_states = jnp.array([terminal_a, terminal_b])

    return embeddings, terminal_states


# ============================================================================
# PseudotimeConfig tests
# ============================================================================


class TestPseudotimeConfig:
    """Tests for PseudotimeConfig defaults and overrides."""

    def test_default_config(self) -> None:
        """Default config should have sensible defaults."""
        config = PseudotimeConfig()
        assert config.n_neighbors == 15
        assert config.n_diffusion_components == 10
        assert config.root_cell_index == 0
        assert config.metric == "euclidean"
        assert config.stochastic is False

    def test_custom_neighbors(self) -> None:
        """Custom n_neighbors should be stored."""
        config = PseudotimeConfig(n_neighbors=30)
        assert config.n_neighbors == 30


# ============================================================================
# DifferentiablePseudotime tests
# ============================================================================


class TestDifferentiablePseudotime:
    """Tests for DifferentiablePseudotime operator."""

    @pytest.fixture
    def config(self) -> PseudotimeConfig:
        """Small config for fast tests."""
        return PseudotimeConfig(
            n_neighbors=5,
            n_diffusion_components=3,
            root_cell_index=0,
        )

    @pytest.fixture
    def linear_data(self) -> dict[str, jax.Array]:
        """Linear trajectory data."""
        embeddings = _make_linear_trajectory(n_cells=30, n_features=10)
        return {"embeddings": embeddings}

    def test_output_keys(self, config: PseudotimeConfig, linear_data: dict) -> None:
        """Output must contain pseudotime, diffusion_components, transition_matrix."""
        op = DifferentiablePseudotime(config)
        result, state, meta = op.apply(linear_data, {}, None)

        assert "pseudotime" in result
        assert "diffusion_components" in result
        assert "transition_matrix" in result

    def test_output_shapes(self, config: PseudotimeConfig, linear_data: dict) -> None:
        """Check output array shapes match spec."""
        n_cells = linear_data["embeddings"].shape[0]
        n_comp = config.n_diffusion_components
        op = DifferentiablePseudotime(config)
        result, _, _ = op.apply(linear_data, {}, None)

        assert result["pseudotime"].shape == (n_cells,)
        assert result["diffusion_components"].shape == (n_cells, n_comp)
        assert result["transition_matrix"].shape == (n_cells, n_cells)

    def test_linear_trajectory(self, config: PseudotimeConfig) -> None:
        """On a clean linear trajectory, pseudotime should roughly increase."""
        embeddings = _make_linear_trajectory(n_cells=30, n_features=10)
        data = {"embeddings": embeddings}
        op = DifferentiablePseudotime(config)
        result, _, _ = op.apply(data, {}, None)

        pt = result["pseudotime"]
        # The first third of cells should generally have lower pseudotime
        # than the last third
        early_mean = jnp.mean(pt[:10])
        late_mean = jnp.mean(pt[20:])
        assert float(late_mean) > float(early_mean)

    def test_root_cell_has_zero_pseudotime(
        self, config: PseudotimeConfig, linear_data: dict
    ) -> None:
        """Root cell should have the smallest pseudotime (approx 0)."""
        op = DifferentiablePseudotime(config)
        result, _, _ = op.apply(linear_data, {}, None)

        pt = result["pseudotime"]
        root_idx = config.root_cell_index
        # Root should be the minimum (or very close)
        assert float(pt[root_idx]) == pytest.approx(0.0, abs=1e-6)

    def test_transition_matrix_row_stochastic(
        self, config: PseudotimeConfig, linear_data: dict
    ) -> None:
        """Transition matrix rows should sum to 1."""
        op = DifferentiablePseudotime(config)
        result, _, _ = op.apply(linear_data, {}, None)

        row_sums = jnp.sum(result["transition_matrix"], axis=1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-5)


# ============================================================================
# FateProbabilityConfig tests
# ============================================================================


class TestFateProbabilityConfig:
    """Tests for FateProbabilityConfig defaults."""

    def test_default_config(self) -> None:
        """Default config should have sensible defaults."""
        config = FateProbabilityConfig()
        assert config.n_macrostates == 2
        assert config.stochastic is False


# ============================================================================
# DifferentiableFateProbability tests
# ============================================================================


class TestDifferentiableFateProbability:
    """Tests for DifferentiableFateProbability operator."""

    @pytest.fixture
    def branching_data(self) -> dict[str, jax.Array]:
        """Branching trajectory with pre-computed transition matrix."""
        embeddings, terminal_states = _make_branching_trajectory()
        # Build a transition matrix from the pseudotime operator
        pt_config = PseudotimeConfig(
            n_neighbors=5,
            n_diffusion_components=3,
            root_cell_index=0,
        )
        pt_op = DifferentiablePseudotime(pt_config)
        pt_result, _, _ = pt_op.apply({"embeddings": embeddings}, {}, None)
        return {
            "transition_matrix": pt_result["transition_matrix"],
            "terminal_states": terminal_states,
        }

    def test_output_keys(self, branching_data: dict) -> None:
        """Output must contain fate_probabilities and macrostates."""
        config = FateProbabilityConfig(n_macrostates=2)
        op = DifferentiableFateProbability(config)
        result, _, _ = op.apply(branching_data, {}, None)

        assert "fate_probabilities" in result
        assert "macrostates" in result

    def test_fate_probabilities_sum_to_one(self, branching_data: dict) -> None:
        """Each cell's fate probabilities should sum to ~1."""
        config = FateProbabilityConfig(n_macrostates=2)
        op = DifferentiableFateProbability(config)
        result, _, _ = op.apply(branching_data, {}, None)

        row_sums = jnp.sum(result["fate_probabilities"], axis=1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-4)

    def test_absorbing_states_certain(self, branching_data: dict) -> None:
        """Terminal cells should have probability 1 for their own absorbing state."""
        config = FateProbabilityConfig(n_macrostates=2)
        op = DifferentiableFateProbability(config)
        result, _, _ = op.apply(branching_data, {}, None)

        terminal = branching_data["terminal_states"]
        fate = result["fate_probabilities"]

        for i in range(terminal.shape[0]):
            t_idx = int(terminal[i])
            # The terminal cell should have prob ~1 for its own state
            assert float(fate[t_idx, i]) == pytest.approx(1.0, abs=1e-4)

    def test_branching_trajectory(self, branching_data: dict) -> None:
        """Cells near branch A should have higher probability for terminal A."""
        embeddings, terminal_states = _make_branching_trajectory()
        n_per_branch = 15
        n_trunk = n_per_branch

        config = FateProbabilityConfig(n_macrostates=2)
        op = DifferentiableFateProbability(config)
        result, _, _ = op.apply(branching_data, {}, None)

        fate = result["fate_probabilities"]

        # Cells in branch A (indices n_trunk to n_trunk+n_per_branch)
        # should lean toward terminal A (index 0)
        branch_a_fate_for_a = jnp.mean(fate[n_trunk : n_trunk + n_per_branch, 0])

        # Cells in branch B should lean toward terminal B (index 1)
        branch_b_start = n_trunk + n_per_branch
        branch_b_fate_for_b = jnp.mean(fate[branch_b_start:, 1])

        assert float(branch_a_fate_for_a) > 0.5
        assert float(branch_b_fate_for_b) > 0.5


# ============================================================================
# Gradient flow tests
# ============================================================================


class TestGradientFlow:
    """Tests for end-to-end differentiability."""

    def test_pseudotime_gradient(self) -> None:
        """Gradient of pseudotime sum wrt embeddings should be non-zero."""
        config = PseudotimeConfig(
            n_neighbors=5,
            n_diffusion_components=3,
            root_cell_index=0,
        )
        op = DifferentiablePseudotime(config)
        embeddings = _make_linear_trajectory(n_cells=20, n_features=8)

        def loss_fn(emb: jax.Array) -> jax.Array:
            data = {"embeddings": emb}
            result, _, _ = op.apply(data, {}, None)
            return jnp.sum(result["pseudotime"])

        grad = jax.grad(loss_fn)(embeddings)
        assert grad is not None
        assert grad.shape == embeddings.shape
        assert jnp.isfinite(grad).all()
        assert jnp.any(grad != 0.0)

    def test_fate_gradient(self) -> None:
        """Gradient of fate probabilities sum wrt transition_matrix should be non-zero."""
        config = FateProbabilityConfig(n_macrostates=2)
        op = DifferentiableFateProbability(config)

        n_cells = 20
        # Build a simple row-stochastic matrix
        key = jax.random.key(10)
        raw = jax.random.uniform(key, (n_cells, n_cells)) + 0.01
        transition_matrix = raw / raw.sum(axis=1, keepdims=True)
        terminal_states = jnp.array([n_cells - 2, n_cells - 1])

        def loss_fn(t_matrix: jax.Array) -> jax.Array:
            data = {
                "transition_matrix": t_matrix,
                "terminal_states": terminal_states,
            }
            result, _, _ = op.apply(data, {}, None)
            return jnp.sum(result["fate_probabilities"])

        grad = jax.grad(loss_fn)(transition_matrix)
        assert grad is not None
        assert grad.shape == transition_matrix.shape
        assert jnp.isfinite(grad).all()
        assert jnp.any(grad != 0.0)


# ============================================================================
# JIT compatibility tests
# ============================================================================


class TestJITCompatibility:
    """Tests for JAX JIT compilation."""

    def test_jit_pseudotime(self) -> None:
        """Pseudotime operator should compile under jax.jit."""
        config = PseudotimeConfig(
            n_neighbors=5,
            n_diffusion_components=3,
            root_cell_index=0,
        )
        op = DifferentiablePseudotime(config)
        embeddings = _make_linear_trajectory(n_cells=20, n_features=8)
        data = {"embeddings": embeddings}

        @jax.jit
        def run(d: dict) -> tuple:
            return op.apply(d, {}, None)

        result, _, _ = run(data)
        assert jnp.isfinite(result["pseudotime"]).all()

    def test_jit_fate(self) -> None:
        """Fate probability operator should compile under jax.jit."""
        config = FateProbabilityConfig(n_macrostates=2)
        op = DifferentiableFateProbability(config)

        n_cells = 20
        key = jax.random.key(10)
        raw = jax.random.uniform(key, (n_cells, n_cells)) + 0.01
        transition_matrix = raw / raw.sum(axis=1, keepdims=True)
        terminal_states = jnp.array([n_cells - 2, n_cells - 1])
        data = {
            "transition_matrix": transition_matrix,
            "terminal_states": terminal_states,
        }

        @jax.jit
        def run(d: dict) -> tuple:
            return op.apply(d, {}, None)

        result, _, _ = run(data)
        assert jnp.isfinite(result["fate_probabilities"]).all()


# ============================================================================
# Edge case tests
# ============================================================================


class TestEdgeCases:
    """Tests for boundary and degenerate inputs."""

    def test_small_graph(self) -> None:
        """Pseudotime should work with as few as 5 cells."""
        config = PseudotimeConfig(
            n_neighbors=3,
            n_diffusion_components=2,
            root_cell_index=0,
        )
        key = jax.random.key(99)
        embeddings = jax.random.normal(key, (5, 4))
        data = {"embeddings": embeddings}

        op = DifferentiablePseudotime(config)
        result, _, _ = op.apply(data, {}, None)

        assert result["pseudotime"].shape == (5,)
        assert jnp.isfinite(result["pseudotime"]).all()

    def test_single_terminal_state(self) -> None:
        """With one terminal state, all fate goes to that state."""
        config = FateProbabilityConfig(n_macrostates=1)
        op = DifferentiableFateProbability(config)

        n_cells = 10
        key = jax.random.key(20)
        raw = jax.random.uniform(key, (n_cells, n_cells)) + 0.01
        transition_matrix = raw / raw.sum(axis=1, keepdims=True)
        terminal_states = jnp.array([n_cells - 1])
        data = {
            "transition_matrix": transition_matrix,
            "terminal_states": terminal_states,
        }

        result, _, _ = op.apply(data, {}, None)
        fate = result["fate_probabilities"]

        # All probabilities should be 1 for the single terminal state
        assert jnp.allclose(fate, 1.0, atol=1e-4)
