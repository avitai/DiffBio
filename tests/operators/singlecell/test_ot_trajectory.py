"""Tests for diffbio.operators.singlecell.ot_trajectory module.

These tests define the expected behavior of the DifferentiableOTTrajectory
operator, which uses entropy-regularised optimal transport to infer cell
lineage trajectories between two timepoints (Waddington-OT style).
"""

import jax
import jax.numpy as jnp
import pytest

from diffbio.operators.singlecell.ot_trajectory import (
    DifferentiableOTTrajectory,
    OTTrajectoryConfig,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CELLS_T1 = 20
N_CELLS_T2 = 25
N_GENES = 15
SEED = 42


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_config() -> OTTrajectoryConfig:
    """Return an OTTrajectoryConfig with defaults."""
    return OTTrajectoryConfig()


@pytest.fixture()
def small_config() -> OTTrajectoryConfig:
    """Return a small config suitable for testing."""
    return OTTrajectoryConfig(
        n_genes=N_GENES,
        sinkhorn_epsilon=0.1,
        sinkhorn_iters=50,
        growth_rate_regularization=1.0,
        interpolation_time=0.5,
    )


@pytest.fixture()
def sample_data() -> dict[str, jax.Array]:
    """Generate small two-timepoint expression data."""
    key = jax.random.key(SEED)
    k1, k2 = jax.random.split(key)
    counts_t1 = jnp.abs(jax.random.normal(k1, (N_CELLS_T1, N_GENES)))
    counts_t2 = jnp.abs(jax.random.normal(k2, (N_CELLS_T2, N_GENES)))
    return {"counts_t1": counts_t1, "counts_t2": counts_t2}


@pytest.fixture()
def operator(small_config: OTTrajectoryConfig) -> DifferentiableOTTrajectory:
    """Create an OT trajectory operator with test config."""
    return DifferentiableOTTrajectory(small_config)


# ===========================================================================
# TestOTTrajectoryConfig
# ===========================================================================


class TestOTTrajectoryConfig:
    """Tests for OTTrajectoryConfig defaults and custom values."""

    def test_defaults(self, default_config: OTTrajectoryConfig) -> None:
        """Default config values should match the specification."""
        assert default_config.n_genes == 200
        assert default_config.sinkhorn_epsilon == 0.1
        assert default_config.sinkhorn_iters == 100
        assert default_config.growth_rate_regularization == 1.0
        assert default_config.interpolation_time == 0.5

    def test_custom_values(self) -> None:
        """Custom config values should be stored correctly."""
        config = OTTrajectoryConfig(
            n_genes=50,
            sinkhorn_epsilon=0.05,
            sinkhorn_iters=200,
            growth_rate_regularization=2.0,
            interpolation_time=0.3,
        )
        assert config.n_genes == 50
        assert config.sinkhorn_epsilon == 0.05
        assert config.sinkhorn_iters == 200
        assert config.growth_rate_regularization == 2.0
        assert config.interpolation_time == 0.3

    def test_non_stochastic(self, default_config: OTTrajectoryConfig) -> None:
        """OT trajectory should not be stochastic by default."""
        assert not default_config.stochastic


# ===========================================================================
# TestOTTrajectory
# ===========================================================================


class TestOTTrajectory:
    """Tests for DifferentiableOTTrajectory output keys, shapes, and values."""

    def test_output_keys(
        self,
        operator: DifferentiableOTTrajectory,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Apply should return transport_plan, growth_rates, interpolated_counts."""
        result, state, metadata = operator.apply(sample_data, {}, None)

        assert "transport_plan" in result
        assert "growth_rates" in result
        assert "interpolated_counts" in result
        # Original keys are preserved
        assert "counts_t1" in result
        assert "counts_t2" in result

    def test_output_shapes(
        self,
        operator: DifferentiableOTTrajectory,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Output arrays should have correct shapes."""
        result, _, _ = operator.apply(sample_data, {}, None)

        assert result["transport_plan"].shape == (N_CELLS_T1, N_CELLS_T2)
        assert result["growth_rates"].shape == (N_CELLS_T1,)
        assert result["interpolated_counts"].shape == (N_CELLS_T1, N_GENES)

    def test_transport_plan_non_negative(
        self,
        operator: DifferentiableOTTrajectory,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Transport plan entries should be non-negative."""
        result, _, _ = operator.apply(sample_data, {}, None)
        assert jnp.all(result["transport_plan"] >= 0.0)

    def test_growth_rates_positive(
        self,
        operator: DifferentiableOTTrajectory,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Growth rates should be strictly positive."""
        result, _, _ = operator.apply(sample_data, {}, None)
        assert jnp.all(result["growth_rates"] > 0.0)

    def test_growth_rates_mean_near_one(
        self,
        operator: DifferentiableOTTrajectory,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Growth rates should be normalized to have mean approximately 1."""
        result, _, _ = operator.apply(sample_data, {}, None)
        mean_growth = jnp.mean(result["growth_rates"])
        assert jnp.abs(mean_growth - 1.0) < 0.1

    def test_interpolated_counts_finite(
        self,
        operator: DifferentiableOTTrajectory,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Interpolated counts should contain only finite values."""
        result, _, _ = operator.apply(sample_data, {}, None)
        assert jnp.all(jnp.isfinite(result["interpolated_counts"]))

    def test_state_and_metadata_passthrough(
        self,
        operator: DifferentiableOTTrajectory,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """State and metadata should be passed through unchanged."""
        input_state = {"some_state": 42}
        input_meta = {"some_meta": "value"}
        _, state, metadata = operator.apply(sample_data, input_state, input_meta)
        assert state == input_state
        assert metadata == input_meta


# ===========================================================================
# TestTransportPlan
# ===========================================================================


class TestTransportPlan:
    """Tests for the transport plan marginal constraints."""

    def test_row_marginals_approximate_uniform(
        self,
        operator: DifferentiableOTTrajectory,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Row sums of the transport plan should approximate uniform source marginal."""
        result, _, _ = operator.apply(sample_data, {}, None)
        plan = result["transport_plan"]
        row_sums = jnp.sum(plan, axis=1)
        expected = 1.0 / N_CELLS_T1
        # Allow tolerance due to entropy regularization
        assert jnp.allclose(row_sums, expected, atol=0.05)

    def test_column_marginals_approximate_uniform(
        self,
        operator: DifferentiableOTTrajectory,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Column sums should approximate uniform target marginal."""
        result, _, _ = operator.apply(sample_data, {}, None)
        plan = result["transport_plan"]
        col_sums = jnp.sum(plan, axis=0)
        expected = 1.0 / N_CELLS_T2
        assert jnp.allclose(col_sums, expected, atol=0.05)

    def test_plan_total_mass(
        self,
        operator: DifferentiableOTTrajectory,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Total mass of the transport plan should be approximately 1."""
        result, _, _ = operator.apply(sample_data, {}, None)
        total = jnp.sum(result["transport_plan"])
        assert jnp.abs(total - 1.0) < 0.05


# ===========================================================================
# TestGradientFlow
# ===========================================================================


class TestGradientFlow:
    """Tests for gradient flow through the operator."""

    def test_grads_through_transport_plan_cost(
        self,
        operator: DifferentiableOTTrajectory,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Gradients should flow from transport plan back to input counts."""
        counts_t1 = sample_data["counts_t1"]
        counts_t2 = sample_data["counts_t2"]

        def loss_fn(t1: jax.Array) -> jax.Array:
            data = {"counts_t1": t1, "counts_t2": counts_t2}
            result, _, _ = operator.apply(data, {}, None)
            return jnp.sum(result["transport_plan"])

        grad = jax.grad(loss_fn)(counts_t1)
        assert grad is not None
        assert grad.shape == counts_t1.shape
        assert jnp.isfinite(grad).all()

    def test_grads_through_growth_rates(
        self,
        operator: DifferentiableOTTrajectory,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Gradients should flow from growth rates back to input counts."""
        counts_t1 = sample_data["counts_t1"]
        counts_t2 = sample_data["counts_t2"]

        def loss_fn(t1: jax.Array) -> jax.Array:
            data = {"counts_t1": t1, "counts_t2": counts_t2}
            result, _, _ = operator.apply(data, {}, None)
            return jnp.sum(result["growth_rates"])

        grad = jax.grad(loss_fn)(counts_t1)
        assert grad is not None
        assert grad.shape == counts_t1.shape
        assert jnp.isfinite(grad).all()

    def test_grads_through_interpolated_counts(
        self,
        operator: DifferentiableOTTrajectory,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Gradients should flow from interpolated counts back to input counts."""
        counts_t1 = sample_data["counts_t1"]
        counts_t2 = sample_data["counts_t2"]

        def loss_fn(t1: jax.Array) -> jax.Array:
            data = {"counts_t1": t1, "counts_t2": counts_t2}
            result, _, _ = operator.apply(data, {}, None)
            return jnp.sum(result["interpolated_counts"])

        grad = jax.grad(loss_fn)(counts_t1)
        assert grad is not None
        assert grad.shape == counts_t1.shape
        assert jnp.isfinite(grad).all()


# ===========================================================================
# TestJITCompatibility
# ===========================================================================


class TestJITCompatibility:
    """Tests for JIT compilation of the operator."""

    def test_jit_apply(
        self,
        operator: DifferentiableOTTrajectory,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Apply should work under JIT compilation."""

        @jax.jit
        def run(data: dict[str, jax.Array]) -> tuple:
            return operator.apply(data, {}, None)

        result, _, _ = run(sample_data)
        assert result["transport_plan"].shape == (N_CELLS_T1, N_CELLS_T2)
        assert jnp.all(jnp.isfinite(result["transport_plan"]))

    def test_jit_gradient(
        self,
        operator: DifferentiableOTTrajectory,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Gradient computation should work under JIT compilation."""
        counts_t1 = sample_data["counts_t1"]
        counts_t2 = sample_data["counts_t2"]

        @jax.jit
        def grad_fn(t1: jax.Array) -> jax.Array:
            def loss(x: jax.Array) -> jax.Array:
                data = {"counts_t1": x, "counts_t2": counts_t2}
                result, _, _ = operator.apply(data, {}, None)
                return jnp.sum(result["interpolated_counts"])

            return jax.grad(loss)(t1)

        grad = grad_fn(counts_t1)
        assert jnp.isfinite(grad).all()


# ===========================================================================
# TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_equal_size_timepoints(self) -> None:
        """Operator should work when t1 and t2 have the same number of cells."""
        n = 15
        config = OTTrajectoryConfig(n_genes=N_GENES, sinkhorn_iters=50)
        op = DifferentiableOTTrajectory(config)

        key = jax.random.key(99)
        k1, k2 = jax.random.split(key)
        data = {
            "counts_t1": jnp.abs(jax.random.normal(k1, (n, N_GENES))),
            "counts_t2": jnp.abs(jax.random.normal(k2, (n, N_GENES))),
        }

        result, _, _ = op.apply(data, {}, None)
        assert result["transport_plan"].shape == (n, n)
        assert result["growth_rates"].shape == (n,)
        assert result["interpolated_counts"].shape == (n, N_GENES)
        assert jnp.all(jnp.isfinite(result["interpolated_counts"]))

    def test_single_cell_at_t1(self) -> None:
        """Operator should handle a single cell at timepoint 1."""
        config = OTTrajectoryConfig(n_genes=N_GENES, sinkhorn_iters=50)
        op = DifferentiableOTTrajectory(config)

        key = jax.random.key(77)
        k1, k2 = jax.random.split(key)
        n2 = 10
        data = {
            "counts_t1": jnp.abs(jax.random.normal(k1, (1, N_GENES))),
            "counts_t2": jnp.abs(jax.random.normal(k2, (n2, N_GENES))),
        }

        result, _, _ = op.apply(data, {}, None)
        assert result["transport_plan"].shape == (1, n2)
        assert result["growth_rates"].shape == (1,)
        assert result["interpolated_counts"].shape == (1, N_GENES)
        assert jnp.all(jnp.isfinite(result["interpolated_counts"]))

    def test_custom_interpolation_time(self) -> None:
        """Custom interpolation time should produce different results from default."""
        config_early = OTTrajectoryConfig(
            n_genes=N_GENES, sinkhorn_iters=50, interpolation_time=0.1
        )
        config_late = OTTrajectoryConfig(
            n_genes=N_GENES, sinkhorn_iters=50, interpolation_time=0.9
        )
        op_early = DifferentiableOTTrajectory(config_early)
        op_late = DifferentiableOTTrajectory(config_late)

        key = jax.random.key(55)
        k1, k2 = jax.random.split(key)
        data = {
            "counts_t1": jnp.abs(jax.random.normal(k1, (10, N_GENES))),
            "counts_t2": jnp.abs(jax.random.normal(k2, (12, N_GENES))),
        }

        result_early, _, _ = op_early.apply(data, {}, None)
        result_late, _, _ = op_late.apply(data, {}, None)

        # Different interpolation times should give different interpolated counts
        assert not jnp.allclose(
            result_early["interpolated_counts"],
            result_late["interpolated_counts"],
            atol=1e-3,
        )
