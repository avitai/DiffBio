"""Tests for diffbio.operators.singlecell.simulation module.

These tests define the expected behavior of the DifferentiableSimulator
operator, a Splatter-style single-cell count simulator with fully
differentiable Gamma-Poisson model, DE groups, batch effects, and dropout.
"""

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.singlecell.simulation import (
    DifferentiableSimulator,
    SimulationConfig,
)

N_CELLS = 50
N_GENES = 30
N_GROUPS = 2
N_BATCHES = 1


@pytest.fixture
def default_config() -> SimulationConfig:
    """Provide default simulation configuration with small dimensions."""
    return SimulationConfig(
        n_cells=N_CELLS,
        n_genes=N_GENES,
        n_groups=N_GROUPS,
        n_batches=N_BATCHES,
    )


@pytest.fixture
def simulator(default_config: SimulationConfig, rngs: nnx.Rngs) -> DifferentiableSimulator:
    """Provide an initialized simulator."""
    return DifferentiableSimulator(default_config, rngs=rngs)


@pytest.fixture
def sample_output(
    simulator: DifferentiableSimulator,
) -> tuple[dict, dict, dict[str, Any] | None]:
    """Provide a sample apply() output for reuse."""
    rng = jax.random.key(99)
    random_params = simulator.generate_random_params(rng, {})
    return simulator.apply({}, {}, None, random_params=random_params)


class TestSimulationConfig:
    """Tests for SimulationConfig defaults and custom values."""

    def test_defaults(self) -> None:
        """Test that default configuration matches Splatter defaults."""
        config = SimulationConfig()
        assert config.n_cells == 500
        assert config.n_genes == 200
        assert config.n_groups == 3
        assert config.n_batches == 1
        assert config.mean_shape == 0.6
        assert config.mean_rate == 0.3
        assert config.lib_loc == 11.0
        assert config.lib_scale == 0.2
        assert config.de_prob == 0.1
        assert config.de_fac_loc == 0.1
        assert config.de_fac_scale == 0.4
        assert config.dropout_mid == -1.0
        assert config.dropout_shape == -0.5
        assert config.stochastic is True
        assert config.stream_name == "sample"

    def test_custom(self) -> None:
        """Test custom configuration values are stored correctly."""
        config = SimulationConfig(
            n_cells=100,
            n_genes=50,
            n_groups=4,
            n_batches=3,
            mean_shape=1.0,
            mean_rate=0.5,
            de_prob=0.2,
        )
        assert config.n_cells == 100
        assert config.n_genes == 50
        assert config.n_groups == 4
        assert config.n_batches == 3
        assert config.mean_shape == 1.0
        assert config.mean_rate == 0.5
        assert config.de_prob == 0.2


class TestDifferentiableSimulator:
    """Tests for output keys, shapes, and basic properties."""

    def test_output_keys(self, sample_output: tuple) -> None:
        """Test that apply returns all expected output keys."""
        result, _, _ = sample_output
        expected_keys = {"counts", "group_labels", "batch_labels", "gene_means", "de_mask"}
        assert expected_keys.issubset(set(result.keys()))

    def test_counts_shape(self, sample_output: tuple) -> None:
        """Test that counts have shape (n_cells, n_genes)."""
        result, _, _ = sample_output
        assert result["counts"].shape == (N_CELLS, N_GENES)

    def test_group_labels_shape(self, sample_output: tuple) -> None:
        """Test that group_labels have shape (n_cells,)."""
        result, _, _ = sample_output
        assert result["group_labels"].shape == (N_CELLS,)

    def test_batch_labels_shape(self, sample_output: tuple) -> None:
        """Test that batch_labels have shape (n_cells,)."""
        result, _, _ = sample_output
        assert result["batch_labels"].shape == (N_CELLS,)

    def test_gene_means_shape(self, sample_output: tuple) -> None:
        """Test that gene_means have shape (n_genes,)."""
        result, _, _ = sample_output
        assert result["gene_means"].shape == (N_GENES,)

    def test_de_mask_shape(self, sample_output: tuple) -> None:
        """Test that de_mask has shape (n_groups, n_genes)."""
        result, _, _ = sample_output
        assert result["de_mask"].shape == (N_GROUPS, N_GENES)

    def test_counts_non_negative(self, sample_output: tuple) -> None:
        """All simulated counts must be non-negative."""
        result, _, _ = sample_output
        assert jnp.all(result["counts"] >= 0.0)

    def test_counts_finite(self, sample_output: tuple) -> None:
        """All simulated counts must be finite."""
        result, _, _ = sample_output
        assert jnp.all(jnp.isfinite(result["counts"]))

    def test_group_labels_valid(self, sample_output: tuple) -> None:
        """Group labels must be in [0, n_groups)."""
        result, _, _ = sample_output
        labels = result["group_labels"]
        assert jnp.all(labels >= 0)
        assert jnp.all(labels < N_GROUPS)

    def test_batch_labels_valid(self, sample_output: tuple) -> None:
        """Batch labels must be in [0, n_batches)."""
        result, _, _ = sample_output
        labels = result["batch_labels"]
        assert jnp.all(labels >= 0)
        assert jnp.all(labels < N_BATCHES)

    def test_gene_means_positive(self, sample_output: tuple) -> None:
        """Gene means must be strictly positive."""
        result, _, _ = sample_output
        assert jnp.all(result["gene_means"] > 0.0)

    def test_state_and_metadata_passthrough(self, simulator: DifferentiableSimulator) -> None:
        """State and metadata should pass through unchanged."""
        rng = jax.random.key(0)
        rp = simulator.generate_random_params(rng, {})
        in_state = {"key": "value"}
        in_meta = {"info": 42}
        _, out_state, out_meta = simulator.apply({}, in_state, in_meta, random_params=rp)
        assert out_state is in_state
        assert out_meta is in_meta

    def test_generate_random_params(self, simulator: DifferentiableSimulator) -> None:
        """generate_random_params should return a dict of JAX arrays."""
        rng = jax.random.key(0)
        rp = simulator.generate_random_params(rng, {})
        assert isinstance(rp, dict)
        # Should have keys for each sampling step
        assert "gene_means_key" in rp
        assert "lib_sizes_key" in rp
        assert "group_key" in rp
        assert "de_mask_key" in rp
        assert "de_fold_key" in rp
        assert "poisson_key" in rp


class TestDEGroups:
    """Tests verifying that DE genes have different means across groups."""

    def test_de_genes_differ_across_groups(self, rngs: nnx.Rngs) -> None:
        """DE genes should produce different per-group fold-changes."""
        config = SimulationConfig(
            n_cells=N_CELLS,
            n_genes=N_GENES,
            n_groups=N_GROUPS,
            n_batches=1,
            de_prob=0.5,  # High DE fraction to ensure some DE genes
            de_fac_loc=1.0,  # Strong fold-change
            de_fac_scale=0.1,
        )
        sim = DifferentiableSimulator(config, rngs=rngs)
        rng = jax.random.key(7)
        rp = sim.generate_random_params(rng, {})
        result, _, _ = sim.apply({}, {}, None, random_params=rp)

        de_mask = result["de_mask"]
        # At least some genes should be DE (marked in de_mask)
        total_de = jnp.sum(de_mask)
        assert total_de > 0, "No DE genes were generated"

    def test_de_mask_binary(self, sample_output: tuple) -> None:
        """DE mask should contain values in [0, 1]."""
        result, _, _ = sample_output
        de_mask = result["de_mask"]
        assert jnp.all(de_mask >= 0.0)
        assert jnp.all(de_mask <= 1.0)


class TestGradientFlow:
    """Tests verifying gradients flow through learnable parameters."""

    def test_grads_through_gene_means_params(
        self, default_config: SimulationConfig, rngs: nnx.Rngs
    ) -> None:
        """Gradients should flow through gene_means learnable parameters."""
        sim = DifferentiableSimulator(default_config, rngs=rngs)
        rng = jax.random.key(0)
        rp = sim.generate_random_params(rng, {})

        @nnx.value_and_grad
        def loss_fn(model: DifferentiableSimulator) -> jax.Array:
            result, _, _ = model.apply({}, {}, None, random_params=rp)
            return jnp.mean(result["counts"])

        loss, grads = loss_fn(sim)
        assert jnp.isfinite(loss)
        # The gene_means_logits parameter should receive gradients
        assert hasattr(grads, "gene_means_logits")
        grad_vals = grads.gene_means_logits[...]
        assert jnp.any(grad_vals != 0.0), "Gene means gradients are all zero"

    def test_grads_through_batch_shift_params(self, rngs: nnx.Rngs) -> None:
        """Gradients should flow through batch_shift learnable parameters."""
        config = SimulationConfig(
            n_cells=N_CELLS,
            n_genes=N_GENES,
            n_groups=N_GROUPS,
            n_batches=2,  # Need >1 batch to test batch effects
        )
        sim = DifferentiableSimulator(config, rngs=rngs)
        rng = jax.random.key(0)
        rp = sim.generate_random_params(rng, {})

        @nnx.value_and_grad
        def loss_fn(model: DifferentiableSimulator) -> jax.Array:
            result, _, _ = model.apply({}, {}, None, random_params=rp)
            return jnp.mean(result["counts"])

        loss, grads = loss_fn(sim)
        assert jnp.isfinite(loss)
        assert hasattr(grads, "batch_shift")
        grad_vals = grads.batch_shift[...]
        assert jnp.any(grad_vals != 0.0), "Batch shift gradients are all zero"


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    def test_jit_apply(self, simulator: DifferentiableSimulator) -> None:
        """jax.jit(apply) should compile and produce correct shapes."""
        rng = jax.random.key(0)
        rp = simulator.generate_random_params(rng, {})

        @jax.jit
        def jit_apply(data: dict, state: dict) -> tuple[dict, dict, dict[str, Any] | None]:
            return simulator.apply(data, state, None, random_params=rp)

        result, _, _ = jit_apply({}, {})
        assert result["counts"].shape == (N_CELLS, N_GENES)
        assert jnp.all(jnp.isfinite(result["counts"]))

    def test_jit_gradient(self, default_config: SimulationConfig, rngs: nnx.Rngs) -> None:
        """jax.jit + nnx.value_and_grad should work together."""
        sim = DifferentiableSimulator(default_config, rngs=rngs)
        rng = jax.random.key(0)
        rp = sim.generate_random_params(rng, {})

        @jax.jit
        @nnx.value_and_grad
        def jit_loss(model: DifferentiableSimulator) -> jax.Array:
            result, _, _ = model.apply({}, {}, None, random_params=rp)
            return jnp.mean(result["counts"])

        loss, grads = jit_loss(sim)
        assert jnp.isfinite(loss)
        assert hasattr(grads, "gene_means_logits")


class TestEdgeCases:
    """Tests for edge cases: single group, single batch, etc."""

    def test_single_group_no_de(self, rngs: nnx.Rngs) -> None:
        """Single group should produce no DE effects (all fold-changes = 1)."""
        config = SimulationConfig(
            n_cells=N_CELLS,
            n_genes=N_GENES,
            n_groups=1,
            n_batches=1,
        )
        sim = DifferentiableSimulator(config, rngs=rngs)
        rng = jax.random.key(0)
        rp = sim.generate_random_params(rng, {})
        result, _, _ = sim.apply({}, {}, None, random_params=rp)

        assert result["counts"].shape == (N_CELLS, N_GENES)
        assert jnp.all(result["counts"] >= 0.0)
        # All cells belong to group 0
        assert jnp.all(result["group_labels"] == 0)

    def test_single_batch_no_batch_effects(self, rngs: nnx.Rngs) -> None:
        """Single batch should produce no batch effects."""
        config = SimulationConfig(
            n_cells=N_CELLS,
            n_genes=N_GENES,
            n_groups=N_GROUPS,
            n_batches=1,
        )
        sim = DifferentiableSimulator(config, rngs=rngs)
        rng = jax.random.key(0)
        rp = sim.generate_random_params(rng, {})
        result, _, _ = sim.apply({}, {}, None, random_params=rp)

        # All cells belong to batch 0
        assert jnp.all(result["batch_labels"] == 0)
        assert result["counts"].shape == (N_CELLS, N_GENES)

    def test_multi_batch_produces_batch_labels(self, rngs: nnx.Rngs) -> None:
        """Multiple batches should assign cells to different batches."""
        config = SimulationConfig(
            n_cells=N_CELLS,
            n_genes=N_GENES,
            n_groups=N_GROUPS,
            n_batches=3,
        )
        sim = DifferentiableSimulator(config, rngs=rngs)
        rng = jax.random.key(0)
        rp = sim.generate_random_params(rng, {})
        result, _, _ = sim.apply({}, {}, None, random_params=rp)

        labels = result["batch_labels"]
        assert jnp.all(labels >= 0)
        assert jnp.all(labels < 3)

    def test_input_data_passthrough(self, simulator: DifferentiableSimulator) -> None:
        """Existing data keys should be preserved in output."""
        rng = jax.random.key(0)
        rp = simulator.generate_random_params(rng, {})
        input_data = {"extra_key": jnp.array(42.0)}
        result, _, _ = simulator.apply(input_data, {}, None, random_params=rp)
        assert "extra_key" in result
        assert result["extra_key"] == 42.0
