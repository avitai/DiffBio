"""Integration tests for scVI-style VAE benchmark.

Validates that:
- Synthetic data generation produces correct shapes and labels
- The benchmark runs end-to-end and returns expected metric keys
- All metrics are finite and within expected ranges
- The training loop is JIT-compiled
"""

import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from benchmarks.scvi_benchmark import (
    create_jit_train_step,
    generate_synthetic_pbmc_data,
    run_scvi_benchmark,
)
from diffbio.operators.normalization.vae_normalizer import VAENormalizer, VAENormalizerConfig


class TestSyntheticData:
    """Tests for synthetic PBMC data generation."""

    def test_data_shapes(self) -> None:
        """Verify count matrix and library size have correct shapes."""
        data = generate_synthetic_pbmc_data(n_cells=100, n_genes=50)
        assert data["counts"].shape == (100, 50)
        assert data["library_size"].shape == (100,)

    def test_batch_labels_present(self) -> None:
        """Verify batch labels have correct shape and range."""
        data = generate_synthetic_pbmc_data(n_cells=100, n_batches=3)
        assert data["batch_labels"].shape == (100,)
        unique_batches = jnp.unique(data["batch_labels"])
        assert unique_batches.shape[0] == 3
        assert int(jnp.min(data["batch_labels"])) == 0
        assert int(jnp.max(data["batch_labels"])) == 2

    def test_cell_type_labels_present(self) -> None:
        """Verify cell type labels have correct shape and range."""
        data = generate_synthetic_pbmc_data(n_cells=120, n_types=4)
        assert data["cell_type_labels"].shape == (120,)
        unique_types = jnp.unique(data["cell_type_labels"])
        assert unique_types.shape[0] == 4
        assert int(jnp.min(data["cell_type_labels"])) == 0
        assert int(jnp.max(data["cell_type_labels"])) == 3

    def test_counts_nonnegative(self) -> None:
        """Verify counts are non-negative (they are Poisson-sampled)."""
        data = generate_synthetic_pbmc_data(n_cells=50, n_genes=30)
        assert float(jnp.min(data["counts"])) >= 0.0

    def test_library_size_matches_counts(self) -> None:
        """Verify library size equals row sums of counts."""
        data = generate_synthetic_pbmc_data(n_cells=50, n_genes=30)
        expected_lib = jnp.sum(data["counts"], axis=-1)
        assert jnp.allclose(data["library_size"], expected_lib)

    def test_metadata_keys(self) -> None:
        """Verify all expected metadata keys are present."""
        data = generate_synthetic_pbmc_data()
        for key in ("n_cells", "n_genes", "n_batches", "n_types"):
            assert key in data


class TestBenchmarkRuns:
    """Tests for end-to-end benchmark execution."""

    @pytest.fixture()
    def benchmark_results(self) -> dict[str, float]:
        """Run benchmark once with minimal epochs for testing."""
        return run_scvi_benchmark(n_epochs=5, seed=42)

    def test_benchmark_completes(self, benchmark_results: dict[str, float]) -> None:
        """Verify benchmark returns a dictionary."""
        assert isinstance(benchmark_results, dict)

    def test_result_keys(self, benchmark_results: dict[str, float]) -> None:
        """Verify all expected metric keys are present."""
        expected_keys = {
            "elbo",
            "reconstruction_mse",
            "silhouette",
            "batch_asw",
            "ari",
            "nmi",
        }
        for key in expected_keys:
            assert key in benchmark_results, f"Missing key: {key}"

    def test_elbo_finite(self, benchmark_results: dict[str, float]) -> None:
        """Verify ELBO is a finite number."""
        elbo = benchmark_results["elbo"]
        assert jnp.isfinite(elbo), f"ELBO is not finite: {elbo}"

    def test_metrics_in_range(self, benchmark_results: dict[str, float]) -> None:
        """Verify metrics are within their theoretical ranges."""
        sil = benchmark_results["silhouette"]
        assert -1.0 <= sil <= 1.0, f"Silhouette out of range: {sil}"

        ari = benchmark_results["ari"]
        assert -1.0 <= ari <= 1.0, f"ARI out of range: {ari}"

        nmi = benchmark_results["nmi"]
        assert -0.01 <= nmi <= 1.01, f"NMI out of range: {nmi}"

        batch_asw = benchmark_results["batch_asw"]
        assert -0.01 <= batch_asw <= 1.01, f"Batch ASW out of range: {batch_asw}"

    def test_reconstruction_mse_nonnegative(
        self, benchmark_results: dict[str, float]
    ) -> None:
        """Verify reconstruction MSE is non-negative."""
        mse = benchmark_results["reconstruction_mse"]
        assert mse >= 0.0, f"Reconstruction MSE is negative: {mse}"
        assert jnp.isfinite(mse), f"Reconstruction MSE is not finite: {mse}"


class TestJITTraining:
    """Tests for JIT-compiled training step."""

    def test_training_loop_jit(self) -> None:
        """Verify the training step function is JIT-compiled."""
        data = generate_synthetic_pbmc_data(n_cells=50, n_genes=30, seed=0)
        config = VAENormalizerConfig(
            n_genes=30,
            latent_dim=5,
            hidden_dims=[32],
            likelihood="zinb",
        )
        model = VAENormalizer(config, rngs=nnx.Rngs(0))
        nnx_opt = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
        train_step = create_jit_train_step(model, nnx_opt)

        # First call triggers JIT compilation
        loss = train_step(model, nnx_opt, data["counts"], data["library_size"])
        assert jnp.isfinite(loss), f"JIT step loss is not finite: {loss}"

        # Second call uses cached compilation
        loss2 = train_step(model, nnx_opt, data["counts"], data["library_size"])
        assert jnp.isfinite(loss2), f"Second JIT step loss is not finite: {loss2}"

    def test_loss_decreases_over_steps(self) -> None:
        """Verify that loss decreases over multiple training steps."""
        data = generate_synthetic_pbmc_data(n_cells=50, n_genes=30, seed=0)
        config = VAENormalizerConfig(
            n_genes=30,
            latent_dim=5,
            hidden_dims=[32],
            likelihood="zinb",
        )
        model = VAENormalizer(config, rngs=nnx.Rngs(0))
        nnx_opt = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
        train_step = create_jit_train_step(model, nnx_opt)

        losses: list[float] = []
        for _ in range(10):
            loss = train_step(model, nnx_opt, data["counts"], data["library_size"])
            losses.append(float(loss))

        # Loss at end should be less than loss at start
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"
        )
