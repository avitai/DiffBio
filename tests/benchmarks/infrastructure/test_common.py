"""Tests for benchmarks._common shared utilities."""

import jax.numpy as jnp
import pytest
from flax import nnx

from benchmarks._common import (
    BaseBenchmarkResult,
    GradientFlowResult,
    check_gradient_flow,
    collect_platform_info,
    generate_synthetic_coverage,
    generate_synthetic_expression,
    generate_synthetic_sequences,
    measure_throughput,
    save_benchmark_result,
)


class TestCollectPlatformInfo:
    """Tests for platform info collection."""

    def test_returns_dict_with_required_keys(self) -> None:
        info = collect_platform_info()
        assert isinstance(info, dict)
        assert "jax_version" in info
        assert "python_version" in info
        assert "platform" in info
        assert "device" in info

    def test_values_are_nonempty_strings(self) -> None:
        info = collect_platform_info()
        for key, value in info.items():
            assert isinstance(value, str), f"{key} is not a string"
            assert len(value) > 0, f"{key} is empty"


class TestCheckGradientFlow:
    """Tests for gradient flow verification."""

    def test_nonzero_gradient_for_differentiable_model(self) -> None:
        model = nnx.Linear(4, 2, rngs=nnx.Rngs(0))
        data = jnp.ones((1, 4))

        def loss_fn(m: nnx.Linear, x: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(m(x))

        result = check_gradient_flow(loss_fn, model, data)
        assert result.gradient_nonzero is True
        assert result.gradient_norm > 0.0

    def test_returns_gradient_flow_result(self) -> None:
        model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))
        data = jnp.ones((1, 2))

        def loss_fn(m: nnx.Linear, x: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(m(x))

        result = check_gradient_flow(loss_fn, model, data)
        assert isinstance(result, GradientFlowResult)
        assert isinstance(result.gradient_norm, float)
        assert isinstance(result.gradient_nonzero, bool)


class TestMeasureThroughput:
    """Tests for throughput measurement."""

    def test_returns_timing_metrics(self) -> None:
        def fn(x: jnp.ndarray) -> jnp.ndarray:
            return x + 1

        result = measure_throughput(fn, (jnp.ones(10),), n_iterations=10, warmup=2)
        assert "total_time_s" in result
        assert "per_item_ms" in result
        assert "items_per_sec" in result
        assert result["total_time_s"] > 0.0
        assert result["items_per_sec"] > 0.0

    def test_warmup_excluded_from_timing(self) -> None:
        call_count = {"value": 0}

        def fn(x: jnp.ndarray) -> jnp.ndarray:
            call_count["value"] += 1
            return x + 1

        measure_throughput(fn, (jnp.ones(10),), n_iterations=5, warmup=3)
        assert call_count["value"] == 8  # 3 warmup + 5 measured


class TestSaveBenchmarkResult:
    """Tests for result saving."""

    def test_saves_json_file(self, tmp_path) -> None:
        result = {"test": "value", "score": 1.0}
        path = save_benchmark_result(result, "test_domain", "test_bench", output_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".json"
        assert "test_domain" in str(path)

    def test_creates_domain_subdirectory(self, tmp_path) -> None:
        result = {"test": "value"}
        path = save_benchmark_result(result, "new_domain", "test_bench", output_dir=tmp_path)
        assert (tmp_path / "new_domain").is_dir()
        assert path.parent == tmp_path / "new_domain"


class TestGenerateSyntheticExpression:
    """Tests for synthetic single-cell data generation."""

    def test_output_shapes(self) -> None:
        data = generate_synthetic_expression(n_cells=100, n_genes=50, n_types=3, n_batches=2)
        assert data["counts"].shape == (100, 50)
        assert data["library_size"].shape == (100,)
        assert data["batch_labels"].shape == (100,)
        assert data["cell_type_labels"].shape == (100,)

    def test_correct_number_of_types(self) -> None:
        data = generate_synthetic_expression(n_cells=90, n_genes=10, n_types=3)
        unique_types = jnp.unique(data["cell_type_labels"])
        assert len(unique_types) == 3

    def test_correct_number_of_batches(self) -> None:
        data = generate_synthetic_expression(n_cells=100, n_genes=10, n_batches=4)
        unique_batches = jnp.unique(data["batch_labels"])
        assert len(unique_batches) == 4

    def test_counts_are_nonnegative(self) -> None:
        data = generate_synthetic_expression(n_cells=50, n_genes=20)
        assert jnp.all(data["counts"] >= 0)

    def test_library_size_matches_counts(self) -> None:
        data = generate_synthetic_expression(n_cells=50, n_genes=20)
        expected = jnp.sum(data["counts"], axis=1)
        assert jnp.allclose(data["library_size"], expected)

    def test_reproducible_with_same_seed(self) -> None:
        d1 = generate_synthetic_expression(seed=42)
        d2 = generate_synthetic_expression(seed=42)
        assert jnp.array_equal(d1["counts"], d2["counts"])

    def test_different_with_different_seed(self) -> None:
        d1 = generate_synthetic_expression(seed=42)
        d2 = generate_synthetic_expression(seed=99)
        assert not jnp.array_equal(d1["counts"], d2["counts"])

    def test_returns_embeddings(self) -> None:
        data = generate_synthetic_expression(n_cells=100, n_genes=50)
        assert "embeddings" in data
        assert data["embeddings"].shape[0] == 100


class TestGenerateSyntheticSequences:
    """Tests for synthetic sequence generation."""

    def test_output_shape(self) -> None:
        seqs = generate_synthetic_sequences(n_seqs=10, seq_len=50, alphabet_size=4)
        assert seqs.shape == (10, 50, 4)

    def test_one_hot_encoding(self) -> None:
        seqs = generate_synthetic_sequences(n_seqs=5, seq_len=20, alphabet_size=4)
        row_sums = jnp.sum(seqs, axis=-1)
        assert jnp.allclose(row_sums, 1.0)

    def test_reproducible(self) -> None:
        s1 = generate_synthetic_sequences(seed=42)
        s2 = generate_synthetic_sequences(seed=42)
        assert jnp.array_equal(s1, s2)


class TestGenerateSyntheticCoverage:
    """Tests for synthetic coverage signal generation."""

    def test_output_shapes(self) -> None:
        signal, truth = generate_synthetic_coverage(length=1000, n_peaks=5)
        assert signal.shape == (1000,)
        assert truth.shape == (1000,)

    def test_truth_mask_is_binary(self) -> None:
        _, truth = generate_synthetic_coverage(length=500, n_peaks=3)
        unique_vals = jnp.unique(truth)
        assert all(v in (0.0, 1.0) for v in unique_vals)

    def test_signal_is_nonnegative(self) -> None:
        signal, _ = generate_synthetic_coverage(length=500)
        assert jnp.all(signal >= 0)

    def test_peak_regions_have_higher_signal(self) -> None:
        signal, truth = generate_synthetic_coverage(
            length=5000, n_peaks=10, background_rate=2.0, peak_height_range=(50.0, 100.0)
        )
        peak_mean = jnp.mean(signal[truth > 0])
        background_mean = jnp.mean(signal[truth == 0])
        assert peak_mean > background_mean


class TestBaseBenchmarkResult:
    """Tests for the base result dataclass."""

    def test_frozen(self) -> None:
        result = BaseBenchmarkResult(
            timestamp="2026-01-01T00:00:00",
            domain="test",
            benchmark_name="test_bench",
            gradient_nonzero=True,
            gradient_norm=1.0,
            throughput=100.0,
            throughput_unit="items/sec",
            wall_time_seconds=1.0,
        )
        with pytest.raises(AttributeError):
            result.domain = "changed"  # type: ignore[misc]
