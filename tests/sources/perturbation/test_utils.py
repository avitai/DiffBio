"""Tests for perturbation utility functions."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from diffbio.sources.perturbation._utils import (
    generate_onehot_map,
    is_discrete_counts,
    is_log_transformed,
    safe_decode_array,
    split_perturbations_by_cell_fraction,
)


class TestSafeDecodeArray:
    """Tests for safe_decode_array."""

    def test_decode_byte_strings(self) -> None:
        arr = np.array([b"hello", b"world"])
        result = safe_decode_array(arr)
        assert result.dtype.kind == "U"
        assert list(result) == ["hello", "world"]

    def test_decode_regular_strings(self) -> None:
        arr = np.array(["foo", "bar"])
        result = safe_decode_array(arr)
        assert list(result) == ["foo", "bar"]

    def test_decode_mixed(self) -> None:
        arr = [b"bytes", "string", 42]
        result = safe_decode_array(arr)
        assert result[0] == "bytes"
        assert result[1] == "string"
        assert result[2] == "42"

    def test_empty_array(self) -> None:
        arr = np.array([], dtype=bytes)
        result = safe_decode_array(arr)
        assert len(result) == 0


class TestGenerateOnehotMap:
    """Tests for generate_onehot_map."""

    def test_basic_onehot(self) -> None:
        keys = ["A", "B", "C"]
        result = generate_onehot_map(keys)
        assert len(result) == 3
        assert result["A"].shape == (3,)
        np.testing.assert_array_equal(result["A"], jnp.array([1.0, 0.0, 0.0]))
        np.testing.assert_array_equal(result["B"], jnp.array([0.0, 1.0, 0.0]))
        np.testing.assert_array_equal(result["C"], jnp.array([0.0, 0.0, 1.0]))

    def test_duplicate_keys_deduplicated(self) -> None:
        keys = ["X", "Y", "X", "Y", "Z"]
        result = generate_onehot_map(keys)
        assert len(result) == 3
        assert result["X"].shape == (3,)

    def test_single_key(self) -> None:
        result = generate_onehot_map(["only"])
        assert len(result) == 1
        np.testing.assert_array_equal(result["only"], jnp.array([1.0]))

    def test_output_is_jax_array(self) -> None:
        result = generate_onehot_map(["a", "b"])
        assert isinstance(result["a"], jnp.ndarray)

    def test_sorted_order(self) -> None:
        keys = ["C", "A", "B"]
        result = generate_onehot_map(keys)
        np.testing.assert_array_equal(result["A"], jnp.array([1.0, 0.0, 0.0]))
        np.testing.assert_array_equal(result["B"], jnp.array([0.0, 1.0, 0.0]))
        np.testing.assert_array_equal(result["C"], jnp.array([0.0, 0.0, 1.0]))


class TestIsDiscreteCounts:
    """Tests for is_discrete_counts."""

    def test_integer_counts_detected(self) -> None:
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert is_discrete_counts(x) is True

    def test_float_values_not_discrete(self) -> None:
        x = jnp.array([[1.5, 2.3, 0.7], [0.1, 0.2, 0.3]])
        assert is_discrete_counts(x) is False

    def test_log_transformed_not_discrete(self) -> None:
        raw = jnp.array([[100.0, 200.0, 50.0]])
        logged = jnp.log1p(raw)
        assert is_discrete_counts(logged) is False

    def test_n_cells_parameter(self) -> None:
        x = jnp.ones((200, 10))
        assert is_discrete_counts(x, n_cells=50) is True


class TestIsLogTransformed:
    """Tests for is_log_transformed."""

    def test_log_data_detected(self) -> None:
        x = jnp.log1p(jnp.array([[100.0, 200.0], [50.0, 80.0]]))
        assert is_log_transformed(x) is True

    def test_raw_counts_not_log(self) -> None:
        x = jnp.array([[1000.0, 5000.0], [20000.0, 100.0]])
        assert is_log_transformed(x) is False

    def test_small_counts_ambiguous(self) -> None:
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        assert is_log_transformed(x) is True


class TestSplitPerturbationsByCellFraction:
    """Tests for split_perturbations_by_cell_fraction."""

    def test_basic_split(self) -> None:
        pert_groups = {
            "A": np.arange(100),
            "B": np.arange(50),
            "C": np.arange(30),
            "D": np.arange(20),
        }
        rng = np.random.default_rng(42)
        train, val = split_perturbations_by_cell_fraction(pert_groups, 0.2, rng)

        # All perturbations should be assigned
        assert set(train) | set(val) == {"A", "B", "C", "D"}
        assert len(set(train) & set(val)) == 0

    def test_val_fraction_approximately_correct(self) -> None:
        pert_groups = {f"p{i}": np.arange(50) for i in range(10)}
        rng = np.random.default_rng(42)
        train, val = split_perturbations_by_cell_fraction(pert_groups, 0.3, rng)

        total = sum(len(pert_groups[p]) for p in train + val)
        val_cells = sum(len(pert_groups[p]) for p in val)
        val_frac = val_cells / total
        assert 0.1 < val_frac < 0.5

    def test_zero_fraction_all_train(self) -> None:
        pert_groups = {"A": np.arange(10), "B": np.arange(10)}
        rng = np.random.default_rng(42)
        train, val = split_perturbations_by_cell_fraction(pert_groups, 0.0, rng)
        assert len(val) == 0
        assert len(train) == 2

    def test_deterministic_with_seed(self) -> None:
        pert_groups = {f"p{i}": np.arange(50) for i in range(10)}
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        train1, val1 = split_perturbations_by_cell_fraction(pert_groups, 0.2, rng1)
        train2, val2 = split_perturbations_by_cell_fraction(pert_groups, 0.2, rng2)
        assert set(train1) == set(train2)
        assert set(val1) == set(val2)
