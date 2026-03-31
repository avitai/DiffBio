"""Tests for SINDy-based gene regulatory network inference."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.singlecell.sindy_grn import (
    SINDyGRNConfig,
    SINDyGRNOperator,
    build_polynomial_library,
)


@pytest.fixture()
def rngs() -> nnx.Rngs:
    """Shared RNG for tests."""
    return nnx.Rngs(42)


@pytest.fixture()
def simple_counts() -> jnp.ndarray:
    """Synthetic time-ordered expression (20 timepoints, 5 genes)."""
    key = jax.random.key(0)
    return jax.random.normal(key, (20, 5)) * 0.5 + 1.0


class TestBuildPolynomialLibrary:
    """Tests for the polynomial library builder."""

    def test_degree_1_identity(self) -> None:
        """Degree-1 library is the identity (just raw features)."""
        x = jnp.ones((10, 3))
        lib = build_polynomial_library(x, degree=1)
        assert lib.shape == (10, 3)

    def test_degree_2_adds_interactions(self) -> None:
        """Degree-2 library includes pairwise products."""
        x = jnp.ones((10, 3))
        lib = build_polynomial_library(x, degree=2)
        # 3 linear + 6 quadratic (3 squared + 3 cross) = 9
        assert lib.shape[0] == 10
        assert lib.shape[1] == 9

    def test_output_is_finite(self) -> None:
        """Library output should contain no NaN or Inf."""
        key = jax.random.key(1)
        x = jax.random.normal(key, (15, 4))
        lib = build_polynomial_library(x, degree=2)
        assert jnp.all(jnp.isfinite(lib))


class TestSINDyGRNConfig:
    """Tests for SINDy GRN configuration."""

    def test_default_values(self) -> None:
        """Config has sensible defaults."""
        config = SINDyGRNConfig()
        assert config.n_genes == 100
        assert config.polynomial_degree == 2
        assert config.sparsity_threshold > 0

    def test_frozen(self) -> None:
        """Config is immutable."""
        config = SINDyGRNConfig()
        with pytest.raises(AttributeError):
            config.n_genes = 50  # type: ignore[misc]


class TestSINDyGRNOperator:
    """Tests for the SINDy GRN operator."""

    def test_output_keys(self, rngs: nnx.Rngs, simple_counts: jnp.ndarray) -> None:
        """Operator produces expected output keys."""
        config = SINDyGRNConfig(n_genes=5, polynomial_degree=1)
        op = SINDyGRNOperator(config, rngs=rngs)
        data = {"counts": simple_counts}
        result, state, meta = op.apply(data, {}, None)
        assert "grn_coefficients" in result
        assert "grn_equations" in result
        assert "counts" in result  # original data preserved

    def test_coefficient_shape(self, rngs: nnx.Rngs, simple_counts: jnp.ndarray) -> None:
        """Coefficient matrix has correct shape."""
        config = SINDyGRNConfig(n_genes=5, polynomial_degree=1)
        op = SINDyGRNOperator(config, rngs=rngs)
        result, _, _ = op.apply({"counts": simple_counts}, {}, None)
        coeff = result["grn_coefficients"]
        # For degree=1: library has 5 features, output has 5 genes
        assert coeff.shape == (5, 5)

    def test_sparsity(self, rngs: nnx.Rngs, simple_counts: jnp.ndarray) -> None:
        """Higher sparsity threshold produces more zeros."""
        config_sparse = SINDyGRNConfig(n_genes=5, polynomial_degree=1, sparsity_threshold=10.0)
        op_sparse = SINDyGRNOperator(config_sparse, rngs=rngs)
        result_sparse, _, _ = op_sparse.apply({"counts": simple_counts}, {}, None)

        config_dense = SINDyGRNConfig(n_genes=5, polynomial_degree=1, sparsity_threshold=0.001)
        op_dense = SINDyGRNOperator(config_dense, rngs=rngs)
        result_dense, _, _ = op_dense.apply({"counts": simple_counts}, {}, None)

        sparse_zeros = jnp.sum(jnp.abs(result_sparse["grn_coefficients"]) < 0.01)
        dense_zeros = jnp.sum(jnp.abs(result_dense["grn_coefficients"]) < 0.01)
        assert sparse_zeros >= dense_zeros

    def test_differentiable(self, rngs: nnx.Rngs, simple_counts: jnp.ndarray) -> None:
        """Output is differentiable with respect to input counts."""
        config = SINDyGRNConfig(n_genes=5, polynomial_degree=1)
        op = SINDyGRNOperator(config, rngs=rngs)

        def loss_fn(counts: jnp.ndarray) -> jnp.ndarray:
            result, _, _ = op.apply({"counts": counts}, {}, None)
            return jnp.sum(result["grn_coefficients"] ** 2)

        grads = jax.grad(loss_fn)(simple_counts)
        assert grads is not None
        assert grads.shape == simple_counts.shape
        assert jnp.all(jnp.isfinite(grads))

    def test_jit_compatible(self, rngs: nnx.Rngs, simple_counts: jnp.ndarray) -> None:
        """Test JIT compilation works for SINDyGRNOperator."""
        config = SINDyGRNConfig(n_genes=5, polynomial_degree=1)
        op = SINDyGRNOperator(config, rngs=rngs)

        @jax.jit
        def compute(counts: jnp.ndarray) -> jnp.ndarray:
            result, _, _ = op.apply({"counts": counts}, {}, None)
            return result["grn_coefficients"]

        coefficients = compute(simple_counts)
        assert coefficients.shape == (5, 5)
        assert jnp.all(jnp.isfinite(coefficients))

    def test_degree_2_library(self, rngs: nnx.Rngs, simple_counts: jnp.ndarray) -> None:
        """Degree-2 produces larger coefficient matrix."""
        config = SINDyGRNConfig(n_genes=5, polynomial_degree=2)
        op = SINDyGRNOperator(config, rngs=rngs)
        result, _, _ = op.apply({"counts": simple_counts}, {}, None)
        coeff = result["grn_coefficients"]
        # Library has 5 + 15 = 20 features for degree=2 with 5 genes
        # But coefficient matrix maps library_size -> n_genes
        assert coeff.shape[1] == 5
        assert coeff.shape[0] > 5
