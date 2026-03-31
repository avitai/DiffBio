"""Tests for uncertainty quantification wrappers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.core.uncertainty import (
    ConformalUQOperator,
    ConformalUQConfig,
    EnsembleUQOperator,
    EnsembleUQConfig,
)


@pytest.fixture()
def rngs() -> nnx.Rngs:
    """Shared RNG."""
    return nnx.Rngs(42)


def _make_simple_operator(rngs: nnx.Rngs):
    """Create a simple clustering operator for wrapping."""
    from diffbio.operators.singlecell import (
        SoftClusteringConfig,
        SoftKMeansClustering,
    )

    config = SoftClusteringConfig(n_clusters=3, n_features=10, temperature=1.0)
    return SoftKMeansClustering(config, rngs=rngs)


def _make_embeddings(n: int = 30, d: int = 10) -> dict[str, jnp.ndarray]:
    """Synthetic embeddings."""
    key = jax.random.key(0)
    return {"embeddings": jax.random.normal(key, (n, d))}


class TestEnsembleUQOperator:
    """Tests for ensemble-based uncertainty quantification."""

    def test_output_has_uncertainty_keys(self, rngs: nnx.Rngs) -> None:
        """Output dict includes uncertainty and confidence_interval."""
        base_op = _make_simple_operator(rngs)
        config = EnsembleUQConfig(n_members=3)
        uq_op = EnsembleUQOperator(config, base_operator=base_op, rngs=rngs)

        data = _make_embeddings()
        result, state, meta = uq_op.apply(data, {}, None)

        assert "uncertainty" in result
        assert "confidence_interval_lower" in result
        assert "confidence_interval_upper" in result

    def test_uncertainty_is_non_negative(self, rngs: nnx.Rngs) -> None:
        """Uncertainty values are non-negative."""
        base_op = _make_simple_operator(rngs)
        config = EnsembleUQConfig(n_members=3)
        uq_op = EnsembleUQOperator(config, base_operator=base_op, rngs=rngs)

        result, _, _ = uq_op.apply(_make_embeddings(), {}, None)
        assert jnp.all(result["uncertainty"] >= 0)

    def test_original_keys_preserved(self, rngs: nnx.Rngs) -> None:
        """Original operator output keys are preserved."""
        base_op = _make_simple_operator(rngs)
        config = EnsembleUQConfig(n_members=2)
        uq_op = EnsembleUQOperator(config, base_operator=base_op, rngs=rngs)

        result, _, _ = uq_op.apply(_make_embeddings(), {}, None)
        assert "cluster_assignments" in result
        assert "embeddings" in result

    def test_more_members_different_uncertainty(self, rngs: nnx.Rngs) -> None:
        """Different ensemble sizes produce output (smoke test)."""
        base_op = _make_simple_operator(rngs)

        config_2 = EnsembleUQConfig(n_members=2)
        uq_2 = EnsembleUQOperator(config_2, base_operator=base_op, rngs=rngs)

        config_5 = EnsembleUQConfig(n_members=5)
        uq_5 = EnsembleUQOperator(config_5, base_operator=base_op, rngs=nnx.Rngs(99))

        r2, _, _ = uq_2.apply(_make_embeddings(), {}, None)
        r5, _, _ = uq_5.apply(_make_embeddings(), {}, None)

        assert r2["uncertainty"].shape == r5["uncertainty"].shape

    def test_config_frozen(self) -> None:
        """Config is immutable."""
        config = EnsembleUQConfig(n_members=3)
        with pytest.raises(AttributeError):
            config.n_members = 5  # type: ignore[misc]


class TestConformalUQOperator:
    """Tests for conformal prediction-based uncertainty quantification."""

    def test_output_has_uncertainty_keys(self, rngs: nnx.Rngs) -> None:
        """Output dict includes uncertainty and confidence intervals."""
        base_op = _make_simple_operator(rngs)
        config = ConformalUQConfig(alpha=0.1, num_samples=5)
        uq_op = ConformalUQOperator(config, base_operator=base_op, rngs=rngs)

        result, _, _ = uq_op.apply(_make_embeddings(), {}, None)

        assert "uncertainty" in result
        assert "confidence_interval_lower" in result
        assert "confidence_interval_upper" in result

    def test_confidence_interval_ordering(self, rngs: nnx.Rngs) -> None:
        """Lower bound <= mean <= upper bound."""
        base_op = _make_simple_operator(rngs)
        config = ConformalUQConfig(alpha=0.1, num_samples=5)
        uq_op = ConformalUQOperator(config, base_operator=base_op, rngs=rngs)

        result, _, _ = uq_op.apply(_make_embeddings(), {}, None)

        lower = result["confidence_interval_lower"]
        upper = result["confidence_interval_upper"]
        assert jnp.all(lower <= upper + 1e-6)

    def test_alpha_controls_width(self, rngs: nnx.Rngs) -> None:
        """Smaller alpha (higher confidence) -> wider intervals."""
        base_op = _make_simple_operator(rngs)
        data = _make_embeddings()

        config_wide = ConformalUQConfig(alpha=0.01, num_samples=10)
        uq_wide = ConformalUQOperator(config_wide, base_operator=base_op, rngs=rngs)
        r_wide, _, _ = uq_wide.apply(data, {}, None)

        config_narrow = ConformalUQConfig(alpha=0.5, num_samples=10)
        uq_narrow = ConformalUQOperator(config_narrow, base_operator=base_op, rngs=rngs)
        r_narrow, _, _ = uq_narrow.apply(data, {}, None)

        width_wide = jnp.mean(
            r_wide["confidence_interval_upper"] - r_wide["confidence_interval_lower"]
        )
        width_narrow = jnp.mean(
            r_narrow["confidence_interval_upper"] - r_narrow["confidence_interval_lower"]
        )
        assert width_wide >= width_narrow - 1e-6


class TestJITCompatibility:
    """Tests for JIT compatibility of UQ operators."""

    def test_ensemble_jit_apply(self, rngs: nnx.Rngs) -> None:
        """Test EnsembleUQOperator works under JIT."""
        base_op = _make_simple_operator(rngs)
        config = EnsembleUQConfig(n_members=2)
        uq_op = EnsembleUQOperator(config, base_operator=base_op, rngs=rngs)

        data = _make_embeddings()

        @jax.jit
        def forward(embeddings: jnp.ndarray) -> jnp.ndarray:
            result, _, _ = uq_op.apply({"embeddings": embeddings}, {}, None)
            return result["uncertainty"]

        uncertainty = forward(data["embeddings"])
        assert uncertainty is not None
        assert jnp.all(jnp.isfinite(uncertainty))

    def test_conformal_jit_apply(self, rngs: nnx.Rngs) -> None:
        """Test ConformalUQOperator works under JIT."""
        base_op = _make_simple_operator(rngs)
        config = ConformalUQConfig(alpha=0.1, num_samples=3)
        uq_op = ConformalUQOperator(config, base_operator=base_op, rngs=rngs)

        data = _make_embeddings()

        @jax.jit
        def forward(embeddings: jnp.ndarray) -> jnp.ndarray:
            result, _, _ = uq_op.apply({"embeddings": embeddings}, {}, None)
            return result["uncertainty"]

        uncertainty = forward(data["embeddings"])
        assert uncertainty is not None
        assert jnp.all(jnp.isfinite(uncertainty))


class TestGradientFlow:
    """Tests for differentiability of UQ operators."""

    def test_ensemble_gradient_through_input(self, rngs: nnx.Rngs) -> None:
        """Test gradients flow through EnsembleUQOperator."""
        base_op = _make_simple_operator(rngs)
        config = EnsembleUQConfig(n_members=2)
        uq_op = EnsembleUQOperator(config, base_operator=base_op, rngs=rngs)

        def loss_fn(embeddings: jnp.ndarray) -> jnp.ndarray:
            result, _, _ = uq_op.apply({"embeddings": embeddings}, {}, None)
            return result["cluster_assignments"].sum()

        data = _make_embeddings()
        grads = jax.grad(loss_fn)(data["embeddings"])
        assert grads is not None
        assert grads.shape == data["embeddings"].shape
        assert jnp.all(jnp.isfinite(grads))

    def test_conformal_gradient_through_input(self, rngs: nnx.Rngs) -> None:
        """Test gradients flow through ConformalUQOperator."""
        base_op = _make_simple_operator(rngs)
        config = ConformalUQConfig(alpha=0.1, num_samples=3)
        uq_op = ConformalUQOperator(config, base_operator=base_op, rngs=rngs)

        def loss_fn(embeddings: jnp.ndarray) -> jnp.ndarray:
            result, _, _ = uq_op.apply({"embeddings": embeddings}, {}, None)
            return result["cluster_assignments"].sum()

        data = _make_embeddings()
        grads = jax.grad(loss_fn)(data["embeddings"])
        assert grads is not None
        assert grads.shape == data["embeddings"].shape
        assert jnp.all(jnp.isfinite(grads))

    def test_ensemble_gradient_through_params(self, rngs: nnx.Rngs) -> None:
        """Test gradients flow to base operator parameters through ensemble wrapper."""
        from flax import nnx

        base_op = _make_simple_operator(rngs)
        config = EnsembleUQConfig(n_members=2)
        uq_op = EnsembleUQOperator(config, base_operator=base_op, rngs=rngs)

        data = _make_embeddings()

        @nnx.value_and_grad
        def loss_fn(uq_operator: EnsembleUQOperator) -> jnp.ndarray:
            result, _, _ = uq_operator.apply(data, {}, None)
            return result["cluster_assignments"].sum()

        loss, grads = loss_fn(uq_op)
        assert jnp.isfinite(loss)
        assert grads is not None
