"""Tests for differentiable splicing PSI calculation operator.

Following TDD principles, these tests define the expected behavior
of the SplicingPSI operator for alternative splicing analysis.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest


class TestSplicingPSIConfig:
    """Tests for SplicingPSIConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from diffbio.operators.rnaseq.splicing_psi import SplicingPSIConfig

        config = SplicingPSIConfig(stream_name=None)

        assert config.pseudocount == 1.0
        assert config.temperature == 1.0
        assert config.min_total_reads == 10
        assert config.stochastic is False

    def test_custom_config(self):
        """Test custom configuration values."""
        from diffbio.operators.rnaseq.splicing_psi import SplicingPSIConfig

        config = SplicingPSIConfig(
            pseudocount=0.5,
            temperature=0.1,
            min_total_reads=20,
            stream_name=None,
        )

        assert config.pseudocount == 0.5
        assert config.temperature == 0.1
        assert config.min_total_reads == 20


class TestSplicingPSI:
    """Tests for SplicingPSI operator."""

    @pytest.fixture
    def rngs(self):
        """Provide RNG fixture."""
        return nnx.Rngs(42)

    @pytest.fixture
    def config(self):
        """Provide default config."""
        from diffbio.operators.rnaseq.splicing_psi import SplicingPSIConfig

        return SplicingPSIConfig(
            pseudocount=1.0,
            temperature=1.0,
            min_total_reads=10,
            stream_name=None,
        )

    @pytest.fixture
    def psi_operator(self, config, rngs):
        """Create PSI operator instance."""
        from diffbio.operators.rnaseq.splicing_psi import SplicingPSI

        return SplicingPSI(config, rngs=rngs)

    def test_initialization(self, config, rngs):
        """Test operator initialization."""
        from diffbio.operators.rnaseq.splicing_psi import SplicingPSI

        psi_op = SplicingPSI(config, rngs=rngs)

        assert psi_op.config == config
        assert hasattr(psi_op, "pseudocount")
        assert hasattr(psi_op, "temperature")

    def test_initialization_without_rngs(self, config):
        """Test initialization without providing RNGs."""
        from diffbio.operators.rnaseq.splicing_psi import SplicingPSI

        psi_op = SplicingPSI(config, rngs=None)
        assert psi_op is not None

    def test_apply_single_event(self, psi_operator):
        """Test PSI calculation for a single splicing event."""
        # Inclusion junction reads and exclusion junction reads
        # PSI = inclusion / (inclusion + exclusion)
        inclusion_reads = jnp.array([50.0])
        exclusion_reads = jnp.array([50.0])

        data = {
            "inclusion_counts": inclusion_reads,
            "exclusion_counts": exclusion_reads,
        }
        result, state, metadata = psi_operator.apply(data, {}, None)

        assert "psi" in result
        assert "psi_confidence" in result
        # With equal counts, PSI should be ~0.5
        assert jnp.isclose(result["psi"], 0.5, atol=0.1)

    def test_apply_batch_events(self, psi_operator):
        """Test PSI calculation for multiple events (batch)."""
        batch_size = 10

        inclusion_reads = jax.random.uniform(
            jax.random.key(0), (batch_size,), minval=10, maxval=100
        )
        exclusion_reads = jax.random.uniform(
            jax.random.key(1), (batch_size,), minval=10, maxval=100
        )

        data = {
            "inclusion_counts": inclusion_reads,
            "exclusion_counts": exclusion_reads,
        }
        result, state, metadata = psi_operator.apply(data, {}, None)

        assert result["psi"].shape == (batch_size,)
        assert result["psi_confidence"].shape == (batch_size,)

    def test_psi_range(self, psi_operator):
        """Test that PSI values are in [0, 1]."""
        # Various count combinations
        inclusion_reads = jnp.array([100.0, 0.0, 50.0, 10.0])
        exclusion_reads = jnp.array([0.0, 100.0, 50.0, 90.0])

        data = {
            "inclusion_counts": inclusion_reads,
            "exclusion_counts": exclusion_reads,
        }
        result, _, _ = psi_operator.apply(data, {}, None)

        assert jnp.all(result["psi"] >= 0.0)
        assert jnp.all(result["psi"] <= 1.0)

    def test_confidence_range(self, psi_operator):
        """Test that confidence values are in [0, 1]."""
        inclusion_reads = jnp.array([100.0, 10.0, 5.0])
        exclusion_reads = jnp.array([100.0, 10.0, 5.0])

        data = {
            "inclusion_counts": inclusion_reads,
            "exclusion_counts": exclusion_reads,
        }
        result, _, _ = psi_operator.apply(data, {}, None)

        assert jnp.all(result["psi_confidence"] >= 0.0)
        assert jnp.all(result["psi_confidence"] <= 1.0)

    def test_high_inclusion_gives_high_psi(self, psi_operator):
        """Test that high inclusion reads give high PSI."""
        inclusion_reads = jnp.array([100.0])
        exclusion_reads = jnp.array([10.0])

        data = {
            "inclusion_counts": inclusion_reads,
            "exclusion_counts": exclusion_reads,
        }
        result, _, _ = psi_operator.apply(data, {}, None)

        assert result["psi"] > 0.7

    def test_high_exclusion_gives_low_psi(self, psi_operator):
        """Test that high exclusion reads give low PSI."""
        inclusion_reads = jnp.array([10.0])
        exclusion_reads = jnp.array([100.0])

        data = {
            "inclusion_counts": inclusion_reads,
            "exclusion_counts": exclusion_reads,
        }
        result, _, _ = psi_operator.apply(data, {}, None)

        assert result["psi"] < 0.3

    def test_output_finite(self, psi_operator):
        """Test that all outputs are finite."""
        inclusion_reads = jax.random.uniform(jax.random.key(0), (20,), minval=0, maxval=100)
        exclusion_reads = jax.random.uniform(jax.random.key(1), (20,), minval=0, maxval=100)

        data = {
            "inclusion_counts": inclusion_reads,
            "exclusion_counts": exclusion_reads,
        }
        result, _, _ = psi_operator.apply(data, {}, None)

        assert jnp.all(jnp.isfinite(result["psi"]))
        assert jnp.all(jnp.isfinite(result["psi_confidence"]))

    def test_preserves_original_data(self, psi_operator):
        """Test that original data is preserved in output."""
        inclusion_reads = jnp.array([50.0, 50.0])
        exclusion_reads = jnp.array([50.0, 50.0])
        extra_data = jnp.array([1.0, 2.0, 3.0])

        data = {
            "inclusion_counts": inclusion_reads,
            "exclusion_counts": exclusion_reads,
            "extra": extra_data,
        }
        result, _, _ = psi_operator.apply(data, {}, None)

        assert "extra" in result
        assert jnp.allclose(result["extra"], extra_data)


class TestSplicingPSIDifferentiability:
    """Tests for gradient flow through the PSI operator."""

    @pytest.fixture
    def rngs(self):
        """Provide RNG fixture."""
        return nnx.Rngs(42)

    @pytest.fixture
    def config(self):
        """Provide default config."""
        from diffbio.operators.rnaseq.splicing_psi import SplicingPSIConfig

        return SplicingPSIConfig(stream_name=None)

    def test_gradient_flow_through_operator(self, config, rngs):
        """Test that gradients flow through the operator."""
        from diffbio.operators.rnaseq.splicing_psi import SplicingPSI

        psi_op = SplicingPSI(config, rngs=rngs)

        def loss_fn(op, inclusion, exclusion):
            data = {
                "inclusion_counts": inclusion,
                "exclusion_counts": exclusion,
            }
            result, _, _ = op.apply(data, {}, None)
            return result["psi"].sum()

        inclusion = jnp.array([50.0, 30.0, 70.0])
        exclusion = jnp.array([50.0, 70.0, 30.0])

        grads = nnx.grad(loss_fn)(psi_op, inclusion, exclusion)
        assert grads is not None

    def test_gradient_wrt_pseudocount(self, config, rngs):
        """Test gradient with respect to pseudocount parameter."""
        from diffbio.operators.rnaseq.splicing_psi import SplicingPSI

        psi_op = SplicingPSI(config, rngs=rngs)

        def loss_fn(op, inclusion, exclusion):
            data = {
                "inclusion_counts": inclusion,
                "exclusion_counts": exclusion,
            }
            result, _, _ = op.apply(data, {}, None)
            return result["psi"].mean()

        inclusion = jnp.array([50.0, 30.0])
        exclusion = jnp.array([50.0, 70.0])

        grads = nnx.grad(loss_fn)(psi_op, inclusion, exclusion)

        assert hasattr(grads, "pseudocount")
        assert grads.pseudocount.value is not None

    def test_gradient_wrt_input_counts(self, config, rngs):
        """Test gradient with respect to input counts."""
        from diffbio.operators.rnaseq.splicing_psi import SplicingPSI

        psi_op = SplicingPSI(config, rngs=rngs)

        def loss_fn(inclusion, exclusion):
            data = {
                "inclusion_counts": inclusion,
                "exclusion_counts": exclusion,
            }
            result, _, _ = psi_op.apply(data, {}, None)
            return result["psi"].sum()

        inclusion = jnp.array([50.0, 30.0])
        exclusion = jnp.array([50.0, 70.0])

        grad_inc, grad_exc = jax.grad(loss_fn, argnums=(0, 1))(inclusion, exclusion)

        assert grad_inc.shape == inclusion.shape
        assert grad_exc.shape == exclusion.shape
        assert jnp.all(jnp.isfinite(grad_inc))
        assert jnp.all(jnp.isfinite(grad_exc))


class TestSplicingPSIJITCompatibility:
    """Tests for JIT compilation compatibility."""

    @pytest.fixture
    def rngs(self):
        """Provide RNG fixture."""
        return nnx.Rngs(42)

    @pytest.fixture
    def config(self):
        """Provide default config."""
        from diffbio.operators.rnaseq.splicing_psi import SplicingPSIConfig

        return SplicingPSIConfig(stream_name=None)

    def test_jit_apply(self, config, rngs):
        """Test JIT compilation of apply method."""
        from diffbio.operators.rnaseq.splicing_psi import SplicingPSI

        psi_op = SplicingPSI(config, rngs=rngs)

        @jax.jit
        def jit_apply(inclusion, exclusion):
            data = {
                "inclusion_counts": inclusion,
                "exclusion_counts": exclusion,
            }
            result, _, _ = psi_op.apply(data, {}, None)
            return result["psi"]

        inclusion = jnp.array([50.0, 30.0])
        exclusion = jnp.array([50.0, 70.0])

        # Should compile and run without error
        result = jit_apply(inclusion, exclusion)
        assert result.shape == (2,)

    def test_jit_gradient(self, config, rngs):
        """Test JIT compilation of gradient computation."""
        from diffbio.operators.rnaseq.splicing_psi import SplicingPSI

        psi_op = SplicingPSI(config, rngs=rngs)

        @jax.jit
        def loss_and_grad(inclusion, exclusion):
            def loss_fn(inc, exc):
                data = {
                    "inclusion_counts": inc,
                    "exclusion_counts": exc,
                }
                result, _, _ = psi_op.apply(data, {}, None)
                return result["psi"].sum()

            return jax.value_and_grad(loss_fn, argnums=(0, 1))(inclusion, exclusion)

        inclusion = jnp.array([50.0, 30.0])
        exclusion = jnp.array([50.0, 70.0])

        # Should compile and run without error
        (loss, (grad_inc, grad_exc)) = loss_and_grad(inclusion, exclusion)
        assert jnp.isfinite(loss)
        assert jnp.all(jnp.isfinite(grad_inc))
        assert jnp.all(jnp.isfinite(grad_exc))


class TestDeltaPSI:
    """Tests for delta PSI calculation (differential splicing)."""

    @pytest.fixture
    def rngs(self):
        """Provide RNG fixture."""
        return nnx.Rngs(42)

    @pytest.fixture
    def config(self):
        """Provide default config."""
        from diffbio.operators.rnaseq.splicing_psi import SplicingPSIConfig

        return SplicingPSIConfig(stream_name=None)

    def test_delta_psi_calculation(self, config, rngs):
        """Test delta PSI between two conditions."""
        from diffbio.operators.rnaseq.splicing_psi import SplicingPSI

        psi_op = SplicingPSI(config, rngs=rngs)

        # Condition 1: high inclusion
        data1 = {
            "inclusion_counts": jnp.array([80.0]),
            "exclusion_counts": jnp.array([20.0]),
        }
        result1, _, _ = psi_op.apply(data1, {}, None)

        # Condition 2: low inclusion
        data2 = {
            "inclusion_counts": jnp.array([20.0]),
            "exclusion_counts": jnp.array([80.0]),
        }
        result2, _, _ = psi_op.apply(data2, {}, None)

        delta_psi = result1["psi"] - result2["psi"]

        # Delta should be positive (high inclusion - low inclusion)
        assert delta_psi > 0.3
