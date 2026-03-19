"""Tests for diffbio.operators.variant.cnv_segmentation module.

These tests define the expected behavior of the DifferentiableCNVSegmentation
operator for soft changepoint detection in copy number analysis, including
enhanced multi-signal fusion, pyramidal smoothing, dynamic thresholding,
and HMM state mapping.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.variant.cnv_segmentation import (
    CNVSegmentationConfig,
    DifferentiableCNVSegmentation,
    EnhancedCNVSegmentation,
    EnhancedCNVSegmentationConfig,
)


class TestCNVSegmentationConfig:
    """Tests for CNVSegmentationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CNVSegmentationConfig()
        assert config.max_segments == 100
        assert config.hidden_dim == 64
        assert config.attention_heads == 4
        assert config.temperature == 1.0
        assert config.stochastic is False

    def test_custom_max_segments(self):
        """Test custom max segments."""
        config = CNVSegmentationConfig(max_segments=50)
        assert config.max_segments == 50

    def test_custom_hidden_dim(self):
        """Test custom hidden dimension."""
        config = CNVSegmentationConfig(hidden_dim=128)
        assert config.hidden_dim == 128


class TestDifferentiableCNVSegmentation:
    """Tests for DifferentiableCNVSegmentation operator."""

    @pytest.fixture
    def sample_coverage(self):
        """Provide sample coverage signal."""
        # Coverage signal along genome
        key = jax.random.key(0)
        n_positions = 1000
        # Simulate coverage with some copy number changes
        base_coverage = jnp.ones(n_positions)
        # Add a deletion region
        base_coverage = base_coverage.at[300:500].set(0.5)
        # Add a duplication region
        base_coverage = base_coverage.at[700:800].set(2.0)
        # Add noise
        noise = jax.random.normal(key, (n_positions,)) * 0.1
        coverage = base_coverage + noise
        return {"coverage": coverage}

    @pytest.fixture
    def small_config(self):
        """Provide small config for faster tests."""
        return CNVSegmentationConfig(
            max_segments=20,
            hidden_dim=32,
            attention_heads=2,
        )

    def test_initialization(self, rngs, small_config):
        """Test operator initialization."""
        op = DifferentiableCNVSegmentation(small_config, rngs=rngs)
        assert op is not None

    def test_output_contains_segment_means(self, rngs, small_config, sample_coverage):
        """Test that output contains segment mean values."""
        op = DifferentiableCNVSegmentation(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_coverage, {}, None, None)

        assert "segment_means" in transformed

    def test_output_contains_boundaries(self, rngs, small_config, sample_coverage):
        """Test that output contains soft segment boundaries."""
        op = DifferentiableCNVSegmentation(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_coverage, {}, None, None)

        assert "boundary_probs" in transformed
        # Boundary probs should be per position
        assert transformed["boundary_probs"].shape[0] == 1000

    def test_boundary_probs_valid(self, rngs, small_config, sample_coverage):
        """Test that boundary probabilities are in [0, 1]."""
        op = DifferentiableCNVSegmentation(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_coverage, {}, None, None)

        probs = transformed["boundary_probs"]
        assert jnp.all(probs >= 0)
        assert jnp.all(probs <= 1)

    def test_output_contains_segment_assignments(self, rngs, small_config, sample_coverage):
        """Test that output contains soft segment assignments."""
        op = DifferentiableCNVSegmentation(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_coverage, {}, None, None)

        assert "segment_assignments" in transformed
        # Shape: (n_positions, max_segments)
        assert transformed["segment_assignments"].shape == (1000, small_config.max_segments)

    def test_segment_assignments_sum_to_one(self, rngs, small_config, sample_coverage):
        """Test that segment assignments sum to 1 per position."""
        op = DifferentiableCNVSegmentation(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_coverage, {}, None, None)

        assignments_sum = jnp.sum(transformed["segment_assignments"], axis=-1)
        assert jnp.allclose(assignments_sum, 1.0, atol=1e-5)

    def test_smoothed_signal(self, rngs, small_config, sample_coverage):
        """Test that smoothed/segmented signal is returned."""
        op = DifferentiableCNVSegmentation(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_coverage, {}, None, None)

        assert "smoothed_coverage" in transformed
        assert transformed["smoothed_coverage"].shape == (1000,)


class TestGradientFlow:
    """Tests for gradient flow through CNV segmentation."""

    @pytest.fixture
    def small_config(self):
        return CNVSegmentationConfig(
            max_segments=10,
            hidden_dim=16,
            attention_heads=2,
        )

    def test_gradient_flows_through_segmentation(self, rngs, small_config):
        """Test that gradients flow through segmentation."""
        op = DifferentiableCNVSegmentation(small_config, rngs=rngs)

        key = jax.random.key(0)
        coverage = jax.random.uniform(key, (500,))

        def loss_fn(cov):
            data = {"coverage": cov}
            transformed, _, _ = op.apply(data, {}, None, None)
            return transformed["smoothed_coverage"].sum()

        grad = jax.grad(loss_fn)(coverage)
        assert grad is not None
        assert grad.shape == coverage.shape
        assert jnp.isfinite(grad).all()

    def test_attention_parameters_learnable(self, rngs, small_config):
        """Test that attention parameters are learnable."""
        op = DifferentiableCNVSegmentation(small_config, rngs=rngs)

        key = jax.random.key(0)
        coverage = jax.random.uniform(key, (500,))
        data = {"coverage": coverage}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return transformed["smoothed_coverage"].sum()

        loss, grads = loss_fn(op)

        # Check attention layers have gradients
        assert hasattr(grads, "query_proj")
        assert hasattr(grads, "key_proj")


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture
    def small_config(self):
        return CNVSegmentationConfig(
            max_segments=10,
            hidden_dim=16,
            attention_heads=2,
        )

    def test_apply_is_jit_compatible(self, rngs, small_config):
        """Test that apply method works with JIT."""
        op = DifferentiableCNVSegmentation(small_config, rngs=rngs)

        key = jax.random.key(0)
        coverage = jax.random.uniform(key, (500,))
        data = {"coverage": coverage}
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, _, _ = jit_apply(data, state)
        assert jnp.isfinite(transformed["smoothed_coverage"]).all()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_short_signal(self, rngs):
        """Test with short coverage signal."""
        config = CNVSegmentationConfig(
            max_segments=5,
            hidden_dim=16,
            attention_heads=2,
        )
        op = DifferentiableCNVSegmentation(config, rngs=rngs)

        key = jax.random.key(0)
        coverage = jax.random.uniform(key, (50,))
        data = {"coverage": coverage}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["smoothed_coverage"]).all()

    def test_long_signal(self, rngs):
        """Test with long coverage signal."""
        config = CNVSegmentationConfig(
            max_segments=50,
            hidden_dim=32,
            attention_heads=2,
        )
        op = DifferentiableCNVSegmentation(config, rngs=rngs)

        key = jax.random.key(0)
        coverage = jax.random.uniform(key, (5000,))
        data = {"coverage": coverage}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["smoothed_coverage"]).all()

    def test_uniform_signal(self, rngs):
        """Test with uniform coverage (no CNVs)."""
        config = CNVSegmentationConfig(
            max_segments=10,
            hidden_dim=16,
            attention_heads=2,
        )
        op = DifferentiableCNVSegmentation(config, rngs=rngs)

        # Uniform signal should have minimal boundaries
        coverage = jnp.ones(500)
        data = {"coverage": coverage}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["smoothed_coverage"]).all()

    def test_high_temperature(self, rngs):
        """Test with high temperature."""
        config = CNVSegmentationConfig(
            max_segments=10,
            hidden_dim=16,
            attention_heads=2,
            temperature=10.0,
        )
        op = DifferentiableCNVSegmentation(config, rngs=rngs)

        key = jax.random.key(0)
        coverage = jax.random.uniform(key, (500,))
        data = {"coverage": coverage}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["smoothed_coverage"]).all()


# =========================================================================
# Enhanced CNV Segmentation Tests
# =========================================================================


class TestEnhancedCNVConfig:
    """Tests for EnhancedCNVSegmentationConfig new fields."""

    def test_default_enhanced_config(self) -> None:
        """Test default values for all enhanced config fields."""
        config = EnhancedCNVSegmentationConfig()
        assert config.use_baf is False
        assert config.baf_weight == 0.3
        assert config.smoothing_window == 100
        assert config.threshold_scale == 1.5
        assert config.n_copy_states == 5
        # Inherited defaults still work
        assert config.max_segments == 100
        assert config.hidden_dim == 64

    def test_custom_baf_config(self) -> None:
        """Test custom BAF settings."""
        config = EnhancedCNVSegmentationConfig(use_baf=True, baf_weight=0.5)
        assert config.use_baf is True
        assert config.baf_weight == 0.5

    def test_custom_smoothing_window(self) -> None:
        """Test custom smoothing window size."""
        config = EnhancedCNVSegmentationConfig(smoothing_window=50)
        assert config.smoothing_window == 50

    def test_custom_threshold_scale(self) -> None:
        """Test custom dynamic threshold scale."""
        config = EnhancedCNVSegmentationConfig(threshold_scale=2.0)
        assert config.threshold_scale == 2.0

    def test_custom_copy_states(self) -> None:
        """Test custom number of copy number states."""
        config = EnhancedCNVSegmentationConfig(n_copy_states=7)
        assert config.n_copy_states == 7


class TestMultiSignalFusion:
    """Tests for multi-signal fusion: log-ratio + BAF + SNP density."""

    @pytest.fixture
    def enhanced_config(self) -> EnhancedCNVSegmentationConfig:
        """Provide small enhanced config for tests."""
        return EnhancedCNVSegmentationConfig(
            max_segments=10,
            hidden_dim=16,
            attention_heads=2,
            use_baf=True,
            baf_weight=0.3,
            smoothing_window=21,
            n_copy_states=5,
        )

    @pytest.fixture
    def multi_signal_data(self) -> dict[str, jax.Array]:
        """Provide multi-signal input data with BAF."""
        key = jax.random.key(7)
        n_positions = 200
        keys = jax.random.split(key, 3)
        coverage = jax.random.uniform(keys[0], (n_positions,)) + 0.5
        baf_signal = jax.random.uniform(keys[1], (n_positions,))
        snp_density = jax.random.uniform(keys[2], (n_positions,)) * 0.5
        return {
            "coverage": coverage,
            "baf_signal": baf_signal,
            "snp_density": snp_density,
        }

    def test_fusion_with_baf_enabled(
        self,
        rngs: nnx.Rngs,
        enhanced_config: EnhancedCNVSegmentationConfig,
        multi_signal_data: dict[str, jax.Array],
    ) -> None:
        """Test that BAF signal is integrated when use_baf is True."""
        op = EnhancedCNVSegmentation(enhanced_config, rngs=rngs)
        transformed, _, _ = op.apply(multi_signal_data, {}, None, None)

        assert "fused_signal" in transformed
        assert transformed["fused_signal"].shape == (200,)
        assert jnp.isfinite(transformed["fused_signal"]).all()

    def test_fusion_weights_are_learnable(
        self,
        rngs: nnx.Rngs,
        enhanced_config: EnhancedCNVSegmentationConfig,
        multi_signal_data: dict[str, jax.Array],
    ) -> None:
        """Test that fusion weights receive gradients and are learnable."""
        op = EnhancedCNVSegmentation(enhanced_config, rngs=rngs)

        @nnx.value_and_grad
        def loss_fn(model: EnhancedCNVSegmentation) -> jax.Array:
            transformed, _, _ = model.apply(multi_signal_data, {}, None, None)
            return transformed["smoothed_coverage"].sum()

        _, grads = loss_fn(op)
        assert hasattr(grads, "signal_fusion")

    def test_fusion_without_baf(self, rngs: nnx.Rngs) -> None:
        """Test that operator works without BAF signal."""
        config = EnhancedCNVSegmentationConfig(
            max_segments=10,
            hidden_dim=16,
            attention_heads=2,
            use_baf=False,
            smoothing_window=21,
            n_copy_states=5,
        )
        op = EnhancedCNVSegmentation(config, rngs=rngs)

        key = jax.random.key(8)
        coverage = jax.random.uniform(key, (200,)) + 0.5
        data = {"coverage": coverage}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert "smoothed_coverage" in transformed
        assert jnp.isfinite(transformed["smoothed_coverage"]).all()

    def test_baf_integration_changes_output(
        self,
        rngs: nnx.Rngs,
        multi_signal_data: dict[str, jax.Array],
    ) -> None:
        """Test that enabling BAF produces different output than without it."""
        config_no_baf = EnhancedCNVSegmentationConfig(
            max_segments=10,
            hidden_dim=16,
            attention_heads=2,
            use_baf=False,
            smoothing_window=21,
            n_copy_states=5,
        )
        config_baf = EnhancedCNVSegmentationConfig(
            max_segments=10,
            hidden_dim=16,
            attention_heads=2,
            use_baf=True,
            smoothing_window=21,
            n_copy_states=5,
        )
        op_no_baf = EnhancedCNVSegmentation(config_no_baf, rngs=nnx.Rngs(42))
        op_baf = EnhancedCNVSegmentation(config_baf, rngs=nnx.Rngs(42))

        out_no_baf, _, _ = op_no_baf.apply(multi_signal_data, {}, None, None)
        out_baf, _, _ = op_baf.apply(multi_signal_data, {}, None, None)

        # With BAF, the fused signal should differ from coverage-only
        assert not jnp.allclose(out_no_baf["smoothed_coverage"], out_baf["smoothed_coverage"])


class TestPyramidalSmoothing:
    """Tests for pyramidal (triangular) smoothing convolution."""

    @pytest.fixture
    def enhanced_config(self) -> EnhancedCNVSegmentationConfig:
        """Provide config for smoothing tests."""
        return EnhancedCNVSegmentationConfig(
            max_segments=10,
            hidden_dim=16,
            attention_heads=2,
            use_baf=False,
            smoothing_window=21,
            n_copy_states=5,
        )

    def test_smoothed_output_smoother_than_input(
        self,
        rngs: nnx.Rngs,
        enhanced_config: EnhancedCNVSegmentationConfig,
    ) -> None:
        """Test that pyramidal smoothing reduces signal variance."""
        op = EnhancedCNVSegmentation(enhanced_config, rngs=rngs)

        key = jax.random.key(10)
        # Noisy signal
        noise = jax.random.normal(key, (300,)) * 2.0
        base = jnp.ones(300)
        coverage = base + noise
        data = {"coverage": coverage}

        transformed, _, _ = op.apply(data, {}, None, None)

        raw_var = jnp.var(coverage)
        smoothed_var = jnp.var(transformed["pyramidal_smoothed"])
        # Smoothed signal should have lower variance than raw input
        assert smoothed_var < raw_var

    def test_smoothed_output_same_length(
        self,
        rngs: nnx.Rngs,
        enhanced_config: EnhancedCNVSegmentationConfig,
    ) -> None:
        """Test that smoothing preserves signal length."""
        op = EnhancedCNVSegmentation(enhanced_config, rngs=rngs)

        key = jax.random.key(11)
        coverage = jax.random.uniform(key, (250,))
        data = {"coverage": coverage}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["pyramidal_smoothed"].shape == (250,)

    def test_smoothing_is_finite(
        self,
        rngs: nnx.Rngs,
        enhanced_config: EnhancedCNVSegmentationConfig,
    ) -> None:
        """Test that smoothing produces finite values."""
        op = EnhancedCNVSegmentation(enhanced_config, rngs=rngs)

        key = jax.random.key(12)
        coverage = jax.random.uniform(key, (200,))
        data = {"coverage": coverage}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["pyramidal_smoothed"]).all()


class TestDynamicThreshold:
    """Tests for standard-deviation-based dynamic thresholding."""

    @pytest.fixture
    def enhanced_config(self) -> EnhancedCNVSegmentationConfig:
        """Provide config for threshold tests."""
        return EnhancedCNVSegmentationConfig(
            max_segments=10,
            hidden_dim=16,
            attention_heads=2,
            use_baf=False,
            smoothing_window=21,
            threshold_scale=1.5,
            n_copy_states=5,
        )

    def test_threshold_adapts_to_high_variance(
        self,
        rngs: nnx.Rngs,
    ) -> None:
        """Test that threshold is higher for higher-variance signals."""
        config = EnhancedCNVSegmentationConfig(
            max_segments=10,
            hidden_dim=16,
            attention_heads=2,
            use_baf=False,
            smoothing_window=21,
            threshold_scale=1.5,
            n_copy_states=5,
        )
        op = EnhancedCNVSegmentation(config, rngs=rngs)

        # Low-variance signal
        key = jax.random.key(20)
        low_var = jnp.ones(200) + jax.random.normal(key, (200,)) * 0.01
        data_low = {"coverage": low_var}

        # High-variance signal
        high_var = jnp.ones(200) + jax.random.normal(key, (200,)) * 2.0
        data_high = {"coverage": high_var}

        out_low, _, _ = op.apply(data_low, {}, None, None)
        out_high, _, _ = op.apply(data_high, {}, None, None)

        assert out_high["dynamic_threshold"] > out_low["dynamic_threshold"]

    def test_threshold_output_present(
        self,
        rngs: nnx.Rngs,
        enhanced_config: EnhancedCNVSegmentationConfig,
    ) -> None:
        """Test that dynamic threshold value is present in output."""
        op = EnhancedCNVSegmentation(enhanced_config, rngs=rngs)

        key = jax.random.key(21)
        coverage = jax.random.uniform(key, (200,))
        data = {"coverage": coverage}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert "dynamic_threshold" in transformed
        assert transformed["dynamic_threshold"].shape == ()

    def test_thresholded_signal_present(
        self,
        rngs: nnx.Rngs,
        enhanced_config: EnhancedCNVSegmentationConfig,
    ) -> None:
        """Test that the thresholded signal is in the output."""
        op = EnhancedCNVSegmentation(enhanced_config, rngs=rngs)

        key = jax.random.key(22)
        coverage = jax.random.uniform(key, (200,))
        data = {"coverage": coverage}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert "thresholded_signal" in transformed
        assert transformed["thresholded_signal"].shape == (200,)


class TestHMMStateMapping:
    """Tests for HMM state to copy number mapping."""

    @pytest.fixture
    def enhanced_config(self) -> EnhancedCNVSegmentationConfig:
        """Provide config for HMM tests."""
        return EnhancedCNVSegmentationConfig(
            max_segments=10,
            hidden_dim=16,
            attention_heads=2,
            use_baf=False,
            smoothing_window=21,
            n_copy_states=5,
        )

    def test_copy_number_posteriors_present(
        self,
        rngs: nnx.Rngs,
        enhanced_config: EnhancedCNVSegmentationConfig,
    ) -> None:
        """Test that copy number posteriors are output."""
        op = EnhancedCNVSegmentation(enhanced_config, rngs=rngs)

        key = jax.random.key(30)
        coverage = jax.random.uniform(key, (200,))
        data = {"coverage": coverage}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert "copy_number_posteriors" in transformed
        assert transformed["copy_number_posteriors"].shape == (200, 5)

    def test_copy_number_posteriors_sum_to_one(
        self,
        rngs: nnx.Rngs,
        enhanced_config: EnhancedCNVSegmentationConfig,
    ) -> None:
        """Test that copy number posteriors form valid distributions."""
        op = EnhancedCNVSegmentation(enhanced_config, rngs=rngs)

        key = jax.random.key(31)
        coverage = jax.random.uniform(key, (200,))
        data = {"coverage": coverage}

        transformed, _, _ = op.apply(data, {}, None, None)
        row_sums = jnp.sum(transformed["copy_number_posteriors"], axis=-1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-5)

    def test_copy_number_values_present(
        self,
        rngs: nnx.Rngs,
        enhanced_config: EnhancedCNVSegmentationConfig,
    ) -> None:
        """Test that expected copy number values are in output."""
        op = EnhancedCNVSegmentation(enhanced_config, rngs=rngs)

        key = jax.random.key(32)
        coverage = jax.random.uniform(key, (200,))
        data = {"coverage": coverage}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert "expected_copy_number" in transformed
        assert transformed["expected_copy_number"].shape == (200,)


class TestEnhancedGradientFlow:
    """Tests for gradient flow through fusion + smoothing + thresholding."""

    @pytest.fixture
    def enhanced_config(self) -> EnhancedCNVSegmentationConfig:
        """Provide config for gradient tests."""
        return EnhancedCNVSegmentationConfig(
            max_segments=10,
            hidden_dim=16,
            attention_heads=2,
            use_baf=True,
            smoothing_window=21,
            n_copy_states=5,
        )

    def test_gradients_through_full_enhanced_pipeline(
        self,
        rngs: nnx.Rngs,
        enhanced_config: EnhancedCNVSegmentationConfig,
    ) -> None:
        """Test that gradients flow through the full enhanced pipeline."""
        op = EnhancedCNVSegmentation(enhanced_config, rngs=rngs)

        key = jax.random.key(40)
        keys = jax.random.split(key, 3)
        coverage = jax.random.uniform(keys[0], (150,)) + 0.5
        baf_signal = jax.random.uniform(keys[1], (150,))
        snp_density = jax.random.uniform(keys[2], (150,)) * 0.5

        def loss_fn(cov: jax.Array) -> jax.Array:
            data = {
                "coverage": cov,
                "baf_signal": baf_signal,
                "snp_density": snp_density,
            }
            transformed, _, _ = op.apply(data, {}, None, None)
            return transformed["expected_copy_number"].sum()

        grad = jax.grad(loss_fn)(coverage)
        assert grad is not None
        assert grad.shape == coverage.shape
        assert jnp.isfinite(grad).all()

    def test_model_parameters_receive_gradients(
        self,
        rngs: nnx.Rngs,
        enhanced_config: EnhancedCNVSegmentationConfig,
    ) -> None:
        """Test that model parameters receive gradients for learning."""
        op = EnhancedCNVSegmentation(enhanced_config, rngs=rngs)

        key = jax.random.key(41)
        keys = jax.random.split(key, 3)
        data = {
            "coverage": jax.random.uniform(keys[0], (150,)) + 0.5,
            "baf_signal": jax.random.uniform(keys[1], (150,)),
            "snp_density": jax.random.uniform(keys[2], (150,)) * 0.5,
        }

        @nnx.value_and_grad
        def loss_fn(model: EnhancedCNVSegmentation) -> jax.Array:
            transformed, _, _ = model.apply(data, {}, None, None)
            return transformed["expected_copy_number"].sum()

        _, grads = loss_fn(op)
        # Check that fusion, smoothing, and HMM layers receive gradients
        assert hasattr(grads, "signal_fusion")
        assert hasattr(grads, "copy_number_head")


class TestEnhancedJIT:
    """Tests for JIT compatibility of enhanced features."""

    @pytest.fixture
    def enhanced_config(self) -> EnhancedCNVSegmentationConfig:
        """Provide config for JIT tests."""
        return EnhancedCNVSegmentationConfig(
            max_segments=10,
            hidden_dim=16,
            attention_heads=2,
            use_baf=True,
            smoothing_window=21,
            n_copy_states=5,
        )

    def test_enhanced_jit_compiles(
        self,
        rngs: nnx.Rngs,
        enhanced_config: EnhancedCNVSegmentationConfig,
    ) -> None:
        """Test that the enhanced operator compiles with JIT."""
        op = EnhancedCNVSegmentation(enhanced_config, rngs=rngs)

        key = jax.random.key(50)
        keys = jax.random.split(key, 3)
        data = {
            "coverage": jax.random.uniform(keys[0], (150,)) + 0.5,
            "baf_signal": jax.random.uniform(keys[1], (150,)),
            "snp_density": jax.random.uniform(keys[2], (150,)) * 0.5,
        }

        @jax.jit
        def jit_apply(
            data: dict[str, jax.Array],
            state: dict,
        ) -> tuple:
            return op.apply(data, state, None, None)

        transformed, _, _ = jit_apply(data, {})
        assert jnp.isfinite(transformed["smoothed_coverage"]).all()
        assert jnp.isfinite(transformed["expected_copy_number"]).all()
        assert jnp.isfinite(transformed["fused_signal"]).all()

    def test_enhanced_jit_without_baf(self, rngs: nnx.Rngs) -> None:
        """Test JIT when BAF is disabled."""
        config = EnhancedCNVSegmentationConfig(
            max_segments=10,
            hidden_dim=16,
            attention_heads=2,
            use_baf=False,
            smoothing_window=21,
            n_copy_states=5,
        )
        op = EnhancedCNVSegmentation(config, rngs=rngs)

        key = jax.random.key(51)
        data = {"coverage": jax.random.uniform(key, (150,)) + 0.5}

        @jax.jit
        def jit_apply(
            data: dict[str, jax.Array],
            state: dict,
        ) -> tuple:
            return op.apply(data, state, None, None)

        transformed, _, _ = jit_apply(data, {})
        assert jnp.isfinite(transformed["smoothed_coverage"]).all()
        assert jnp.isfinite(transformed["expected_copy_number"]).all()
