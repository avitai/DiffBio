"""Tests for DeepVariantStylePileup operator.

Test-driven development: These tests define the expected behavior.
Implementation must pass these tests without modification.

Based on DeepVariant's pileup image format:
- Image dimensions: (height, width, channels) = (max_reads, window_size, 6+)
- 6 core channels: read_base, base_quality, mapping_quality, strand,
  read_supports_variant, base_differs_from_ref

References:
    - https://google.github.io/deepvariant/posts/2020-02-20-looking-through-deepvariants-eyes/
    - https://google.github.io/deepvariant/posts/2022-06-09-adding-custom-channels/
"""

import jax
import jax.numpy as jnp
import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_reads():
    """Generate sample one-hot encoded reads."""
    key = jax.random.PRNGKey(42)
    num_reads = 20
    read_length = 50
    # Generate random base indices and convert to one-hot
    indices = jax.random.randint(key, (num_reads, read_length), 0, 4)
    return jax.nn.one_hot(indices, 4).astype(jnp.float32)


@pytest.fixture
def sample_reference():
    """Generate sample one-hot encoded reference sequence."""
    key = jax.random.PRNGKey(123)
    ref_length = 221  # DeepVariant default window
    indices = jax.random.randint(key, (ref_length,), 0, 4)
    return jax.nn.one_hot(indices, 4).astype(jnp.float32)


@pytest.fixture
def sample_quality_scores():
    """Generate sample Phred quality scores."""
    key = jax.random.PRNGKey(456)
    num_reads = 20
    read_length = 50
    # Phred scores typically 0-40
    return jax.random.uniform(key, (num_reads, read_length), minval=10.0, maxval=40.0)


@pytest.fixture
def sample_mapping_qualities():
    """Generate sample mapping quality scores."""
    key = jax.random.PRNGKey(789)
    num_reads = 20
    # MAPQ typically 0-60
    return jax.random.uniform(key, (num_reads,), minval=20.0, maxval=60.0)


@pytest.fixture
def sample_strands():
    """Generate sample strand information (0=forward, 1=reverse)."""
    key = jax.random.PRNGKey(101)
    num_reads = 20
    return jax.random.bernoulli(key, 0.5, (num_reads,)).astype(jnp.float32)


@pytest.fixture
def sample_positions():
    """Generate sample read positions within the window."""
    key = jax.random.PRNGKey(202)
    num_reads = 20
    # Positions within a 221bp window, accounting for read length
    return jax.random.randint(key, (num_reads,), 0, 171)


# =============================================================================
# Tests for DeepVariantPileupConfig
# =============================================================================


class TestDeepVariantPileupConfigImport:
    """Tests for DeepVariantPileupConfig module imports."""

    def test_import(self):
        """Test that DeepVariantPileupConfig can be imported."""
        from diffbio.operators.variant import (
            DeepVariantPileupConfig,
            DeepVariantStylePileup,
        )

        assert DeepVariantPileupConfig is not None
        assert DeepVariantStylePileup is not None


class TestDeepVariantPileupConfig:
    """Tests for DeepVariantPileupConfig."""

    def test_config_defaults(self):
        """Test config default values match DeepVariant."""
        from diffbio.operators.variant import DeepVariantPileupConfig

        config = DeepVariantPileupConfig()
        # DeepVariant defaults
        assert config.window_size == 221
        assert config.max_reads == 100
        assert config.channels == (
            "base",
            "base_quality",
            "mapping_quality",
            "strand",
            "supports_variant",
            "differs_from_ref",
        )

    def test_config_custom_values(self):
        """Test config with custom values."""
        from diffbio.operators.variant import DeepVariantPileupConfig

        config = DeepVariantPileupConfig(
            window_size=101,
            max_reads=50,
            channels=("base", "base_quality", "supports_variant"),
        )
        assert config.window_size == 101
        assert config.max_reads == 50
        assert config.channels == ("base", "base_quality", "supports_variant")

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("include_base_channels", False),
            ("include_base_quality", False),
            ("include_mapping_quality", False),
            ("include_strand", False),
            ("include_supports_variant", False),
            ("include_differs_from_ref", False),
        ],
    )
    def test_rejects_removed_legacy_channel_flags(self, field_name, value):
        """Legacy channel booleans should fail immediately."""
        from diffbio.operators.variant import DeepVariantPileupConfig

        with pytest.raises(TypeError, match=field_name):
            DeepVariantPileupConfig(**{field_name: value})

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"window_size": 0}, "window_size"),
            ({"max_reads": 0}, "max_reads"),
            ({"quality_max": 0.0}, "quality_max"),
            ({"mapq_max": 0.0}, "mapq_max"),
            ({"channels": ()}, "channels"),
            ({"channels": ("base", "unknown")}, "unknown"),
        ],
    )
    def test_rejects_invalid_config(self, kwargs, message):
        """Config should fail fast for invalid values."""
        from diffbio.operators.variant import DeepVariantPileupConfig

        with pytest.raises(ValueError, match=message):
            DeepVariantPileupConfig(**kwargs)

    def test_config_inheritance(self):
        """Test that config inherits from TemperatureConfig."""
        from diffbio.configs import TemperatureConfig
        from diffbio.operators.variant import DeepVariantPileupConfig

        config = DeepVariantPileupConfig()
        assert isinstance(config, TemperatureConfig)
        # Should have temperature field from TemperatureConfig
        assert hasattr(config, "temperature")


# =============================================================================
# Tests for DeepVariantStylePileup Basic Functionality
# =============================================================================


class TestDeepVariantStylePileupBasic:
    """Basic functionality tests for DeepVariantStylePileup."""

    def test_initialization(self, rngs):
        """Test DeepVariantStylePileup initialization."""
        from diffbio.operators.variant import (
            DeepVariantPileupConfig,
            DeepVariantStylePileup,
        )

        config = DeepVariantPileupConfig()
        pileup = DeepVariantStylePileup(config, rngs=rngs)
        assert pileup is not None

    def test_num_channels_calculation(self, rngs):
        """Test that number of channels is calculated correctly."""
        from diffbio.operators.variant import (
            DeepVariantPileupConfig,
            DeepVariantStylePileup,
        )

        # All channels enabled: 4 (base) + 1 (qual) + 1 (mapq) + 1 (strand) + 1 (var) + 1 (diff) = 9
        config = DeepVariantPileupConfig()
        pileup = DeepVariantStylePileup(config, rngs=rngs)
        assert pileup.num_channels == 9

        # Disable some channels
        config2 = DeepVariantPileupConfig(
            channels=("base", "base_quality", "supports_variant", "differs_from_ref"),
        )
        pileup2 = DeepVariantStylePileup(config2, rngs=rngs)
        assert pileup2.num_channels == 7  # 9 - 2

    def test_output_shape(
        self,
        rngs,
        sample_reads,
        sample_reference,
        sample_quality_scores,
        sample_mapping_qualities,
        sample_strands,
        sample_positions,
    ):
        """Test pileup image output shape matches DeepVariant format."""
        from diffbio.operators.variant import (
            DeepVariantPileupConfig,
            DeepVariantStylePileup,
        )

        config = DeepVariantPileupConfig(window_size=221, max_reads=100)
        pileup = DeepVariantStylePileup(config, rngs=rngs)

        data = {
            "reads": sample_reads,
            "reference": sample_reference,
            "base_qualities": sample_quality_scores,
            "mapping_qualities": sample_mapping_qualities,
            "strands": sample_strands,
            "positions": sample_positions,
        }

        result, _, _ = pileup.apply(data, {}, None)

        # Output shape should be (max_reads, window_size, num_channels)
        assert "pileup_image" in result
        assert result["pileup_image"].shape == (100, 221, pileup.num_channels)

    def test_output_values_finite(
        self,
        rngs,
        sample_reads,
        sample_reference,
        sample_quality_scores,
        sample_mapping_qualities,
        sample_strands,
        sample_positions,
    ):
        """Test all output values are finite."""
        from diffbio.operators.variant import (
            DeepVariantPileupConfig,
            DeepVariantStylePileup,
        )

        config = DeepVariantPileupConfig()
        pileup = DeepVariantStylePileup(config, rngs=rngs)

        data = {
            "reads": sample_reads,
            "reference": sample_reference,
            "base_qualities": sample_quality_scores,
            "mapping_qualities": sample_mapping_qualities,
            "strands": sample_strands,
            "positions": sample_positions,
        }

        result, _, _ = pileup.apply(data, {}, None)
        assert jnp.all(jnp.isfinite(result["pileup_image"]))

    def test_output_values_normalized(
        self,
        rngs,
        sample_reads,
        sample_reference,
        sample_quality_scores,
        sample_mapping_qualities,
        sample_strands,
        sample_positions,
    ):
        """Test output values are in expected range [0, 1]."""
        from diffbio.operators.variant import (
            DeepVariantPileupConfig,
            DeepVariantStylePileup,
        )

        config = DeepVariantPileupConfig()
        pileup = DeepVariantStylePileup(config, rngs=rngs)

        data = {
            "reads": sample_reads,
            "reference": sample_reference,
            "base_qualities": sample_quality_scores,
            "mapping_qualities": sample_mapping_qualities,
            "strands": sample_strands,
            "positions": sample_positions,
        }

        result, _, _ = pileup.apply(data, {}, None)
        pileup_image = result["pileup_image"]

        # All values should be in [0, 1] range
        assert jnp.all(pileup_image >= 0.0)
        assert jnp.all(pileup_image <= 1.0)


# =============================================================================
# Tests for Individual Channels
# =============================================================================


class TestDeepVariantPileupChannels:
    """Tests for individual pileup channels."""

    def test_base_channels_one_hot(self, rngs, sample_reads, sample_positions):
        """Test base channels contain one-hot-like values."""
        from diffbio.operators.variant import (
            DeepVariantPileupConfig,
            DeepVariantStylePileup,
        )

        # Only enable base channels
        config = DeepVariantPileupConfig(
            channels=("base",),
        )
        pileup = DeepVariantStylePileup(config, rngs=rngs)

        # Minimal reference (won't be used for base channels)
        reference = jnp.zeros((config.window_size, 4))

        data = {
            "reads": sample_reads,
            "reference": reference,
            "base_qualities": jnp.ones((sample_reads.shape[0], sample_reads.shape[1])),
            "mapping_qualities": jnp.ones((sample_reads.shape[0],)),
            "strands": jnp.zeros((sample_reads.shape[0],)),
            "positions": sample_positions,
        }

        result, _, _ = pileup.apply(data, {}, None)
        pileup_image = result["pileup_image"]

        # Base channels should be first 4 channels
        base_channels = pileup_image[:, :, :4]
        # Each position should have at most one base (one-hot)
        # Sum across base channels should be <= 1
        base_sums = jnp.sum(base_channels, axis=-1)
        assert jnp.all(base_sums <= 1.0 + 1e-5)

    def test_quality_channel_normalized(
        self, rngs, sample_reads, sample_quality_scores, sample_positions
    ):
        """Test base quality channel is normalized to [0, 1]."""
        from diffbio.operators.variant import (
            DeepVariantPileupConfig,
            DeepVariantStylePileup,
        )

        config = DeepVariantPileupConfig(
            channels=("base_quality",),
        )
        pileup = DeepVariantStylePileup(config, rngs=rngs)

        reference = jnp.zeros((config.window_size, 4))

        data = {
            "reads": sample_reads,
            "reference": reference,
            "base_qualities": sample_quality_scores,
            "mapping_qualities": jnp.ones((sample_reads.shape[0],)),
            "strands": jnp.zeros((sample_reads.shape[0],)),
            "positions": sample_positions,
        }

        result, _, _ = pileup.apply(data, {}, None)
        quality_channel = result["pileup_image"][:, :, 0]

        # Quality should be normalized to [0, 1]
        assert jnp.all(quality_channel >= 0.0)
        assert jnp.all(quality_channel <= 1.0)

    def test_strand_channel_binary(self, rngs, sample_reads, sample_strands, sample_positions):
        """Test strand channel contains binary-like values."""
        from diffbio.operators.variant import (
            DeepVariantPileupConfig,
            DeepVariantStylePileup,
        )

        config = DeepVariantPileupConfig(
            channels=("strand",),
        )
        pileup = DeepVariantStylePileup(config, rngs=rngs)

        reference = jnp.zeros((config.window_size, 4))

        data = {
            "reads": sample_reads,
            "reference": reference,
            "base_qualities": jnp.ones((sample_reads.shape[0], sample_reads.shape[1])),
            "mapping_qualities": jnp.ones((sample_reads.shape[0],)),
            "strands": sample_strands,
            "positions": sample_positions,
        }

        result, _, _ = pileup.apply(data, {}, None)
        strand_channel = result["pileup_image"][:, :, 0]

        # Strand values should be 0 or 1 (with small tolerance for soft values)
        assert jnp.all(strand_channel >= 0.0)
        assert jnp.all(strand_channel <= 1.0)


# =============================================================================
# Tests for Differentiability
# =============================================================================


class TestDeepVariantPileupDifferentiability:
    """Tests for differentiability of DeepVariantStylePileup."""

    def test_differentiable_through_reads(
        self,
        rngs,
        sample_reads,
        sample_reference,
        sample_quality_scores,
        sample_mapping_qualities,
        sample_strands,
        sample_positions,
    ):
        """Test gradients flow through reads."""
        from diffbio.operators.variant import (
            DeepVariantPileupConfig,
            DeepVariantStylePileup,
        )

        config = DeepVariantPileupConfig()
        pileup = DeepVariantStylePileup(config, rngs=rngs)

        def loss_fn(reads):
            data = {
                "reads": reads,
                "reference": sample_reference,
                "base_qualities": sample_quality_scores,
                "mapping_qualities": sample_mapping_qualities,
                "strands": sample_strands,
                "positions": sample_positions,
            }
            result, _, _ = pileup.apply(data, {}, None)
            return jnp.sum(result["pileup_image"])

        grad = jax.grad(loss_fn)(sample_reads)

        assert grad is not None
        assert grad.shape == sample_reads.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_differentiable_through_quality(
        self,
        rngs,
        sample_reads,
        sample_reference,
        sample_quality_scores,
        sample_mapping_qualities,
        sample_strands,
        sample_positions,
    ):
        """Test gradients flow through quality scores."""
        from diffbio.operators.variant import (
            DeepVariantPileupConfig,
            DeepVariantStylePileup,
        )

        config = DeepVariantPileupConfig()
        pileup = DeepVariantStylePileup(config, rngs=rngs)

        def loss_fn(qualities):
            data = {
                "reads": sample_reads,
                "reference": sample_reference,
                "base_qualities": qualities,
                "mapping_qualities": sample_mapping_qualities,
                "strands": sample_strands,
                "positions": sample_positions,
            }
            result, _, _ = pileup.apply(data, {}, None)
            return jnp.sum(result["pileup_image"])

        grad = jax.grad(loss_fn)(sample_quality_scores)

        assert grad is not None
        assert grad.shape == sample_quality_scores.shape
        assert jnp.all(jnp.isfinite(grad))


# =============================================================================
# Tests for JIT Compatibility
# =============================================================================


class TestDeepVariantPileupJIT:
    """Tests for JIT compilation compatibility."""

    def test_jit_compatible(
        self,
        rngs,
        sample_reads,
        sample_reference,
        sample_quality_scores,
        sample_mapping_qualities,
        sample_strands,
        sample_positions,
    ):
        """Test operator works with JIT compilation."""
        from diffbio.operators.variant import (
            DeepVariantPileupConfig,
            DeepVariantStylePileup,
        )

        config = DeepVariantPileupConfig()
        pileup = DeepVariantStylePileup(config, rngs=rngs)

        @jax.jit
        def compute(reads, ref, qual, mapq, strand, pos):
            data = {
                "reads": reads,
                "reference": ref,
                "base_qualities": qual,
                "mapping_qualities": mapq,
                "strands": strand,
                "positions": pos,
            }
            result, _, _ = pileup.apply(data, {}, None)
            return result["pileup_image"]

        result = compute(
            sample_reads,
            sample_reference,
            sample_quality_scores,
            sample_mapping_qualities,
            sample_strands,
            sample_positions,
        )

        assert result.shape == (config.max_reads, config.window_size, pileup.num_channels)
        assert jnp.all(jnp.isfinite(result))


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestDeepVariantPileupEdgeCases:
    """Edge case tests for DeepVariantStylePileup."""

    def test_fewer_reads_than_max(self, rngs):
        """Test handling when num_reads < max_reads."""
        from diffbio.operators.variant import (
            DeepVariantPileupConfig,
            DeepVariantStylePileup,
        )

        config = DeepVariantPileupConfig(max_reads=100, window_size=50)
        pileup = DeepVariantStylePileup(config, rngs=rngs)

        # Only 5 reads, but max_reads=100
        num_reads = 5
        read_length = 30
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 5)

        reads = jax.nn.one_hot(jax.random.randint(keys[0], (num_reads, read_length), 0, 4), 4)
        reference = jax.nn.one_hot(jax.random.randint(keys[1], (config.window_size,), 0, 4), 4)
        qualities = jax.random.uniform(keys[2], (num_reads, read_length), minval=10.0, maxval=40.0)
        mapq = jax.random.uniform(keys[3], (num_reads,), minval=20.0, maxval=60.0)
        strands = jax.random.bernoulli(keys[4], 0.5, (num_reads,)).astype(jnp.float32)
        positions = jnp.zeros((num_reads,), dtype=jnp.int32)

        data = {
            "reads": reads,
            "reference": reference,
            "base_qualities": qualities,
            "mapping_qualities": mapq,
            "strands": strands,
            "positions": positions,
        }

        result, _, _ = pileup.apply(data, {}, None)

        # Output should still be (max_reads, window_size, channels)
        # Empty read slots should be zero-padded
        assert result["pileup_image"].shape == (100, 50, pileup.num_channels)
        # First 5 rows may have data, rest should be zeros
        assert jnp.all(jnp.isfinite(result["pileup_image"]))

    def test_single_read(self, rngs):
        """Test with single read."""
        from diffbio.operators.variant import (
            DeepVariantPileupConfig,
            DeepVariantStylePileup,
        )

        config = DeepVariantPileupConfig(max_reads=10, window_size=30)
        pileup = DeepVariantStylePileup(config, rngs=rngs)

        key = jax.random.PRNGKey(42)
        reads = jax.nn.one_hot(jax.random.randint(key, (1, 20), 0, 4), 4)
        reference = jax.nn.one_hot(jax.random.randint(key, (30,), 0, 4), 4)

        data = {
            "reads": reads,
            "reference": reference,
            "base_qualities": jnp.ones((1, 20)) * 30.0,
            "mapping_qualities": jnp.array([60.0]),
            "strands": jnp.array([0.0]),
            "positions": jnp.array([5]),
        }

        result, _, _ = pileup.apply(data, {}, None)

        assert result["pileup_image"].shape == (10, 30, pileup.num_channels)
        assert jnp.all(jnp.isfinite(result["pileup_image"]))

    def test_preserves_input_data(
        self,
        rngs,
        sample_reads,
        sample_reference,
        sample_quality_scores,
        sample_mapping_qualities,
        sample_strands,
        sample_positions,
    ):
        """Test that input data is preserved in output."""
        from diffbio.operators.variant import (
            DeepVariantPileupConfig,
            DeepVariantStylePileup,
        )

        config = DeepVariantPileupConfig()
        pileup = DeepVariantStylePileup(config, rngs=rngs)

        data = {
            "reads": sample_reads,
            "reference": sample_reference,
            "base_qualities": sample_quality_scores,
            "mapping_qualities": sample_mapping_qualities,
            "strands": sample_strands,
            "positions": sample_positions,
        }

        result, _, _ = pileup.apply(data, {}, None)

        # Original data should be preserved
        assert "reads" in result
        assert "reference" in result
        assert jnp.array_equal(result["reads"], sample_reads)
