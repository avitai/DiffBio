"""Tests for diffbio.operators.singlecell.imputation module.

These tests define the expected behavior of the DifferentiableDiffusionImputer
operator for MAGIC-style diffusion-based data imputation, and the
DifferentiableTransformerDenoiser for transformer-based gene denoising.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.singlecell.imputation import (
    DifferentiableDiffusionImputer,
    DifferentiableTransformerDenoiser,
    DiffusionImputerConfig,
    TransformerDenoiserConfig,
)


class TestDiffusionImputerConfig:
    """Tests for DiffusionImputerConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = DiffusionImputerConfig()
        assert config.n_neighbors == 15
        assert config.diffusion_t == 3
        assert config.n_pca_components == 100
        assert config.metric == "euclidean"
        assert config.stochastic is False
        assert config.stream_name is None

    def test_custom_neighbors(self) -> None:
        """Test custom number of neighbors."""
        config = DiffusionImputerConfig(n_neighbors=30, diffusion_t=5)
        assert config.n_neighbors == 30
        assert config.diffusion_t == 5


class TestDifferentiableDiffusionImputer:
    """Tests for DifferentiableDiffusionImputer operator."""

    @pytest.fixture()
    def noisy_counts(self) -> dict[str, jax.Array]:
        """Provide noisy count data with known structure."""
        key = jax.random.key(0)
        n_cells, n_genes = 20, 10
        # Create structured data with added noise
        k1, k2 = jax.random.split(key)
        clean = jnp.abs(jax.random.normal(k1, (n_cells, n_genes))) * 5.0
        noise = jax.random.normal(k2, (n_cells, n_genes)) * 2.0
        counts = clean + noise
        return {"counts": counts}

    @pytest.fixture()
    def default_config(self) -> DiffusionImputerConfig:
        """Provide default config for tests."""
        return DiffusionImputerConfig(n_neighbors=5, diffusion_t=2)

    def test_output_keys(
        self,
        rngs: nnx.Rngs,
        default_config: DiffusionImputerConfig,
        noisy_counts: dict[str, jax.Array],
    ) -> None:
        """Test that apply returns imputed_counts and diffusion_operator."""
        op = DifferentiableDiffusionImputer(default_config, rngs=rngs)
        result, state, metadata = op.apply(noisy_counts, {}, None)

        assert "imputed_counts" in result
        assert "diffusion_operator" in result
        # Original data should be preserved
        assert "counts" in result

    def test_output_shapes(
        self,
        rngs: nnx.Rngs,
        default_config: DiffusionImputerConfig,
        noisy_counts: dict[str, jax.Array],
    ) -> None:
        """Test correct output shapes."""
        op = DifferentiableDiffusionImputer(default_config, rngs=rngs)
        result, _, _ = op.apply(noisy_counts, {}, None)

        n_cells, n_genes = noisy_counts["counts"].shape
        assert result["imputed_counts"].shape == (n_cells, n_genes)
        assert result["diffusion_operator"].shape == (n_cells, n_cells)

    def test_imputed_smoother(self, rngs: nnx.Rngs) -> None:
        """Test that imputed data has lower variance than noisy input."""
        key = jax.random.key(10)
        n_cells, n_genes = 30, 8
        k1, k2 = jax.random.split(key)
        # All cells share same underlying signal, with added noise
        signal = jnp.ones((1, n_genes)) * 5.0
        signal = jnp.broadcast_to(signal, (n_cells, n_genes))
        noise = jax.random.normal(k2, (n_cells, n_genes)) * 3.0
        noisy = signal + noise

        config = DiffusionImputerConfig(n_neighbors=10, diffusion_t=3)
        op = DifferentiableDiffusionImputer(config, rngs=rngs)
        result, _, _ = op.apply({"counts": noisy}, {}, None)

        input_var = jnp.var(noisy, axis=0).mean()
        imputed_var = jnp.var(result["imputed_counts"], axis=0).mean()
        assert imputed_var < input_var

    def test_preserves_mean(
        self,
        rngs: nnx.Rngs,
        default_config: DiffusionImputerConfig,
        noisy_counts: dict[str, jax.Array],
    ) -> None:
        """Test that overall mean is approximately preserved."""
        op = DifferentiableDiffusionImputer(default_config, rngs=rngs)
        result, _, _ = op.apply(noisy_counts, {}, None)

        input_mean = jnp.mean(noisy_counts["counts"], axis=0)
        imputed_mean = jnp.mean(result["imputed_counts"], axis=0)
        # Row-stochastic matrix preserves column means
        assert jnp.allclose(input_mean, imputed_mean, atol=0.5)

    def test_idempotent_clean(self, rngs: nnx.Rngs) -> None:
        """Test that clean (uniform) data barely changes after imputation."""
        n_cells, n_genes = 20, 6
        # All cells identical -> diffusion should not change them
        clean = jnp.ones((n_cells, n_genes)) * 3.0

        config = DiffusionImputerConfig(n_neighbors=5, diffusion_t=2)
        op = DifferentiableDiffusionImputer(config, rngs=rngs)
        result, _, _ = op.apply({"counts": clean}, {}, None)

        assert jnp.allclose(result["imputed_counts"], clean, atol=1e-3)


class TestGradientFlow:
    """Tests for gradient flow through imputation."""

    @pytest.fixture()
    def config(self) -> DiffusionImputerConfig:
        """Provide config for gradient tests."""
        return DiffusionImputerConfig(n_neighbors=5, diffusion_t=2)

    def test_gradient_wrt_input(self, rngs: nnx.Rngs, config: DiffusionImputerConfig) -> None:
        """Test that gradients flow from imputed_counts back to input counts."""
        op = DifferentiableDiffusionImputer(config, rngs=rngs)
        key = jax.random.key(0)
        counts = jnp.abs(jax.random.normal(key, (15, 8))) + 0.1

        def loss_fn(c: jax.Array) -> jax.Array:
            result, _, _ = op.apply({"counts": c}, {}, None)
            return jnp.sum(result["imputed_counts"])

        grad = jax.grad(loss_fn)(counts)
        assert grad is not None
        assert grad.shape == counts.shape
        assert jnp.any(grad != 0.0)
        assert jnp.isfinite(grad).all()

    def test_gradient_wrt_operator_params(
        self, rngs: nnx.Rngs, config: DiffusionImputerConfig
    ) -> None:
        """Test that grad through operator does not error (no learnable params expected)."""
        op = DifferentiableDiffusionImputer(config, rngs=rngs)
        key = jax.random.key(0)
        counts = jnp.abs(jax.random.normal(key, (15, 8))) + 0.1
        data = {"counts": counts}

        # Even without learnable params, value_and_grad should not raise
        @nnx.value_and_grad
        def loss_fn(model: DifferentiableDiffusionImputer) -> jax.Array:
            result, _, _ = model.apply(data, {}, None)
            return jnp.sum(result["imputed_counts"])

        loss, grads = loss_fn(op)
        assert jnp.isfinite(loss)


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture()
    def config(self) -> DiffusionImputerConfig:
        """Provide config for JIT tests."""
        return DiffusionImputerConfig(n_neighbors=5, diffusion_t=2)

    def test_jit_apply(self, rngs: nnx.Rngs, config: DiffusionImputerConfig) -> None:
        """Test that jax.jit compiles and runs apply."""
        op = DifferentiableDiffusionImputer(config, rngs=rngs)
        key = jax.random.key(0)
        counts = jnp.abs(jax.random.normal(key, (15, 8))) + 0.1
        data = {"counts": counts}

        @jax.jit
        def jit_apply(
            d: dict[str, jax.Array],
            s: dict,
        ) -> tuple:
            return op.apply(d, s, None)

        result, _, _ = jit_apply(data, {})
        assert jnp.isfinite(result["imputed_counts"]).all()

    def test_jit_gradient(self, rngs: nnx.Rngs, config: DiffusionImputerConfig) -> None:
        """Test that jax.jit + jax.grad works together."""
        op = DifferentiableDiffusionImputer(config, rngs=rngs)
        key = jax.random.key(0)
        counts = jnp.abs(jax.random.normal(key, (15, 8))) + 0.1

        @jax.jit
        def grad_fn(c: jax.Array) -> jax.Array:
            def loss(x: jax.Array) -> jax.Array:
                result, _, _ = op.apply({"counts": x}, {}, None)
                return jnp.sum(result["imputed_counts"])

            return jax.grad(loss)(c)

        grad = grad_fn(counts)
        assert jnp.isfinite(grad).all()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_small_n_cells(self, rngs: nnx.Rngs) -> None:
        """Test that imputation works with only 5 cells."""
        config = DiffusionImputerConfig(n_neighbors=3, diffusion_t=2)
        op = DifferentiableDiffusionImputer(config, rngs=rngs)
        key = jax.random.key(0)
        counts = jnp.abs(jax.random.normal(key, (5, 4))) + 0.1

        result, _, _ = op.apply({"counts": counts}, {}, None)
        assert result["imputed_counts"].shape == (5, 4)
        assert jnp.isfinite(result["imputed_counts"]).all()

    def test_diffusion_t_zero(self, rngs: nnx.Rngs) -> None:
        """Test that t=0 returns identity (original data unchanged)."""
        config = DiffusionImputerConfig(n_neighbors=5, diffusion_t=0)
        op = DifferentiableDiffusionImputer(config, rngs=rngs)
        key = jax.random.key(0)
        counts = jnp.abs(jax.random.normal(key, (10, 6))) + 0.1

        result, _, _ = op.apply({"counts": counts}, {}, None)
        assert jnp.allclose(result["imputed_counts"], counts, atol=1e-5)

    def test_diffusion_t_one(self, rngs: nnx.Rngs) -> None:
        """Test that t=1 does one step of smoothing."""
        config = DiffusionImputerConfig(n_neighbors=5, diffusion_t=1)
        op = DifferentiableDiffusionImputer(config, rngs=rngs)
        key = jax.random.key(0)
        counts = jnp.abs(jax.random.normal(key, (10, 6))) + 0.1

        result, _, _ = op.apply({"counts": counts}, {}, None)
        # With t=1 and noisy data, result should differ from input
        assert not jnp.allclose(result["imputed_counts"], counts, atol=1e-3)
        assert jnp.isfinite(result["imputed_counts"]).all()


# =============================================================================
# Transformer Denoiser Tests
# =============================================================================

N_GENES = 50
N_CELLS = 10
HIDDEN_DIM = 32
NUM_LAYERS = 1
NUM_HEADS = 2


class TestTransformerDenoiserConfig:
    """Tests for TransformerDenoiserConfig defaults and customization."""

    def test_default_config(self) -> None:
        """Test default configuration values match specification."""
        config = TransformerDenoiserConfig()
        assert config.n_genes == 2000
        assert config.hidden_dim == 128
        assert config.num_layers == 2
        assert config.num_heads == 4
        assert config.mask_ratio == 0.15
        assert config.dropout_rate == 0.1
        assert config.stochastic is True
        assert config.stream_name == "sample"

    def test_custom_hidden_dim(self) -> None:
        """Test that custom hidden_dim is stored correctly."""
        config = TransformerDenoiserConfig(hidden_dim=64, num_heads=2)
        assert config.hidden_dim == 64
        assert config.num_heads == 2


class TestDifferentiableTransformerDenoiser:
    """Tests for the DifferentiableTransformerDenoiser operator."""

    @pytest.fixture()
    def config(self) -> TransformerDenoiserConfig:
        """Provide a small config for fast tests."""
        return TransformerDenoiserConfig(
            n_genes=N_GENES,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            mask_ratio=0.15,
            dropout_rate=0.0,
        )

    @pytest.fixture()
    def sample_data(self) -> dict[str, jax.Array]:
        """Provide sample count data and gene IDs."""
        key = jax.random.key(7)
        counts = jnp.abs(jax.random.normal(key, (N_CELLS, N_GENES))) + 0.1
        gene_ids = jnp.arange(N_GENES)
        return {"counts": counts, "gene_ids": gene_ids}

    def test_output_keys(
        self,
        config: TransformerDenoiserConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that imputed_counts is present in the output."""
        rngs = nnx.Rngs(params=0, sample=1, dropout=2)
        op = DifferentiableTransformerDenoiser(config, rngs=rngs)
        rng = jax.random.key(99)
        rp = op.generate_random_params(rng, {"counts": (N_CELLS, N_GENES)})
        result, _, _ = op.apply(sample_data, {}, None, random_params=rp)

        assert "imputed_counts" in result
        # Original data preserved
        assert "counts" in result
        assert "gene_ids" in result

    def test_output_shapes(
        self,
        config: TransformerDenoiserConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that imputed_counts has shape (n_cells, n_genes)."""
        rngs = nnx.Rngs(params=0, sample=1, dropout=2)
        op = DifferentiableTransformerDenoiser(config, rngs=rngs)
        rng = jax.random.key(99)
        rp = op.generate_random_params(rng, {"counts": (N_CELLS, N_GENES)})
        result, _, _ = op.apply(sample_data, {}, None, random_params=rp)

        assert result["imputed_counts"].shape == (N_CELLS, N_GENES)

    def test_imputed_finite(
        self,
        config: TransformerDenoiserConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that all imputed values are finite."""
        rngs = nnx.Rngs(params=0, sample=1, dropout=2)
        op = DifferentiableTransformerDenoiser(config, rngs=rngs)
        rng = jax.random.key(99)
        rp = op.generate_random_params(rng, {"counts": (N_CELLS, N_GENES)})
        result, _, _ = op.apply(sample_data, {}, None, random_params=rp)

        assert jnp.isfinite(result["imputed_counts"]).all()

    def test_masked_genes_predicted(
        self,
        config: TransformerDenoiserConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that masked gene positions are filled with predictions."""
        rngs = nnx.Rngs(params=0, sample=1, dropout=2)
        op = DifferentiableTransformerDenoiser(config, rngs=rngs)
        rng = jax.random.key(99)
        rp = op.generate_random_params(rng, {"counts": (N_CELLS, N_GENES)})
        result, _, _ = op.apply(sample_data, {}, None, random_params=rp)

        imputed = result["imputed_counts"]
        mask = result["mask"]

        # Where mask is True (masked), imputed should come from predictions
        # (not zero, since the transformer predicts values for masked positions)
        masked_values = imputed * mask[None, :]
        # At least some masked predictions should be non-zero
        assert jnp.any(masked_values != 0.0)


class TestTransformerDenoiserGradientFlow:
    """Tests for gradient flow through the transformer denoiser."""

    @pytest.fixture()
    def config(self) -> TransformerDenoiserConfig:
        """Provide a small config for gradient tests."""
        return TransformerDenoiserConfig(
            n_genes=N_GENES,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            mask_ratio=0.15,
            dropout_rate=0.0,
        )

    def test_gradient_wrt_input(self, config: TransformerDenoiserConfig) -> None:
        """Test that gradients flow from imputed_counts back to input counts."""
        rngs = nnx.Rngs(params=0, sample=1, dropout=2)
        op = DifferentiableTransformerDenoiser(config, rngs=rngs)
        key = jax.random.key(7)
        counts = jnp.abs(jax.random.normal(key, (N_CELLS, N_GENES))) + 0.1
        gene_ids = jnp.arange(N_GENES)
        rng = jax.random.key(99)
        rp = op.generate_random_params(rng, {"counts": (N_CELLS, N_GENES)})

        def loss_fn(c: jax.Array) -> jax.Array:
            data = {"counts": c, "gene_ids": gene_ids}
            result, _, _ = op.apply(data, {}, None, random_params=rp)
            return jnp.sum(result["imputed_counts"])

        grad = jax.grad(loss_fn)(counts)
        assert grad is not None
        assert grad.shape == counts.shape
        assert jnp.any(grad != 0.0)
        assert jnp.isfinite(grad).all()

    def test_gradient_wrt_params(self, config: TransformerDenoiserConfig) -> None:
        """Test that gradients flow through transformer parameters."""
        rngs = nnx.Rngs(params=0, sample=1, dropout=2)
        op = DifferentiableTransformerDenoiser(config, rngs=rngs)
        key = jax.random.key(7)
        counts = jnp.abs(jax.random.normal(key, (N_CELLS, N_GENES))) + 0.1
        gene_ids = jnp.arange(N_GENES)
        rng = jax.random.key(99)
        rp = op.generate_random_params(rng, {"counts": (N_CELLS, N_GENES)})
        data = {"counts": counts, "gene_ids": gene_ids}

        @nnx.value_and_grad
        def loss_fn(
            model: DifferentiableTransformerDenoiser,
        ) -> jax.Array:
            result, _, _ = model.apply(data, {}, None, random_params=rp)
            return jnp.sum(result["imputed_counts"])

        loss, grads = loss_fn(op)
        assert jnp.isfinite(loss)
        # Check that at least the output head has non-zero gradients
        assert jnp.any(grads.output_head.kernel[...] != 0.0)


class TestTransformerDenoiserJIT:
    """Tests for JIT compilation of the transformer denoiser."""

    @pytest.fixture()
    def config(self) -> TransformerDenoiserConfig:
        """Provide a small config for JIT tests."""
        return TransformerDenoiserConfig(
            n_genes=N_GENES,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            mask_ratio=0.15,
            dropout_rate=0.0,
        )

    def test_jit_apply(self, config: TransformerDenoiserConfig) -> None:
        """Test that jax.jit compiles and runs apply."""
        rngs = nnx.Rngs(params=0, sample=1, dropout=2)
        op = DifferentiableTransformerDenoiser(config, rngs=rngs)
        key = jax.random.key(7)
        counts = jnp.abs(jax.random.normal(key, (N_CELLS, N_GENES))) + 0.1
        gene_ids = jnp.arange(N_GENES)
        data = {"counts": counts, "gene_ids": gene_ids}
        rng = jax.random.key(99)
        rp = op.generate_random_params(rng, {"counts": (N_CELLS, N_GENES)})

        @jax.jit
        def jit_apply(
            d: dict[str, jax.Array],
            random_params: jax.Array,
        ) -> tuple:
            return op.apply(d, {}, None, random_params=random_params)

        result, _, _ = jit_apply(data, rp)
        assert jnp.isfinite(result["imputed_counts"]).all()

    def test_jit_gradient(self, config: TransformerDenoiserConfig) -> None:
        """Test that jax.jit + jax.grad works together."""
        rngs = nnx.Rngs(params=0, sample=1, dropout=2)
        op = DifferentiableTransformerDenoiser(config, rngs=rngs)
        key = jax.random.key(7)
        counts = jnp.abs(jax.random.normal(key, (N_CELLS, N_GENES))) + 0.1
        gene_ids = jnp.arange(N_GENES)
        rng = jax.random.key(99)
        rp = op.generate_random_params(rng, {"counts": (N_CELLS, N_GENES)})

        @jax.jit
        def grad_fn(c: jax.Array) -> jax.Array:
            def loss(x: jax.Array) -> jax.Array:
                data = {"counts": x, "gene_ids": gene_ids}
                result, _, _ = op.apply(data, {}, None, random_params=rp)
                return jnp.sum(result["imputed_counts"])

            return jax.grad(loss)(c)

        grad = grad_fn(counts)
        assert jnp.isfinite(grad).all()


class TestTransformerDenoiserEdgeCases:
    """Tests for edge cases of the transformer denoiser."""

    def test_single_cell(self) -> None:
        """Test that the denoiser works with n_cells=1."""
        config = TransformerDenoiserConfig(
            n_genes=N_GENES,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            mask_ratio=0.15,
            dropout_rate=0.0,
        )
        rngs = nnx.Rngs(params=0, sample=1, dropout=2)
        op = DifferentiableTransformerDenoiser(config, rngs=rngs)
        key = jax.random.key(7)
        counts = jnp.abs(jax.random.normal(key, (1, N_GENES))) + 0.1
        gene_ids = jnp.arange(N_GENES)
        data = {"counts": counts, "gene_ids": gene_ids}
        rng = jax.random.key(99)
        rp = op.generate_random_params(rng, {"counts": (1, N_GENES)})

        result, _, _ = op.apply(data, {}, None, random_params=rp)
        assert result["imputed_counts"].shape == (1, N_GENES)
        assert jnp.isfinite(result["imputed_counts"]).all()

    def test_no_masking(self) -> None:
        """Test that mask_ratio=0 returns data close to input."""
        config = TransformerDenoiserConfig(
            n_genes=N_GENES,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            mask_ratio=0.0,
            dropout_rate=0.0,
        )
        rngs = nnx.Rngs(params=0, sample=1, dropout=2)
        op = DifferentiableTransformerDenoiser(config, rngs=rngs)
        key = jax.random.key(7)
        counts = jnp.abs(jax.random.normal(key, (N_CELLS, N_GENES))) + 0.1
        gene_ids = jnp.arange(N_GENES)
        data = {"counts": counts, "gene_ids": gene_ids}
        rng = jax.random.key(99)
        rp = op.generate_random_params(rng, {"counts": (N_CELLS, N_GENES)})

        result, _, _ = op.apply(data, {}, None, random_params=rp)
        # With mask_ratio=0, no genes are masked so imputed == original counts
        # (the unmasked positions are kept from original counts)
        assert jnp.allclose(result["imputed_counts"], counts, atol=1e-5)
