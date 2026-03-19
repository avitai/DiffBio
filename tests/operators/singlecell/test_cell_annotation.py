"""Tests for diffbio.operators.singlecell.cell_annotation module.

These tests define the expected behavior of the DifferentiableCellAnnotator
operator for cell type annotation using three modes: celltypist, cellassign,
and scanvi.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.singlecell.cell_annotation import (
    CellAnnotatorConfig,
    DifferentiableCellAnnotator,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_CELLS = 20
N_GENES = 50
N_TYPES = 5
LATENT_DIM = 8
HIDDEN_DIMS = [32, 16]


@pytest.fixture
def celltypist_config() -> CellAnnotatorConfig:
    """Default celltypist-mode config."""
    return CellAnnotatorConfig(
        annotation_mode="celltypist",
        n_cell_types=N_TYPES,
        n_genes=N_GENES,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
        stochastic=True,
        stream_name="sample",
    )


@pytest.fixture
def cellassign_config() -> CellAnnotatorConfig:
    """Default cellassign-mode config."""
    return CellAnnotatorConfig(
        annotation_mode="cellassign",
        n_cell_types=N_TYPES,
        n_genes=N_GENES,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
        marker_matrix_shape=(N_TYPES, N_GENES),
        stochastic=True,
        stream_name="sample",
    )


@pytest.fixture
def scanvi_config() -> CellAnnotatorConfig:
    """Default scanvi-mode config."""
    return CellAnnotatorConfig(
        annotation_mode="scanvi",
        n_cell_types=N_TYPES,
        n_genes=N_GENES,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
        stochastic=True,
        stream_name="sample",
    )


@pytest.fixture
def counts_data() -> dict[str, jax.Array]:
    """Synthetic count matrix."""
    key = jax.random.key(0)
    counts = jax.random.poisson(key, lam=5.0, shape=(N_CELLS, N_GENES)).astype(jnp.float32)
    return {"counts": counts}


@pytest.fixture
def marker_matrix() -> jax.Array:
    """Binary marker matrix where each type has distinct marker genes."""
    m = jnp.zeros((N_TYPES, N_GENES), dtype=jnp.float32)
    genes_per_type = N_GENES // N_TYPES
    for t in range(N_TYPES):
        m = m.at[t, t * genes_per_type : (t + 1) * genes_per_type].set(1.0)
    return m


# ===========================================================================
# Config tests
# ===========================================================================


class TestCellAnnotatorConfig:
    """Tests for CellAnnotatorConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = CellAnnotatorConfig(stochastic=True, stream_name="sample")
        assert config.annotation_mode == "celltypist"
        assert config.n_cell_types == 10
        assert config.n_genes == 2000
        assert config.latent_dim == 10
        assert config.hidden_dims == [128, 64]
        assert config.marker_matrix_shape is None
        assert config.stochastic is True
        assert config.stream_name == "sample"

    def test_scanvi_mode(self) -> None:
        """Test scanvi mode configuration."""
        config = CellAnnotatorConfig(
            annotation_mode="scanvi",
            stochastic=True,
            stream_name="sample",
        )
        assert config.annotation_mode == "scanvi"

    def test_cellassign_mode(self) -> None:
        """Test cellassign mode configuration with marker shape."""
        config = CellAnnotatorConfig(
            annotation_mode="cellassign",
            marker_matrix_shape=(10, 2000),
            stochastic=True,
            stream_name="sample",
        )
        assert config.annotation_mode == "cellassign"
        assert config.marker_matrix_shape == (10, 2000)


# ===========================================================================
# Celltypist mode tests
# ===========================================================================


class TestCelltypistMode:
    """Tests for celltypist annotation mode."""

    def test_output_keys(self, rngs, celltypist_config, counts_data) -> None:
        """Output must contain cell_type_probabilities, cell_type_labels, latent."""
        op = DifferentiableCellAnnotator(celltypist_config, rngs=rngs)
        result, _, _ = op.apply(counts_data, {}, None)
        assert "cell_type_probabilities" in result
        assert "cell_type_labels" in result
        assert "latent" in result

    def test_output_shapes(self, rngs, celltypist_config, counts_data) -> None:
        """Shapes: probabilities (n, n_types), labels (n,), latent (n, latent_dim)."""
        op = DifferentiableCellAnnotator(celltypist_config, rngs=rngs)
        result, _, _ = op.apply(counts_data, {}, None)
        assert result["cell_type_probabilities"].shape == (N_CELLS, N_TYPES)
        assert result["cell_type_labels"].shape == (N_CELLS,)
        assert result["latent"].shape == (N_CELLS, LATENT_DIM)

    def test_probabilities_sum_to_one(self, rngs, celltypist_config, counts_data) -> None:
        """Each row of cell_type_probabilities must sum to 1."""
        op = DifferentiableCellAnnotator(celltypist_config, rngs=rngs)
        result, _, _ = op.apply(counts_data, {}, None)
        row_sums = jnp.sum(result["cell_type_probabilities"], axis=-1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-5)

    def test_labels_in_range(self, rngs, celltypist_config, counts_data) -> None:
        """All labels must be in [0, n_types)."""
        op = DifferentiableCellAnnotator(celltypist_config, rngs=rngs)
        result, _, _ = op.apply(counts_data, {}, None)
        labels = result["cell_type_labels"]
        assert jnp.all(labels >= 0)
        assert jnp.all(labels < N_TYPES)


# ===========================================================================
# Cellassign mode tests
# ===========================================================================


class TestCellassignMode:
    """Tests for cellassign annotation mode."""

    def test_output_keys(self, rngs, cellassign_config, counts_data, marker_matrix) -> None:
        """Output must contain required keys."""
        op = DifferentiableCellAnnotator(cellassign_config, rngs=rngs)
        data = {**counts_data, "marker_matrix": marker_matrix}
        result, _, _ = op.apply(data, {}, None)
        assert "cell_type_probabilities" in result
        assert "cell_type_labels" in result
        assert "latent" in result

    def test_output_shapes(self, rngs, cellassign_config, counts_data, marker_matrix) -> None:
        """Output shapes must match spec."""
        op = DifferentiableCellAnnotator(cellassign_config, rngs=rngs)
        data = {**counts_data, "marker_matrix": marker_matrix}
        result, _, _ = op.apply(data, {}, None)
        assert result["cell_type_probabilities"].shape == (N_CELLS, N_TYPES)
        assert result["cell_type_labels"].shape == (N_CELLS,)
        assert result["latent"].shape == (N_CELLS, LATENT_DIM)

    def test_marker_influence(self, rngs, cellassign_config, marker_matrix) -> None:
        """Cells expressing marker genes for type A should get higher prob for A."""
        op = DifferentiableCellAnnotator(cellassign_config, rngs=rngs)

        genes_per_type = N_GENES // N_TYPES
        # Create cells strongly expressing markers for type 0
        counts_type0 = jnp.zeros((4, N_GENES), dtype=jnp.float32)
        counts_type0 = counts_type0.at[:, :genes_per_type].set(100.0)

        # Create cells strongly expressing markers for type 1
        counts_type1 = jnp.zeros((4, N_GENES), dtype=jnp.float32)
        counts_type1 = counts_type1.at[:, genes_per_type : 2 * genes_per_type].set(100.0)

        counts = jnp.concatenate([counts_type0, counts_type1], axis=0)
        data = {"counts": counts, "marker_matrix": marker_matrix}
        result, _, _ = op.apply(data, {}, None)

        probs = result["cell_type_probabilities"]
        # Type-0 cells should have higher probability for type 0 than type 1
        assert jnp.mean(probs[:4, 0]) > jnp.mean(probs[:4, 1])
        # Type-1 cells should have higher probability for type 1 than type 0
        assert jnp.mean(probs[4:, 1]) > jnp.mean(probs[4:, 0])


# ===========================================================================
# Scanvi mode tests
# ===========================================================================


class TestScanviMode:
    """Tests for scanvi annotation mode."""

    def test_output_keys(self, rngs, scanvi_config, counts_data) -> None:
        """Output must contain required keys."""
        op = DifferentiableCellAnnotator(scanvi_config, rngs=rngs)
        result, _, _ = op.apply(counts_data, {}, None)
        assert "cell_type_probabilities" in result
        assert "cell_type_labels" in result
        assert "latent" in result

    def test_output_shapes(self, rngs, scanvi_config, counts_data) -> None:
        """Output shapes must match spec."""
        op = DifferentiableCellAnnotator(scanvi_config, rngs=rngs)
        result, _, _ = op.apply(counts_data, {}, None)
        assert result["cell_type_probabilities"].shape == (N_CELLS, N_TYPES)
        assert result["cell_type_labels"].shape == (N_CELLS,)
        assert result["latent"].shape == (N_CELLS, LATENT_DIM)

    def test_labeled_cells_guide_predictions(self, rngs, scanvi_config) -> None:
        """Labeled cells should receive high probability for their known type."""
        op = DifferentiableCellAnnotator(scanvi_config, rngs=rngs)

        key = jax.random.key(7)
        counts = jax.random.poisson(key, lam=5.0, shape=(N_CELLS, N_GENES)).astype(jnp.float32)

        # Label first 10 cells as type 2
        known_labels = jnp.full((10,), 2, dtype=jnp.int32)
        label_indices = jnp.arange(10, dtype=jnp.int32)

        data = {
            "counts": counts,
            "known_labels": known_labels,
            "label_indices": label_indices,
        }
        result, _, _ = op.apply(data, {}, None)

        probs = result["cell_type_probabilities"]
        # Labeled cells should have elevated probability for type 2
        labeled_prob_for_type2 = jnp.mean(probs[:10, 2])
        labeled_prob_for_others = jnp.mean(probs[:10, :2])
        assert labeled_prob_for_type2 > labeled_prob_for_others

    def test_has_type_conditioned_prior_params(self, rngs, scanvi_config) -> None:
        """Scanvi must have learnable prior_means and prior_logvars."""
        op = DifferentiableCellAnnotator(scanvi_config, rngs=rngs)
        assert hasattr(op, "prior_means")
        assert hasattr(op, "prior_logvars")
        assert op.prior_means[...].shape == (N_TYPES, LATENT_DIM)
        assert op.prior_logvars[...].shape == (N_TYPES, LATENT_DIM)

    def test_type_conditioned_kl_differs_from_standard(self, rngs, scanvi_config) -> None:
        """Type-conditioned KL must differ from standard N(0,I) KL."""
        op = DifferentiableCellAnnotator(scanvi_config, rngs=rngs)

        # Shift one prior mean away from zero so the two KLs diverge
        op.prior_means = nnx.Param(op.prior_means[...].at[0, :].set(jnp.ones(LATENT_DIM) * 3.0))

        key = jax.random.key(11)
        counts = jax.random.poisson(key, lam=5.0, shape=(N_CELLS, N_GENES)).astype(jnp.float32)

        mean, logvar = op.encode(counts)
        uniform_probs = jnp.ones((N_CELLS, N_TYPES)) / N_TYPES

        kl_conditioned = op._type_conditioned_kl(mean, logvar, uniform_probs)
        kl_standard = op.kl_divergence(mean, logvar)

        assert jnp.isfinite(kl_conditioned)
        assert jnp.isfinite(kl_standard)
        # They should NOT be equal because prior_means[0] != 0
        assert not jnp.allclose(kl_conditioned, kl_standard, atol=1e-3)

    def test_type_conditioned_kl_reduces_to_standard_at_zero_prior(
        self, rngs, scanvi_config
    ) -> None:
        """When all priors are N(0,I), type-conditioned KL equals standard KL."""
        op = DifferentiableCellAnnotator(scanvi_config, rngs=rngs)

        # Reset priors to N(0, I) -- means=0, logvars=0
        op.prior_means = nnx.Param(jnp.zeros((N_TYPES, LATENT_DIM)))
        op.prior_logvars = nnx.Param(jnp.zeros((N_TYPES, LATENT_DIM)))

        key = jax.random.key(12)
        counts = jax.random.poisson(key, lam=5.0, shape=(N_CELLS, N_GENES)).astype(jnp.float32)

        mean, logvar = op.encode(counts)
        uniform_probs = jnp.ones((N_CELLS, N_TYPES)) / N_TYPES

        kl_conditioned = op._type_conditioned_kl(mean, logvar, uniform_probs)
        kl_standard = op.kl_divergence(mean, logvar)

        assert jnp.allclose(kl_conditioned, kl_standard, atol=1e-3)

    def test_scanvi_elbo_uses_type_conditioned_kl(self, rngs, scanvi_config) -> None:
        """Scanvi ELBO must use type-conditioned KL, not standard KL."""
        op = DifferentiableCellAnnotator(scanvi_config, rngs=rngs)

        # Push prior for type 0 far away so the ELBO changes measurably
        op.prior_means = nnx.Param(op.prior_means[...].at[0, :].set(jnp.ones(LATENT_DIM) * 5.0))

        key = jax.random.key(13)
        counts = jax.random.poisson(key, lam=5.0, shape=(N_CELLS, N_GENES)).astype(jnp.float32)

        elbo_shifted = op.compute_elbo_loss(counts)

        # Reset priors to N(0,I) and recompute
        op.prior_means = nnx.Param(jnp.zeros((N_TYPES, LATENT_DIM)))
        op.prior_logvars = nnx.Param(jnp.zeros((N_TYPES, LATENT_DIM)))

        elbo_standard = op.compute_elbo_loss(counts)

        assert jnp.isfinite(elbo_shifted)
        assert jnp.isfinite(elbo_standard)
        # Shifted prior should produce a different ELBO
        assert not jnp.allclose(elbo_shifted, elbo_standard, atol=1e-2)


# ===========================================================================
# Gradient flow tests
# ===========================================================================


class TestGradientFlow:
    """Tests for gradient flow through each mode."""

    def test_gradient_celltypist(self, rngs, celltypist_config, counts_data) -> None:
        """Gradients must flow through celltypist mode."""
        op = DifferentiableCellAnnotator(celltypist_config, rngs=rngs)

        @nnx.value_and_grad
        def loss_fn(model: DifferentiableCellAnnotator) -> jax.Array:
            result, _, _ = model.apply(counts_data, {}, None)
            return jnp.sum(result["cell_type_probabilities"])

        loss, grads = loss_fn(op)
        assert jnp.isfinite(loss)
        # Classifier head must receive gradients
        assert hasattr(grads, "classifier_head")
        grad_vals = grads.classifier_head.kernel[...]
        assert jnp.any(grad_vals != 0.0)

    def test_gradient_cellassign(self, rngs, cellassign_config, counts_data, marker_matrix) -> None:
        """Gradients must flow to mu params in cellassign mode."""
        op = DifferentiableCellAnnotator(cellassign_config, rngs=rngs)
        data = {**counts_data, "marker_matrix": marker_matrix}

        @nnx.value_and_grad
        def loss_fn(model: DifferentiableCellAnnotator) -> jax.Array:
            result, _, _ = model.apply(data, {}, None)
            return jnp.sum(result["cell_type_probabilities"])

        loss, grads = loss_fn(op)
        assert jnp.isfinite(loss)
        assert hasattr(grads, "log_mu")
        grad_vals = grads.log_mu[...]
        assert jnp.any(grad_vals != 0.0)

    def test_gradient_scanvi(self, rngs, scanvi_config, counts_data) -> None:
        """Gradients must flow through scanvi mode."""
        op = DifferentiableCellAnnotator(scanvi_config, rngs=rngs)

        @nnx.value_and_grad
        def loss_fn(model: DifferentiableCellAnnotator) -> jax.Array:
            result, _, _ = model.apply(counts_data, {}, None)
            return jnp.sum(result["cell_type_probabilities"])

        loss, grads = loss_fn(op)
        assert jnp.isfinite(loss)
        assert hasattr(grads, "classifier_head")

    def test_gradient_scanvi_prior_params(self, rngs, scanvi_config) -> None:
        """Gradients must flow through prior_means and prior_logvars via ELBO."""
        op = DifferentiableCellAnnotator(scanvi_config, rngs=rngs)

        key = jax.random.key(14)
        counts = jax.random.poisson(key, lam=5.0, shape=(N_CELLS, N_GENES)).astype(jnp.float32)

        @nnx.value_and_grad
        def loss_fn(model: DifferentiableCellAnnotator) -> jax.Array:
            return model.compute_elbo_loss(counts)

        loss, grads = loss_fn(op)
        assert jnp.isfinite(loss)

        # Prior parameters must receive gradients
        assert hasattr(grads, "prior_means")
        assert hasattr(grads, "prior_logvars")
        grad_means = grads.prior_means[...]
        grad_logvars = grads.prior_logvars[...]
        assert jnp.any(grad_means != 0.0)
        assert jnp.any(grad_logvars != 0.0)

    def test_gradient_scanvi_elbo_with_labels(self, rngs, scanvi_config) -> None:
        """Gradients flow through scanvi ELBO when labels are provided."""
        op = DifferentiableCellAnnotator(scanvi_config, rngs=rngs)

        key = jax.random.key(15)
        counts = jax.random.poisson(key, lam=5.0, shape=(N_CELLS, N_GENES)).astype(jnp.float32)

        known_labels = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
        label_indices = jnp.arange(5, dtype=jnp.int32)

        @nnx.value_and_grad
        def loss_fn(model: DifferentiableCellAnnotator) -> jax.Array:
            return model.compute_elbo_loss(counts, known_labels, label_indices)

        loss, grads = loss_fn(op)
        assert jnp.isfinite(loss)
        # Both classifier and prior params should get gradients
        assert hasattr(grads, "prior_means")
        assert hasattr(grads, "classifier_head")
        assert jnp.any(grads.prior_means[...] != 0.0)


# ===========================================================================
# JIT compatibility tests
# ===========================================================================


class TestJITCompatibility:
    """Tests for JAX JIT compilation."""

    def test_jit_celltypist(self, rngs, celltypist_config, counts_data) -> None:
        """Celltypist mode must be JIT-compilable."""
        op = DifferentiableCellAnnotator(celltypist_config, rngs=rngs)

        @jax.jit
        def run(data: dict[str, jax.Array]) -> dict[str, jax.Array]:
            result, _, _ = op.apply(data, {}, None)
            return result

        result = run(counts_data)
        assert jnp.isfinite(result["cell_type_probabilities"]).all()

    def test_jit_cellassign(self, rngs, cellassign_config, counts_data, marker_matrix) -> None:
        """Cellassign mode must be JIT-compilable."""
        op = DifferentiableCellAnnotator(cellassign_config, rngs=rngs)
        data = {**counts_data, "marker_matrix": marker_matrix}

        @jax.jit
        def run(data: dict[str, jax.Array]) -> dict[str, jax.Array]:
            result, _, _ = op.apply(data, {}, None)
            return result

        result = run(data)
        assert jnp.isfinite(result["cell_type_probabilities"]).all()

    def test_jit_scanvi(self, rngs, scanvi_config, counts_data) -> None:
        """Scanvi mode must be JIT-compilable."""
        op = DifferentiableCellAnnotator(scanvi_config, rngs=rngs)

        @jax.jit
        def run(data: dict[str, jax.Array]) -> dict[str, jax.Array]:
            result, _, _ = op.apply(data, {}, None)
            return result

        result = run(counts_data)
        assert jnp.isfinite(result["cell_type_probabilities"]).all()

    def test_jit_scanvi_elbo(self, rngs, scanvi_config) -> None:
        """Scanvi ELBO with type-conditioned KL must be JIT-compilable."""
        op = DifferentiableCellAnnotator(scanvi_config, rngs=rngs)

        key = jax.random.key(16)
        counts = jax.random.poisson(key, lam=5.0, shape=(N_CELLS, N_GENES)).astype(jnp.float32)

        @jax.jit
        def run_elbo(c: jax.Array) -> jax.Array:
            return op.compute_elbo_loss(c)

        loss = run_elbo(counts)
        assert jnp.isfinite(loss)


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_cell_type(self, rngs) -> None:
        """n_types=1 should degenerate gracefully (all prob = 1)."""
        config = CellAnnotatorConfig(
            annotation_mode="celltypist",
            n_cell_types=1,
            n_genes=N_GENES,
            latent_dim=LATENT_DIM,
            hidden_dims=HIDDEN_DIMS,
            stochastic=True,
            stream_name="sample",
        )
        op = DifferentiableCellAnnotator(config, rngs=rngs)
        key = jax.random.key(99)
        counts = jax.random.poisson(key, lam=5.0, shape=(4, N_GENES)).astype(jnp.float32)
        data = {"counts": counts}
        result, _, _ = op.apply(data, {}, None)
        assert result["cell_type_probabilities"].shape == (4, 1)
        assert jnp.allclose(result["cell_type_probabilities"], 1.0, atol=1e-5)

    def test_two_cells(self, rngs, celltypist_config) -> None:
        """Minimal input of 2 cells must produce valid output."""
        op = DifferentiableCellAnnotator(celltypist_config, rngs=rngs)
        key = jax.random.key(10)
        counts = jax.random.poisson(key, lam=3.0, shape=(2, N_GENES)).astype(jnp.float32)
        data = {"counts": counts}
        result, _, _ = op.apply(data, {}, None)
        assert result["cell_type_probabilities"].shape == (2, N_TYPES)
        row_sums = jnp.sum(result["cell_type_probabilities"], axis=-1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-5)


# ===========================================================================
# ZINB likelihood tests
# ===========================================================================


class TestScanviZINBLikelihood:
    """Tests for ZINB likelihood option in scanvi mode."""

    def test_scanvi_zinb_likelihood(self, rngs) -> None:
        """ZINB mode must produce finite ELBO with non-zero gradients."""
        config = CellAnnotatorConfig(
            annotation_mode="scanvi",
            n_cell_types=N_TYPES,
            n_genes=N_GENES,
            latent_dim=LATENT_DIM,
            hidden_dims=HIDDEN_DIMS,
            gene_likelihood="zinb",
            stochastic=True,
            stream_name="sample",
        )
        op = DifferentiableCellAnnotator(config, rngs=rngs)

        key = jax.random.key(20)
        counts = jax.random.poisson(key, lam=5.0, shape=(N_CELLS, N_GENES)).astype(jnp.float32)

        # ELBO must be finite
        elbo = op.compute_elbo_loss(counts)
        assert jnp.isfinite(elbo)

        # Gradients must flow through ZINB heads
        @nnx.value_and_grad
        def loss_fn(model: DifferentiableCellAnnotator) -> jax.Array:
            return model.compute_elbo_loss(counts)

        loss, grads = loss_fn(op)
        assert jnp.isfinite(loss)
        # ZINB decoder heads must receive gradients
        assert hasattr(grads, "fc_log_theta")
        assert hasattr(grads, "fc_pi_logit")
        assert jnp.any(grads.fc_log_theta.kernel[...] != 0.0)
        assert jnp.any(grads.fc_pi_logit.kernel[...] != 0.0)

    def test_zinb_has_decoder_heads(self, rngs) -> None:
        """ZINB mode must create fc_log_theta and fc_pi_logit layers."""
        config = CellAnnotatorConfig(
            annotation_mode="scanvi",
            n_cell_types=N_TYPES,
            n_genes=N_GENES,
            latent_dim=LATENT_DIM,
            hidden_dims=HIDDEN_DIMS,
            gene_likelihood="zinb",
            stochastic=True,
            stream_name="sample",
        )
        op = DifferentiableCellAnnotator(config, rngs=rngs)
        assert hasattr(op, "fc_log_theta")
        assert hasattr(op, "fc_pi_logit")

    def test_poisson_default_no_zinb_heads(self, rngs, scanvi_config) -> None:
        """Default poisson mode must NOT create ZINB decoder heads."""
        op = DifferentiableCellAnnotator(scanvi_config, rngs=rngs)
        assert not hasattr(op, "fc_log_theta")
        assert not hasattr(op, "fc_pi_logit")
