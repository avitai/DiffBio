"""Integration tests for single-cell operator composition.

Verifies that DiffBio single-cell operators can be chained end-to-end
via dictionary-based data flow. Each operator consumes and produces
data dicts, and downstream operators read keys produced upstream.

Pipeline under test:
    Simulation -> VAE Normalization -> Imputation -> Trajectory -> Cell Annotation
                                                                -> Doublet Detection

Critical validation:
    - JIT compilation of full operator chains
    - End-to-end gradient flow through chained operators
"""

import jax
import jax.numpy as jnp
from flax import nnx

from diffbio.operators.normalization.vae_normalizer import (
    VAENormalizer,
    VAENormalizerConfig,
)
from diffbio.operators.singlecell.cell_annotation import (
    CellAnnotatorConfig,
    DifferentiableCellAnnotator,
)
from diffbio.operators.singlecell.doublet_detection import (
    DifferentiableDoubletScorer,
    DoubletScorerConfig,
)
from diffbio.operators.singlecell.imputation import (
    DifferentiableDiffusionImputer,
    DiffusionImputerConfig,
)
from diffbio.operators.singlecell.simulation import (
    DifferentiableSimulator,
    SimulationConfig,
)
from diffbio.operators.singlecell.trajectory import (
    DifferentiableFateProbability,
    DifferentiablePseudotime,
    FateProbabilityConfig,
    PseudotimeConfig,
)

# Small dims for fast tests
N_CELLS = 20
N_GENES = 30
LATENT_DIM = 5
HIDDEN_DIMS = [16]
N_GROUPS = 2
N_CELL_TYPES = 3


class TestSingleCellPipeline:
    """Integration test: operators compose end-to-end."""

    def test_simulate_normalize_impute(self, rngs: nnx.Rngs) -> None:
        """Simulate data, VAE normalize, then diffusion impute."""
        # Step 1: Simulate counts
        sim_config = SimulationConfig(
            n_cells=N_CELLS,
            n_genes=N_GENES,
            n_groups=N_GROUPS,
            stochastic=True,
            stream_name="sample",
        )
        simulator = DifferentiableSimulator(sim_config, rngs=rngs)
        rp = simulator.generate_random_params(jax.random.key(0), {})
        sim_data, sim_state, sim_meta = simulator.apply({}, {}, None, random_params=rp)

        assert sim_data["counts"].shape == (N_CELLS, N_GENES)
        assert sim_data["group_labels"].shape == (N_CELLS,)

        # Step 2: VAE normalize -- operates per-cell (unbatched)
        counts = sim_data["counts"]
        cell_counts = counts[0]  # single cell
        library_size = jnp.sum(cell_counts)

        vae_config = VAENormalizerConfig(
            n_genes=N_GENES,
            latent_dim=LATENT_DIM,
            hidden_dims=HIDDEN_DIMS,
        )
        normalizer = VAENormalizer(vae_config, rngs=rngs)
        vae_data, _, _ = normalizer.apply(
            {"counts": cell_counts, "library_size": library_size}, {}, None
        )

        assert vae_data["normalized"].shape == (N_GENES,)
        assert vae_data["latent_z"].shape == (LATENT_DIM,)

        # Step 3: Diffusion impute on the original counts
        imp_config = DiffusionImputerConfig(n_neighbors=3, diffusion_t=2)
        imputer = DifferentiableDiffusionImputer(imp_config, rngs=rngs)
        imp_data, _, _ = imputer.apply({"counts": counts}, {}, None)

        assert imp_data["imputed_counts"].shape == (N_CELLS, N_GENES)
        assert imp_data["diffusion_operator"].shape == (N_CELLS, N_CELLS)
        assert jnp.all(jnp.isfinite(imp_data["imputed_counts"]))

    def test_normalize_annotate(self, rngs: nnx.Rngs) -> None:
        """VAE normalize then cell type annotation."""
        key = jax.random.key(1)
        counts = jax.random.uniform(key, (N_CELLS, N_GENES), minval=0.0, maxval=50.0)

        # Step 1: VAE normalize (per-cell operator, use first cell)
        cell_counts = counts[0]
        library_size = jnp.sum(cell_counts)

        vae_config = VAENormalizerConfig(
            n_genes=N_GENES,
            latent_dim=LATENT_DIM,
            hidden_dims=HIDDEN_DIMS,
        )
        normalizer = VAENormalizer(vae_config, rngs=rngs)
        vae_data, _, _ = normalizer.apply(
            {"counts": cell_counts, "library_size": library_size}, {}, None
        )

        assert "latent_z" in vae_data
        assert vae_data["latent_z"].shape == (LATENT_DIM,)
        assert "normalized" in vae_data

        # Step 2: Cell annotation (celltypist mode, operates on batched counts)
        ann_config = CellAnnotatorConfig(
            annotation_mode="celltypist",
            n_cell_types=N_CELL_TYPES,
            n_genes=N_GENES,
            latent_dim=LATENT_DIM,
            hidden_dims=HIDDEN_DIMS,
        )
        annotator = DifferentiableCellAnnotator(ann_config, rngs=rngs)
        ann_data, _, _ = annotator.apply({"counts": counts}, {}, None)

        assert ann_data["cell_type_probabilities"].shape == (N_CELLS, N_CELL_TYPES)
        assert ann_data["cell_type_labels"].shape == (N_CELLS,)
        # Probabilities sum to 1 per cell
        prob_sums = jnp.sum(ann_data["cell_type_probabilities"], axis=-1)
        assert jnp.allclose(prob_sums, 1.0, atol=1e-5)

    def test_trajectory_from_embeddings(self, rngs: nnx.Rngs) -> None:
        """Normalize then trajectory inference: pseudotime and fate."""
        key = jax.random.key(2)
        # Create synthetic embeddings with some structure
        k1, k2 = jax.random.split(key)
        embeddings = jax.random.normal(k1, (N_CELLS, LATENT_DIM))
        # Add a gradient so pseudotime is not degenerate
        gradient = jnp.linspace(0, 1, N_CELLS)[:, None] * jnp.ones(LATENT_DIM)
        embeddings = embeddings * 0.3 + gradient

        # Step 1: Pseudotime
        pt_config = PseudotimeConfig(
            n_neighbors=5,
            n_diffusion_components=3,
            root_cell_index=0,
        )
        pseudotime_op = DifferentiablePseudotime(pt_config, rngs=rngs)
        pt_data, _, _ = pseudotime_op.apply({"embeddings": embeddings}, {}, None)

        assert pt_data["pseudotime"].shape == (N_CELLS,)
        assert pt_data["transition_matrix"].shape == (N_CELLS, N_CELLS)
        assert pt_data["diffusion_components"].shape[0] == N_CELLS
        # Root cell should have pseudotime 0
        assert jnp.isclose(pt_data["pseudotime"][0], 0.0, atol=1e-6)

        # Step 2: Fate probability -- consumes transition_matrix from step 1
        n_terminal = 2
        terminal_states = jnp.array([N_CELLS - 2, N_CELLS - 1])
        fate_config = FateProbabilityConfig(n_macrostates=n_terminal)
        fate_op = DifferentiableFateProbability(fate_config, rngs=rngs)
        fate_data, _, _ = fate_op.apply(
            {
                "transition_matrix": pt_data["transition_matrix"],
                "terminal_states": terminal_states,
            },
            {},
            None,
        )

        assert fate_data["fate_probabilities"].shape == (N_CELLS, n_terminal)
        assert fate_data["macrostates"].shape == (N_CELLS,)
        # Terminal states should have fate probability 1 for themselves
        for i, ts in enumerate(terminal_states):
            assert jnp.isclose(fate_data["fate_probabilities"][ts, i], 1.0, atol=1e-4)

    def test_full_pipeline_jit(self, rngs: nnx.Rngs) -> None:
        """Full pipeline under JIT: impute -> pseudotime -> fate."""
        imputer = DifferentiableDiffusionImputer(
            DiffusionImputerConfig(n_neighbors=3, diffusion_t=2), rngs=rngs
        )
        pseudotime_op = DifferentiablePseudotime(
            PseudotimeConfig(n_neighbors=5, n_diffusion_components=3), rngs=rngs
        )
        fate_op = DifferentiableFateProbability(FateProbabilityConfig(n_macrostates=2), rngs=rngs)

        terminal_states = jnp.array([N_CELLS - 2, N_CELLS - 1])

        @jax.jit
        def pipeline(counts: jax.Array) -> dict[str, jax.Array]:
            """Chain three operators under JIT."""
            # Impute
            imp_data, _, _ = imputer.apply({"counts": counts}, {}, None)
            # Use imputed counts as embeddings for trajectory
            pt_data, _, _ = pseudotime_op.apply(
                {"embeddings": imp_data["imputed_counts"]}, {}, None
            )
            # Fate from transition matrix
            fate_data, _, _ = fate_op.apply(
                {
                    "transition_matrix": pt_data["transition_matrix"],
                    "terminal_states": terminal_states,
                },
                {},
                None,
            )
            return {
                "imputed_counts": imp_data["imputed_counts"],
                "pseudotime": pt_data["pseudotime"],
                "fate_probabilities": fate_data["fate_probabilities"],
            }

        key = jax.random.key(3)
        counts = jax.random.uniform(key, (N_CELLS, N_GENES), minval=0.0, maxval=20.0)
        result = pipeline(counts)

        assert result["imputed_counts"].shape == (N_CELLS, N_GENES)
        assert result["pseudotime"].shape == (N_CELLS,)
        assert result["fate_probabilities"].shape == (N_CELLS, 2)
        assert jnp.all(jnp.isfinite(result["pseudotime"]))
        assert jnp.all(jnp.isfinite(result["fate_probabilities"]))

    def test_full_pipeline_gradient(self, rngs: nnx.Rngs) -> None:
        """Gradient flows through chained operators end-to-end."""
        imputer = DifferentiableDiffusionImputer(
            DiffusionImputerConfig(n_neighbors=3, diffusion_t=2), rngs=rngs
        )
        pseudotime_op = DifferentiablePseudotime(
            PseudotimeConfig(
                n_neighbors=5,
                n_diffusion_components=3,
                root_cell_index=0,
            ),
            rngs=rngs,
        )

        def loss_fn(counts: jax.Array) -> jax.Array:
            """Loss over imputation -> pseudotime chain."""
            # Step 1: Impute
            imp_data, _, _ = imputer.apply({"counts": counts}, {}, None)
            # Step 2: Pseudotime from imputed counts
            pt_data, _, _ = pseudotime_op.apply(
                {"embeddings": imp_data["imputed_counts"]}, {}, None
            )
            return jnp.sum(pt_data["pseudotime"])

        key = jax.random.key(4)
        counts = jax.random.uniform(key, (N_CELLS, N_GENES), minval=1.0, maxval=30.0)

        grad = jax.grad(loss_fn)(counts)

        assert grad.shape == counts.shape
        assert jnp.all(jnp.isfinite(grad))
        # Gradient should be non-zero (end-to-end differentiability)
        assert jnp.any(grad != 0.0)

    def test_doublet_detection_in_pipeline(self, rngs: nnx.Rngs) -> None:
        """Doublet detection after simulation."""
        key = jax.random.key(5)
        counts = jax.random.uniform(key, (N_CELLS, N_GENES), minval=0.0, maxval=50.0)

        scorer_config = DoubletScorerConfig(
            n_neighbors=5,
            n_pca_components=5,
            n_genes=N_GENES,
            sim_doublet_ratio=1.0,
        )
        scorer = DifferentiableDoubletScorer(scorer_config, rngs=rngs)
        rp = scorer.generate_random_params(jax.random.key(10), {"counts": counts.shape})
        result, _, _ = scorer.apply({"counts": counts}, {}, None, random_params=rp)

        assert result["doublet_scores"].shape == (N_CELLS,)
        assert result["predicted_doublets"].shape == (N_CELLS,)
        assert jnp.all(jnp.isfinite(result["doublet_scores"]))

    def test_annotation_gradient(self, rngs: nnx.Rngs) -> None:
        """Gradient flows through cell annotation."""
        ann_config = CellAnnotatorConfig(
            annotation_mode="celltypist",
            n_cell_types=N_CELL_TYPES,
            n_genes=N_GENES,
            latent_dim=LATENT_DIM,
            hidden_dims=HIDDEN_DIMS,
        )
        annotator = DifferentiableCellAnnotator(ann_config, rngs=rngs)

        def loss_fn(counts: jax.Array) -> jax.Array:
            """Loss on cell-type probabilities."""
            data, _, _ = annotator.apply({"counts": counts}, {}, None)
            return jnp.sum(data["cell_type_probabilities"])

        key = jax.random.key(6)
        counts = jax.random.uniform(key, (N_CELLS, N_GENES), minval=0.0, maxval=50.0)

        grad = jax.grad(loss_fn)(counts)

        assert grad.shape == counts.shape
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.any(grad != 0.0)
