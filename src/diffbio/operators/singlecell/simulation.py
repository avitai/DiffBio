"""Splatter-style differentiable single-cell count simulator.

This module provides a fully differentiable implementation of the Splatter
simulation algorithm (Zappia et al., 2017) using a Gamma-Poisson model with
learnable parameters. The simulator generates realistic scRNA-seq count
matrices with group-specific differential expression, batch effects, and
expression-dependent dropout.

Key techniques:
- Gamma-distributed gene means with softplus-parameterized learnable logits
- LogNormal library sizes for cell-level sequencing depth variation
- Soft group assignments via learnable logits and softmax
- Logistic dropout model as a function of mean expression
- Continuous relaxation of Poisson sampling for differentiability

Applications: Benchmarking single-cell methods, data augmentation for
downstream analysis, parameter estimation via gradient-based optimization.
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, Int, PyTree

from diffbio.constants import EPSILON

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SimulationSizeConfig:
    """Cell, gene, group, and batch sizing for the simulator."""

    n_cells: int = 500
    n_genes: int = 200
    n_groups: int = 3
    n_batches: int = 1


@dataclass(frozen=True)
class _SimulationDistributionConfig:
    """Sampling distributions for means, library sizes, and DE effects."""

    mean_shape: float = 0.6
    mean_rate: float = 0.3
    lib_loc: float = 11.0
    lib_scale: float = 0.2
    de_prob: float = 0.1
    de_fac_loc: float = 0.1
    de_fac_scale: float = 0.4


@dataclass(frozen=True)
class _SimulationDropoutConfig:
    """Expression-dependent dropout parameters."""

    dropout_mid: float = -1.0
    dropout_shape: float = -0.5


@dataclass(frozen=True)
class SimulationConfig(
    _SimulationSizeConfig,
    _SimulationDistributionConfig,
    _SimulationDropoutConfig,
    OperatorConfig,
):
    """Configuration for DifferentiableSimulator."""

    def __post_init__(self) -> None:
        """Set stochastic defaults and validate simulation assumptions."""
        object.__setattr__(self, "stochastic", True)
        if self.stream_name is None:
            object.__setattr__(self, "stream_name", "sample")
        if self.n_cells <= 0:
            raise ValueError("n_cells must be positive")
        if self.n_genes <= 0:
            raise ValueError("n_genes must be positive")
        if self.n_groups <= 0:
            raise ValueError("n_groups must be positive")
        if self.n_groups > self.n_cells:
            raise ValueError("n_groups cannot exceed n_cells for even cell-group assignment")
        if self.n_batches <= 0:
            raise ValueError("n_batches must be positive")
        if self.n_batches > self.n_cells:
            raise ValueError("n_batches cannot exceed n_cells for even batch assignment")
        if self.mean_shape <= 0.0:
            raise ValueError("mean_shape must be positive")
        if self.mean_rate <= 0.0:
            raise ValueError("mean_rate must be positive")
        if self.lib_scale < 0.0:
            raise ValueError("lib_scale must be non-negative")
        if not 0.0 <= self.de_prob <= 1.0:
            raise ValueError("de_prob must be in [0, 1]")
        if self.de_fac_scale < 0.0:
            raise ValueError("de_fac_scale must be non-negative")
        if self.dropout_shape >= 0.0:
            raise ValueError("dropout_shape must be negative to model lower-expression dropout")
        super().__post_init__()


class DifferentiableSimulator(OperatorModule):
    """Splatter-style differentiable single-cell count simulator.

    Generates realistic scRNA-seq count matrices following the Splatter
    generative model (Zappia et al., 2017), with all steps implemented
    as differentiable JAX operations.

    Algorithm:
        1. Gene means: softplus-transformed learnable logits, scaled by
           Gamma(shape, rate) random perturbation.
        2. Cell library sizes: LogNormal(lib_loc, lib_scale) sampling.
        3. Group assignments: cells divided evenly across groups, with
           learnable group logits enabling soft assignment.
        4. DE fold-changes: per-group per-gene LogNormal fold-changes,
           masked by a Bernoulli(de_prob) DE indicator.
        5. Cell means: lib_sizes * gene_means * group_fold_change * batch_effect.
        6. Batch effects: exp(learnable batch_shift) multiplicative scaling.
        7. Dropout: sigmoid-based keep probability as function of log(cell_means).
        8. Counts: cell_means * keep_prob (continuous relaxation of Poisson).

    Args:
        config: SimulationConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = SimulationConfig(n_cells=100, n_genes=50, n_groups=2)
        >>> sim = DifferentiableSimulator(config, rngs=nnx.Rngs(0, sample=1))
        >>> rng = jax.random.key(0)
        >>> rp = sim.generate_random_params(rng, {})
        >>> result, state, meta = sim.apply({}, {}, None, random_params=rp)
        >>> result["counts"].shape
        (100, 50)
    """

    def __init__(
        self,
        config: SimulationConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the differentiable simulator.

        Args:
            config: Simulation configuration.
            rngs: Random number generators for parameter initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        safe_rngs = rngs or nnx.Rngs(0)

        # Learnable gene-mean logits: softplus maps these to positive means
        key = safe_rngs.params()
        init_logits = jax.random.normal(key, (config.n_genes,)) * 0.5
        self.gene_means_logits = nnx.Param(init_logits)

        # Learnable group logits for soft cell-group assignment
        key = safe_rngs.params()
        init_group_logits = jax.random.normal(key, (config.n_groups,)) * 0.1
        self.group_logits = nnx.Param(init_group_logits)

        # Learnable batch shift (additive on log scale)
        self.batch_shift = nnx.Param(jnp.zeros(config.n_batches))

    def generate_random_params(
        self,
        rng: jax.Array,
        data_shapes: PyTree,
    ) -> dict[str, jax.Array]:
        """Generate random keys for all stochastic sampling steps.

        Args:
            rng: JAX random key.
            data_shapes: PyTree with shapes (unused, kept for interface).

        Returns:
            Dictionary of JAX random keys for each sampling step.
        """
        keys = jax.random.split(rng, 6)
        return {
            "gene_means_key": keys[0],
            "lib_sizes_key": keys[1],
            "group_key": keys[2],
            "de_mask_key": keys[3],
            "de_fold_key": keys[4],
            "poisson_key": keys[5],
        }

    def _sample_gene_means(
        self,
        key: jax.Array,
    ) -> Float[Array, "n_genes"]:
        """Sample gene means from Gamma distribution scaled by learnable logits.

        The base gene means come from softplus(learnable_logits), then are
        perturbed by Gamma(shape, 1) / rate to maintain the Splatter prior.

        Args:
            key: JAX random key for Gamma sampling.

        Returns:
            Positive gene mean expression levels of shape (n_genes,).
        """
        config = self.config

        # Learnable base means via softplus (always positive)
        base_means = jax.nn.softplus(self.gene_means_logits[...])

        # Stochastic Gamma perturbation: gamma(shape) / rate
        gamma_samples = jax.random.gamma(key, config.mean_shape, shape=(config.n_genes,))
        gamma_factor = gamma_samples / (config.mean_rate + EPSILON)

        # Combine learnable and stochastic components
        return base_means * gamma_factor + EPSILON

    def _sample_library_sizes(
        self,
        key: jax.Array,
    ) -> Float[Array, "n_cells"]:
        """Sample cell library sizes from LogNormal distribution.

        Args:
            key: JAX random key for Normal sampling.

        Returns:
            Positive library sizes of shape (n_cells,).
        """
        config = self.config
        log_lib = config.lib_loc + config.lib_scale * jax.random.normal(key, (config.n_cells,))
        return jnp.exp(log_lib)

    def _assign_groups(
        self,
        key: jax.Array,
    ) -> tuple[Int[Array, "n_cells"], Float[Array, "n_cells n_groups"]]:
        """Assign cells to groups using even division with learnable logits.

        Cells are divided evenly across groups. The soft assignment
        probabilities are derived from learnable group logits via softmax,
        enabling gradient flow through group membership.

        Args:
            key: JAX random key (unused, kept for interface consistency).

        Returns:
            Tuple of (hard_labels, soft_assignments) where:
                - hard_labels: Integer group labels of shape (n_cells,).
                - soft_assignments: Soft probabilities of shape (n_cells, n_groups).
        """
        config = self.config

        # Even division of cells across groups
        cells_per_group = config.n_cells // config.n_groups
        hard_labels = jnp.repeat(jnp.arange(config.n_groups), cells_per_group)
        # Handle remainder cells
        remainder = config.n_cells - len(hard_labels)
        if remainder > 0:
            extra = jnp.full(remainder, config.n_groups - 1)
            hard_labels = jnp.concatenate([hard_labels, extra])

        # Soft assignments from learnable group logits
        group_probs = jax.nn.softmax(self.group_logits[...])  # (n_groups,)
        # Expand to full soft assignment matrix via one-hot weighting
        one_hot = jax.nn.one_hot(hard_labels, config.n_groups)  # (n_cells, n_groups)
        # Scale by learnable logits for gradient flow
        soft_assignments_full = one_hot * group_probs[None, :]  # (n_cells, n_groups)
        soft_assignments_full = soft_assignments_full / (
            jnp.sum(soft_assignments_full, axis=-1, keepdims=True) + EPSILON
        )

        return hard_labels, soft_assignments_full

    def _compute_de_fold_changes(
        self,
        de_mask_key: jax.Array,
        de_fold_key: jax.Array,
    ) -> tuple[Float[Array, "n_groups n_genes"], Float[Array, "n_groups n_genes"]]:
        """Compute per-group DE fold-changes with Bernoulli masking.

        For each group, a subset of genes (determined by de_prob) receives
        a LogNormal fold-change; non-DE genes have fold-change 1.0.

        Args:
            de_mask_key: JAX random key for DE mask sampling.
            de_fold_key: JAX random key for fold-change sampling.

        Returns:
            Tuple of (fold_changes, de_mask) where:
                - fold_changes: Multiplicative fold-changes (n_groups, n_genes).
                - de_mask: Binary DE indicator (n_groups, n_genes).
        """
        config = self.config

        # Bernoulli mask: which genes are DE in each group
        de_mask = jax.random.bernoulli(
            de_mask_key, config.de_prob, shape=(config.n_groups, config.n_genes)
        ).astype(jnp.float32)

        # LogNormal fold-changes for DE genes
        log_fc = config.de_fac_loc + config.de_fac_scale * jax.random.normal(
            de_fold_key, (config.n_groups, config.n_genes)
        )
        fold_changes_raw = jnp.exp(log_fc)

        # Non-DE genes get fold-change of 1.0
        fold_changes = jnp.where(de_mask > 0.5, fold_changes_raw, 1.0)

        return fold_changes, de_mask

    def _apply_batch_effects(
        self,
        cell_means: Float[Array, "n_cells n_genes"],
        batch_labels: Int[Array, "n_cells"],
    ) -> Float[Array, "n_cells n_genes"]:
        """Apply multiplicative batch effects to cell means.

        Batch effects are exp(learnable_shift), applied multiplicatively.

        Args:
            cell_means: Pre-batch cell expression means.
            batch_labels: Integer batch assignment per cell.

        Returns:
            Batch-corrected cell means of shape (n_cells, n_genes).
        """
        batch_factors = jnp.exp(self.batch_shift[...])[batch_labels]  # (n_cells,)
        return cell_means * batch_factors[:, None]

    def _apply_dropout(
        self,
        cell_means: Float[Array, "n_cells n_genes"],
    ) -> Float[Array, "n_cells n_genes"]:
        """Apply expression-dependent dropout via logistic function.

        keep_prob = sigmoid(dropout_shape * (log(cell_means) - dropout_mid))

        At low expression, keep_prob is low (more zeros); at high expression,
        keep_prob approaches 1.

        Args:
            cell_means: Cell expression means (positive values).

        Returns:
            Dropout-adjusted cell means.
        """
        config = self.config
        log_means = jnp.log(cell_means + EPSILON)
        keep_prob = jax.nn.sigmoid(config.dropout_shape * (log_means - config.dropout_mid))
        return cell_means * keep_prob

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Simulate a single-cell count matrix.

        Follows the Splatter generative model with all steps differentiable:
        gene means, library sizes, group DE, batch effects, dropout, and
        Poisson count generation (continuous relaxation).

        Args:
            data: Input dictionary (may be empty; existing keys are preserved).
            state: Element state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: Dictionary of JAX random keys from generate_random_params.
            stats: Not used.

        Returns:
            Tuple of (output_data, state, metadata) where output_data contains:
                - All original data keys preserved.
                - "counts": Simulated count matrix (n_cells, n_genes).
                - "group_labels": Hard group assignments (n_cells,).
                - "batch_labels": Batch assignments (n_cells,).
                - "gene_means": Per-gene expression means (n_genes,).
                - "de_mask": Binary DE indicator (n_groups, n_genes).
        """
        rp = random_params or {}

        # Step 1: Gene means from learnable logits + Gamma perturbation
        gene_means = self._sample_gene_means(rp["gene_means_key"])

        # Step 2: Library sizes from LogNormal
        lib_sizes = self._sample_library_sizes(rp["lib_sizes_key"])

        # Step 3: Group assignments
        group_labels, soft_assignments = self._assign_groups(rp["group_key"])

        # Step 4: DE fold-changes per group
        fold_changes, de_mask = self._compute_de_fold_changes(rp["de_mask_key"], rp["de_fold_key"])

        # Step 5: Compute cell means = lib_size * gene_mean * group_fold_change
        # Use soft assignments for differentiability:
        # effective_fold = sum_g(soft_assign[c,g] * fold_change[g,:])
        effective_fold = jnp.einsum(
            "cg,gn->cn", soft_assignments, fold_changes
        )  # (n_cells, n_genes)
        cell_means = lib_sizes[:, None] * gene_means[None, :] * effective_fold

        # Step 6: Batch effects
        # Assign cells evenly across batches
        config = self.config
        cells_per_batch = config.n_cells // config.n_batches
        batch_labels = jnp.repeat(jnp.arange(config.n_batches), cells_per_batch)
        remainder = config.n_cells - len(batch_labels)
        if remainder > 0:
            extra = jnp.full(remainder, config.n_batches - 1)
            batch_labels = jnp.concatenate([batch_labels, extra])

        cell_means = self._apply_batch_effects(cell_means, batch_labels)

        # Step 7: Dropout
        cell_means = self._apply_dropout(cell_means)

        # Step 8: Continuous relaxation of Poisson sampling
        # Use the reparameterization: counts ~ Poisson(lambda) ≈ lambda + sqrt(lambda) * noise
        noise = jax.random.normal(rp["poisson_key"], cell_means.shape)
        counts = cell_means + jnp.sqrt(cell_means + EPSILON) * noise
        # Ensure non-negative counts
        counts = jax.nn.relu(counts)

        # Build output
        output_data = {
            **data,
            "counts": counts,
            "group_labels": group_labels,
            "batch_labels": batch_labels,
            "gene_means": gene_means,
            "de_mask": de_mask,
        }

        return output_data, state, metadata
