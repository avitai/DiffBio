"""Differentiable differential distribution operator for single-cell analysis.

This module provides a differentiable implementation of the KS-test and
pattern classification for detecting distributional differences between
two conditions in single-cell expression data, inspired by scDD
(Korthauer et al., Genome Biology 2016).

Key technique: Replace the hard empirical CDF step function with a
sigmoid-smoothed soft CDF, and replace the hard max in the KS statistic
with logsumexp-based soft_max from TemperatureOperator.

Applications: Identifying genes with differential distributions (shift,
scale, both, or none) between conditions in scRNA-seq experiments.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from flax import nnx
from jaxtyping import Array, Float, PyTree

from diffbio.constants import EPSILON
from diffbio.core.base_operators import TemperatureOperator
from diffbio.utils.nn_utils import ensure_rngs


@dataclass
class DifferentialDistributionConfig(OperatorConfig):
    """Configuration for differentiable differential distribution testing.

    Attributes:
        n_genes: Number of genes to analyse.
        temperature: Temperature controlling sigmoid smoothness in the soft
            CDF and logsumexp soft max. Lower values yield sharper
            approximations closer to the true KS statistic.
        learnable_temperature: Whether temperature is a learnable parameter.
        n_pattern_classes: Number of distributional pattern categories.
            Default 4 corresponds to (shift, scale, both, none).
    """

    n_genes: int = 2000
    temperature: float = 1.0
    learnable_temperature: bool = False
    n_pattern_classes: int = 4


class DifferentiableDifferentialDistribution(TemperatureOperator):
    """Differentiable KS-test with learned pattern classification.

    For each gene, this operator:

    1. Splits cells into two conditions based on binary condition labels.
    2. Computes a soft empirical CDF using sigmoid smoothing:
       ``soft_CDF(x, values) = mean(sigmoid((x - values) / temperature))``
    3. Computes a soft KS statistic as the smooth maximum of
       ``|CDF_A(x) - CDF_B(x)|`` over evaluation points, using logsumexp.
    4. Extracts distributional features (mean shift, variance ratio,
       zero-proportion difference) and passes them through a learned
       linear head to classify each gene into one of the pattern categories
       (shift, scale, both, none).

    Inherits from TemperatureOperator to get:

    - _temperature property for temperature-controlled smoothing
    - soft_max() for logsumexp-based smooth maximum

    Args:
        config: DifferentialDistributionConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        ```python
        config = DifferentialDistributionConfig(n_genes=2000, temperature=1.0)
        op = DifferentiableDifferentialDistribution(config, rngs=nnx.Rngs(42))
        data = {"counts": counts, "condition_labels": labels}
        result, state, meta = op.apply(data, {}, None)
        ```
    """

    # Number of features extracted per gene for pattern classification:
    # mean_shift, variance_ratio, zero_proportion_diff
    _N_PATTERN_FEATURES: int = 3

    def __init__(
        self,
        config: DifferentialDistributionConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the differentiable differential distribution operator.

        Args:
            config: Differential distribution configuration.
            rngs: Random number generators for parameter initialisation.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        self.n_genes = config.n_genes
        self.n_pattern_classes = config.n_pattern_classes

        rngs_safe = ensure_rngs(rngs)

        # Learned linear head: pattern features -> pattern logits
        self.pattern_head = nnx.Linear(
            in_features=self._N_PATTERN_FEATURES,
            out_features=self.n_pattern_classes,
            rngs=rngs_safe,
        )

    def _process_single_gene(
        self,
        gene_values: Float[Array, "n_cells"],
        condition_mask: Float[Array, "n_cells"],
    ) -> tuple[Float[Array, ""], Float[Array, "n_patterns"]]:
        """Process a single gene: compute KS stat and pattern logits.

        Args:
            gene_values: Expression values for one gene across all cells.
            condition_mask: Binary mask (0/1) indicating condition membership.

        Returns:
            Tuple of (ks_statistic, pattern_logits).
        """
        # Soft splitting: weight contributions by condition membership
        # condition_mask=0 -> condition A, condition_mask=1 -> condition B
        mask_a = 1.0 - condition_mask
        mask_b = condition_mask

        n_a = jnp.sum(mask_a) + EPSILON
        n_b = jnp.sum(mask_b) + EPSILON

        # Weighted values for each condition using soft masks
        # For the CDF computation, we use all values but weight by condition
        # To handle variable-size splits in a JIT-compatible way, we compute
        # weighted statistics instead of explicit splits.

        # For KS: evaluate soft CDF using the full set of values, but weight
        # the indicator functions by condition membership.
        eval_points = gene_values  # Evaluate at all cell values

        temp = self._temperature
        # diff: (n_cells, n_cells) -- eval_points[i] vs gene_values[j]
        diff = eval_points[:, None] - gene_values[None, :]
        sigmoid_vals = jax.nn.sigmoid(diff / temp)

        # Weighted CDF for condition A: sum(sigmoid * mask_a) / n_a
        cdf_a = jnp.sum(sigmoid_vals * mask_a[None, :], axis=1) / n_a
        # Weighted CDF for condition B: sum(sigmoid * mask_b) / n_b
        cdf_b = jnp.sum(sigmoid_vals * mask_b[None, :], axis=1) / n_b

        abs_diff = jnp.abs(cdf_a - cdf_b)
        # Use softmax-weighted sum as smooth max: sum_i(x_i * softmax(x_i/T))
        # This stays within [min(x), max(x)] unlike logsumexp which overshoots.
        temp = self._temperature
        weights = jax.nn.softmax(abs_diff / (temp + EPSILON))
        ks_stat = jnp.sum(abs_diff * weights)

        # Pattern features using weighted statistics
        mean_a = jnp.sum(gene_values * mask_a) / n_a
        mean_b = jnp.sum(gene_values * mask_b) / n_b
        mean_shift = jnp.abs(mean_a - mean_b)

        var_a = jnp.sum(mask_a * (gene_values - mean_a) ** 2) / n_a + EPSILON
        var_b = jnp.sum(mask_b * (gene_values - mean_b) ** 2) / n_b + EPSILON
        variance_ratio = jax.nn.sigmoid(jnp.log(var_a / var_b))

        # Soft zero fraction per condition
        soft_zero = jax.nn.sigmoid(-gene_values / (temp + EPSILON))
        frac_zero_a = jnp.sum(soft_zero * mask_a) / n_a
        frac_zero_b = jnp.sum(soft_zero * mask_b) / n_b
        zero_diff = jnp.abs(frac_zero_a - frac_zero_b)

        features = jnp.stack([mean_shift, variance_ratio, zero_diff])
        pattern_logits = self.pattern_head(features)

        return ks_stat, pattern_logits

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply differentiable differential distribution testing.

        For each gene, computes a soft KS statistic and classifies the
        distributional difference pattern using a learned linear head.

        Args:
            data: Dictionary containing:
                - "counts": Gene expression matrix (n_cells, n_genes)
                - "condition_labels": Binary condition labels (n_cells,)
            state: Element state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: Not used.
            stats: Not used.

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - "counts": Original expression counts
                    - "condition_labels": Original condition labels
                    - "ks_statistics": Soft KS statistic per gene (n_genes,)
                    - "pattern_logits": Pattern class logits (n_genes, n_patterns)
                    - "pattern_labels": Predicted pattern labels (n_genes,)
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        counts = data["counts"]
        condition_labels = data["condition_labels"]

        # Process all genes in parallel using vmap over gene dimension (axis 1)
        def process_gene(gene_col: Float[Array, "n_cells"]) -> tuple[
            Float[Array, ""], Float[Array, "n_patterns"]
        ]:
            return self._process_single_gene(gene_col, condition_labels)

        # vmap over columns (genes) of counts: (n_cells, n_genes) -> per-gene
        ks_statistics, pattern_logits = jax.vmap(
            process_gene, in_axes=1
        )(counts)

        pattern_labels = jnp.argmax(pattern_logits, axis=-1)

        transformed_data = {
            **data,
            "ks_statistics": ks_statistics,
            "pattern_logits": pattern_logits,
            "pattern_labels": pattern_labels,
        }

        return transformed_data, state, metadata
