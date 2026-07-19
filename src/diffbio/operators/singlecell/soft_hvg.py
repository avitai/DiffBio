"""Differentiable soft highly-variable-gene (HVG) selection operator.

Drop-in for scanpy ``highly_variable_genes``: it ranks genes by normalized
dispersion (variance / mean, the Seurat measure), modulates the ranking with a
learnable per-gene weight, and emits a continuous soft selection mask so a
downstream loss can tune *which* genes survive instead of freezing a global
top-k up front.

The default configuration is a straight-through estimator: the forward pass
produces the exact hard top-k boolean mask (matching the frozen scanpy/baseline
selection), while the backward pass flows gradients to the gene weights through
the soft relaxation. A fully smooth mode is available for end-to-end smoothness.

Soft top-k and the straight-through estimator are reused from
:mod:`diffbio.core.soft_ops`; MarkerMap-style Gumbel selection is reference-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jax.typing import ArrayLike

from diffbio.core import soft_ops

_DEFAULT_SOFTNESS = 0.1


class HVGFlavor(StrEnum):
    """Highly-variable-gene selection flavor, mirroring scanpy's ``flavor`` argument.

    Only :attr:`SEURAT` is currently implemented, and as the unbinned
    normalized-dispersion ranking of the frozen baseline -- an approximation of
    scanpy's ``seurat`` flavor, which additionally bins by mean and z-scores
    dispersion within each bin. The other members are accepted for signature
    parity and rejected at construction so migration code fails fast rather than
    silently diverging.
    """

    SEURAT = "seurat"
    CELL_RANGER = "cell_ranger"
    SEURAT_V3 = "seurat_v3"


def gene_dispersion(features: ArrayLike) -> jnp.ndarray:
    """Normalized dispersion (variance / mean) per gene across cells.

    This is the unbinned normalized-dispersion measure used by the frozen baseline
    selector, an approximation of scanpy's ``seurat`` flavor (which additionally
    bins genes by mean and z-scores dispersion within each bin). Genes with zero
    mean use a unit denominator, so a dropped gene yields a finite zero dispersion
    rather than ``0 / 0``.

    Args:
        features: A ``(n_cells, n_genes)`` expression matrix.

    Returns:
        Per-gene normalized dispersion ``(n_genes,)``.
    """
    features = jnp.asarray(features)
    gene_mean = jnp.mean(features, axis=0)
    gene_variance = jnp.var(features, axis=0)
    safe_mean = jnp.where(gene_mean == 0.0, 1.0, gene_mean)
    return gene_variance / safe_mean


def _weighted_dispersion(
    features: ArrayLike, gene_weights: ArrayLike | None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the ``(ranking_score, dispersion)`` for highly-variable-gene selection.

    The score is the normalized dispersion scaled by a strictly-positive learnable
    per-gene weight (``softplus(gene_weights)``); a uniform weight leaves the
    ranking unchanged. Returns the raw dispersion alongside for reporting.
    """
    dispersion = gene_dispersion(features)
    if gene_weights is None:
        return dispersion, dispersion
    score = dispersion * jax.nn.softplus(jnp.asarray(gene_weights))
    return score, dispersion


def soft_hvg_mask(
    features: ArrayLike,
    n_top_genes: int,
    *,
    gene_weights: ArrayLike | None = None,
    softness: float = _DEFAULT_SOFTNESS,
    hard: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute a differentiable highly-variable-gene selection mask.

    Genes are ranked by normalized dispersion scaled by a strictly-positive,
    learnable per-gene weight (``softplus(gene_weights)``), so a downstream loss
    can promote or suppress individual genes. A uniform weight leaves the ranking
    unchanged, so the default reproduces the pure-dispersion top-k. Selection uses
    :func:`diffbio.core.soft_ops.top_k_mask`, whose mask is bounded in ``[0, 1]``.

    Args:
        features: A ``(n_cells, n_genes)`` expression matrix.
        n_top_genes: Number of genes to select; clamped to ``n_genes``.
        gene_weights: Optional per-gene ``(n_genes,)`` weights; ``None`` selects
            purely by dispersion.
        softness: Sharpness of the soft top-k (``> 0``); smaller is sharper.
        hard: If ``True``, use the straight-through estimator (exact 0/1 mask
            forward, soft gradient backward); if ``False``, return a smooth mask.

    Returns:
        Tuple ``(mask, dispersion)`` where ``mask`` is the ``(n_genes,)`` soft
        selection mask in ``[0, 1]`` and ``dispersion`` is the ranking score.
    """
    score, dispersion = _weighted_dispersion(features, gene_weights)
    if hard:
        mask = soft_ops.top_k_mask_st(score, n_top_genes, softness=softness)
    else:
        mask = soft_ops.top_k_mask(score, n_top_genes, softness=softness, mode="smooth")
    return mask, dispersion


def highly_variable_genes(
    features: ArrayLike,
    n_top_genes: int,
    *,
    gene_weights: ArrayLike | None = None,
    softness: float = _DEFAULT_SOFTNESS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Scanpy-style HVG helper returning both soft weights and the hard mask.

    Mirrors the intent of scanpy ``highly_variable_genes`` at the array level; the
    AnnData-signature wrapper lives in the migration surface. The soft weights
    carry gradients for joint optimization while the boolean mask reproduces the
    frozen top-k selection.

    Args:
        features: A ``(n_cells, n_genes)`` expression matrix.
        n_top_genes: Number of highly-variable genes to select.
        gene_weights: Optional per-gene ``(n_genes,)`` weights.
        softness: Sharpness of the soft top-k relaxation.

    Returns:
        Tuple ``(soft_weights, hard_mask)`` of a smooth ``(n_genes,)`` mask and a
        boolean ``(n_genes,)`` selection mask.
    """
    score, _ = _weighted_dispersion(features, gene_weights)
    soft_weights = soft_ops.top_k_mask(score, n_top_genes, softness=softness, mode="smooth")
    hard_mask = soft_ops.top_k_mask(score, n_top_genes, mode="hard")
    return soft_weights, hard_mask > 0.5


@dataclass(frozen=True)
class SoftHVGConfig(OperatorConfig):
    """Configuration for :class:`SoftHVG`.

    Attributes:
        n_genes: Number of input genes; sizes the learnable weight vector.
        n_top_genes: Number of highly-variable genes to select.
        flavor: Selection flavor; only :attr:`HVGFlavor.SEURAT` is implemented.
        softness: Sharpness of the soft top-k relaxation (``> 0``).
        hard: If ``True``, emit the straight-through hard mask (parity forward,
            soft gradient); if ``False``, emit a fully smooth mask.
    """

    n_genes: int = 2000
    n_top_genes: int = 2000
    flavor: HVGFlavor = HVGFlavor.SEURAT
    softness: float = _DEFAULT_SOFTNESS
    hard: bool = True

    def __post_init__(self) -> None:
        """Validate the configuration at construction, failing fast on bad values.

        Raises:
            ValueError: If ``n_genes`` or ``n_top_genes`` is not strictly positive.
            NotImplementedError: If ``flavor`` is not :attr:`HVGFlavor.SEURAT`.
        """
        super().__post_init__()
        if self.n_genes <= 0:
            raise ValueError(f"n_genes must be strictly positive, got {self.n_genes}")
        if self.n_top_genes <= 0:
            raise ValueError(f"n_top_genes must be strictly positive, got {self.n_top_genes}")
        if self.flavor is not HVGFlavor.SEURAT:
            raise NotImplementedError(
                f"Only the 'seurat' HVG flavor is implemented, got flavor={self.flavor.value!r}"
            )


class SoftHVG(OperatorModule):
    """Differentiable highly-variable-gene selection with learnable per-gene weights.

    Like :class:`DifferentiablePCA`, this is a whole-matrix (cross-cell) operator:
    one element is the full ``(n_cells, n_genes)`` matrix, and ``apply`` reduces
    over the cell axis to form per-gene dispersion. Only ``apply`` is implemented,
    per the ``OperatorModule`` contract; the framework's ``apply_batch`` maps it
    over a batch of such matrices (``apply`` is vmap-safe).
    """

    def __init__(
        self,
        config: SoftHVGConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the operator with a learnable per-gene weight vector.

        Args:
            config: HVG selection configuration.
            rngs: Optional RNG state (unused; kept for interface compatibility).
            name: Optional module name.
        """
        super().__init__(config, rngs=rngs, name=name)
        # Zero-initialized weights give a uniform multiplier, so the frozen
        # operator ranks purely by dispersion (scanpy/baseline parity).
        self.gene_weights = nnx.Param(jnp.zeros(config.n_genes, dtype=jnp.float32))

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict | None,
        random_params: dict | None = None,
        stats: dict | None = None,
    ) -> tuple[dict, dict, dict | None]:
        """Gate ``data["features"]`` by the highly-variable-gene mask.

        Args:
            data: Dictionary containing ``"features"`` ``(n_cells, n_genes)``.
            state: Operator state dictionary.
            metadata: Optional metadata dictionary.
            random_params: Optional random parameters (unused).
            stats: Optional statistics dictionary (unused).

        Returns:
            Tuple of ``(output_data, state, metadata)`` where ``output_data`` gates
            ``"features"`` by the mask and adds ``"hvg_weights"`` (the soft mask)
            and ``"hvg_dispersion"`` (the ranking score).
        """
        del random_params, stats
        config: SoftHVGConfig = self.config
        mask, dispersion = soft_hvg_mask(
            data["features"],
            config.n_top_genes,
            gene_weights=self.gene_weights[...],
            softness=config.softness,
            hard=config.hard,
        )
        gated = data["features"] * mask[None, :]
        output_data = {
            **data,
            "features": gated,
            "hvg_weights": mask,
            "hvg_dispersion": dispersion,
        }
        return output_data, state, metadata
