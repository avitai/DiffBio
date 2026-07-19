"""Joint-optimizable single-cell preprocessing-to-annotation pipeline.

Composes the standard single-cell transform stack -- learnable normalization,
soft highly-variable-gene selection, per-gene scaling, differentiable PCA -- and
a linear annotation probe into one sequential pipeline. Every stage is
differentiable, so a downstream classification loss can tune the preprocessing
end-to-end rather than freezing it up front.

With its default configuration the stack pins the standard settings
(logCP10k + log1p, hard HVG top-k, z-score + clip, PCA) and reproduces the frozen
scanpy baseline's annotation features -- the Gate 1 parity floor. Composition is
delegated to datarax ``CompositeOperatorModule`` with the ``SEQUENTIAL`` strategy;
:class:`~diffbio.pipelines.adapters.RenameField` adapters bridge the data-dict key
names between stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from datarax.operators import (
    CompositeOperatorConfig,
    CompositeOperatorModule,
    CompositionStrategy,
)
from flax import nnx

from diffbio.operators.foundation_models import EmbeddingProbeConfig, LinearEmbeddingProbe
from diffbio.operators.normalization.differentiable_pca import (
    DifferentiablePCA,
    DifferentiablePCAConfig,
)
from diffbio.operators.normalization.learnable_normalization import (
    LearnableNormalization,
    LearnableNormalizationConfig,
)
from diffbio.operators.normalization.scaling import DifferentiableScaler, ScalerConfig
from diffbio.operators.singlecell.soft_hvg import SoftHVG, SoftHVGConfig
from diffbio.pipelines.adapters import RenameField, RenameFieldConfig

_OperatorT = TypeVar("_OperatorT", bound=OperatorModule)

_DEFAULT_TARGET_SUM = 1.0e4
_DEFAULT_N_TOP_GENES = 2000
_DEFAULT_N_COMPONENTS = 50
_DEFAULT_SCALE_CLIP = 10.0
_DEFAULT_HVG_SOFTNESS = 0.1


@dataclass(frozen=True)
class JointPreprocessingPipelineConfig(OperatorConfig):
    """Configuration for :class:`JointPreprocessingPipeline`.

    The defaults pin the standard (frozen) preprocessing stack that reproduces the
    scanpy baseline; ``hvg_hard=False`` switches highly-variable-gene selection to
    the fully smooth mask for joint optimization.

    Attributes:
        n_genes: Number of input genes; sizes the HVG gene-weight vector.
        n_classes: Number of annotation classes for the probe.
        n_top_genes: Number of highly-variable genes to keep.
        n_components: Number of principal components (and probe input dim). Must not
            exceed ``n_genes`` (validated) nor the number of cells at run time --
            PCA yields ``min(n_components, n_genes, n_cells)`` components, so fewer
            cells than ``n_components`` would shape-mismatch the probe.
        target_sum: Per-cell target library size for normalization.
        scale_clip: Symmetric clip bound after per-gene standardization.
        hidden_dim: Probe hidden width; ``None`` gives a linear (logistic) head.
        hvg_softness: Sharpness of the soft HVG mask.
        hvg_hard: If ``True`` (frozen default), emit the exact top-k HVG mask.
    """

    n_genes: int = _DEFAULT_N_TOP_GENES
    n_classes: int = 2
    n_top_genes: int = _DEFAULT_N_TOP_GENES
    n_components: int = _DEFAULT_N_COMPONENTS
    target_sum: float = _DEFAULT_TARGET_SUM
    scale_clip: float = _DEFAULT_SCALE_CLIP
    hidden_dim: int | None = None
    hvg_softness: float = _DEFAULT_HVG_SOFTNESS
    hvg_hard: bool = True

    def __post_init__(self) -> None:
        """Validate the configuration at construction, failing fast on bad values.

        Raises:
            ValueError: If any size field is not strictly positive, or if
                ``n_components`` exceeds ``n_genes``.
        """
        super().__post_init__()
        for field_name in ("n_genes", "n_classes", "n_top_genes", "n_components"):
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be strictly positive, got {value}")
        if self.n_components > self.n_genes:
            raise ValueError(
                f"n_components ({self.n_components}) cannot exceed n_genes ({self.n_genes})"
            )


class JointPreprocessingPipeline(OperatorModule):
    """Sequential preprocessing-to-annotation pipeline over the single-cell stack.

    One element is the full ``(n_cells, n_genes)`` count matrix. ``apply`` threads
    it through ``LearnableNormalization -> SoftHVG -> DifferentiableScaler ->
    DifferentiablePCA -> LinearEmbeddingProbe`` (with key-rename adapters between
    stages) via a datarax sequential ``CompositeOperatorModule``. The child
    operators' learnable parameters are preserved, so ``nnx.split``/``grad`` reach
    them for end-to-end joint optimization.
    """

    def __init__(
        self,
        config: JointPreprocessingPipelineConfig,
        *,
        rngs: nnx.Rngs,
        name: str | None = None,
    ) -> None:
        """Build the composed pipeline from ``config``.

        Args:
            config: Pipeline configuration.
            rngs: RNG state used to initialize the child operators.
            name: Optional module name.
        """
        super().__init__(config, rngs=rngs, name=name)
        operators: list[OperatorModule] = [
            LearnableNormalization(
                LearnableNormalizationConfig(target_sum=config.target_sum), rngs=rngs
            ),
            RenameField(RenameFieldConfig(source="normalized", target="features"), rngs=rngs),
            SoftHVG(
                SoftHVGConfig(
                    n_genes=config.n_genes,
                    n_top_genes=config.n_top_genes,
                    softness=config.hvg_softness,
                    hard=config.hvg_hard,
                ),
                rngs=rngs,
            ),
            DifferentiableScaler(ScalerConfig(clip=config.scale_clip), rngs=rngs),
            DifferentiablePCA(DifferentiablePCAConfig(n_components=config.n_components), rngs=rngs),
            RenameField(RenameFieldConfig(source="pca", target="embeddings"), rngs=rngs),
            LinearEmbeddingProbe(
                EmbeddingProbeConfig(
                    input_dim=config.n_components,
                    n_classes=config.n_classes,
                    hidden_dim=config.hidden_dim,
                ),
                rngs=rngs,
            ),
        ]
        self.composite = CompositeOperatorModule(
            CompositeOperatorConfig(
                strategy=CompositionStrategy.SEQUENTIAL,
                operators=operators,
            ),
            rngs=rngs,
        )

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict | None,
        random_params: dict | None = None,
        stats: dict | None = None,
    ) -> tuple[dict, dict, dict | None]:
        """Run the full preprocessing-to-annotation pipeline on ``data``.

        Args:
            data: Dictionary containing ``"counts"`` ``(n_cells, n_genes)``.
            state: Operator state dictionary.
            metadata: Optional metadata dictionary.
            random_params: Optional random parameters (forwarded to the composite).
            stats: Optional statistics dictionary (forwarded to the composite).

        Returns:
            Tuple of ``(output_data, state, metadata)`` where ``output_data`` adds
            the PCA ``"embeddings"`` and the probe's ``"logits"``,
            ``"probabilities"``, and ``"predicted_labels"``.
        """
        return self.composite.apply(data, state, metadata, random_params, stats)

    def _child(self, operator_type: type[_OperatorT]) -> _OperatorT:
        """Return the first composed child operator of ``operator_type``.

        Args:
            operator_type: The ``OperatorModule`` subclass to find.

        Returns:
            The matching child operator.

        Raises:
            ValueError: If no child of that type is composed.
        """
        for operator in self.composite.operators:  # type: ignore[union-attr]
            if isinstance(operator, operator_type):
                return operator
        msg = f"pipeline has no composed {operator_type.__name__} operator"
        raise ValueError(msg)

    @property
    def normalization(self) -> LearnableNormalization:
        """The learnable count-normalization stage."""
        return self._child(LearnableNormalization)

    @property
    def soft_hvg(self) -> SoftHVG:
        """The soft highly-variable-gene selection stage."""
        return self._child(SoftHVG)
