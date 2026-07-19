"""Differentiable soft selection of the number of PCA components (learnable rank).

In a genuine-PCA reduction the eigenvector *directions* are fixed, but the number of
components retained -- equivalently the cumulative explained-variance coverage -- is
itself a parameter. This operator makes that choice differentiable: it keeps the PCA
directions frozen and learns a single **coverage threshold** that softly gates the
eigenvalue-ranked components, so the effective dimensionality is trained end-to-end
against the downstream loss ("learn-the-dimension", as opposed to the learnable
projection's "learn-the-directions").

The soft gate is a temperature-controlled sigmoid over the cumulative variance ratio --
a differentiable relaxation of a hard rank truncation, in the lineage of soft singular-
value thresholding (soft-impute; Mazumder, Hastie, Tibshirani 2010) and of automatic
PCA dimensionality selection (Minka, NeurIPS 2000). As the temperature goes to zero the
gate recovers the exact top-k' truncation implied by the coverage target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jax.typing import ArrayLike


@dataclass(frozen=True)
class SoftComponentSelectionConfig(OperatorConfig):
    """Configuration for :class:`SoftComponentSelection`.

    Attributes:
        n_components: Number of input principal components to gate.
        init_coverage: Initial cumulative-variance coverage target in ``(0, 1)``.
        temperature: Sigmoid sharpness of the soft rank gate (smaller = harder).
    """

    n_components: int = 50
    init_coverage: float = 0.9
    temperature: float = 0.05

    def __post_init__(self) -> None:
        """Validate the configuration, failing fast on out-of-range values.

        Raises:
            ValueError: If ``n_components`` is not positive, ``init_coverage`` is not
                in ``(0, 1)``, or ``temperature`` is not positive.
        """
        super().__post_init__()
        if self.n_components <= 0:
            raise ValueError(f"n_components must be strictly positive, got {self.n_components}")
        if not 0.0 < self.init_coverage < 1.0:
            raise ValueError(f"init_coverage must be in (0, 1), got {self.init_coverage}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be strictly positive, got {self.temperature}")


class SoftComponentSelection(OperatorModule):
    """Softly gate PCA components by a learnable cumulative-variance coverage target.

    The eigenvalues fix a frozen cumulative-variance profile over the components; the
    single learnable parameter is the coverage threshold (stored as a logit). The keep-
    gate ``sigmoid((coverage - cumulative_variance) / temperature)`` is monotone
    decreasing over the ranked components, so raising the coverage keeps more of them.
    """

    def __init__(
        self,
        config: SoftComponentSelectionConfig,
        *,
        eigenvalues: ArrayLike,
        rngs: nnx.Rngs,
        name: str | None = None,
    ) -> None:
        """Initialize the soft component selector.

        Args:
            config: Selection configuration.
            eigenvalues: ``(n_components,)`` PCA explained variances (descending); their
                normalized cumulative sum defines the frozen coverage profile.
            rngs: RNG state (unused; the operator is deterministic).
            name: Optional module name.

        Raises:
            ValueError: If ``eigenvalues`` does not have length ``n_components``.
        """
        super().__init__(config, rngs=rngs, name=name)
        spectrum = jnp.asarray(eigenvalues, dtype=jnp.float32)
        if spectrum.shape != (config.n_components,):
            raise ValueError(
                f"eigenvalues must have shape ({config.n_components},), got {spectrum.shape}"
            )
        cumulative = jnp.cumsum(spectrum) / jnp.sum(spectrum)
        self.cumulative_variance = nnx.Variable(cumulative)
        self.temperature = float(config.temperature)
        logit = jnp.log(config.init_coverage / (1.0 - config.init_coverage))
        self.raw_coverage = nnx.Param(jnp.asarray(logit, dtype=jnp.float32))

    def _keep_gate(self) -> jnp.ndarray:
        """Return the ``(n_components,)`` soft keep-gate at the current coverage."""
        coverage = jax.nn.sigmoid(self.raw_coverage[...])
        return jax.nn.sigmoid((coverage - self.cumulative_variance[...]) / self.temperature)

    def effective_dimension(self) -> jax.Array:
        """Return the soft effective number of retained components (the gate sum)."""
        return jnp.sum(self._keep_gate())

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict | None,
        random_params: dict | None = None,
        stats: dict | None = None,
    ) -> tuple[dict, dict, dict | None]:
        """Gate ``data["projection"]`` by the learnable soft rank.

        Args:
            data: Dictionary containing ``"projection"`` ``(n_cells, n_components)``.
            state: Operator state dictionary.
            metadata: Optional metadata dictionary.
            random_params: Optional random parameters (unused).
            stats: Optional statistics dictionary (unused).

        Returns:
            Tuple of ``(output_data, state, metadata)`` with ``"projection"`` scaled by
            the soft keep-gate.
        """
        del random_params, stats
        gated = jnp.asarray(data["projection"]) * self._keep_gate()[None, :]
        return {**data, "projection": gated}, state, metadata
