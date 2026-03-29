"""Uncertainty quantification wrappers for DiffBio operators.

Provides ensemble and conformal prediction wrappers that add
``uncertainty``, ``confidence_interval_lower``, and
``confidence_interval_upper`` keys to any operator's output dict.

Uses ``nnx.vmap`` with ``nnx.StateAxes`` to vectorize multiple forward
passes across different RNG seeds (ensemble) or dropout samples
(conformal), computing mean predictions and uncertainty estimates
without Python for-loops.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import PyTree

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnsembleUQConfig(OperatorConfig):
    """Configuration for ensemble-based uncertainty quantification.

    Attributes:
        n_members: Number of ensemble members (forward passes with
            different random seeds).
        confidence_level: Confidence level for intervals (0 to 1).
    """

    n_members: int = 5
    confidence_level: float = 0.95


@dataclass(frozen=True)
class ConformalUQConfig(OperatorConfig):
    """Configuration for conformal prediction-based UQ.

    Attributes:
        alpha: Significance level (1 - confidence). Smaller alpha
            gives wider intervals.
        num_samples: Number of Monte Carlo samples for interval estimation.
    """

    alpha: float = 0.1
    num_samples: int = 20


def _find_primary_output_key(result: dict[str, Any], data: dict[str, Any]) -> str | None:
    """Find the main output key added by the operator.

    Returns the first key in result that is not in data and holds a JAX array.
    """
    for key in result:
        if key not in data and isinstance(result[key], jax.Array):
            return key
    return None


def _run_n_samples(
    operator: OperatorModule,
    data: dict[str, Any],
    state: dict[str, Any],
    metadata: dict[str, Any] | None,
    n_samples: int,
    primary_key: str,
) -> jax.Array:
    """Run operator n_samples times and stack the primary output.

    Uses jax.lax.scan to avoid Python for-loops. The operator is
    deterministic per call (same params, same data), so all samples
    are identical for non-stochastic operators. For stochastic operators
    (dropout, noise), each call samples fresh randomness via NNX RNG state.

    Args:
        operator: The base operator to run.
        data: Input data dict.
        state: Element state.
        metadata: Element metadata.
        n_samples: Number of forward passes.
        primary_key: Key to stack from the output dict.

    Returns:
        Stacked array of shape (n_samples, *output_shape).
    """

    def _step(_carry: None, _xs: None) -> tuple[None, jax.Array]:
        result, _, _ = operator.apply(data, state, metadata)
        return None, result[primary_key]

    _, stacked = jax.lax.scan(_step, None, None, length=n_samples)
    return stacked


def _compute_uq_stats(
    stacked: jax.Array,
    confidence_level: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute mean, std, and confidence interval from stacked samples.

    Args:
        stacked: Array of shape (n_samples, *output_shape).
        confidence_level: Confidence level for quantile intervals.

    Returns:
        Tuple of (mean, std, lower, upper).
    """
    mean = jnp.mean(stacked, axis=0)
    std = jnp.std(stacked, axis=0)
    alpha = 1.0 - confidence_level
    lower = jnp.quantile(stacked, alpha / 2, axis=0)
    upper = jnp.quantile(stacked, 1 - alpha / 2, axis=0)
    return mean, std, lower, upper


class EnsembleUQOperator(OperatorModule):
    """Ensemble-based uncertainty quantification wrapper.

    Runs the base operator multiple times and aggregates outputs to produce
    mean predictions with uncertainty estimates (standard deviation) and
    confidence intervals (quantile-based). Uses ``jax.lax.scan`` to avoid
    Python for-loops.

    Output adds:
        - ``uncertainty``: Standard deviation across ensemble members.
        - ``confidence_interval_lower``: Lower bound of confidence interval.
        - ``confidence_interval_upper``: Upper bound of confidence interval.

    Args:
        config: EnsembleUQConfig with ensemble parameters.
        base_operator: The operator to wrap.
        rngs: Random number generators.
        name: Optional operator name.
    """

    def __init__(
        self,
        config: EnsembleUQConfig,
        *,
        base_operator: OperatorModule,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize ensemble UQ operator."""
        super().__init__(config, rngs=rngs, name=name)
        self.config: EnsembleUQConfig = config
        self.base_operator = base_operator

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,  # noqa: ARG002
        stats: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Run ensemble forward passes and aggregate with uncertainty.

        Args:
            data: Input data dict for the base operator.
            state: Element state (passed through).
            metadata: Element metadata (passed through).
            random_params: Unused.
            stats: Unused.

        Returns:
            Tuple of (output_with_uncertainty, state, metadata).
        """
        # Get a single result to find output keys and structure
        base_result, _, _ = self.base_operator.apply(data, state, metadata)
        primary_key = _find_primary_output_key(base_result, data)
        if primary_key is None:
            return base_result, state, metadata

        stacked = _run_n_samples(
            self.base_operator,
            data,
            state,
            metadata,
            self.config.n_members,
            primary_key,
        )
        mean, std, lower, upper = _compute_uq_stats(
            stacked,
            self.config.confidence_level,
        )

        result = dict(base_result)
        result[primary_key] = mean
        result["uncertainty"] = std
        result["confidence_interval_lower"] = lower
        result["confidence_interval_upper"] = upper
        return result, state, metadata


class ConformalUQOperator(OperatorModule):
    """Conformal prediction-based uncertainty quantification wrapper.

    Uses Monte Carlo sampling to estimate empirical prediction intervals.
    Runs the base operator multiple times via ``jax.lax.scan`` and computes
    quantile-based intervals at the specified confidence level (1 - alpha).

    Output adds:
        - ``uncertainty``: Standard deviation across samples.
        - ``confidence_interval_lower``: Lower quantile bound.
        - ``confidence_interval_upper``: Upper quantile bound.

    Args:
        config: ConformalUQConfig with sampling parameters.
        base_operator: The operator to wrap.
        rngs: Random number generators.
        name: Optional operator name.
    """

    def __init__(
        self,
        config: ConformalUQConfig,
        *,
        base_operator: OperatorModule,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize conformal UQ operator."""
        super().__init__(config, rngs=rngs, name=name)
        self.config: ConformalUQConfig = config
        self.base_operator = base_operator

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,  # noqa: ARG002
        stats: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Run conformal prediction sampling.

        Args:
            data: Input data dict for the base operator.
            state: Element state (passed through).
            metadata: Element metadata (passed through).
            random_params: Unused.
            stats: Unused.

        Returns:
            Tuple of (output_with_intervals, state, metadata).
        """
        base_result, _, _ = self.base_operator.apply(data, state, metadata)
        primary_key = _find_primary_output_key(base_result, data)
        if primary_key is None:
            return base_result, state, metadata

        stacked = _run_n_samples(
            self.base_operator,
            data,
            state,
            metadata,
            self.config.num_samples,
            primary_key,
        )
        confidence_level = 1.0 - self.config.alpha
        mean, std, lower, upper = _compute_uq_stats(stacked, confidence_level)

        result = dict(base_result)
        result[primary_key] = mean
        result["uncertainty"] = std
        result["confidence_interval_lower"] = lower
        result["confidence_interval_upper"] = upper
        return result, state, metadata
