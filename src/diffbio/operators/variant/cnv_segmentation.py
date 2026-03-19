"""Differentiable CNV Segmentation operator.

This module provides differentiable implementations of copy number
variation segmentation:

- ``DifferentiableCNVSegmentation``: Attention-based soft changepoint detection.
- ``EnhancedCNVSegmentation``: Multi-signal fusion (log-ratio + BAF + SNP
  density), pyramidal smoothing (infercnvpy-style), STDDEV-based dynamic
  thresholding, and HMM state-to-copy-number mapping.

Key techniques:
- Attention mechanism identifies segment boundaries softly.
- Pyramidal (triangular) convolution for robust spatial smoothing.
- Dynamic threshold = scale * std(smoothed_signal) filters noise.
- Learnable linear fusion of heterogeneous signals.
- Soft copy-number state posteriors via learned emission model.

Applications: CNV analysis, coverage depth segmentation, breakpoint detection.
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


@dataclass
class CNVSegmentationConfig(OperatorConfig):
    """Configuration for DifferentiableCNVSegmentation.

    Attributes:
        max_segments: Maximum number of segments to detect.
        hidden_dim: Hidden dimension for attention layers.
        attention_heads: Number of attention heads.
        temperature: Temperature for softmax operations.
    """

    max_segments: int = 100
    hidden_dim: int = 64
    attention_heads: int = 4
    temperature: float = 1.0


class DifferentiableCNVSegmentation(TemperatureOperator):
    """Soft CNV segmentation using attention-based changepoint detection.

    This operator identifies segment boundaries in coverage data using
    attention mechanisms, replacing hard Circular Binary Segmentation
    with differentiable soft assignments.

    Algorithm:
    1. Project coverage signal into hidden space
    2. Use self-attention to identify changepoint positions
    3. Compute soft segment assignments via attention
    4. Compute segment means as weighted averages

    Inherits from TemperatureOperator to get:

    - _temperature property for temperature-controlled smoothing
    - soft_max() for logsumexp-based smooth maximum
    - soft_argmax() for soft position selection

    Args:
        config: CNVSegmentationConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        ```python
        config = CNVSegmentationConfig(max_segments=50)
        segmenter = DifferentiableCNVSegmentation(config, rngs=nnx.Rngs(42))
        data = {"coverage": coverage_signal}  # (n_positions,)
        result, state, meta = segmenter.apply(data, {}, None)
        ```
    """

    def __init__(
        self,
        config: CNVSegmentationConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the CNV segmentation operator.

        Args:
            config: Segmentation configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.max_segments = config.max_segments
        self.hidden_dim = config.hidden_dim
        self.attention_heads = config.attention_heads
        # Temperature is now managed by TemperatureOperator via self._temperature

        # Input projection: coverage value -> hidden
        self.input_proj = nnx.Linear(1, config.hidden_dim, rngs=rngs)

        # Positional encoding projection
        self.pos_proj = nnx.Linear(1, config.hidden_dim, rngs=rngs)

        # Attention projections (for boundary detection)
        self.query_proj = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
        self.key_proj = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
        self.value_proj = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)

        # Boundary detection head
        self.boundary_head = nnx.Linear(config.hidden_dim, 1, rngs=rngs)

        # Segment centroids (learnable)
        key = rngs.params()
        init_centroids = jax.random.normal(key, (config.max_segments, config.hidden_dim)) * 0.1
        self.segment_centroids = nnx.Param(init_centroids)

    def compute_embeddings(
        self,
        coverage: Float[Array, "n_positions"],
    ) -> Float[Array, "n_positions hidden_dim"]:
        """Compute position embeddings from coverage signal.

        Args:
            coverage: Coverage values at each position.

        Returns:
            Embedded representation of each position.
        """
        n_positions = coverage.shape[0]

        # Project coverage values
        coverage_emb = self.input_proj(coverage[:, None])  # (n_positions, hidden_dim)

        # Add positional encoding
        positions = jnp.arange(n_positions, dtype=jnp.float32) / n_positions
        pos_emb = self.pos_proj(positions[:, None])  # (n_positions, hidden_dim)

        embeddings = coverage_emb + pos_emb

        return embeddings

    def compute_boundary_probs(
        self,
        embeddings: Float[Array, "n_positions hidden_dim"],
    ) -> Float[Array, "n_positions"]:
        """Compute soft boundary probabilities using self-attention.

        Args:
            embeddings: Position embeddings.

        Returns:
            Probability of being a segment boundary at each position.
        """
        n_positions = embeddings.shape[0]

        # Self-attention to detect changepoints
        Q = self.query_proj(embeddings)  # (n_positions, hidden_dim)
        K = self.key_proj(embeddings)  # (n_positions, hidden_dim)
        V = self.value_proj(embeddings)  # (n_positions, hidden_dim)

        # Compute attention scores
        head_dim = self.hidden_dim // self.attention_heads
        scale = jnp.sqrt(head_dim).astype(embeddings.dtype)

        # Reshape for multi-head attention
        Q = Q.reshape(n_positions, self.attention_heads, head_dim)
        K = K.reshape(n_positions, self.attention_heads, head_dim)
        V = V.reshape(n_positions, self.attention_heads, head_dim)

        # Attention: (n_positions, n_heads, n_positions)
        attn_scores = jnp.einsum("nhd,mhd->nhm", Q, K) / scale

        # Soft attention weights
        attn_weights = jax.nn.softmax(attn_scores / self._temperature, axis=-1)

        # Attend to values
        attended = jnp.einsum("nhm,mhd->nhd", attn_weights, V)
        attended = attended.reshape(n_positions, self.hidden_dim)

        # Compute boundary probability from attended features
        # Look at how much attention pattern changes
        boundary_logits = self.boundary_head(attended).squeeze(-1)  # (n_positions,)

        # Boundaries at positions where signal changes
        boundary_probs = jax.nn.sigmoid(boundary_logits)

        return boundary_probs

    def compute_segment_assignments(
        self,
        embeddings: Float[Array, "n_positions hidden_dim"],
    ) -> Float[Array, "n_positions max_segments"]:
        """Compute soft segment assignments via attention to centroids.

        Args:
            embeddings: Position embeddings.

        Returns:
            Soft assignment probability to each segment.
        """
        centroids = self.segment_centroids[...]  # (max_segments, hidden_dim)

        # Compute similarity to segment centroids
        # (n_positions, hidden_dim) x (hidden_dim, max_segments) -> (n_positions, max_segments)
        similarities = jnp.einsum("nh,sh->ns", embeddings, centroids)

        # Soft assignments via softmax
        assignments = jax.nn.softmax(similarities / self._temperature, axis=-1)

        return assignments

    def compute_segment_means(
        self,
        coverage: Float[Array, "n_positions"],
        assignments: Float[Array, "n_positions max_segments"],
    ) -> Float[Array, "max_segments"]:
        """Compute segment mean values.

        Args:
            coverage: Coverage values.
            assignments: Soft segment assignments.

        Returns:
            Mean coverage for each segment.
        """
        # Weighted sum of coverage / sum of weights
        weighted_sum = jnp.einsum("n,ns->s", coverage, assignments)  # (max_segments,)
        weight_sum = jnp.sum(assignments, axis=0) + 1e-10  # (max_segments,)

        segment_means = weighted_sum / weight_sum

        return segment_means

    def compute_smoothed_coverage(
        self,
        coverage: Float[Array, "n_positions"],
        assignments: Float[Array, "n_positions max_segments"],
        segment_means: Float[Array, "max_segments"],
    ) -> Float[Array, "n_positions"]:
        """Compute smoothed coverage from segment assignments.

        Args:
            coverage: Original coverage values.
            assignments: Soft segment assignments.
            segment_means: Mean value for each segment.

        Returns:
            Smoothed coverage (soft segmentation result).
        """
        # Weighted combination of segment means
        smoothed = jnp.einsum("ns,s->n", assignments, segment_means)

        return smoothed

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply CNV segmentation to coverage data.

        Args:
            data: Dictionary containing:
                - "coverage": Coverage signal (n_positions,)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - "coverage": Original coverage
                    - "boundary_probs": Soft boundary probabilities
                    - "segment_assignments": Soft segment memberships
                    - "segment_means": Mean value per segment
                    - "smoothed_coverage": Segmented/smoothed signal
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        coverage = data["coverage"]

        # Compute embeddings
        embeddings = self.compute_embeddings(coverage)

        # Detect boundaries
        boundary_probs = self.compute_boundary_probs(embeddings)

        # Soft segment assignments
        segment_assignments = self.compute_segment_assignments(embeddings)

        # Segment statistics
        segment_means = self.compute_segment_means(coverage, segment_assignments)

        # Smoothed signal
        smoothed_coverage = self.compute_smoothed_coverage(
            coverage, segment_assignments, segment_means
        )

        # Build output data
        transformed_data = {
            "coverage": coverage,
            "boundary_probs": boundary_probs,
            "segment_assignments": segment_assignments,
            "segment_means": segment_means,
            "smoothed_coverage": smoothed_coverage,
        }

        return transformed_data, state, metadata


# =========================================================================
# Enhanced CNV Segmentation
# =========================================================================


@dataclass
class EnhancedCNVSegmentationConfig(OperatorConfig):
    """Configuration for EnhancedCNVSegmentation.

    Extends the base CNV segmentation with multi-signal fusion, pyramidal
    smoothing, dynamic thresholding, and HMM copy-number state mapping.

    Attributes:
        max_segments: Maximum number of segments to detect.
        hidden_dim: Hidden dimension for attention layers.
        attention_heads: Number of attention heads.
        temperature: Temperature for softmax operations.
        use_baf: Whether to incorporate B-allele frequency signal.
        baf_weight: Initial weight for BAF signal in fusion.
        smoothing_window: Window size for pyramidal smoothing convolution.
        threshold_scale: Multiplier for STDDEV-based dynamic threshold.
        n_copy_states: Number of discrete copy-number states (0-somy to N-somy).
    """

    max_segments: int = 100
    hidden_dim: int = 64
    attention_heads: int = 4
    temperature: float = 1.0
    use_baf: bool = False
    baf_weight: float = 0.3
    smoothing_window: int = 100
    threshold_scale: float = 1.5
    n_copy_states: int = 5


def _build_pyramidal_kernel(window_size: int) -> Float[Array, "window_size"]:
    """Build a normalized pyramidal (triangular) smoothing kernel.

    Mirrors the infercnvpy approach: ``min(r, r[::-1])`` creates a
    triangle that peaks at the centre and tapers linearly to the edges.

    Args:
        window_size: Length of the kernel (must be >= 1).

    Returns:
        Normalised 1-D pyramidal kernel that sums to 1.
    """
    r = jnp.arange(1, window_size + 1, dtype=jnp.float32)
    pyramid = jnp.minimum(r, r[::-1])
    return pyramid / jnp.sum(pyramid)


class EnhancedCNVSegmentation(TemperatureOperator):
    """Enhanced CNV segmentation with multi-signal fusion and pyramidal smoothing.

    Builds on DifferentiableCNVSegmentation by adding:

    1. **Multi-signal fusion** -- learnable linear combination of log-ratio
       coverage, BAF, and SNP density signals.
    2. **Pyramidal smoothing** -- infercnvpy-style triangular convolution
       for spatial noise reduction.
    3. **Dynamic thresholding** -- ``threshold_scale * std(smoothed)``
       filters low-amplitude noise.
    4. **HMM state mapping** -- soft copy-number posteriors (0-somy to
       4-somy by default) via learned emission model.

    Args:
        config: EnhancedCNVSegmentationConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        ```python
        config = EnhancedCNVSegmentationConfig(
            max_segments=50, use_baf=True, smoothing_window=100,
        )
        op = EnhancedCNVSegmentation(config, rngs=nnx.Rngs(0))
        data = {"coverage": cov, "baf_signal": baf, "snp_density": snp}
        result, state, meta = op.apply(data, {}, None)
        ```
    """

    def __init__(
        self,
        config: EnhancedCNVSegmentationConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the enhanced CNV segmentation operator.

        Args:
            config: Enhanced segmentation configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.max_segments = config.max_segments
        self.hidden_dim = config.hidden_dim
        self.attention_heads = config.attention_heads
        self.use_baf = config.use_baf
        self.smoothing_window = config.smoothing_window
        self.threshold_scale = config.threshold_scale
        self.n_copy_states = config.n_copy_states

        # --- Signal fusion ---
        # Number of input channels: coverage is always present.
        # BAF and SNP density are optional extras.
        n_signals = 3 if self.use_baf else 1
        self.signal_fusion = nnx.Linear(n_signals, 1, rngs=rngs)

        # --- Base segmentation layers (same as DifferentiableCNVSegmentation) ---
        self.input_proj = nnx.Linear(1, config.hidden_dim, rngs=rngs)
        self.pos_proj = nnx.Linear(1, config.hidden_dim, rngs=rngs)
        self.query_proj = nnx.Linear(
            config.hidden_dim,
            config.hidden_dim,
            rngs=rngs,
        )
        self.key_proj = nnx.Linear(
            config.hidden_dim,
            config.hidden_dim,
            rngs=rngs,
        )
        self.value_proj = nnx.Linear(
            config.hidden_dim,
            config.hidden_dim,
            rngs=rngs,
        )
        self.boundary_head = nnx.Linear(config.hidden_dim, 1, rngs=rngs)

        key = rngs.params()
        init_centroids = jax.random.normal(key, (config.max_segments, config.hidden_dim)) * 0.1
        self.segment_centroids = nnx.Param(init_centroids)

        # --- Copy-number state mapping head ---
        self.copy_number_head = nnx.Linear(
            config.hidden_dim,
            config.n_copy_states,
            rngs=rngs,
        )

    # -----------------------------------------------------------------
    # Multi-signal fusion
    # -----------------------------------------------------------------

    def fuse_signals(
        self,
        data: dict[str, Float[Array, "n_positions"]],
    ) -> Float[Array, "n_positions"]:
        """Fuse multiple genomic signals via learned linear combination.

        When ``use_baf`` is enabled, the operator expects ``baf_signal``
        and ``snp_density`` alongside ``coverage`` in the data dict.
        A learnable ``nnx.Linear(n_signals, 1)`` combines them.

        Args:
            data: Dictionary with at least ``"coverage"`` key.

        Returns:
            Fused 1-D signal of shape ``(n_positions,)``.
        """
        coverage = data["coverage"]

        if self.use_baf:
            baf = data.get("baf_signal", jnp.zeros_like(coverage))
            snp = data.get("snp_density", jnp.zeros_like(coverage))
            # Stack into (n_positions, 3)
            stacked = jnp.stack([coverage, baf, snp], axis=-1)
        else:
            stacked = coverage[:, None]  # (n_positions, 1)

        # Linear fusion -> (n_positions, 1) -> squeeze
        fused = self.signal_fusion(stacked).squeeze(-1)
        return fused

    # -----------------------------------------------------------------
    # Pyramidal smoothing (infercnvpy-style)
    # -----------------------------------------------------------------

    def pyramidal_smooth(
        self,
        signal: Float[Array, "n_positions"],
    ) -> Float[Array, "n_positions"]:
        """Apply pyramidal (triangular) convolution smoothing.

        Mirrors the infercnvpy ``_running_mean`` approach: a triangular
        kernel ``min(r, r[::-1])`` is convolved with the signal using
        ``'same'`` mode so the output length matches the input.

        The convolution is implemented via ``jax.numpy.convolve`` which
        is fully differentiable and JIT-compatible.

        Args:
            signal: 1-D signal to smooth.

        Returns:
            Smoothed signal of the same length.
        """
        kernel = _build_pyramidal_kernel(self.smoothing_window)
        # 'same' mode preserves signal length
        smoothed = jnp.convolve(signal, kernel, mode="same")
        return smoothed

    # -----------------------------------------------------------------
    # Dynamic thresholding (infercnvpy-style)
    # -----------------------------------------------------------------

    def dynamic_threshold_filter(
        self,
        signal: Float[Array, "n_positions"],
    ) -> tuple[Float[Array, "n_positions"], Float[Array, ""]]:
        """Apply STDDEV-based dynamic noise filtering.

        Following infercnvpy Step 5: ``threshold = scale * std(signal)``.
        Values with absolute magnitude below the threshold are pushed
        toward zero using a soft sigmoid gate for differentiability.

        Args:
            signal: Smoothed signal to filter.

        Returns:
            Tuple of (filtered_signal, threshold_value).
        """
        noise_threshold = self.threshold_scale * jnp.std(signal)
        # Soft gate: sigmoid((|x| - threshold) / temperature)
        # Outputs near 0 when |x| < threshold, near 1 when |x| > threshold
        gate = jax.nn.sigmoid((jnp.abs(signal) - noise_threshold) / self._temperature)
        filtered = signal * gate
        return filtered, noise_threshold

    # -----------------------------------------------------------------
    # Attention-based segmentation (reuse DifferentiableCNVSegmentation logic)
    # -----------------------------------------------------------------

    def _compute_embeddings(
        self,
        signal: Float[Array, "n_positions"],
    ) -> Float[Array, "n_positions hidden_dim"]:
        """Compute position embeddings from a 1-D signal.

        Args:
            signal: Fused/smoothed signal values at each position.

        Returns:
            Embedded representation of each position.
        """
        n_positions = signal.shape[0]
        signal_emb = self.input_proj(signal[:, None])
        positions = jnp.arange(n_positions, dtype=jnp.float32) / n_positions
        pos_emb = self.pos_proj(positions[:, None])
        return signal_emb + pos_emb

    def _compute_boundary_probs(
        self,
        embeddings: Float[Array, "n_positions hidden_dim"],
    ) -> Float[Array, "n_positions"]:
        """Compute soft boundary probabilities using self-attention.

        Args:
            embeddings: Position embeddings.

        Returns:
            Probability of being a segment boundary at each position.
        """
        n_positions = embeddings.shape[0]
        head_dim = self.hidden_dim // self.attention_heads
        scale = jnp.sqrt(jnp.array(head_dim, dtype=embeddings.dtype))

        q = self.query_proj(embeddings).reshape(
            n_positions,
            self.attention_heads,
            head_dim,
        )
        k = self.key_proj(embeddings).reshape(
            n_positions,
            self.attention_heads,
            head_dim,
        )
        v = self.value_proj(embeddings).reshape(
            n_positions,
            self.attention_heads,
            head_dim,
        )

        attn_scores = jnp.einsum("nhd,mhd->nhm", q, k) / scale
        attn_weights = jax.nn.softmax(
            attn_scores / self._temperature,
            axis=-1,
        )
        attended = jnp.einsum("nhm,mhd->nhd", attn_weights, v)
        attended = attended.reshape(n_positions, self.hidden_dim)

        logits = self.boundary_head(attended).squeeze(-1)
        return jax.nn.sigmoid(logits)

    def _compute_segment_assignments(
        self,
        embeddings: Float[Array, "n_positions hidden_dim"],
    ) -> Float[Array, "n_positions max_segments"]:
        """Compute soft segment assignments via attention to centroids.

        Args:
            embeddings: Position embeddings.

        Returns:
            Soft assignment probability to each segment.
        """
        centroids = self.segment_centroids[...]
        similarities = jnp.einsum("nh,sh->ns", embeddings, centroids)
        return jax.nn.softmax(similarities / self._temperature, axis=-1)

    def _compute_segment_means(
        self,
        signal: Float[Array, "n_positions"],
        assignments: Float[Array, "n_positions max_segments"],
    ) -> Float[Array, "max_segments"]:
        """Compute weighted mean signal per segment.

        Args:
            signal: Input signal.
            assignments: Soft segment assignments.

        Returns:
            Mean signal value for each segment.
        """
        weighted_sum = jnp.einsum("n,ns->s", signal, assignments)
        weight_sum = jnp.sum(assignments, axis=0) + EPSILON
        return weighted_sum / weight_sum

    def _compute_smoothed_coverage(
        self,
        assignments: Float[Array, "n_positions max_segments"],
        segment_means: Float[Array, "max_segments"],
    ) -> Float[Array, "n_positions"]:
        """Compute smoothed coverage from segment assignments.

        Args:
            assignments: Soft segment assignments.
            segment_means: Mean value for each segment.

        Returns:
            Smoothed coverage signal.
        """
        return jnp.einsum("ns,s->n", assignments, segment_means)

    # -----------------------------------------------------------------
    # HMM state mapping
    # -----------------------------------------------------------------

    def compute_copy_number_posteriors(
        self,
        embeddings: Float[Array, "n_positions hidden_dim"],
    ) -> Float[Array, "n_positions n_copy_states"]:
        """Map embeddings to soft copy-number state posteriors.

        Uses a learned linear head followed by temperature-scaled softmax
        to produce per-position posterior probabilities over discrete
        copy-number states (0-somy through (n_copy_states-1)-somy).

        Args:
            embeddings: Position embeddings.

        Returns:
            Copy-number state posteriors, shape ``(n_positions, n_copy_states)``.
        """
        logits = self.copy_number_head(embeddings)
        return jax.nn.softmax(logits / self._temperature, axis=-1)

    def compute_expected_copy_number(
        self,
        posteriors: Float[Array, "n_positions n_copy_states"],
    ) -> Float[Array, "n_positions"]:
        """Compute expected copy number as posterior-weighted state values.

        E[CN] = sum_k( k * P(state=k) ) for k in 0..n_copy_states-1.

        Args:
            posteriors: Copy-number state posteriors.

        Returns:
            Expected copy number at each position.
        """
        state_values = jnp.arange(self.n_copy_states, dtype=jnp.float32)
        return jnp.einsum("ns,s->n", posteriors, state_values)

    # -----------------------------------------------------------------
    # Main apply
    # -----------------------------------------------------------------

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply enhanced CNV segmentation to genomic signal data.

        Args:
            data: Dictionary containing:
                - ``"coverage"``: Log-ratio coverage signal ``(n_positions,)``
                - ``"baf_signal"`` (optional): B-allele frequency ``(n_positions,)``
                - ``"snp_density"`` (optional): SNP density ``(n_positions,)``
            state: Element state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: Not used.
            stats: Not used.

        Returns:
            Tuple of ``(transformed_data, state, metadata)`` where
            ``transformed_data`` contains:

            - ``"coverage"``: Original coverage
            - ``"fused_signal"``: Fused multi-signal output
            - ``"pyramidal_smoothed"``: After pyramidal convolution
            - ``"thresholded_signal"``: After dynamic noise filtering
            - ``"dynamic_threshold"``: Scalar threshold value
            - ``"boundary_probs"``: Soft boundary probabilities
            - ``"segment_assignments"``: Soft segment memberships
            - ``"segment_means"``: Mean value per segment
            - ``"smoothed_coverage"``: Final segmented/smoothed signal
            - ``"copy_number_posteriors"``: Per-position CN state posteriors
            - ``"expected_copy_number"``: Expected copy number per position
        """
        coverage = data["coverage"]

        # 1. Multi-signal fusion
        fused = self.fuse_signals(data)

        # 2. Pyramidal smoothing
        smoothed = self.pyramidal_smooth(fused)

        # 3. Dynamic thresholding
        thresholded, threshold_val = self.dynamic_threshold_filter(smoothed)

        # 4. Attention-based segmentation on thresholded signal
        embeddings = self._compute_embeddings(thresholded)
        boundary_probs = self._compute_boundary_probs(embeddings)
        segment_assignments = self._compute_segment_assignments(embeddings)
        segment_means = self._compute_segment_means(
            thresholded,
            segment_assignments,
        )
        smoothed_coverage = self._compute_smoothed_coverage(
            segment_assignments,
            segment_means,
        )

        # 5. HMM state mapping
        cn_posteriors = self.compute_copy_number_posteriors(embeddings)
        expected_cn = self.compute_expected_copy_number(cn_posteriors)

        transformed_data = {
            "coverage": coverage,
            "fused_signal": fused,
            "pyramidal_smoothed": smoothed,
            "thresholded_signal": thresholded,
            "dynamic_threshold": threshold_val,
            "boundary_probs": boundary_probs,
            "segment_assignments": segment_assignments,
            "segment_means": segment_means,
            "smoothed_coverage": smoothed_coverage,
            "copy_number_posteriors": cn_posteriors,
            "expected_copy_number": expected_cn,
        }

        return transformed_data, state, metadata
