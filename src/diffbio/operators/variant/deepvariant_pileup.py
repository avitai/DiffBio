"""DeepVariant-style pileup image generation for variant calling."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int, PyTree

from diffbio.configs import TemperatureConfig
from diffbio.core.base_operators import TemperatureOperator

_DEFAULT_DEEPVARIANT_CHANNELS = (
    "base",
    "base_quality",
    "mapping_quality",
    "strand",
    "supports_variant",
    "differs_from_ref",
)
_DEEPVARIANT_CHANNEL_WIDTHS = {
    "base": 4,
    "base_quality": 1,
    "mapping_quality": 1,
    "strand": 1,
    "supports_variant": 1,
    "differs_from_ref": 1,
}


@dataclass(frozen=True)
class DeepVariantPileupConfig(TemperatureConfig):
    """Configuration for DeepVariant-style pileup generation.

    Inherits from TemperatureConfig to get temperature parameter for
    soft/differentiable operations.

    Attributes:
        window_size: Width of pileup image in base pairs (default: 221)
        max_reads: Height of pileup image / max reads to include (default: 100)
        channels: Ordered channel set to emit in the pileup image.
        quality_max: Maximum quality score for normalization (default: 40)
        mapq_max: Maximum mapping quality for normalization (default: 60)
    """

    window_size: int = 221
    max_reads: int = 100
    channels: tuple[str, ...] = _DEFAULT_DEEPVARIANT_CHANNELS
    quality_max: float = 40.0
    mapq_max: float = 60.0

    def __post_init__(self) -> None:
        """Validate the supported DeepVariant pileup configuration surface."""
        super().__post_init__()

        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if self.max_reads <= 0:
            raise ValueError(f"max_reads must be positive, got {self.max_reads}")
        if self.quality_max <= 0.0:
            raise ValueError(f"quality_max must be positive, got {self.quality_max}")
        if self.mapq_max <= 0.0:
            raise ValueError(f"mapq_max must be positive, got {self.mapq_max}")
        if not self.channels:
            raise ValueError("channels must contain at least one DeepVariant channel")

        invalid_channels = tuple(
            channel for channel in self.channels if channel not in _DEEPVARIANT_CHANNEL_WIDTHS
        )
        if invalid_channels:
            invalid = ", ".join(invalid_channels)
            raise ValueError(f"channels contains unsupported values: {invalid}")

        if len(set(self.channels)) != len(self.channels):
            raise ValueError("channels must not contain duplicates")


class DeepVariantStylePileup(TemperatureOperator):
    """DeepVariant-style multi-channel pileup image generator.

    Generates pileup images compatible with DeepVariant's CNN architecture
    while maintaining full differentiability for end-to-end training.

    The pileup image has shape (max_reads, window_size, num_channels) where
    each read occupies a row and each column represents a base position.

    Inherits from TemperatureOperator to get:

    - _temperature property for temperature-controlled smoothing
    - soft_max() for logsumexp-based smooth maximum
    - soft_argmax() for soft position selection

    Example:
        ```python
        config = DeepVariantPileupConfig(window_size=101, max_reads=50)
        pileup = DeepVariantStylePileup(config)
        data = {
            "reads": reads,  # (num_reads, read_length, 4)
            "reference": reference,  # (window_size, 4)
            "base_qualities": qualities,  # (num_reads, read_length)
            "mapping_qualities": mapq,  # (num_reads,)
            "strands": strands,  # (num_reads,)
            "positions": positions,  # (num_reads,)
        }
        result, _, _ = pileup.apply(data, {}, None)
        pileup_image = result["pileup_image"]  # (50, 101, num_channels)
        ```
    """

    def __init__(
        self,
        config: DeepVariantPileupConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize DeepVariantStylePileup.

        Args:
            config: Pileup configuration
            rngs: Random number generators (optional)
            name: Optional operator name
        """
        super().__init__(config, rngs=rngs, name=name)

        # Calculate number of output channels
        self._num_channels = sum(
            _DEEPVARIANT_CHANNEL_WIDTHS[channel] for channel in config.channels
        )

    @property
    def num_channels(self) -> int:
        """Return the number of output channels."""
        return self._num_channels

    def _scatter_reads_to_image(
        self,
        values: Float[Array, "num_reads read_length ..."],
        positions: Int[Array, "num_reads"],
        read_length: int,
    ) -> Float[Array, "max_reads window_size ..."]:
        """Scatter per-read, per-position values into a 2D (or 3D) pileup image.

        This is the generic helper that handles the common scan+fori_loop pattern
        used by all channel computation methods. It processes reads one at a time
        via lax.scan, and for each read scatters its values into the correct
        window positions via lax.fori_loop.

        Supports both scalar per-position values (image shape: max_reads x window_size)
        and vector per-position values (image shape: max_reads x window_size x D).

        Args:
            values: Pre-computed values to scatter. Shape is either
                (num_reads, read_length) for scalar channels or
                (num_reads, read_length, D) for vector channels (e.g., one-hot bases).
            positions: Starting position of each read in the window (num_reads,).
            read_length: Length of each read.

        Returns:
            Image with scattered values. Shape is (max_reads, window_size) for
            scalar values or (max_reads, window_size, D) for vector values.
        """
        config = self.config
        is_vector = values.ndim == 3
        if is_vector:
            value_dim = values.shape[2]
            image = jnp.zeros((config.max_reads, config.window_size, value_dim), dtype=jnp.float32)
        else:
            image = jnp.zeros((config.max_reads, config.window_size), dtype=jnp.float32)

        def scatter_read(carry, read_data):
            img, read_idx = carry
            read_vals, pos = read_data

            valid = read_idx < config.max_reads
            read_positions = pos + jnp.arange(read_length)
            in_bounds = (read_positions >= 0) & (read_positions < config.window_size)

            def update_position(i, current_img):
                pos_idx = jnp.clip(read_positions[i], 0, config.window_size - 1)
                is_valid = valid & in_bounds[i]
                if is_vector:
                    return jnp.where(
                        is_valid,
                        current_img.at[read_idx, pos_idx, :].set(read_vals[i]),
                        current_img,
                    )
                return jnp.where(
                    is_valid,
                    current_img.at[read_idx, pos_idx].set(read_vals[i]),
                    current_img,
                )

            new_img = jax.lax.fori_loop(0, read_length, update_position, img)
            return (new_img, read_idx + 1), None

        (image, _), _ = jax.lax.scan(
            scatter_read,
            (image, 0),
            (values, positions),
        )

        return image

    def compute_pileup_image(
        self,
        reads: Float[Array, "num_reads read_length 4"],
        reference: Float[Array, "window_size 4"],
        base_qualities: Float[Array, "num_reads read_length"],
        mapping_qualities: Float[Array, "num_reads"],
        strands: Float[Array, "num_reads"],
        positions: Int[Array, "num_reads"],
    ) -> Float[Array, "max_reads window_size num_channels"]:
        """Compute DeepVariant-style pileup image.

        Args:
            reads: One-hot encoded reads (num_reads, read_length, 4)
            reference: One-hot encoded reference (window_size, 4)
            base_qualities: Phred quality scores (num_reads, read_length)
            mapping_qualities: Mapping quality scores (num_reads,)
            strands: Strand orientation, 0=forward, 1=reverse (num_reads,)
            positions: Starting position of each read in window (num_reads,)

        Returns:
            Pileup image of shape (max_reads, window_size, num_channels)
        """
        config = self.config
        read_length = reads.shape[1]

        # Initialize output image with zeros
        pileup_image = jnp.zeros(
            (config.max_reads, config.window_size, self._num_channels),
            dtype=jnp.float32,
        )

        # Build the pileup image channel by channel
        channel_idx = 0

        for channel in config.channels:
            if channel == "base":
                base_image = self._compute_base_channels(reads, positions, read_length)
                pileup_image = pileup_image.at[:, :, channel_idx : channel_idx + 4].set(base_image)
                channel_idx += 4
            elif channel == "base_quality":
                quality_image = self._compute_quality_channel(
                    base_qualities, positions, read_length
                )
                pileup_image = pileup_image.at[:, :, channel_idx].set(quality_image)
                channel_idx += 1
            elif channel == "mapping_quality":
                mapq_image = self._compute_mapq_channel(mapping_qualities, positions, read_length)
                pileup_image = pileup_image.at[:, :, channel_idx].set(mapq_image)
                channel_idx += 1
            elif channel == "strand":
                strand_image = self._compute_strand_channel(strands, positions, read_length)
                pileup_image = pileup_image.at[:, :, channel_idx].set(strand_image)
                channel_idx += 1
            elif channel == "supports_variant":
                variant_image = self._compute_variant_support_channel(
                    reads, reference, positions, read_length
                )
                pileup_image = pileup_image.at[:, :, channel_idx].set(variant_image)
                channel_idx += 1
            else:
                diff_image = self._compute_diff_from_ref_channel(
                    reads, reference, positions, read_length
                )
                pileup_image = pileup_image.at[:, :, channel_idx].set(diff_image)
                channel_idx += 1

        return pileup_image

    def _compute_base_channels(
        self,
        reads: Float[Array, "num_reads read_length 4"],
        positions: Int[Array, "num_reads"],
        read_length: int,
    ) -> Float[Array, "max_reads window_size 4"]:
        """Compute base identity channels (one-hot A/C/G/T).

        Places each read's bases at the correct position in the image.
        Values are the one-hot base vectors from the reads directly.
        """
        return self._scatter_reads_to_image(reads, positions, read_length)

    def _compute_quality_channel(
        self,
        base_qualities: Float[Array, "num_reads read_length"],
        positions: Int[Array, "num_reads"],
        read_length: int,
    ) -> Float[Array, "max_reads window_size"]:
        """Compute base quality channel normalized to [0, 1]."""
        normalized_qual = jnp.clip(base_qualities / self.config.quality_max, 0.0, 1.0)
        return self._scatter_reads_to_image(normalized_qual, positions, read_length)

    def _compute_mapq_channel(
        self,
        mapping_qualities: Float[Array, "num_reads"],
        positions: Int[Array, "num_reads"],
        read_length: int,
    ) -> Float[Array, "max_reads window_size"]:
        """Compute mapping quality channel normalized to [0, 1].

        MAPQ is constant across a read, so we broadcast to all positions.
        """
        normalized_mapq = jnp.clip(mapping_qualities / self.config.mapq_max, 0.0, 1.0)
        # Broadcast constant MAPQ to all positions: (num_reads,) -> (num_reads, read_length)
        mapq_per_position = jnp.broadcast_to(
            normalized_mapq[:, None], (normalized_mapq.shape[0], read_length)
        )
        return self._scatter_reads_to_image(mapq_per_position, positions, read_length)

    def _compute_strand_channel(
        self,
        strands: Float[Array, "num_reads"],
        positions: Int[Array, "num_reads"],
        read_length: int,
    ) -> Float[Array, "max_reads window_size"]:
        """Compute strand channel (0=forward, 1=reverse).

        Strand is constant across a read, so we broadcast to all positions.
        """
        # Broadcast constant strand to all positions: (num_reads,) -> (num_reads, read_length)
        strand_per_position = jnp.broadcast_to(strands[:, None], (strands.shape[0], read_length))
        return self._scatter_reads_to_image(strand_per_position, positions, read_length)

    def _compute_variant_support_channel(
        self,
        reads: Float[Array, "num_reads read_length 4"],
        reference: Float[Array, "window_size 4"],
        positions: Int[Array, "num_reads"],
        read_length: int,
    ) -> Float[Array, "max_reads window_size"]:
        """Compute variant support channel.

        This is a soft indicator of whether a read base differs from reference,
        which can indicate support for a variant allele.

        Uses soft comparison for differentiability: mismatch = 1 - dot(read, ref).
        """
        # Pre-compute mismatch values for all reads and positions
        # ref_positions[r, i] = positions[r] + i
        ref_positions = positions[:, None] + jnp.arange(read_length)[None, :]
        clipped_ref_positions = jnp.clip(ref_positions, 0, self.config.window_size - 1)

        # Look up reference bases at each position: (num_reads, read_length, 4)
        ref_bases = reference[clipped_ref_positions]

        # Soft mismatch: 1 - dot product of one-hot vectors
        match_scores = jnp.sum(reads * ref_bases, axis=-1)  # (num_reads, read_length)
        mismatch_values = 1.0 - match_scores

        return self._scatter_reads_to_image(mismatch_values, positions, read_length)

    def _compute_diff_from_ref_channel(
        self,
        reads: Float[Array, "num_reads read_length 4"],
        reference: Float[Array, "window_size 4"],
        positions: Int[Array, "num_reads"],
        read_length: int,
    ) -> Float[Array, "max_reads window_size"]:
        """Compute 'differs from reference' channel.

        Similar to variant support but provides a direct mismatch signal.
        In practice, this is equivalent to variant_support for standard pileups.
        """
        # For standard pileups, this is the same as variant support
        # DeepVariant distinguishes them for multi-allelic calling
        return self._compute_variant_support_channel(reads, reference, positions, read_length)

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,  # noqa: ARG002
        stats: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply DeepVariant-style pileup generation.

        Args:
            data: Dictionary containing:
                - "reads": One-hot encoded reads (num_reads, read_length, 4)
                - "reference": One-hot encoded reference (window_size, 4)
                - "base_qualities": Phred quality scores (num_reads, read_length)
                - "mapping_qualities": Mapping quality scores (num_reads,)
                - "strands": Strand orientation (num_reads,)
                - "positions": Read start positions in window (num_reads,)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used (deterministic operator)
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains input data plus pileup_image
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        del random_params, stats  # Unused parameters

        reads = data["reads"]
        reference = data["reference"]
        base_qualities = data["base_qualities"]
        mapping_qualities = data["mapping_qualities"]
        strands = data["strands"]
        positions = data["positions"]

        # Compute pileup image
        pileup_image = self.compute_pileup_image(
            reads=reads,
            reference=reference,
            base_qualities=base_qualities,
            mapping_qualities=mapping_qualities,
            strands=strands,
            positions=positions,
        )

        # Build output data - preserve input keys for Datarax compatibility
        transformed_data = {
            **data,
            "pileup_image": pileup_image,
        }

        return transformed_data, state, metadata
