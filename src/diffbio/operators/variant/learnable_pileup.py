"""Learnable DeepVariant-style pileup encoding (task-aware pileup images).

DeepVariant encodes reads into a fixed, hand-designed multi-channel pileup image: the
base one-hot, a quality intensity ``q / quality_max``, a mapping-quality intensity, a
strand indicator, and reference-mismatch channels. Those encoding rules never sit on
the gradient path -- they are frozen featurization. This operator makes them learnable.

Each frozen rule is replaced by a parameterized encoder initialized to reproduce it
exactly: a ``(4, 4)`` base embedding initialized to the identity, and a per-channel
affine ``gain * value + bias`` initialized to ``gain = 1, bias = 0``. At initialization
the pileup image is bit-for-bit the frozen DeepVariant image, so joint training starts
at the hand-designed baseline and can only improve. This is the variant-calling analogue
of a learnable feature frontend (LEAF for audio, the learnable projection for scRNA):
when the reduction/encoding is hand-designed and the head is complex, jointly optimizing
the encoding recovers task-relevant structure the fixed rules discard.
"""

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int

from diffbio.operators.variant.deepvariant_pileup import (
    DeepVariantPileupConfig,
    DeepVariantStylePileup,
)


class LearnablePileup(DeepVariantStylePileup):
    """DeepVariant-style pileup whose channel encodings are jointly trainable.

    Reuses the scatter machinery, channel routing, and configuration surface of
    :class:`DeepVariantStylePileup`; only the per-channel value encodings are replaced
    by learnable parameters. Every parameter is initialized so the emitted pileup image
    exactly matches the frozen operator at initialization:

    - ``base_embedding`` ``(4, 4)`` initialized to the identity, applied as
      ``reads @ base_embedding`` so base intensities start as the one-hot encoding.
    - ``quality_affine``, ``mapq_affine``, ``strand_affine``, ``mismatch_affine`` --
      each a ``(gain, bias)`` pair initialized to ``(1, 0)``, applied to the normalized
      scalar value before scatter (so empty read slots stay zero). ``mismatch_affine``
      is shared by the ``supports_variant`` and ``differs_from_ref`` channels, which are
      identical in the frozen encoding.
    """

    def __init__(
        self,
        config: DeepVariantPileupConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the learnable pileup encoders at their frozen values.

        Args:
            config: Pileup configuration (shared with the frozen operator).
            rngs: Random number generators (optional; encoders are deterministically
                initialized, so no randomness is consumed).
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)
        self.base_embedding = nnx.Param(jnp.eye(4, dtype=jnp.float32))
        identity_affine = jnp.array([1.0, 0.0], dtype=jnp.float32)
        self.quality_affine = nnx.Param(identity_affine)
        self.mapq_affine = nnx.Param(jnp.array([1.0, 0.0], dtype=jnp.float32))
        self.strand_affine = nnx.Param(jnp.array([1.0, 0.0], dtype=jnp.float32))
        self.mismatch_affine = nnx.Param(jnp.array([1.0, 0.0], dtype=jnp.float32))

    def _apply_affine(self, values: Float[Array, "..."], affine: nnx.Param) -> Float[Array, "..."]:
        """Apply a learnable ``gain * value + bias`` map to already-normalized values."""
        gain, bias = affine[0], affine[1]
        return gain * values + bias

    def _compute_base_channels(
        self,
        reads: Float[Array, "num_reads read_length 4"],
        positions: Int[Array, "num_reads"],
        read_length: int,
    ) -> Float[Array, "max_reads window_size 4"]:
        """Scatter learnably-embedded base vectors (``reads @ base_embedding``)."""
        embedded = reads @ self.base_embedding[...]
        return self._scatter_reads_to_image(embedded, positions, read_length)

    def _compute_quality_channel(
        self,
        base_qualities: Float[Array, "num_reads read_length"],
        positions: Int[Array, "num_reads"],
        read_length: int,
    ) -> Float[Array, "max_reads window_size"]:
        """Learnable-affine base-quality channel over the normalized Phred score."""
        normalized_qual = jnp.clip(base_qualities / self.config.quality_max, 0.0, 1.0)
        encoded = self._apply_affine(normalized_qual, self.quality_affine)
        return self._scatter_reads_to_image(encoded, positions, read_length)

    def _compute_mapq_channel(
        self,
        mapping_qualities: Float[Array, "num_reads"],
        positions: Int[Array, "num_reads"],
        read_length: int,
    ) -> Float[Array, "max_reads window_size"]:
        """Learnable-affine mapping-quality channel, broadcast across read positions."""
        normalized_mapq = jnp.clip(mapping_qualities / self.config.mapq_max, 0.0, 1.0)
        encoded = self._apply_affine(normalized_mapq, self.mapq_affine)
        mapq_per_position = jnp.broadcast_to(encoded[:, None], (encoded.shape[0], read_length))
        return self._scatter_reads_to_image(mapq_per_position, positions, read_length)

    def _compute_strand_channel(
        self,
        strands: Float[Array, "num_reads"],
        positions: Int[Array, "num_reads"],
        read_length: int,
    ) -> Float[Array, "max_reads window_size"]:
        """Learnable-affine strand channel, broadcast across read positions."""
        encoded = self._apply_affine(strands, self.strand_affine)
        strand_per_position = jnp.broadcast_to(encoded[:, None], (encoded.shape[0], read_length))
        return self._scatter_reads_to_image(strand_per_position, positions, read_length)

    def _compute_variant_support_channel(
        self,
        reads: Float[Array, "num_reads read_length 4"],
        reference: Float[Array, "window_size 4"],
        positions: Int[Array, "num_reads"],
        read_length: int,
    ) -> Float[Array, "max_reads window_size"]:
        """Learnable-affine reference-mismatch channel (``1 - dot(read, ref)``)."""
        ref_positions = positions[:, None] + jnp.arange(read_length)[None, :]
        clipped_ref_positions = jnp.clip(ref_positions, 0, self.config.window_size - 1)
        ref_bases = reference[clipped_ref_positions]
        match_scores = jnp.sum(reads * ref_bases, axis=-1)
        mismatch_values = 1.0 - match_scores
        encoded = self._apply_affine(mismatch_values, self.mismatch_affine)
        return self._scatter_reads_to_image(encoded, positions, read_length)
