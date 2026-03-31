"""Tests for output space management utilities."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from diffbio.sources.perturbation._types import OutputSpaceMode
from diffbio.sources.perturbation.output_space import (
    select_output_counts,
)


class TestSelectOutputCounts:
    """Tests for select_output_counts."""

    def test_all_mode_returns_full_matrix(self) -> None:
        counts = jnp.ones((10, 100))
        result = select_output_counts(counts, hvg_indices=None, mode=OutputSpaceMode.ALL)
        assert result.shape == (10, 100)

    def test_gene_mode_subsets_to_hvg(self) -> None:
        counts = jnp.arange(50).reshape(5, 10).astype(jnp.float32)
        hvg_idx = np.array([0, 2, 5])
        result = select_output_counts(counts, hvg_indices=hvg_idx, mode=OutputSpaceMode.GENE)
        assert result.shape == (5, 3)
        np.testing.assert_array_equal(result[:, 0], counts[:, 0])
        np.testing.assert_array_equal(result[:, 1], counts[:, 2])
        np.testing.assert_array_equal(result[:, 2], counts[:, 5])

    def test_gene_mode_without_hvg_raises(self) -> None:
        counts = jnp.ones((5, 10))
        with pytest.raises(ValueError, match="hvg_indices"):
            select_output_counts(counts, hvg_indices=None, mode=OutputSpaceMode.GENE)

    def test_embedding_mode_returns_empty(self) -> None:
        counts = jnp.ones((5, 10))
        result = select_output_counts(counts, hvg_indices=None, mode=OutputSpaceMode.EMBEDDING)
        assert result.shape == (5, 0)
