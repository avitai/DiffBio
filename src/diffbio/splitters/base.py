"""Base splitter classes for DiffBio.

This module provides the base classes for dataset splitting:
- SplitResult: NamedTuple containing train/valid/test indices
- SplitterConfig: Base configuration for splitters
- SplitterModule: Base class for all splitters
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.core.structural import StructuralModule


class SplitResult(NamedTuple):
    """Result of a dataset split operation.

    Attributes:
        train_indices: Array of indices for training set
        valid_indices: Array of indices for validation set
        test_indices: Array of indices for test set
    """

    train_indices: jnp.ndarray
    valid_indices: jnp.ndarray
    test_indices: jnp.ndarray

    @property
    def train_size(self) -> int:
        """Return number of training samples."""
        return len(self.train_indices)

    @property
    def valid_size(self) -> int:
        """Return number of validation samples."""
        return len(self.valid_indices)

    @property
    def test_size(self) -> int:
        """Return number of test samples."""
        return len(self.test_indices)


@dataclass
class SplitterConfig(StructuralConfig):
    """Base configuration for splitters.

    Frozen because splitters are non-parametric (StructuralModule).

    Attributes:
        train_frac: Fraction of data for training (default: 0.8)
        valid_frac: Fraction of data for validation (default: 0.1)
        test_frac: Fraction of data for testing (default: 0.1)
        seed: Random seed for reproducibility (optional)
    """

    train_frac: float = 0.8
    valid_frac: float = 0.1
    test_frac: float = 0.1
    seed: int | None = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        super().__post_init__()
        total = self.train_frac + self.valid_frac + self.test_frac
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split fractions must sum to 1.0, got {total}")


class SplitterModule(StructuralModule):
    """Base class for dataset splitters.

    Inherits from StructuralModule because:
    - Non-parametric (no learnable parameters)
    - Frozen config (splitting strategy is fixed)
    - Uses process() method pattern
    - Integrates with Datarax data sources

    Splitters divide data into train/valid/test sets, while Datarax
    SamplerModule controls iteration ORDER within those sets.

    Args:
        config: Splitter configuration
        rngs: Random number generators for stochastic splitting
        name: Optional name for the module
    """

    def __init__(
        self,
        config: SplitterConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize SplitterModule.

        Args:
            config: Splitter configuration
            rngs: Random number generators
            name: Optional module name
        """
        super().__init__(config, rngs=rngs, name=name)

    def split(self, data_source: DataSourceModule) -> SplitResult:
        """Split a data source into train/valid/test indices.

        Subclasses must implement this method.

        Args:
            data_source: Datarax DataSourceModule to split

        Returns:
            SplitResult with train/valid/test indices
        """
        raise NotImplementedError("Subclasses must implement split()")

    def process(self, data_source: DataSourceModule) -> SplitResult:
        """StructuralModule interface - delegates to split().

        Args:
            data_source: Datarax DataSourceModule to split

        Returns:
            SplitResult with train/valid/test indices
        """
        return self.split(data_source)

    def k_fold_split(
        self, data_source: DataSourceModule, k: int = 5
    ) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
        """K-fold cross-validation split.

        Subclasses may implement this method.

        Args:
            data_source: Datarax DataSourceModule to split
            k: Number of folds

        Returns:
            List of (train_indices, val_indices) tuples for each fold
        """
        raise NotImplementedError("Subclasses may implement k_fold_split()")

    def create_split_sources(
        self,
        data_source: DataSourceModule,
        split_result: SplitResult | None = None,
        lazy: bool = True,
    ) -> tuple[DataSourceModule, DataSourceModule, DataSourceModule]:
        """Create separate data sources for each split.

        This creates views into the original data source using the split indices.
        Each returned source can be used with Datarax samplers independently.

        Args:
            data_source: Original data source
            split_result: Pre-computed split (or compute if None)
            lazy: If True, use lazy loading (IndexedViewSource). If False,
                  eagerly load into MemorySource (faster iteration but uses memory).

        Returns:
            Tuple of (train_source, valid_source, test_source)
        """
        if split_result is None:
            split_result = self.split(data_source)

        if lazy:
            # LAZY LOADING: Create view sources that delegate to original
            from diffbio.sources.indexed_view import (
                IndexedViewSource,
                IndexedViewSourceConfig,
            )

            train_config = IndexedViewSourceConfig(shuffle=True, seed=self.config.seed)
            valid_config = IndexedViewSourceConfig(shuffle=False)
            test_config = IndexedViewSourceConfig(shuffle=False)

            return (
                IndexedViewSource(
                    train_config, data_source, split_result.train_indices, rngs=self.rngs
                ),
                IndexedViewSource(
                    valid_config, data_source, split_result.valid_indices, rngs=self.rngs
                ),
                IndexedViewSource(
                    test_config, data_source, split_result.test_indices, rngs=self.rngs
                ),
            )
        else:
            # EAGER LOADING: Load all elements into memory (faster iteration)
            from datarax.sources import MemorySource, MemorySourceConfig

            train_elements = [data_source[int(i)] for i in split_result.train_indices]
            valid_elements = [data_source[int(i)] for i in split_result.valid_indices]
            test_elements = [data_source[int(i)] for i in split_result.test_indices]

            train_config = MemorySourceConfig(shuffle=True, seed=self.config.seed)
            valid_config = MemorySourceConfig(shuffle=False)
            test_config = MemorySourceConfig(shuffle=False)

            return (
                MemorySource(train_config, data=train_elements, rngs=self.rngs),
                MemorySource(valid_config, data=valid_elements, rngs=self.rngs),
                MemorySource(test_config, data=test_elements, rngs=self.rngs),
            )
