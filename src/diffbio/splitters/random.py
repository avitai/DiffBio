"""Random and stratified splitters for DiffBio.

This module provides random splitting utilities:
- RandomSplitter: Simple random permutation-based splitting
- StratifiedSplitter: Stratified splitting preserving class distribution
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core.data_source import DataSourceModule

from diffbio.splitters.base import SplitResult, SplitterConfig, SplitterModule


@dataclass
class RandomSplitterConfig(SplitterConfig):
    """Configuration for random splitter.

    Inherits all fields from SplitterConfig:
        - train_frac: Fraction of data for training (default: 0.8)
        - valid_frac: Fraction of data for validation (default: 0.1)
        - test_frac: Fraction of data for testing (default: 0.1)
        - seed: Random seed for reproducibility (optional)
    """

    pass


class RandomSplitter(SplitterModule):
    """Simple random splitting using JAX RNG.

    Uses JAX random permutation for reproducible splits.
    All data points are randomly assigned to train/valid/test sets
    according to the configured fractions.

    Example:
        >>> config = RandomSplitterConfig(train_frac=0.8, valid_frac=0.1, test_frac=0.1, seed=42)
        >>> splitter = RandomSplitter(config)
        >>> result = splitter.split(data_source)
        >>> print(f"Train size: {result.train_size}")
    """

    def __init__(
        self,
        config: RandomSplitterConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize RandomSplitter.

        Args:
            config: Random splitter configuration
            rngs: Random number generators
            name: Optional module name
        """
        super().__init__(config, rngs=rngs, name=name)

    def split(self, data_source: DataSourceModule) -> SplitResult:
        """Split data source randomly.

        Args:
            data_source: Datarax DataSourceModule to split

        Returns:
            SplitResult with randomly assigned train/valid/test indices
        """
        n = len(data_source)
        train_end = int(self.config.train_frac * n)
        valid_end = int((self.config.train_frac + self.config.valid_frac) * n)

        # Use JAX RNG for reproducibility
        if self.config.seed is not None:
            key = jax.random.key(self.config.seed)
        elif self.rngs is not None and "split" in self.rngs:
            key = self.rngs.split()
        else:
            key = jax.random.key(0)

        indices = jax.random.permutation(key, jnp.arange(n))

        return SplitResult(
            train_indices=indices[:train_end],
            valid_indices=indices[train_end:valid_end],
            test_indices=indices[valid_end:],
        )

    def k_fold_split(
        self, data_source: DataSourceModule, k: int = 5
    ) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
        """K-fold cross-validation split.

        Args:
            data_source: Datarax DataSourceModule to split
            k: Number of folds

        Returns:
            List of (train_indices, val_indices) tuples for each fold
        """
        n = len(data_source)

        if self.config.seed is not None:
            key = jax.random.key(self.config.seed)
        else:
            key = jax.random.key(0)

        indices = jax.random.permutation(key, jnp.arange(n))
        fold_size = n // k

        folds = []
        for i in range(k):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < k - 1 else n

            val_indices = indices[val_start:val_end]
            train_indices = jnp.concatenate([indices[:val_start], indices[val_end:]])
            folds.append((train_indices, val_indices))

        return folds


@dataclass
class StratifiedSplitterConfig(SplitterConfig):
    """Configuration for stratified splitter.

    Attributes:
        label_key: Key in data element containing labels (default: "y")
    """

    label_key: str = "y"


class StratifiedSplitter(SplitterModule):
    """Stratified splitting that preserves class distribution.

    Ensures each split has approximately the same class distribution
    as the original dataset. Useful for imbalanced classification tasks.

    Example:
        >>> config = StratifiedSplitterConfig(seed=42, label_key="target")
        >>> splitter = StratifiedSplitter(config)
        >>> result = splitter.split(data_source)
    """

    def __init__(
        self,
        config: StratifiedSplitterConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize StratifiedSplitter.

        Args:
            config: Stratified splitter configuration
            rngs: Random number generators
            name: Optional module name
        """
        super().__init__(config, rngs=rngs, name=name)

    def split(self, data_source: DataSourceModule) -> SplitResult:
        """Split preserving class distribution.

        Args:
            data_source: Datarax DataSourceModule to split

        Returns:
            SplitResult with stratified train/valid/test indices
        """
        # Extract labels from data source
        labels = jnp.array(
            [data_source[i].data[self.config.label_key] for i in range(len(data_source))]
        )

        # Group indices by class
        unique_labels = jnp.unique(labels)
        class_indices = {int(label): jnp.where(labels == label)[0] for label in unique_labels}

        # Use JAX RNG
        if self.config.seed is not None:
            key = jax.random.key(self.config.seed)
        else:
            key = jax.random.key(0)

        train_inds: list[jnp.ndarray] = []
        valid_inds: list[jnp.ndarray] = []
        test_inds: list[jnp.ndarray] = []

        for _label, indices in class_indices.items():
            key, subkey = jax.random.split(key)
            shuffled = jax.random.permutation(subkey, indices)

            n_class = len(shuffled)
            train_end = int(self.config.train_frac * n_class)
            valid_end = int((self.config.train_frac + self.config.valid_frac) * n_class)

            train_inds.append(shuffled[:train_end])
            valid_inds.append(shuffled[train_end:valid_end])
            test_inds.append(shuffled[valid_end:])

        empty = jnp.array([], dtype=jnp.int32)
        return SplitResult(
            train_indices=jnp.concatenate(train_inds) if train_inds else empty,
            valid_indices=jnp.concatenate(valid_inds) if valid_inds else empty,
            test_indices=jnp.concatenate(test_inds) if test_inds else empty,
        )
