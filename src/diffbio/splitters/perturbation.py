"""Perturbation-aware splitters for zero-shot and few-shot evaluation.

Provides specialized splitting strategies for single-cell perturbation
experiments: holding out entire cell types (zero-shot) or specific
perturbations within cell types (few-shot).

References:
    - cell-load/src/cell_load/config.py (zeroshot/fewshot split logic)
    - cell-load/src/cell_load/utils/data_utils.py (split_perturbations_by_cell_fraction)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from diffbio.splitters.base import SplitResult, SplitterConfig, SplitterModule

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ZeroShotSplitterConfig(SplitterConfig):
    """Configuration for ZeroShotSplitter.

    Attributes:
        held_out_cell_types: Cell types held out entirely for test.
        pert_col: Obs column name for perturbation identity.
        cell_type_col: Obs column name for cell type.
    """

    held_out_cell_types: tuple[str, ...] = ()
    pert_col: str = "perturbation"
    cell_type_col: str = "cell_type"


class ZeroShotSplitter(SplitterModule):
    """Hold out entire cell types for zero-shot evaluation.

    All cells of specified cell types go to the test set. Remaining
    cells are split into train and validation by the configured fractions.

    Args:
        config: Splitter configuration.
        rngs: Optional RNG state.
        name: Optional module name.
    """

    def split(self, data_source: Any) -> SplitResult:
        """Split data source by cell type holdout.

        Args:
            data_source: A PerturbationAnnDataSource or similar source
                providing element dicts with perturbation metadata.

        Returns:
            SplitResult with train/valid/test indices.
        """
        n = len(data_source)
        held_out = set(self.config.held_out_cell_types)

        test_indices: list[int] = []
        remaining_indices: list[int] = []

        for i in range(n):
            elem = data_source[i]
            ct = elem.get(
                "cell_type_name",
                str(elem.get("obs", {}).get(self.config.cell_type_col, "")),
            )
            if ct in held_out:
                test_indices.append(i)
            else:
                remaining_indices.append(i)

        # Split remaining into train/val
        rng = np.random.default_rng(self.config.seed)
        remaining = np.array(remaining_indices)
        rng.shuffle(remaining)

        # Adjust fractions for the remaining subset
        total_remaining = len(remaining)
        train_frac = self.config.train_frac
        val_frac = self.config.valid_frac
        total_frac = train_frac + val_frac
        if total_frac > 0:
            adjusted_train = train_frac / total_frac
        else:
            adjusted_train = 0.5

        n_train = int(total_remaining * adjusted_train)
        train = remaining[:n_train].tolist()
        valid = remaining[n_train:].tolist()

        return SplitResult(
            train_indices=jnp.array(train, dtype=jnp.int32),
            valid_indices=jnp.array(valid, dtype=jnp.int32),
            test_indices=jnp.array(test_indices, dtype=jnp.int32),
        )


@dataclass(frozen=True)
class FewShotSplitterConfig(SplitterConfig):
    """Configuration for FewShotSplitter.

    Attributes:
        held_out_perturbations: Perturbation names assigned to test.
        pert_col: Obs column name for perturbation identity.
        cell_type_col: Obs column name for cell type.
        control_pert: Label identifying control cells (always in train).
        val_subsample_fraction: Fraction of validation data to keep.
    """

    held_out_perturbations: tuple[str, ...] = ()
    pert_col: str = "perturbation"
    cell_type_col: str = "cell_type"
    control_pert: str = "non-targeting"
    val_subsample_fraction: float | None = None


class FewShotSplitter(SplitterModule):
    """Hold out specific perturbations for few-shot evaluation.

    Cells with held-out perturbations go to test. Control cells always
    go to train. Remaining perturbed cells are split between train and
    validation.

    Args:
        config: Splitter configuration.
        rngs: Optional RNG state.
        name: Optional module name.
    """

    def split(self, data_source: Any) -> SplitResult:
        """Split data source by perturbation holdout.

        Args:
            data_source: A PerturbationAnnDataSource or similar source.

        Returns:
            SplitResult with train/valid/test indices.
        """
        n = len(data_source)
        held_out = set(self.config.held_out_perturbations)
        control = self.config.control_pert

        test_indices: list[int] = []
        control_indices: list[int] = []
        remaining_indices: list[int] = []

        for i in range(n):
            elem = data_source[i]
            pert = elem.get(
                "pert_name",
                str(elem.get("obs", {}).get(self.config.pert_col, "")),
            )

            if pert in held_out:
                test_indices.append(i)
            elif pert == control:
                control_indices.append(i)
            else:
                remaining_indices.append(i)

        # Split remaining non-control, non-test cells into train/val
        rng = np.random.default_rng(self.config.seed)
        remaining = np.array(remaining_indices)
        rng.shuffle(remaining)

        train_frac = self.config.train_frac
        val_frac = self.config.valid_frac
        total_frac = train_frac + val_frac
        if total_frac > 0:
            adjusted_train = train_frac / total_frac
        else:
            adjusted_train = 0.5

        n_train = int(len(remaining) * adjusted_train)
        train_from_remaining = remaining[:n_train].tolist()
        valid_from_remaining = remaining[n_train:].tolist()

        # Controls always go to train
        train = control_indices + train_from_remaining

        # Apply validation subsample
        valid = valid_from_remaining
        if self.config.val_subsample_fraction is not None and len(valid) > 0:
            n_keep = max(1, int(len(valid) * self.config.val_subsample_fraction))
            valid = valid[:n_keep]

        return SplitResult(
            train_indices=jnp.array(train, dtype=jnp.int32),
            valid_indices=jnp.array(valid, dtype=jnp.int32),
            test_indices=jnp.array(test_indices, dtype=jnp.int32),
        )
