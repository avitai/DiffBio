"""End-to-end perturbation experiment data pipeline.

Orchestrates the full data setup workflow for single-cell perturbation
experiments: loading, QC filtering, train/val/test splitting, batch sampling,
and control cell mapping.

This pipeline produces ready-to-train data sources with paired
(perturbed, control) cell elements. It is a structural (non-differentiable)
setup pipeline — the differentiable training loop operates on its outputs.

References:
    - cell-load PerturbationDataModule
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from flax import nnx

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule

from diffbio.operators.singlecell.knockdown_filter import (
    KnockdownFilterConfig,
    OnTargetKnockdownFilter,
)
from diffbio.samplers.perturbation_sampler import (
    PerturbationBatchSampler,
    PerturbationSamplerConfig,
)
from diffbio.sources.perturbation.concat_source import PerturbationConcatSource
from diffbio.sources.perturbation.control_mapping import (
    BatchControlMapping,
    ControlMappingConfig,
    RandomControlMapping,
)
from diffbio.sources.perturbation.perturbation_source import (
    PerturbationAnnDataSource,
    PerturbationSourceConfig,
)
from diffbio.splitters.perturbation import (
    FewShotSplitter,
    FewShotSplitterConfig,
    ZeroShotSplitter,
    ZeroShotSplitterConfig,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerturbationPipelineConfig(StructuralConfig):
    """Configuration for PerturbationPipeline.

    Attributes:
        pert_col: Obs column for perturbation identity.
        cell_type_col: Obs column for cell type.
        batch_col: Obs column for batch/plate.
        control_pert: Label identifying control cells.
        output_space: Output representation (``"gene"``, ``"all"``,
            ``"embedding"``).
        embedding_key: Key in obsm for embeddings.
        hvg_col: Column in var marking HVGs.
        mapping_strategy: Control mapping strategy (``"batch"`` or
            ``"random"``).
        n_basal_samples: Number of controls per perturbed cell.
        sentence_size: Cells per sentence in batch sampler.
        sentences_per_batch: Sentences per batch.
        split_mode: Split strategy (``"zeroshot"``, ``"fewshot"``, or
            ``"random"``).
        held_out_cell_types: Cell types for zero-shot test set.
        held_out_perturbations: Perturbations for few-shot test set.
        train_frac: Training fraction.
        valid_frac: Validation fraction.
        enable_knockdown_filter: Whether to apply knockdown QC.
        residual_expression: Stage 1 knockdown threshold.
        cell_residual_expression: Stage 2 per-cell threshold.
        min_cells: Stage 3 minimum cells per perturbation.
        var_gene_col: Column in var for gene names (knockdown filter).
        seed: Global random seed.
    """

    pert_col: str = "perturbation"
    cell_type_col: str = "cell_type"
    batch_col: str = "batch"
    control_pert: str = "non-targeting"
    output_space: str = "all"
    embedding_key: str | None = None
    hvg_col: str | None = None
    mapping_strategy: str = "random"
    n_basal_samples: int = 1
    sentence_size: int = 512
    sentences_per_batch: int = 1
    split_mode: str = "random"
    held_out_cell_types: tuple[str, ...] = ()
    held_out_perturbations: tuple[str, ...] = ()
    train_frac: float = 0.8
    valid_frac: float = 0.1
    enable_knockdown_filter: bool = False
    residual_expression: float = 0.30
    cell_residual_expression: float = 0.50
    min_cells: int = 30
    var_gene_col: str | None = None
    seed: int = 42


class PerturbationPipeline:
    """End-to-end data setup pipeline for perturbation experiments.

    Orchestrates the full workflow from raw H5AD files to ready-to-train
    data sources with paired (perturbed, control) cell output.

    Workflow:
        1. Load H5AD file(s) into PerturbationAnnDataSource(s)
        2. (Optional) Apply on-target knockdown QC filter
        3. Split into train/val/test via zero-shot, few-shot, or random
        4. Build control cell mapping for each split
        5. Create batch sampler for training

    Example::

        config = PerturbationPipelineConfig(
            split_mode="zeroshot",
            held_out_cell_types=("TypeA",),
            mapping_strategy="batch",
        )
        pipeline = PerturbationPipeline(config)
        result = pipeline.setup(["path/to/data.h5ad"])

        for batch_indices in result.train_sampler:
            elements = [result.train_source[i] for i in batch_indices]
            paired = result.get_paired_batch(batch_indices, split="train")
    """

    def __init__(self, config: PerturbationPipelineConfig) -> None:
        self._config = config

    def setup(self, file_paths: list[str | Path]) -> PerturbationPipelineResult:
        """Execute the full data setup workflow.

        Args:
            file_paths: Paths to H5AD files to load.

        Returns:
            PerturbationPipelineResult with sources, splits, samplers,
            and control mappings.
        """
        config = self._config

        # 1. Load sources
        logger.info("Loading %d H5AD file(s)...", len(file_paths))
        sources = []
        for path in file_paths:
            src_config = PerturbationSourceConfig(
                file_path=str(path),
                pert_col=config.pert_col,
                cell_type_col=config.cell_type_col,
                batch_col=config.batch_col,
                control_pert=config.control_pert,
                output_space=config.output_space,
                embedding_key=config.embedding_key,
                hvg_col=config.hvg_col,
                seed=config.seed,
            )
            sources.append(PerturbationAnnDataSource(src_config))

        source: DataSourceModule
        if len(sources) == 1:
            source = sources[0]
        else:
            source = PerturbationConcatSource(sources=sources)

        # 2. Knockdown filter (optional)
        filter_mask: np.ndarray | None = None
        if config.enable_knockdown_filter:
            logger.info("Applying knockdown QC filter...")
            filt = OnTargetKnockdownFilter(
                KnockdownFilterConfig(
                    pert_col=config.pert_col,
                    control_pert=config.control_pert,
                    residual_expression=config.residual_expression,
                    cell_residual_expression=config.cell_residual_expression,
                    min_cells=config.min_cells,
                    var_gene_col=config.var_gene_col,
                )
            )
            filter_mask = filt.process(source)
            n_kept = int(filter_mask.sum())
            logger.info(
                "Knockdown filter: %d / %d cells pass",
                n_kept,
                len(filter_mask),
            )

        # 3. Split
        logger.info("Splitting data (mode=%s)...", config.split_mode)
        if config.split_mode == "zeroshot":
            splitter = ZeroShotSplitter(
                ZeroShotSplitterConfig(
                    held_out_cell_types=config.held_out_cell_types,
                    pert_col=config.pert_col,
                    cell_type_col=config.cell_type_col,
                    train_frac=config.train_frac,
                    valid_frac=config.valid_frac,
                    test_frac=1.0 - config.train_frac - config.valid_frac,
                    seed=config.seed,
                ),
                rngs=nnx.Rngs(config.seed),
            )
        elif config.split_mode == "fewshot":
            splitter = FewShotSplitter(
                FewShotSplitterConfig(
                    held_out_perturbations=config.held_out_perturbations,
                    pert_col=config.pert_col,
                    cell_type_col=config.cell_type_col,
                    control_pert=config.control_pert,
                    train_frac=config.train_frac,
                    valid_frac=config.valid_frac,
                    test_frac=1.0 - config.train_frac - config.valid_frac,
                    seed=config.seed,
                ),
                rngs=nnx.Rngs(config.seed),
            )
        else:
            from diffbio.splitters.random import (  # noqa: PLC0415
                RandomSplitter,
                RandomSplitterConfig,
            )

            splitter = RandomSplitter(
                RandomSplitterConfig(
                    train_frac=config.train_frac,
                    valid_frac=config.valid_frac,
                    test_frac=1.0 - config.train_frac - config.valid_frac,
                    seed=config.seed,
                ),
                rngs=nnx.Rngs(config.seed),
            )

        split_result = splitter.split(source)

        # Apply filter mask to split indices if knockdown filter was used
        if filter_mask is not None:
            passing = set(np.where(filter_mask)[0])
            train_idx = np.array([i for i in split_result.train_indices if int(i) in passing])
            valid_idx = np.array([i for i in split_result.valid_indices if int(i) in passing])
            test_idx = np.array([i for i in split_result.test_indices if int(i) in passing])
        else:
            train_idx = np.array(split_result.train_indices)
            valid_idx = np.array(split_result.valid_indices)
            test_idx = np.array(split_result.test_indices)

        logger.info(
            "Split sizes: train=%d, val=%d, test=%d",
            len(train_idx),
            len(valid_idx),
            len(test_idx),
        )

        # 4. Control mapping
        logger.info(
            "Building control mapping (strategy=%s)...",
            config.mapping_strategy,
        )
        mapping_config = ControlMappingConfig(
            strategy=config.mapping_strategy,
            n_basal_samples=config.n_basal_samples,
            seed=config.seed,
        )
        if config.mapping_strategy == "batch":
            mapper = BatchControlMapping(mapping_config)
        else:
            mapper = RandomControlMapping(mapping_config)

        control_mapping = mapper.build_mapping(source)

        # 5. Train sampler
        group_codes = source.get_group_codes()
        train_group_codes = group_codes[train_idx]
        sampler_config = PerturbationSamplerConfig(
            sentence_size=config.sentence_size,
            sentences_per_batch=config.sentences_per_batch,
            seed=config.seed,
        )
        train_sampler = PerturbationBatchSampler(sampler_config, train_group_codes)

        return PerturbationPipelineResult(
            source=source,
            train_indices=train_idx,
            valid_indices=valid_idx,
            test_indices=test_idx,
            control_mapping=control_mapping,
            train_sampler=train_sampler,
            filter_mask=filter_mask,
        )


class PerturbationPipelineResult:
    """Result of PerturbationPipeline.setup().

    Holds all artifacts needed for training: the data source, split indices,
    control mapping, and train sampler.

    Attributes:
        source: The underlying data source (single or concatenated).
        train_indices: Cell indices for training.
        valid_indices: Cell indices for validation.
        test_indices: Cell indices for testing.
        control_mapping: Array mapping perturbed cell index to control indices.
        train_sampler: Batch sampler for training iteration.
        filter_mask: Boolean QC mask (None if filtering was disabled).
    """

    def __init__(
        self,
        source: Any,
        train_indices: np.ndarray,
        valid_indices: np.ndarray,
        test_indices: np.ndarray,
        control_mapping: np.ndarray,
        train_sampler: PerturbationBatchSampler,
        filter_mask: np.ndarray | None = None,
    ) -> None:
        self.source = source
        self.train_indices = train_indices
        self.valid_indices = valid_indices
        self.test_indices = test_indices
        self.control_mapping = control_mapping
        self.train_sampler = train_sampler
        self.filter_mask = filter_mask

    def get_element(self, global_idx: int) -> dict[str, Any]:
        """Get a single cell element by global index.

        Args:
            global_idx: Index into the full source.

        Returns:
            Per-cell dictionary with counts and metadata.
        """
        return self.source[global_idx]

    def get_var_dims(self) -> dict[str, int]:
        """Return dimensionality info from the source."""
        return self.source.get_var_dims()
