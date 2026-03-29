"""Experiment configuration for perturbation experiments.

Parses TOML configuration files following the cell-load schema and provides
frozen dataclass representations of experiment settings: dataset paths,
training assignments, zero-shot cell types, and few-shot perturbation splits.

References:
    - cell-load/src/cell_load/config.py (ExperimentConfig)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datarax.core.config import StructuralConfig

logger = logging.getLogger(__name__)


def _require_toml() -> Any:
    """Import tomllib (stdlib, Python 3.11+) or tomli fallback."""
    try:
        import tomllib  # noqa: PLC0415

        return tomllib
    except ImportError:
        import tomli  # noqa: PLC0415

        return tomli


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetEntry:
    """A single dataset in an experiment.

    Attributes:
        name: Identifier for the dataset.
        path: Filesystem path to the dataset directory or file.
    """

    name: str
    path: str


@dataclass(frozen=True)
class ZeroshotEntry:
    """A cell type held out entirely for zero-shot evaluation.

    Attributes:
        dataset: Name of the dataset containing this cell type.
        cell_type: Cell type to hold out.
        split: Target split (``"val"`` or ``"test"``).
    """

    dataset: str
    cell_type: str
    split: str


@dataclass(frozen=True)
class FewshotEntry:
    """Perturbations held out within a cell type for few-shot evaluation.

    Attributes:
        dataset: Name of the dataset containing this cell type.
        cell_type: Cell type within which perturbations are split.
        val_perturbations: Perturbation names assigned to validation.
        test_perturbations: Perturbation names assigned to testing.
    """

    dataset: str
    cell_type: str
    val_perturbations: tuple[str, ...] = ()
    test_perturbations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExperimentConfig(StructuralConfig):
    """Top-level experiment configuration.

    Encapsulates all settings parsed from a TOML file: dataset paths,
    training assignments, zero-shot cell type holdouts, and few-shot
    perturbation splits.

    Attributes:
        datasets: Registered dataset entries.
        training_datasets: Names of datasets used for training.
        zeroshot: Cell types held out entirely.
        fewshot: Perturbation-level splits within cell types.
    """

    datasets: tuple[DatasetEntry, ...] = ()
    training_datasets: tuple[str, ...] = ()
    zeroshot: tuple[ZeroshotEntry, ...] = ()
    fewshot: tuple[FewshotEntry, ...] = ()

    def get_all_datasets(self) -> set[str]:
        """Return the set of all dataset names referenced in the config."""
        names = set(self.training_datasets)
        for z in self.zeroshot:
            names.add(z.dataset)
        for f in self.fewshot:
            names.add(f.dataset)
        return names

    def get_zeroshot_celltypes(self, dataset: str) -> dict[str, str]:
        """Get zero-shot cell types and their target splits for a dataset.

        Args:
            dataset: Dataset name to filter by.

        Returns:
            Dict mapping cell type names to split names.
        """
        return {z.cell_type: z.split for z in self.zeroshot if z.dataset == dataset}

    def get_fewshot_celltypes(self, dataset: str) -> dict[str, FewshotEntry]:
        """Get few-shot cell type entries for a dataset.

        Args:
            dataset: Dataset name to filter by.

        Returns:
            Dict mapping cell type names to FewshotEntry objects.
        """
        return {f.cell_type: f for f in self.fewshot if f.dataset == dataset}

    def validate(self) -> None:
        """Validate configuration consistency.

        Raises:
            ValueError: If referenced datasets lack paths or splits are invalid.
        """
        all_referenced = self.get_all_datasets()
        dataset_names = {d.name for d in self.datasets}
        missing = all_referenced - dataset_names
        if missing:
            raise ValueError(f"Missing dataset paths for: {missing}")

        valid_splits = {"train", "val", "test"}
        for z in self.zeroshot:
            if z.split not in valid_splits:
                raise ValueError(
                    f"Invalid split '{z.split}' for zeroshot entry "
                    f"{z.dataset}.{z.cell_type}. Must be one of {valid_splits}"
                )

        logger.info("Configuration validation passed")

    def save_config(self, path: Path) -> None:
        """Save configuration to a TOML file.

        Args:
            path: Output file path.
        """
        lines: list[str] = ["[datasets]"]
        for d in self.datasets:
            lines.append(f'{d.name} = "{d.path}"')

        lines.append("")
        lines.append("[training]")
        for name in self.training_datasets:
            lines.append(f'{name} = "train"')

        if self.zeroshot:
            lines.append("")
            lines.append("[zeroshot]")
            for z in self.zeroshot:
                lines.append(f'"{z.dataset}.{z.cell_type}" = "{z.split}"')

        for f in self.fewshot:
            lines.append("")
            lines.append(f'[fewshot."{f.dataset}.{f.cell_type}"]')
            if f.val_perturbations:
                val_list = ", ".join(f'"{p}"' for p in f.val_perturbations)
                lines.append(f"val = [{val_list}]")
            if f.test_perturbations:
                test_list = ", ".join(f'"{p}"' for p in f.test_perturbations)
                lines.append(f"test = [{test_list}]")

        path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# TOML loading
# ---------------------------------------------------------------------------


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load an experiment configuration from a TOML file.

    Supports the cell-load TOML schema::

        [datasets]
        dataset_name = "/path/to/data"

        [training]
        dataset_name = "train"

        [zeroshot]
        "dataset.celltype" = "test"

        [fewshot."dataset.celltype"]
        val = ["Gene1"]
        test = ["Gene2"]

    Args:
        path: Path to the TOML file.

    Returns:
        Parsed ExperimentConfig.

    Raises:
        FileNotFoundError: If the TOML file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TOML config file not found: {path}")

    tomllib = _require_toml()
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    # Parse [datasets]
    datasets_raw: dict[str, str] = raw.get("datasets", {})
    datasets = tuple(DatasetEntry(name=name, path=p) for name, p in datasets_raw.items())

    # Parse [training]
    training_raw: dict[str, str] = raw.get("training", {})
    training_datasets = tuple(training_raw.keys())

    # Parse [zeroshot] -- keys are "dataset.celltype"
    zeroshot_raw: dict[str, str] = raw.get("zeroshot", {})
    zeroshot_entries: list[ZeroshotEntry] = []
    for key, split in zeroshot_raw.items():
        dataset, cell_type = key.split(".", 1)
        zeroshot_entries.append(ZeroshotEntry(dataset=dataset, cell_type=cell_type, split=split))

    # Parse [fewshot] -- keys are "dataset.celltype", values are {split: [perts]}
    fewshot_raw: dict[str, dict[str, list[str]]] = raw.get("fewshot", {})
    fewshot_entries: list[FewshotEntry] = []
    for key, pert_config in fewshot_raw.items():
        dataset, cell_type = key.split(".", 1)
        fewshot_entries.append(
            FewshotEntry(
                dataset=dataset,
                cell_type=cell_type,
                val_perturbations=tuple(pert_config.get("val", [])),
                test_perturbations=tuple(pert_config.get("test", [])),
            )
        )

    return ExperimentConfig(
        datasets=datasets,
        training_datasets=training_datasets,
        zeroshot=tuple(zeroshot_entries),
        fewshot=tuple(fewshot_entries),
    )
