"""Deterministic drug-target interaction sources and paired-input helpers."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.typing import Element

DTI_DATASET_CONTRACT_KEYS = (
    "pair_ids",
    "protein_ids",
    "protein_sequences",
    "drug_ids",
    "drug_smiles",
    "targets",
    "task_type",
    "dataset_provenance",
)
_DTI_PROVENANCE_KEYS = (
    "dataset_name",
    "split",
    "source_type",
    "source_path",
    "seed",
    "task_type",
    "n_pairs",
    "promotion_eligible",
    "biological_validation",
)


@dataclass(frozen=True, kw_only=True)
class DTISourceConfig(StructuralConfig):
    """Configuration for deterministic DTI sources."""

    dataset_name: Literal["davis", "biosnap"]
    split: Literal["train", "valid", "test"] = "train"
    data_dir: Path | None = None
    seed: int = 42
    use_synthetic_fallback: bool = True


def validate_dti_dataset(data: dict[str, Any]) -> None:
    """Validate the shared DTI paired-input source contract."""
    missing_keys = [key for key in DTI_DATASET_CONTRACT_KEYS if key not in data]
    if missing_keys:
        raise ValueError(f"DTI dataset is missing required keys: {missing_keys}")

    lengths = [
        len(data["pair_ids"]),
        len(data["protein_ids"]),
        len(data["protein_sequences"]),
        len(data["drug_ids"]),
        len(data["drug_smiles"]),
        int(np.asarray(data["targets"]).shape[0]),
    ]
    if len(set(lengths)) != 1:
        raise ValueError("All DTI paired-input fields must have the same length.")

    task_type = str(data["task_type"])
    if task_type not in {"affinity_regression", "binary_interaction"}:
        raise ValueError(f"Unsupported DTI task_type: {task_type!r}")

    targets = np.asarray(data["targets"])
    if targets.ndim != 1:
        raise ValueError("DTI targets must be a rank-1 array.")
    if task_type == "binary_interaction":
        unique_values = set(np.asarray(targets, dtype=np.int32).tolist())
        if not unique_values.issubset({0, 1}):
            raise ValueError("Binary DTI targets must contain only 0/1 labels.")

    provenance = data["dataset_provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("DTI dataset_provenance must be a dict.")
    missing_provenance_keys = [key for key in _DTI_PROVENANCE_KEYS if key not in provenance]
    if missing_provenance_keys:
        raise ValueError(
            f"DTI dataset_provenance is missing required keys: {missing_provenance_keys}"
        )
    if provenance["task_type"] != task_type:
        raise ValueError("DTI dataset_provenance.task_type must match task_type.")
    if int(provenance["n_pairs"]) != lengths[0]:
        raise ValueError("DTI dataset_provenance.n_pairs must match paired field length.")


def deterministic_dti_split(
    n_items: int,
    *,
    seed: int,
    train_frac: float = 2.0 / 3.0,
    valid_frac: float = 1.0 / 6.0,
) -> dict[str, np.ndarray]:
    """Build a deterministic train/valid/test split over interaction indices."""
    indices = np.arange(n_items, dtype=np.int32)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_train = int(round(n_items * train_frac))
    n_valid = int(round(n_items * valid_frac))
    n_train = min(max(n_train, 1), n_items - 2)
    n_valid = min(max(n_valid, 1), n_items - n_train - 1)
    n_test = n_items - n_train - n_valid

    return {
        "train": np.sort(indices[:n_train]),
        "valid": np.sort(indices[n_train : n_train + n_valid]),
        "test": np.sort(indices[n_train + n_valid : n_train + n_valid + n_test]),
    }


def build_paired_dti_batch(
    data: dict[str, Any],
    *,
    indices: np.ndarray,
) -> dict[str, Any]:
    """Slice a DTI dataset into one aligned paired batch."""
    validate_dti_dataset(data)
    batch_indices = np.asarray(indices, dtype=np.int32)
    provenance = dict(data["dataset_provenance"])
    provenance["source_n_pairs"] = int(provenance["n_pairs"])
    provenance["n_pairs"] = int(batch_indices.shape[0])
    batch = {
        "pair_ids": [data["pair_ids"][index] for index in batch_indices],
        "protein_ids": [data["protein_ids"][index] for index in batch_indices],
        "protein_sequences": [data["protein_sequences"][index] for index in batch_indices],
        "drug_ids": [data["drug_ids"][index] for index in batch_indices],
        "drug_smiles": [data["drug_smiles"][index] for index in batch_indices],
        "targets": jnp.asarray(np.asarray(data["targets"])[batch_indices]),
        "task_type": data["task_type"],
        "dataset_provenance": provenance,
    }
    validate_dti_dataset(batch)
    return batch


class _BaseDTISource(DataSourceModule):
    """Shared DTI source implementation for dataset-specific wrappers."""

    _data: list[Element] = nnx.data()

    def __init__(
        self,
        config: DTISourceConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(config, rngs=rngs, name=name)
        self._data = self._load_dataset()
        self._current_idx = 0

    def _load_dataset(self) -> list[Element]:
        records = self._load_records()
        split_indices = deterministic_dti_split(len(records), seed=self.config.seed)[
            self.config.split
        ]
        return [records[int(index)] for index in split_indices]

    def _load_records(self) -> list[Element]:
        dataset_path = _resolve_dataset_path(cast(DTISourceConfig, self.config))
        if dataset_path is not None:
            return _load_records_from_table(dataset_path, dataset_name=self.config.dataset_name)
        if not self.config.use_synthetic_fallback:
            raise FileNotFoundError(
                f"No local {self.config.dataset_name} table found under {self.config.data_dir!r}."
            )
        return _build_synthetic_records(dataset_name=self.config.dataset_name)

    def load(self) -> dict[str, Any]:
        """Load the paired DTI payload for the configured split."""
        data = {
            "pair_ids": [element.data["pair_id"] for element in self._data],
            "protein_ids": [element.data["protein_id"] for element in self._data],
            "protein_sequences": [element.data["protein_sequence"] for element in self._data],
            "drug_ids": [element.data["drug_id"] for element in self._data],
            "drug_smiles": [element.data["drug_smiles"] for element in self._data],
            "targets": jnp.asarray([element.data["target"] for element in self._data]),
            "task_type": self._data[0].data["task_type"],
        }
        data["dataset_provenance"] = _build_dataset_provenance(
            cast(DTISourceConfig, self.config),
            task_type=str(data["task_type"]),
            n_pairs=len(data["pair_ids"]),
        )
        validate_dti_dataset(data)
        return data

    def __len__(self) -> int:
        """Return the number of interactions in the configured split."""
        return len(self._data)

    def __getitem__(self, index: int) -> Element | None:
        """Return one interaction element or ``None`` if out of range."""
        if index < 0 or index >= len(self):
            return None
        return self._data[index]

    def __iter__(self) -> Iterator[Element]:
        """Iterate over paired DTI elements."""
        return iter(self._data)


class DavisDTISource(_BaseDTISource):
    """Deterministic affinity-regression source for the Davis DTI task."""


class BioSNAPDTISource(_BaseDTISource):
    """Deterministic binary-interaction source for the BioSNAP DTI task."""


def _resolve_dataset_path(config: DTISourceConfig) -> Path | None:
    """Return the first matching local dataset table if it exists."""
    if config.data_dir is None:
        return None

    candidates = [
        Path(config.data_dir) / f"{config.dataset_name}.csv",
        Path(config.data_dir) / f"{config.dataset_name}.tsv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _build_dataset_provenance(
    config: DTISourceConfig,
    *,
    task_type: str,
    n_pairs: int,
) -> dict[str, Any]:
    """Build one provenance record for the loaded DTI split."""
    dataset_path = _resolve_dataset_path(config)
    source_type = "local_table" if dataset_path is not None else "synthetic_scaffold"
    biological_validation = (
        "local_table_unverified" if dataset_path is not None else "contract_validation_only"
    )
    return {
        "dataset_name": config.dataset_name,
        "split": config.split,
        "source_type": source_type,
        "source_path": None if dataset_path is None else str(dataset_path),
        "seed": config.seed,
        "task_type": task_type,
        "n_pairs": int(n_pairs),
        "promotion_eligible": False,
        "biological_validation": biological_validation,
    }


def _load_records_from_table(
    dataset_path: Path,
    *,
    dataset_name: Literal["davis", "biosnap"],
) -> list[Element]:
    """Load paired DTI records from a local CSV/TSV table."""
    delimiter = "\t" if dataset_path.suffix == ".tsv" else ","
    task_type = "affinity_regression" if dataset_name == "davis" else "binary_interaction"
    target_columns = ("target", "affinity", "label")

    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        records: list[Element] = []
        for row_index, row in enumerate(reader):
            target_value = None
            for column_name in target_columns:
                if column_name in row and row[column_name] not in {"", None}:
                    target_value = float(row[column_name])
                    break
            if target_value is None:
                raise ValueError(f"Missing target column in {dataset_path} row {row_index}.")

            records.append(
                _build_element(
                    pair_id=row.get("pair_id", f"{dataset_name}_{row_index}"),
                    protein_id=row["protein_id"],
                    protein_sequence=row["protein_sequence"],
                    drug_id=row["drug_id"],
                    drug_smiles=row["drug_smiles"],
                    target=target_value,
                    task_type=task_type,
                )
            )
    return records


def _build_synthetic_records(
    *,
    dataset_name: Literal["davis", "biosnap"],
) -> list[Element]:
    """Build deterministic fallback paired DTI records."""
    proteins = (
        ("P0", "MKTAYI"),
        ("P1", "MNNQKLI"),
        ("P2", "MPEPTIDER"),
    )
    drugs = (
        ("D0", "CCO"),
        ("D1", "CCN"),
        ("D2", "CCCl"),
        ("D3", "c1ccccc1"),
    )

    records: list[Element] = []
    for protein_index, (protein_id, protein_sequence) in enumerate(proteins):
        for drug_index, (drug_id, drug_smiles) in enumerate(drugs):
            if dataset_name == "davis":
                target = 5.5 + 0.45 * protein_index + 0.3 * drug_index
                task_type = "affinity_regression"
            else:
                target = float((protein_index + drug_index) % 2 == 0)
                task_type = "binary_interaction"

            records.append(
                _build_element(
                    pair_id=f"{dataset_name}_{protein_id}_{drug_id}",
                    protein_id=protein_id,
                    protein_sequence=protein_sequence,
                    drug_id=drug_id,
                    drug_smiles=drug_smiles,
                    target=target,
                    task_type=task_type,
                )
            )
    return records


def _build_element(
    *,
    pair_id: str,
    protein_id: str,
    protein_sequence: str,
    drug_id: str,
    drug_smiles: str,
    target: float,
    task_type: str,
) -> Element:
    """Build one DTI interaction element."""
    return Element(
        data={
            "pair_id": pair_id,
            "protein_id": protein_id,
            "protein_sequence": protein_sequence,
            "drug_id": drug_id,
            "drug_smiles": drug_smiles,
            "target": target,
            "task_type": task_type,
        },
    )
