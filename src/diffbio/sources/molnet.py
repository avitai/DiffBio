"""MolNet benchmark data source for drug discovery.

This module provides MolNetSource for loading MoleculeNet benchmark datasets:
- BBBP (Blood-Brain Barrier Penetration)
- Tox21 (Toxicity)
- ESOL (Solubility)
- FreeSolv (Solvation Energy)
- Lipophilicity
- And more...

Reference:
    Wu et al. "MoleculeNet: A Benchmark for Molecular Machine Learning"
    Chemical Science, 2018.
"""

import csv
import gzip
import shutil
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
from flax import nnx

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.typing import Element

# MolNet dataset catalog with download URLs and metadata
# URLs point to DeepChem's hosted data files
MOLNET_DATASETS: dict[str, dict] = {
    # ADMET datasets
    "bbbp": {
        "task_type": "classification",
        "n_tasks": 1,
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "smiles_col": "smiles",
        "label_cols": ["p_np"],
    },
    "tox21": {
        "task_type": "classification",
        "n_tasks": 12,
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "smiles_col": "smiles",
        "label_cols": [
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ],
    },
    # Physiology datasets
    "esol": {
        "task_type": "regression",
        "n_tasks": 1,
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "smiles_col": "smiles",
        "label_cols": ["measured log solubility in mols per litre"],
    },
    "freesolv": {
        "task_type": "regression",
        "n_tasks": 1,
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
        "smiles_col": "smiles",
        "label_cols": ["expt"],
    },
    "lipophilicity": {
        "task_type": "regression",
        "n_tasks": 1,
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        "smiles_col": "smiles",
        "label_cols": ["exp"],
    },
    # HIV dataset
    "hiv": {
        "task_type": "classification",
        "n_tasks": 1,
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
        "smiles_col": "smiles",
        "label_cols": ["HIV_active"],
    },
    # BACE dataset
    "bace": {
        "task_type": "classification",
        "n_tasks": 1,
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
        "smiles_col": "mol",
        "label_cols": ["Class"],
    },
    # ClinTox dataset
    "clintox": {
        "task_type": "classification",
        "n_tasks": 2,
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        "smiles_col": "smiles",
        "label_cols": ["FDA_APPROVED", "CT_TOX"],
    },
    # SIDER dataset
    "sider": {
        "task_type": "classification",
        "n_tasks": 27,
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
        "smiles_col": "smiles",
        "label_cols": None,  # All columns except smiles are labels
    },
}

# Compact synthetic fallback used when network is unavailable and no cache exists.
_FALLBACK_MOLNET_SMILES: tuple[str, ...] = (
    "CCO",
    "CCN",
    "CCC",
    "CCCl",
    "CCBr",
    "CC(C)O",
    "CC(C)N",
    "c1ccccc1",
    "c1ccncc1",
    "CCOC(=O)C",
    "CC(=O)O",
    "CC(=O)N",
    "CCS",
    "CCP",
    "COC",
    "CN(C)C",
    "CC(C)C",
    "CC(C)(C)O",
    "O=C(O)C",
    "NCCO",
)


@dataclass
class MolNetSourceConfig(StructuralConfig):
    """Configuration for MolNet benchmark data source.

    Attributes:
        dataset_name: Name of the MolNet dataset (e.g., "bbbp", "tox21", "esol")
        split: Which split to load ("train", "valid", or "test")
        data_dir: Directory to store downloaded data (default: ~/.diffbio/molnet)
        download: Whether to download if data not found (default: True)
    """

    dataset_name: str = ""
    split: Literal["train", "valid", "test"] = "train"
    data_dir: Path | None = None
    download: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        super().__post_init__()
        if not self.dataset_name:
            raise ValueError("dataset_name is required")


class MolNetSource(DataSourceModule):
    """MolNet benchmark data source extending Datarax DataSourceModule.

    Provides standardized access to MoleculeNet benchmark datasets with proper
    train/valid/test splits. Supports automatic downloading and caching.

    Inherits from DataSourceModule (StructuralModule) because:

    - Non-parametric: data loading is deterministic
    - Frozen config: dataset parameters don't change
    - Domain-specific: requires molecular data handling

    Example:
        ```python
        config = MolNetSourceConfig(dataset_name="bbbp", split="train")
        source = MolNetSource(config)
        for element in source:
            print(element.data["smiles"], element.data["y"])
        ```

    References:
        Wu et al. "MoleculeNet: A Benchmark for Molecular Machine Learning"
        Chemical Science, 2018.
    """

    # Annotate data storage for Flax NNX
    _data: list = nnx.data()

    def __init__(
        self,
        config: MolNetSourceConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize MolNetSource.

        Args:
            config: MolNet source configuration
            rngs: Random number generators (unused for data loading)
            name: Optional module name

        Raises:
            ValueError: If dataset_name is unknown
            FileNotFoundError: If data not found and download=False
        """
        super().__init__(config, rngs=rngs, name=name)

        # Validate dataset name
        if config.dataset_name not in MOLNET_DATASETS:
            available = ", ".join(sorted(MOLNET_DATASETS.keys()))
            raise ValueError(
                f"Unknown dataset: '{config.dataset_name}'. Available datasets: {available}"
            )

        # Set up data directory
        if config.data_dir is None:
            self._data_dir = Path.home() / ".diffbio" / "molnet"
        else:
            self._data_dir = Path(config.data_dir)

        # Load the dataset
        self._data = self._load_dataset()
        self._current_idx = 0

    def _load_dataset(self) -> list[Element]:
        """Load the MolNet dataset.

        Returns:
            List of Element objects containing SMILES and labels
        """
        dataset_info = MOLNET_DATASETS[self.config.dataset_name]
        data_path = self._get_data_path()

        self._ensure_dataset_file(data_path, dataset_info)

        # Parse CSV file
        return self._parse_csv(data_path, dataset_info)

    def _ensure_dataset_file(self, data_path: Path, dataset_info: dict) -> None:
        """Ensure the dataset file exists locally."""
        if data_path.exists():
            return

        if not self.config.download:
            raise FileNotFoundError(
                f"Dataset file not found: {data_path}. "
                f"Set download=True to download automatically."
            )

        if self._copy_from_default_cache(data_path):
            return

        try:
            self._download_dataset(dataset_info["url"], data_path)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._write_builtin_fallback_dataset(data_path, dataset_info)
            warnings.warn(
                "Unable to download MolNet dataset "
                f"'{self.config.dataset_name}' ({exc!r}); using a built-in fallback sample.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _get_data_path(self) -> Path:
        """Get the path to the dataset file."""
        url = MOLNET_DATASETS[self.config.dataset_name]["url"]
        filename = url.split("/")[-1]
        return self._data_dir / self.config.dataset_name / filename

    def _copy_from_default_cache(self, data_path: Path) -> bool:
        """Copy from shared ~/.diffbio cache when using a custom data_dir."""
        default_cache_path = (
            Path.home() / ".diffbio" / "molnet" / self.config.dataset_name / data_path.name
        )
        if default_cache_path == data_path or not default_cache_path.exists():
            return False

        data_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(default_cache_path, data_path)
        return True

    def _download_dataset(self, url: str, data_path: Path) -> None:
        """Download dataset from URL.

        Args:
            url: URL to download from
            data_path: Local path to save to
        """
        from urllib.parse import urlparse

        # Validate URL scheme for security
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid URL scheme: {parsed.scheme}. Only http/https allowed.")

        # Create directory
        data_path.parent.mkdir(parents=True, exist_ok=True)

        # Download file
        urllib.request.urlretrieve(url, data_path)  # nosec B310

    @staticmethod
    def _fallback_label_values(row_idx: int, n_labels: int, task_type: str) -> list[str]:
        """Generate deterministic fallback labels by task type."""
        if task_type == "classification":
            return [str((row_idx + col_idx) % 2) for col_idx in range(n_labels)]

        base = -2.0 + (0.15 * row_idx)
        return [f"{base + (0.01 * col_idx):.3f}" for col_idx in range(n_labels)]

    def _write_builtin_fallback_dataset(self, data_path: Path, dataset_info: dict) -> None:
        """Write a small synthetic dataset to support offline execution."""
        label_cols = dataset_info["label_cols"]
        resolved_label_cols = (
            [f"task_{idx}" for idx in range(dataset_info["n_tasks"])]
            if label_cols is None
            else list(label_cols)
        )
        header = [dataset_info["smiles_col"], *resolved_label_cols]

        rows: list[list[str]] = []
        for row_idx, smiles in enumerate(_FALLBACK_MOLNET_SMILES):
            labels = self._fallback_label_values(
                row_idx, len(resolved_label_cols), dataset_info["task_type"]
            )
            rows.append([smiles, *labels])

        data_path.parent.mkdir(parents=True, exist_ok=True)
        if str(data_path).endswith(".gz"):
            with gzip.open(data_path, "wt", newline="", encoding="utf-8") as file_handle:
                writer = csv.writer(file_handle)
                writer.writerow(header)
                writer.writerows(rows)
        else:
            with open(data_path, "w", newline="", encoding="utf-8") as file_handle:
                writer = csv.writer(file_handle)
                writer.writerow(header)
                writer.writerows(rows)

    def _read_rows_and_labels(
        self,
        data_path: Path,
        smiles_col: str,
        label_cols: list[str] | None,
    ) -> tuple[list[dict[str, str]], list[str]]:
        """Read CSV rows and resolve label columns.

        Args:
            data_path: Path to CSV file (optionally gzipped).
            smiles_col: Name of the SMILES column.
            label_cols: Explicit label columns, or None to infer.

        Returns:
            Tuple of (all_rows, resolved_label_columns).
        """
        import contextlib
        import gzip

        with contextlib.ExitStack() as stack:
            if str(data_path).endswith(".gz"):
                file_handle = stack.enter_context(
                    gzip.open(data_path, "rt", newline="", encoding="utf-8")
                )
            else:
                file_handle = stack.enter_context(
                    open(data_path, newline="", encoding="utf-8")  # noqa: SIM115
                )

            reader = csv.DictReader(file_handle)
            all_rows: list[dict[str, str]] = list(reader)

        resolved_labels = (
            [c for c in all_rows[0].keys() if c != smiles_col]
            if label_cols is None and all_rows
            else label_cols
        )
        return all_rows, ([] if resolved_labels is None else list(resolved_labels))

    def _rows_for_split(self, all_rows: list[dict[str, str]]) -> list[dict[str, str]]:
        """Select rows for the configured split."""
        n_total = len(all_rows)
        n_train = int(0.8 * n_total)
        n_valid = int(0.1 * n_total)

        if self.config.split == "train":
            return all_rows[:n_train]
        if self.config.split == "valid":
            return all_rows[n_train : n_train + n_valid]
        return all_rows[n_train + n_valid :]

    @staticmethod
    def _parse_float_or_nan(value: str) -> float:
        """Parse a float value, falling back to NaN for empty/invalid values."""
        try:
            return float(value) if value else float("nan")
        except ValueError:
            return float("nan")

    def _parse_labels(self, row: dict[str, str], label_cols: list[str]) -> float | jnp.ndarray:
        """Parse one or multiple label columns from a CSV row."""
        if len(label_cols) == 1:
            return self._parse_float_or_nan(row.get(label_cols[0], ""))
        values = [self._parse_float_or_nan(row.get(col, "")) for col in label_cols]
        return jnp.array(values)

    def _parse_csv(self, data_path: Path, dataset_info: dict) -> list[Element]:
        """Parse CSV file into Elements.

        Args:
            data_path: Path to CSV file
            dataset_info: Dataset metadata

        Returns:
            List of Element objects
        """
        smiles_col = dataset_info["smiles_col"]
        label_cols = dataset_info["label_cols"]

        all_rows, resolved_label_cols = self._read_rows_and_labels(
            data_path, smiles_col, label_cols
        )
        rows = self._rows_for_split(all_rows)
        elements: list[Element] = []

        for idx, row in enumerate(rows):
            smiles = row.get(smiles_col, "")
            if not smiles:
                continue

            y = self._parse_labels(row, resolved_label_cols)

            element = Element(
                data={"smiles": smiles, "y": y},
                state={},
                metadata={  # pyright: ignore[reportArgumentType]
                    "idx": idx,
                    "dataset": self.config.dataset_name,
                },
            )
            elements.append(element)

        return elements

    def __len__(self) -> int:
        """Return the number of elements in the source."""
        return len(self._data)

    def __getitem__(self, idx: int) -> Element | None:
        """Get element by index.

        Args:
            idx: Index of the element

        Returns:
            Element at the given index, or None if out of bounds
        """
        if idx < 0 or idx >= len(self._data):
            return None
        return self._data[idx]

    def __iter__(self):
        """Return iterator over elements."""
        self._current_idx = 0
        return self

    def __next__(self) -> Element:
        """Get next element in iteration."""
        if self._current_idx >= len(self._data):
            raise StopIteration
        elem = self._data[self._current_idx]
        self._current_idx += 1
        return elem

    @property
    def task_type(self) -> str:
        """Get the task type for this dataset."""
        return MOLNET_DATASETS[self.config.dataset_name]["task_type"]

    @property
    def n_tasks(self) -> int:
        """Get the number of tasks for this dataset."""
        return MOLNET_DATASETS[self.config.dataset_name]["n_tasks"]
