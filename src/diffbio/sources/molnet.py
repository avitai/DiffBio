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
import urllib.request
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
        >>> config = MolNetSourceConfig(dataset_name="bbbp", split="train")
        >>> source = MolNetSource(config)
        >>> for element in source:
        ...     print(element.data["smiles"], element.data["y"])

    References:
        Wu et al. "MoleculeNet: A Benchmark for Molecular Machine Learning"
        Chemical Science, 2018.
    """

    # Annotate data storage for Flax NNX
    _data: nnx.Data[list]

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

        # Download if needed
        if not data_path.exists():
            if self.config.download:
                self._download_dataset(dataset_info["url"], data_path)
            else:
                raise FileNotFoundError(
                    f"Dataset file not found: {data_path}. "
                    f"Set download=True to download automatically."
                )

        # Parse CSV file
        return self._parse_csv(data_path, dataset_info)

    def _get_data_path(self) -> Path:
        """Get the path to the dataset file."""
        url = MOLNET_DATASETS[self.config.dataset_name]["url"]
        filename = url.split("/")[-1]
        return self._data_dir / self.config.dataset_name / filename

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

    def _parse_csv(self, data_path: Path, dataset_info: dict) -> list[Element]:
        """Parse CSV file into Elements.

        Args:
            data_path: Path to CSV file
            dataset_info: Dataset metadata

        Returns:
            List of Element objects
        """
        import gzip

        smiles_col = dataset_info["smiles_col"]
        label_cols = dataset_info["label_cols"]

        elements: list[Element] = []

        # Handle gzipped files
        if str(data_path).endswith(".gz"):
            file_handle = gzip.open(data_path, "rt", newline="", encoding="utf-8")
        else:
            file_handle = open(data_path, newline="", encoding="utf-8")

        try:
            reader = csv.DictReader(file_handle)

            # If label_cols is None, use all columns except smiles
            if label_cols is None:
                first_row = next(reader)
                label_cols = [c for c in first_row.keys() if c != smiles_col]
                # Reset reader
                file_handle.seek(0)
                reader = csv.DictReader(file_handle)

            all_rows = list(reader)
        finally:
            file_handle.close()

        # Apply random split (80/10/10)
        n_total = len(all_rows)
        n_train = int(0.8 * n_total)
        n_valid = int(0.1 * n_total)

        if self.config.split == "train":
            rows = all_rows[:n_train]
        elif self.config.split == "valid":
            rows = all_rows[n_train : n_train + n_valid]
        else:  # test
            rows = all_rows[n_train + n_valid :]

        for idx, row in enumerate(rows):
            smiles = row.get(smiles_col, "")
            if not smiles:
                continue

            # Extract labels
            if len(label_cols) == 1:
                label_val = row.get(label_cols[0], "")
                try:
                    y = float(label_val) if label_val else float("nan")
                except ValueError:
                    y = float("nan")
            else:
                y = []
                for col in label_cols:
                    val = row.get(col, "")
                    try:
                        y.append(float(val) if val else float("nan"))
                    except ValueError:
                        y.append(float("nan"))
                y = jnp.array(y)

            element = Element(
                data={"smiles": smiles, "y": y},
                state={},
                metadata={"idx": idx, "dataset": self.config.dataset_name},
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
