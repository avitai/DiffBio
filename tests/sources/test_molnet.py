"""Tests for MolNetSource benchmark data loader.

Test-driven development: These tests define the expected behavior.
Implementation must pass these tests without modification.
"""

import urllib.error

import pytest
from datarax.core.config import FrozenInstanceError


# =============================================================================
# Tests for MolNetSourceConfig
# =============================================================================


class TestMolNetSourceConfig:
    """Tests for MolNetSourceConfig."""

    def test_import(self):
        """Test that MolNetSourceConfig can be imported."""
        from diffbio.sources import MolNetSourceConfig

        assert MolNetSourceConfig is not None

    def test_default_values(self):
        """Test default configuration values."""
        from diffbio.sources import MolNetSourceConfig

        config = MolNetSourceConfig(dataset_name="bbbp")
        assert config.dataset_name == "bbbp"
        assert config.split == "train"
        assert config.data_dir is None
        assert config.download is True

    def test_custom_values(self):
        """Test custom configuration values."""
        from pathlib import Path

        from diffbio.sources import MolNetSourceConfig

        config = MolNetSourceConfig(
            dataset_name="tox21",
            split="test",
            data_dir=Path("/tmp/molnet"),
            download=False,
        )
        assert config.dataset_name == "tox21"
        assert config.split == "test"
        assert config.data_dir == Path("/tmp/molnet")
        assert config.download is False

    def test_frozen(self):
        """Test that config is frozen after creation."""
        from diffbio.sources import MolNetSourceConfig

        config = MolNetSourceConfig(dataset_name="bbbp")

        with pytest.raises(FrozenInstanceError):
            config.dataset_name = "tox21"


# =============================================================================
# Tests for MolNetSource Basic Functionality
# =============================================================================


class TestMolNetSourceBasic:
    """Basic functionality tests for MolNetSource."""

    def test_import(self):
        """Test that MolNetSource can be imported."""
        from diffbio.sources import MolNetSource, MolNetSourceConfig

        assert MolNetSource is not None
        assert MolNetSourceConfig is not None

    def test_dataset_catalog(self):
        """Test that MOLNET_DATASETS catalog exists and has expected datasets."""
        from diffbio.sources.molnet import MOLNET_DATASETS

        # Check some key datasets exist
        assert "bbbp" in MOLNET_DATASETS
        assert "tox21" in MOLNET_DATASETS
        assert "esol" in MOLNET_DATASETS
        assert "freesolv" in MOLNET_DATASETS
        assert "lipophilicity" in MOLNET_DATASETS

    def test_dataset_info_structure(self):
        """Test that dataset info has required fields."""
        from diffbio.sources.molnet import MOLNET_DATASETS

        for name, info in MOLNET_DATASETS.items():
            assert "task_type" in info, f"Missing task_type for {name}"
            assert "n_tasks" in info, f"Missing n_tasks for {name}"
            assert "url" in info, f"Missing url for {name}"
            assert info["task_type"] in [
                "classification",
                "regression",
            ], f"Invalid task_type for {name}"

    def test_unknown_dataset_raises(self):
        """Test that unknown dataset name raises ValueError."""
        from diffbio.sources import MolNetSource, MolNetSourceConfig

        config = MolNetSourceConfig(dataset_name="unknown_dataset_xyz")

        with pytest.raises(ValueError, match="Unknown dataset"):
            MolNetSource(config)


# =============================================================================
# Tests for MolNetSource Data Loading (requires network)
# =============================================================================


@pytest.mark.network
class TestMolNetSourceDataLoading:
    """Tests that require network access to download data."""

    @pytest.fixture
    def bbbp_source(self, tmp_path):
        """Create a BBBP source for testing."""
        from diffbio.sources import MolNetSource, MolNetSourceConfig

        config = MolNetSourceConfig(
            dataset_name="bbbp",
            split="train",
            data_dir=tmp_path,
            download=True,
        )
        return MolNetSource(config)

    def test_len_returns_positive(self, bbbp_source):
        """Test that source has positive length."""
        assert len(bbbp_source) > 0

    def test_getitem_returns_element(self, bbbp_source):
        """Test that indexing returns valid Element."""
        from datarax.typing import Element

        elem = bbbp_source[0]
        assert elem is not None
        assert isinstance(elem, Element)

    def test_element_has_smiles(self, bbbp_source):
        """Test that elements have SMILES string."""
        elem = bbbp_source[0]
        assert "smiles" in elem.data
        assert isinstance(elem.data["smiles"], str)
        assert len(elem.data["smiles"]) > 0

    def test_element_has_label(self, bbbp_source):
        """Test that elements have label."""
        elem = bbbp_source[0]
        assert "y" in elem.data

    def test_iteration(self, bbbp_source):
        """Test iteration over source."""
        count = 0
        for elem in bbbp_source:
            assert "smiles" in elem.data
            count += 1
            if count >= 10:  # Just test first 10
                break
        assert count == 10

    def test_out_of_bounds_returns_none(self, bbbp_source):
        """Test out of bounds indexing returns None."""
        assert bbbp_source[-1] is None
        assert bbbp_source[len(bbbp_source) + 100] is None


@pytest.mark.network
class TestMolNetSourceSplits:
    """Tests for different data splits."""

    def test_train_valid_test_splits_exist(self, tmp_path):
        """Test that train/valid/test splits can be loaded."""
        from diffbio.sources import MolNetSource, MolNetSourceConfig

        for split in ["train", "valid", "test"]:
            config = MolNetSourceConfig(
                dataset_name="bbbp",
                split=split,  # pyright: ignore[reportArgumentType]
                data_dir=tmp_path,
                download=True,
            )
            source = MolNetSource(config)
            assert len(source) > 0, f"Split '{split}' has no data"

    def test_splits_have_different_sizes(self, tmp_path):
        """Test that splits have different sizes (not identical)."""
        from diffbio.sources import MolNetSource, MolNetSourceConfig

        sizes = {}
        for split in ["train", "valid", "test"]:
            config = MolNetSourceConfig(
                dataset_name="bbbp",
                split=split,  # pyright: ignore[reportArgumentType]
                data_dir=tmp_path,
                download=True,
            )
            source = MolNetSource(config)
            sizes[split] = len(source)

        # Train should be largest
        assert sizes["train"] >= sizes["valid"]
        assert sizes["train"] >= sizes["test"]


@pytest.mark.network
class TestMolNetSourceMultipleDatasets:
    """Tests for different datasets."""

    @pytest.mark.parametrize("dataset_name", ["bbbp", "esol", "freesolv"])
    def test_load_dataset(self, tmp_path, dataset_name):
        """Test loading different datasets."""
        from diffbio.sources import MolNetSource, MolNetSourceConfig

        config = MolNetSourceConfig(
            dataset_name=dataset_name,
            split="train",
            data_dir=tmp_path,
            download=True,
        )
        source = MolNetSource(config)

        assert len(source) > 0
        elem = source[0]
        assert "smiles" in elem.data
        assert "y" in elem.data


# =============================================================================
# Tests for MolNetSource Integration
# =============================================================================


@pytest.mark.network
class TestMolNetSourceIntegration:
    """Integration tests with splitters."""

    def test_integration_with_random_splitter(self, tmp_path):
        """Test that MolNetSource works with RandomSplitter."""
        from diffbio.sources import MolNetSource, MolNetSourceConfig
        from diffbio.splitters import RandomSplitter, RandomSplitterConfig

        # Load dataset
        source_config = MolNetSourceConfig(
            dataset_name="bbbp",
            split="train",
            data_dir=tmp_path,
            download=True,
        )
        source = MolNetSource(source_config)

        # Split with RandomSplitter
        splitter_config = RandomSplitterConfig(seed=42)
        splitter = RandomSplitter(splitter_config)
        result = splitter.split(source)

        # Verify split covers all data
        total = result.train_size + result.valid_size + result.test_size
        assert total == len(source)

    def test_integration_with_scaffold_splitter(self, tmp_path):
        """Test that MolNetSource works with ScaffoldSplitter."""
        pytest.importorskip("rdkit")
        from diffbio.sources import MolNetSource, MolNetSourceConfig
        from diffbio.splitters import ScaffoldSplitter, ScaffoldSplitterConfig

        # Load dataset
        source_config = MolNetSourceConfig(
            dataset_name="bbbp",
            split="train",
            data_dir=tmp_path,
            download=True,
        )
        source = MolNetSource(source_config)

        # Split with ScaffoldSplitter
        splitter_config = ScaffoldSplitterConfig()
        splitter = ScaffoldSplitter(splitter_config)
        result = splitter.split(source)

        # Verify split covers all data
        total = result.train_size + result.valid_size + result.test_size
        assert total == len(source)


# =============================================================================
# Tests for Local/Cached Data
# =============================================================================


class TestMolNetSourceCaching:
    """Tests for data caching functionality."""

    @pytest.mark.network
    def test_data_cached_after_download(self, tmp_path):
        """Test that data is cached after first download."""
        from diffbio.sources import MolNetSource, MolNetSourceConfig

        # First load - downloads
        config1 = MolNetSourceConfig(
            dataset_name="bbbp",
            split="train",
            data_dir=tmp_path,
            download=True,
        )
        source1 = MolNetSource(config1)
        len1 = len(source1)

        # Second load - should use cache
        config2 = MolNetSourceConfig(
            dataset_name="bbbp",
            split="train",
            data_dir=tmp_path,
            download=False,  # Don't download again
        )
        source2 = MolNetSource(config2)
        len2 = len(source2)

        assert len1 == len2

    def test_missing_data_no_download_raises(self, tmp_path):
        """Test that missing data with download=False raises error."""
        from diffbio.sources import MolNetSource, MolNetSourceConfig

        config = MolNetSourceConfig(
            dataset_name="bbbp",
            split="train",
            data_dir=tmp_path / "nonexistent",
            download=False,
        )

        with pytest.raises((FileNotFoundError, ValueError)):
            MolNetSource(config)


class TestMolNetSourceOfflineBehavior:
    """Tests for deterministic offline behavior."""

    def test_falls_back_to_builtin_data_when_download_fails(self, tmp_path, monkeypatch):
        """Test fallback dataset is used when network download fails."""
        from diffbio.sources import MolNetSource, MolNetSourceConfig
        from diffbio.sources import molnet as molnet_module

        def _raise_url_error(*_args, **_kwargs):
            raise urllib.error.URLError("offline")

        monkeypatch.setattr(molnet_module.MolNetSource, "_download_dataset", _raise_url_error)

        config = MolNetSourceConfig(
            dataset_name="freesolv",
            split="train",
            data_dir=tmp_path,
            download=True,
        )
        with pytest.warns(RuntimeWarning, match="built-in fallback sample"):
            source = MolNetSource(config)

        fallback_path = tmp_path / "freesolv" / "SAMPL.csv"
        assert fallback_path.exists()
        assert len(source) == 16  # 80% of the 20-row fallback dataset
        first = source[0]
        assert first is not None
        assert "smiles" in first.data
        assert "y" in first.data

    def test_prefers_default_cache_before_network(self, tmp_path, monkeypatch):
        """Test source copies cached dataset from ~/.diffbio before downloading."""
        from diffbio.sources import MolNetSource, MolNetSourceConfig
        from diffbio.sources import molnet as molnet_module

        fake_home = tmp_path / "fake_home"
        cached_path = fake_home / ".diffbio" / "molnet" / "esol" / "delaney-processed.csv"
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        cached_path.write_text(
            "smiles,measured log solubility in mols per litre\nCCO,-0.50\nCCN,-0.20\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(
            molnet_module.Path,
            "home",
            classmethod(lambda _cls: fake_home),
        )

        def _fail_download(*_args, **_kwargs):
            raise AssertionError("Download should not be called when default cache exists.")

        monkeypatch.setattr(molnet_module.MolNetSource, "_download_dataset", _fail_download)

        local_data_dir = tmp_path / "local_molnet"
        config = MolNetSourceConfig(
            dataset_name="esol",
            split="train",
            data_dir=local_data_dir,
            download=True,
        )
        source = MolNetSource(config)

        copied_path = local_data_dir / "esol" / "delaney-processed.csv"
        assert copied_path.exists()
        assert copied_path.read_text(encoding="utf-8") == cached_path.read_text(encoding="utf-8")
        assert len(source) == 1
