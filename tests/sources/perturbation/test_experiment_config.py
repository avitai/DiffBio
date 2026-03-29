"""Tests for ExperimentConfig and TOML loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from diffbio.sources.perturbation.experiment_config import (
    DatasetEntry,
    ExperimentConfig,
    FewshotEntry,
    ZeroshotEntry,
    load_experiment_config,
)


@pytest.fixture()
def basic_toml(tmp_path: Path) -> Path:
    """Create a basic TOML config file."""
    config = tmp_path / "experiment.toml"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset_a").mkdir()
    (data_dir / "dataset_b").mkdir()

    config.write_text(f"""\
[datasets]
dataset_a = "{data_dir / "dataset_a"}"
dataset_b = "{data_dir / "dataset_b"}"

[training]
dataset_a = "train"

[zeroshot]
"dataset_a.TypeX" = "test"
"dataset_b.TypeY" = "val"

[fewshot."dataset_a.TypeZ"]
val = ["GeneA", "GeneB"]
test = ["GeneC"]
""")
    return config


@pytest.fixture()
def minimal_toml(tmp_path: Path) -> Path:
    """Create a minimal TOML config with just datasets and training."""
    config = tmp_path / "minimal.toml"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "ds1").mkdir()

    config.write_text(f"""\
[datasets]
ds1 = "{data_dir / "ds1"}"

[training]
ds1 = "train"
""")
    return config


class TestLoadExperimentConfig:
    """Tests for load_experiment_config."""

    def test_parse_basic_toml(self, basic_toml: Path) -> None:
        config = load_experiment_config(basic_toml)
        assert isinstance(config, ExperimentConfig)

    def test_datasets_parsed(self, basic_toml: Path) -> None:
        config = load_experiment_config(basic_toml)
        assert len(config.datasets) == 2
        names = {d.name for d in config.datasets}
        assert names == {"dataset_a", "dataset_b"}

    def test_training_datasets(self, basic_toml: Path) -> None:
        config = load_experiment_config(basic_toml)
        assert config.training_datasets == ("dataset_a",)

    def test_zeroshot_entries(self, basic_toml: Path) -> None:
        config = load_experiment_config(basic_toml)
        assert len(config.zeroshot) == 2
        zs = {(z.dataset, z.cell_type, z.split) for z in config.zeroshot}
        assert ("dataset_a", "TypeX", "test") in zs
        assert ("dataset_b", "TypeY", "val") in zs

    def test_fewshot_entries(self, basic_toml: Path) -> None:
        config = load_experiment_config(basic_toml)
        assert len(config.fewshot) == 1
        fs = config.fewshot[0]
        assert fs.dataset == "dataset_a"
        assert fs.cell_type == "TypeZ"
        assert fs.val_perturbations == ("GeneA", "GeneB")
        assert fs.test_perturbations == ("GeneC",)

    def test_minimal_config(self, minimal_toml: Path) -> None:
        config = load_experiment_config(minimal_toml)
        assert len(config.datasets) == 1
        assert len(config.zeroshot) == 0
        assert len(config.fewshot) == 0

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_experiment_config(tmp_path / "nonexistent.toml")


class TestExperimentConfig:
    """Tests for ExperimentConfig methods."""

    def test_get_all_datasets(self) -> None:
        config = ExperimentConfig(
            datasets=(
                DatasetEntry(name="a", path="/a"),
                DatasetEntry(name="b", path="/b"),
            ),
            training_datasets=("a",),
            zeroshot=(ZeroshotEntry(dataset="b", cell_type="T", split="test"),),
            fewshot=(),
        )
        assert config.get_all_datasets() == {"a", "b"}

    def test_get_zeroshot_celltypes(self) -> None:
        config = ExperimentConfig(
            datasets=(DatasetEntry(name="ds", path="/ds"),),
            training_datasets=("ds",),
            zeroshot=(
                ZeroshotEntry(dataset="ds", cell_type="TypeA", split="test"),
                ZeroshotEntry(dataset="ds", cell_type="TypeB", split="val"),
                ZeroshotEntry(dataset="other", cell_type="TypeC", split="test"),
            ),
            fewshot=(),
        )
        result = config.get_zeroshot_celltypes("ds")
        assert result == {"TypeA": "test", "TypeB": "val"}

    def test_get_fewshot_celltypes(self) -> None:
        config = ExperimentConfig(
            datasets=(DatasetEntry(name="ds", path="/ds"),),
            training_datasets=("ds",),
            zeroshot=(),
            fewshot=(
                FewshotEntry(
                    dataset="ds",
                    cell_type="TypeA",
                    val_perturbations=("G1",),
                    test_perturbations=("G2", "G3"),
                ),
            ),
        )
        result = config.get_fewshot_celltypes("ds")
        assert "TypeA" in result
        assert result["TypeA"].val_perturbations == ("G1",)
        assert result["TypeA"].test_perturbations == ("G2", "G3")

    def test_validate_missing_paths_raises(self) -> None:
        config = ExperimentConfig(
            datasets=(DatasetEntry(name="ds", path="/nonexistent/path"),),
            training_datasets=("ds",),
            zeroshot=(ZeroshotEntry(dataset="missing", cell_type="T", split="test"),),
            fewshot=(),
        )
        with pytest.raises(ValueError, match="Missing dataset paths"):
            config.validate()

    def test_validate_invalid_split_raises(self) -> None:
        config = ExperimentConfig(
            datasets=(DatasetEntry(name="ds", path="/tmp"),),
            training_datasets=("ds",),
            zeroshot=(ZeroshotEntry(dataset="ds", cell_type="T", split="invalid"),),
            fewshot=(),
        )
        with pytest.raises(ValueError, match="Invalid split"):
            config.validate()

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        config = ExperimentConfig(
            datasets=(
                DatasetEntry(name="a", path="/path/a"),
                DatasetEntry(name="b", path="/path/b"),
            ),
            training_datasets=("a",),
            zeroshot=(ZeroshotEntry(dataset="a", cell_type="T1", split="test"),),
            fewshot=(
                FewshotEntry(
                    dataset="b",
                    cell_type="T2",
                    val_perturbations=("G1",),
                    test_perturbations=("G2",),
                ),
            ),
        )
        save_path = tmp_path / "saved.toml"
        config.save_config(save_path)
        assert save_path.exists()

        loaded = load_experiment_config(save_path)
        assert len(loaded.datasets) == len(config.datasets)
        assert loaded.training_datasets == config.training_datasets
        assert len(loaded.zeroshot) == len(config.zeroshot)
        assert len(loaded.fewshot) == len(config.fewshot)
