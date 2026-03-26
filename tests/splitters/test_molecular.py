"""Tests for molecular splitters (ScaffoldSplitter, TanimotoClusterSplitter).

Test-driven development: These tests define the expected behavior.
Implementation must pass these tests without modification.
"""

import jax.numpy as jnp
import pytest
from flax import nnx


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_molecules():
    """Simple molecules with known scaffolds."""
    return [
        "c1ccccc1",  # benzene - scaffold: benzene
        "c1ccc(O)cc1",  # phenol - scaffold: benzene
        "c1ccc(C)cc1",  # toluene - scaffold: benzene
        "C1CCCCC1",  # cyclohexane - scaffold: cyclohexane
        "C1CCC(O)CC1",  # cyclohexanol - scaffold: cyclohexane
        "c1ccc2ccccc2c1",  # naphthalene - scaffold: naphthalene
        "CC",  # ethane - no ring scaffold
        "CCC",  # propane - no ring scaffold
        "CCCC",  # butane - no ring scaffold
    ]


@pytest.fixture
def drug_like_molecules():
    """Drug-like molecules for more realistic testing."""
    return [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin - salicylic acid scaffold
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen - phenylpropanoic acid scaffold
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine - xanthine scaffold
        "CC(=O)Nc1ccc(cc1)O",  # Acetaminophen - aminophenol scaffold
        "c1ccc(cc1)C(=O)O",  # Benzoic acid - benzene scaffold
        "OC(=O)c1ccccc1O",  # Salicylic acid - salicylic acid scaffold
    ]


@pytest.fixture
def molecule_data_source(simple_molecules):
    """Create a mock DataSourceModule with molecular data."""
    from tests.mocks import MockDataSource, MockSourceConfig
    from datarax.typing import Element

    elements = [
        Element(data={"smiles": smiles, "value": jnp.array(i)}, state={}, metadata={"idx": i})  # pyright: ignore[reportArgumentType]
        for i, smiles in enumerate(simple_molecules)
    ]

    config = MockSourceConfig()
    return MockDataSource(config, elements)


@pytest.fixture
def drug_data_source(drug_like_molecules):
    """Create a mock DataSourceModule with drug-like molecules."""
    from tests.mocks import MockDataSource, MockSourceConfig
    from datarax.typing import Element

    elements = [
        Element(data={"smiles": smiles, "value": jnp.array(i)}, state={}, metadata={"idx": i})  # pyright: ignore[reportArgumentType]
        for i, smiles in enumerate(drug_like_molecules)
    ]

    config = MockSourceConfig()
    return MockDataSource(config, elements)


# =============================================================================
# Tests for ScaffoldSplitter
# =============================================================================


class TestScaffoldSplitter:
    """Tests for ScaffoldSplitter."""

    def test_import(self):
        """Test that ScaffoldSplitter can be imported."""
        from diffbio.splitters import ScaffoldSplitter, ScaffoldSplitterConfig

        assert ScaffoldSplitter is not None
        assert ScaffoldSplitterConfig is not None

    def test_requires_rdkit(self):
        """Test that ScaffoldSplitter imports RDKit."""
        # This test verifies RDKit is available
        pytest.importorskip("rdkit")

        from diffbio.splitters import ScaffoldSplitter, ScaffoldSplitterConfig

        config = ScaffoldSplitterConfig()
        # Should not raise ImportError
        _splitter = ScaffoldSplitter(config)

    def test_config_smiles_key(self):
        """Test that config has smiles_key parameter."""
        from diffbio.splitters import ScaffoldSplitterConfig

        config = ScaffoldSplitterConfig(smiles_key="mol_smiles")
        assert config.smiles_key == "mol_smiles"

    def test_split_returns_valid_result(self, molecule_data_source):
        """Test that split returns a valid SplitResult."""
        pytest.importorskip("rdkit")
        from diffbio.splitters import ScaffoldSplitter, ScaffoldSplitterConfig, SplitResult

        config = ScaffoldSplitterConfig()
        splitter = ScaffoldSplitter(config)
        result = splitter.split(molecule_data_source)

        assert isinstance(result, SplitResult)
        assert result.train_size + result.valid_size + result.test_size == len(molecule_data_source)

    def test_split_no_overlap(self, molecule_data_source):
        """Test that scaffold splits have no overlapping indices."""
        pytest.importorskip("rdkit")
        from diffbio.splitters import ScaffoldSplitter, ScaffoldSplitterConfig

        config = ScaffoldSplitterConfig()
        splitter = ScaffoldSplitter(config)
        result = splitter.split(molecule_data_source)

        all_indices = jnp.concatenate(
            [result.train_indices, result.valid_indices, result.test_indices]
        )

        assert len(jnp.unique(all_indices)) == len(all_indices)

    def test_split_covers_all_data(self, molecule_data_source):
        """Test that scaffold splits cover all data."""
        pytest.importorskip("rdkit")
        from diffbio.splitters import ScaffoldSplitter, ScaffoldSplitterConfig

        config = ScaffoldSplitterConfig()
        splitter = ScaffoldSplitter(config)
        result = splitter.split(molecule_data_source)

        all_indices = jnp.concatenate(
            [result.train_indices, result.valid_indices, result.test_indices]
        )

        assert len(all_indices) == len(molecule_data_source)

    def test_same_scaffold_in_same_split(self, molecule_data_source):
        """Test that molecules with same scaffold end up in same split."""
        pytest.importorskip("rdkit")
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        from diffbio.splitters import ScaffoldSplitter, ScaffoldSplitterConfig

        config = ScaffoldSplitterConfig()
        splitter = ScaffoldSplitter(config)
        result = splitter.split(molecule_data_source)

        # Get scaffolds for each molecule
        def get_scaffold(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            try:
                return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            except Exception:
                return ""

        # Map indices to scaffolds
        train_scaffolds = set()
        for idx in result.train_indices:
            smiles = molecule_data_source[int(idx)].data["smiles"]
            scaffold = get_scaffold(smiles)
            if scaffold:
                train_scaffolds.add(scaffold)

        valid_scaffolds = set()
        for idx in result.valid_indices:
            smiles = molecule_data_source[int(idx)].data["smiles"]
            scaffold = get_scaffold(smiles)
            if scaffold:
                valid_scaffolds.add(scaffold)

        test_scaffolds = set()
        for idx in result.test_indices:
            smiles = molecule_data_source[int(idx)].data["smiles"]
            scaffold = get_scaffold(smiles)
            if scaffold:
                test_scaffolds.add(scaffold)

        # Scaffolds should not overlap between splits
        # (except empty scaffolds which can appear in multiple splits)
        non_empty_train = {s for s in train_scaffolds if s}
        non_empty_valid = {s for s in valid_scaffolds if s}
        non_empty_test = {s for s in test_scaffolds if s}

        # Check no overlap between train-valid, train-test, valid-test
        assert len(non_empty_train & non_empty_valid) == 0
        assert len(non_empty_train & non_empty_test) == 0
        assert len(non_empty_valid & non_empty_test) == 0

    def test_custom_smiles_key(self, molecule_data_source):
        """Test using custom smiles_key."""
        pytest.importorskip("rdkit")
        from dataclasses import dataclass

        from datarax.core.config import StructuralConfig
        from datarax.core.data_source import DataSourceModule
        from datarax.typing import Element

        from diffbio.splitters import ScaffoldSplitter, ScaffoldSplitterConfig

        # Create source with different key name
        @dataclass(frozen=True)
        class MockConfig(StructuralConfig):
            pass

        class CustomSource(DataSourceModule):
            _data: list = nnx.data()

            def __init__(self, config, *, rngs=None, name=None):
                super().__init__(config, rngs=rngs, name=name)
                self._data = [
                    Element(data={"mol": "c1ccccc1"}, state={}, metadata={}),  # pyright: ignore[reportArgumentType]
                    Element(data={"mol": "C1CCCCC1"}, state={}, metadata={}),  # pyright: ignore[reportArgumentType]
                    Element(data={"mol": "CC"}, state={}, metadata={}),  # pyright: ignore[reportArgumentType]
                ]
                self._idx = 0

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx] if 0 <= idx < len(self._data) else None

            def __iter__(self):
                self._idx = 0
                return self

            def __next__(self):
                if self._idx >= len(self._data):
                    raise StopIteration
                elem = self._data[self._idx]
                self._idx += 1
                return elem

        config = ScaffoldSplitterConfig(
            smiles_key="mol", train_frac=0.5, valid_frac=0.25, test_frac=0.25
        )
        splitter = ScaffoldSplitter(config)
        source = CustomSource(MockConfig())

        result = splitter.split(source)
        assert result.train_size + result.valid_size + result.test_size == 3


class TestScaffoldSplitterEdgeCases:
    """Edge case tests for ScaffoldSplitter."""

    def test_single_scaffold(self):
        """Test with all molecules having same scaffold."""
        pytest.importorskip("rdkit")
        from dataclasses import dataclass

        from datarax.core.config import StructuralConfig
        from datarax.core.data_source import DataSourceModule
        from datarax.typing import Element

        from diffbio.splitters import ScaffoldSplitter, ScaffoldSplitterConfig

        @dataclass(frozen=True)
        class MockConfig(StructuralConfig):
            pass

        class SingleScaffoldSource(DataSourceModule):
            _data: list = nnx.data()

            def __init__(self, config, *, rngs=None, name=None):
                super().__init__(config, rngs=rngs, name=name)
                # All benzene derivatives
                self._data = [
                    Element(data={"smiles": "c1ccccc1"}, state={}, metadata={}),  # pyright: ignore[reportArgumentType]
                    Element(data={"smiles": "c1ccc(O)cc1"}, state={}, metadata={}),  # pyright: ignore[reportArgumentType]
                    Element(data={"smiles": "c1ccc(C)cc1"}, state={}, metadata={}),  # pyright: ignore[reportArgumentType]
                    Element(data={"smiles": "c1ccc(N)cc1"}, state={}, metadata={}),  # pyright: ignore[reportArgumentType]
                ]
                self._idx = 0

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx] if 0 <= idx < len(self._data) else None

            def __iter__(self):
                self._idx = 0
                return self

            def __next__(self):
                if self._idx >= len(self._data):
                    raise StopIteration
                elem = self._data[self._idx]
                self._idx += 1
                return elem

        config = ScaffoldSplitterConfig()
        splitter = ScaffoldSplitter(config)
        source = SingleScaffoldSource(MockConfig())

        result = splitter.split(source)

        # All should end up in same split (train, as it's assigned first)
        assert result.train_size == 4
        assert result.valid_size == 0
        assert result.test_size == 0

    def test_invalid_smiles_handled(self):
        """Test that invalid SMILES are handled gracefully."""
        pytest.importorskip("rdkit")
        from dataclasses import dataclass

        from datarax.core.config import StructuralConfig
        from datarax.core.data_source import DataSourceModule
        from datarax.typing import Element

        from diffbio.splitters import ScaffoldSplitter, ScaffoldSplitterConfig

        @dataclass(frozen=True)
        class MockConfig(StructuralConfig):
            pass

        class MixedValiditySource(DataSourceModule):
            _data: list = nnx.data()

            def __init__(self, config, *, rngs=None, name=None):
                super().__init__(config, rngs=rngs, name=name)
                self._data = [
                    Element(data={"smiles": "c1ccccc1"}, state={}, metadata={}),  # pyright: ignore[reportArgumentType]  # Valid
                    Element(data={"smiles": "invalid_smiles"}, state={}, metadata={}),  # pyright: ignore[reportArgumentType]  # Invalid
                    Element(data={"smiles": "CC"}, state={}, metadata={}),  # pyright: ignore[reportArgumentType]  # Valid
                ]
                self._idx = 0

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx] if 0 <= idx < len(self._data) else None

            def __iter__(self):
                self._idx = 0
                return self

            def __next__(self):
                if self._idx >= len(self._data):
                    raise StopIteration
                elem = self._data[self._idx]
                self._idx += 1
                return elem

        config = ScaffoldSplitterConfig(train_frac=0.6, valid_frac=0.2, test_frac=0.2)
        splitter = ScaffoldSplitter(config)
        source = MixedValiditySource(MockConfig())

        # Should not raise, should handle invalid SMILES gracefully
        result = splitter.split(source)
        total = result.train_size + result.valid_size + result.test_size
        assert total == 3  # All molecules should be assigned somewhere


# =============================================================================
# Tests for TanimotoClusterSplitter
# =============================================================================


class TestTanimotoClusterSplitter:
    """Tests for TanimotoClusterSplitter."""

    def test_import(self):
        """Test that TanimotoClusterSplitter can be imported."""
        from diffbio.splitters import TanimotoClusterSplitter, TanimotoClusterSplitterConfig

        assert TanimotoClusterSplitter is not None
        assert TanimotoClusterSplitterConfig is not None

    def test_requires_rdkit(self):
        """Test that TanimotoClusterSplitter imports RDKit."""
        pytest.importorskip("rdkit")

        from diffbio.splitters import TanimotoClusterSplitter, TanimotoClusterSplitterConfig

        config = TanimotoClusterSplitterConfig()
        # Should not raise ImportError
        _splitter = TanimotoClusterSplitter(config)

    def test_config_defaults(self):
        """Test default configuration values."""
        from diffbio.splitters import TanimotoClusterSplitterConfig

        config = TanimotoClusterSplitterConfig()
        assert config.smiles_key == "smiles"
        assert config.fingerprint_type == "morgan"
        assert config.fingerprint_radius == 2
        assert config.fingerprint_bits == 2048
        assert config.similarity_cutoff == 0.6

    def test_config_custom_values(self):
        """Test custom configuration values."""
        from diffbio.splitters import TanimotoClusterSplitterConfig

        config = TanimotoClusterSplitterConfig(
            smiles_key="mol_smiles",
            fingerprint_type="maccs",
            fingerprint_radius=3,
            fingerprint_bits=1024,
            similarity_cutoff=0.7,
        )
        assert config.smiles_key == "mol_smiles"
        assert config.fingerprint_type == "maccs"
        assert config.fingerprint_radius == 3
        assert config.fingerprint_bits == 1024
        assert config.similarity_cutoff == 0.7

    def test_split_returns_valid_result(self, molecule_data_source):
        """Test that split returns a valid SplitResult."""
        pytest.importorskip("rdkit")
        from diffbio.splitters import (
            SplitResult,
            TanimotoClusterSplitter,
            TanimotoClusterSplitterConfig,
        )

        config = TanimotoClusterSplitterConfig()
        splitter = TanimotoClusterSplitter(config)
        result = splitter.split(molecule_data_source)

        assert isinstance(result, SplitResult)
        assert result.train_size + result.valid_size + result.test_size == len(molecule_data_source)

    def test_split_no_overlap(self, molecule_data_source):
        """Test that Tanimoto splits have no overlapping indices."""
        pytest.importorskip("rdkit")
        from diffbio.splitters import TanimotoClusterSplitter, TanimotoClusterSplitterConfig

        config = TanimotoClusterSplitterConfig()
        splitter = TanimotoClusterSplitter(config)
        result = splitter.split(molecule_data_source)

        all_indices = jnp.concatenate(
            [result.train_indices, result.valid_indices, result.test_indices]
        )

        assert len(jnp.unique(all_indices)) == len(all_indices)

    def test_split_covers_all_data(self, molecule_data_source):
        """Test that Tanimoto splits cover all data."""
        pytest.importorskip("rdkit")
        from diffbio.splitters import TanimotoClusterSplitter, TanimotoClusterSplitterConfig

        config = TanimotoClusterSplitterConfig()
        splitter = TanimotoClusterSplitter(config)
        result = splitter.split(molecule_data_source)

        all_indices = jnp.concatenate(
            [result.train_indices, result.valid_indices, result.test_indices]
        )

        assert len(all_indices) == len(molecule_data_source)

    def test_different_fingerprint_types(self, molecule_data_source):
        """Test that different fingerprint types work."""
        pytest.importorskip("rdkit")
        from diffbio.splitters import TanimotoClusterSplitter, TanimotoClusterSplitterConfig

        for fp_type in ["morgan", "rdkit", "maccs"]:
            config = TanimotoClusterSplitterConfig(fingerprint_type=fp_type)
            splitter = TanimotoClusterSplitter(config)
            result = splitter.split(molecule_data_source)

            total = result.train_size + result.valid_size + result.test_size
            assert total == len(molecule_data_source), f"Failed for {fp_type}"


class TestTanimotoClusterSplitterEdgeCases:
    """Edge case tests for TanimotoClusterSplitter."""

    def test_invalid_smiles_handled(self):
        """Test that invalid SMILES are handled gracefully."""
        pytest.importorskip("rdkit")
        from dataclasses import dataclass

        from datarax.core.config import StructuralConfig
        from datarax.core.data_source import DataSourceModule
        from datarax.typing import Element

        from diffbio.splitters import TanimotoClusterSplitter, TanimotoClusterSplitterConfig

        @dataclass(frozen=True)
        class MockConfig(StructuralConfig):
            pass

        class MixedValiditySource(DataSourceModule):
            _data: list = nnx.data()

            def __init__(self, config, *, rngs=None, name=None):
                super().__init__(config, rngs=rngs, name=name)
                self._data = [
                    Element(data={"smiles": "c1ccccc1"}, state={}, metadata={}),  # pyright: ignore[reportArgumentType]
                    Element(data={"smiles": "invalid_smiles"}, state={}, metadata={}),  # pyright: ignore[reportArgumentType]
                    Element(data={"smiles": "CC"}, state={}, metadata={}),  # pyright: ignore[reportArgumentType]
                    Element(data={"smiles": "CCC"}, state={}, metadata={}),  # pyright: ignore[reportArgumentType]
                ]
                self._idx = 0

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx] if 0 <= idx < len(self._data) else None

            def __iter__(self):
                self._idx = 0
                return self

            def __next__(self):
                if self._idx >= len(self._data):
                    raise StopIteration
                elem = self._data[self._idx]
                self._idx += 1
                return elem

        config = TanimotoClusterSplitterConfig(train_frac=0.5, valid_frac=0.25, test_frac=0.25)
        splitter = TanimotoClusterSplitter(config)
        source = MixedValiditySource(MockConfig())

        # Should not raise, should handle invalid SMILES gracefully
        result = splitter.split(source)
        total = result.train_size + result.valid_size + result.test_size
        assert total == 4  # All molecules should be assigned somewhere

    def test_high_similarity_cutoff(self, molecule_data_source):
        """Test with high similarity cutoff (more clusters)."""
        pytest.importorskip("rdkit")
        from diffbio.splitters import TanimotoClusterSplitter, TanimotoClusterSplitterConfig

        config = TanimotoClusterSplitterConfig(similarity_cutoff=0.9)
        splitter = TanimotoClusterSplitter(config)
        result = splitter.split(molecule_data_source)

        total = result.train_size + result.valid_size + result.test_size
        assert total == len(molecule_data_source)

    def test_low_similarity_cutoff(self, molecule_data_source):
        """Test with low similarity cutoff (fewer clusters)."""
        pytest.importorskip("rdkit")
        from diffbio.splitters import TanimotoClusterSplitter, TanimotoClusterSplitterConfig

        config = TanimotoClusterSplitterConfig(similarity_cutoff=0.3)
        splitter = TanimotoClusterSplitter(config)
        result = splitter.split(molecule_data_source)

        total = result.train_size + result.valid_size + result.test_size
        assert total == len(molecule_data_source)
