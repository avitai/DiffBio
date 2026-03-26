"""Molecular splitters for drug discovery applications.

This module provides molecular-aware splitting utilities:
- ScaffoldSplitter: Split by Bemis-Murcko scaffold for drug discovery
- TanimotoClusterSplitter: Split by fingerprint similarity clustering
"""

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import jax.numpy as jnp
from flax import nnx

from datarax.core.data_source import DataSourceModule

from diffbio.splitters.base import SplitResult, SplitterConfig, SplitterModule

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScaffoldSplitterConfig(SplitterConfig):
    """Configuration for scaffold splitter.

    Attributes:
        smiles_key: Key in data element containing SMILES string (default: "smiles")
    """

    smiles_key: str = "smiles"


class ScaffoldSplitter(SplitterModule):
    """Split molecules by Bemis-Murcko scaffold.

    Inherits from SplitterModule (StructuralModule) because:

    - Non-parametric: scaffold extraction is deterministic
    - Frozen config: splitting strategy doesn't change
    - Domain-specific: requires RDKit and molecular knowledge

    Ensures train/test sets have different molecular scaffolds,
    preventing data leakage from structurally similar molecules.
    This is the industry standard for drug discovery benchmarks.

    Requires RDKit installation.

    Example:
        ```python
        config = ScaffoldSplitterConfig(smiles_key="mol_smiles")
        splitter = ScaffoldSplitter(config)
        result = splitter.split(molecule_source)
        ```

    References:
        Bemis, Guy W., and Mark A. Murcko. "The properties of known drugs.
        1. Molecular frameworks." Journal of medicinal chemistry 39.15 (1996): 2887-2893.
    """

    def __init__(
        self,
        config: ScaffoldSplitterConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize ScaffoldSplitter.

        Args:
            config: Scaffold splitter configuration
            rngs: Random number generators (unused for scaffold splitting)
            name: Optional module name

        Raises:
            ImportError: If RDKit is not installed
        """
        super().__init__(config, rngs=rngs, name=name)
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold

            self._Chem = Chem
            self._MurckoScaffold = MurckoScaffold
        except ImportError as e:
            raise ImportError("ScaffoldSplitter requires RDKit: pip install rdkit") from e

    def _generate_scaffolds(self, smiles_list: Sequence[str]) -> dict[str, list[int]]:
        """Generate Bemis-Murcko scaffolds for molecules.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Dictionary mapping scaffold SMILES to list of molecule indices
        """
        scaffolds: dict[str, list[int]] = {}
        for idx, smiles in enumerate(smiles_list):
            mol = self._Chem.MolFromSmiles(smiles)
            if mol is None:
                # Invalid SMILES - assign to empty scaffold group
                scaffold = ""
            else:
                try:
                    scaffold = self._MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                except (ValueError, RuntimeError):
                    # Some molecules may not have a valid scaffold
                    scaffold = ""

            if scaffold not in scaffolds:
                scaffolds[scaffold] = []
            scaffolds[scaffold].append(idx)

        return scaffolds

    def split(self, data_source: DataSourceModule) -> SplitResult:
        """Split by scaffold, largest scaffolds go to train first.

        Args:
            data_source: Datarax DataSourceModule to split

        Returns:
            SplitResult with scaffold-based train/valid/test indices
        """
        # Extract SMILES from data source
        smiles_list = [data_source[i].data[self.config.smiles_key] for i in range(len(data_source))]

        scaffolds = self._generate_scaffolds(smiles_list)

        # Sort scaffold groups by size (largest first)
        scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
        return self.assign_groups_to_splits(scaffold_sets, len(data_source))


@dataclass(frozen=True)
class TanimotoClusterSplitterConfig(SplitterConfig):
    """Configuration for Tanimoto cluster splitter.

    Attributes:
        smiles_key: Key in data element containing SMILES string (default: "smiles")
        fingerprint_type: Type of fingerprint ("morgan", "rdkit", "maccs")
        fingerprint_radius: Radius for Morgan fingerprints (default: 2)
        fingerprint_bits: Number of bits for fingerprints (default: 2048)
        similarity_cutoff: Tanimoto similarity cutoff for clustering (default: 0.6)
    """

    smiles_key: str = "smiles"
    fingerprint_type: str = "morgan"
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048
    similarity_cutoff: float = 0.6


class TanimotoClusterSplitter(SplitterModule):
    """Split by Tanimoto similarity clustering (Butina algorithm).

    Groups similar molecules together using fingerprint similarity,
    then assigns clusters to train/valid/test to ensure structural
    diversity between splits.

    Inherits from SplitterModule (StructuralModule) because:

    - Non-parametric: clustering is deterministic given fingerprints
    - Frozen config: splitting strategy doesn't change
    - Domain-specific: requires RDKit fingerprints

    Requires RDKit installation.

    Example:
        ```python
        config = TanimotoClusterSplitterConfig(similarity_cutoff=0.6)
        splitter = TanimotoClusterSplitter(config)
        result = splitter.split(molecule_source)
        ```

    References:
        Butina, Darko. "Unsupervised data base clustering based on daylight's
        fingerprint and Tanimoto similarity." JCICS 39.4 (1999): 747-750.
    """

    def __init__(
        self,
        config: TanimotoClusterSplitterConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize TanimotoClusterSplitter.

        Args:
            config: Tanimoto cluster splitter configuration
            rngs: Random number generators (unused)
            name: Optional module name

        Raises:
            ImportError: If RDKit is not installed
        """
        super().__init__(config, rngs=rngs, name=name)
        try:
            from rdkit import Chem, DataStructs
            from rdkit.Chem import AllChem, MACCSkeys
            from rdkit.ML.Cluster import Butina

            self._Chem = Chem
            self._DataStructs = DataStructs
            self._AllChem = AllChem
            self._MACCSkeys = MACCSkeys
            self._Butina = Butina
        except ImportError as e:
            raise ImportError("TanimotoClusterSplitter requires RDKit: pip install rdkit") from e

    def _compute_fingerprint(self, mol: Any) -> Any | None:
        """Compute fingerprint for a molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            Fingerprint object or None if computation fails
        """
        if mol is None:
            return None

        try:
            if self.config.fingerprint_type == "morgan":
                return self._AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    self.config.fingerprint_radius,
                    nBits=self.config.fingerprint_bits,
                )
            elif self.config.fingerprint_type == "rdkit":
                return self._Chem.RDKFingerprint(mol, fpSize=self.config.fingerprint_bits)
            elif self.config.fingerprint_type == "maccs":
                return self._MACCSkeys.GenMACCSKeys(mol)
            else:
                raise ValueError(f"Unknown fingerprint type: {self.config.fingerprint_type}")
        except (ValueError, RuntimeError):
            return None

    def _compute_fingerprints(self, smiles_list: Sequence[str]) -> list[tuple[int, Any]]:
        """Compute fingerprints for all molecules.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            List of (index, fingerprint) tuples for valid molecules
        """
        valid_fps: list[tuple[int, Any]] = []
        for idx, smiles in enumerate(smiles_list):
            mol = self._Chem.MolFromSmiles(smiles)
            fp = self._compute_fingerprint(mol)
            if fp is not None:
                valid_fps.append((idx, fp))
        return valid_fps

    @staticmethod
    def _all_train_result(size: int) -> SplitResult:
        """Create a split result with all items assigned to train."""
        return SplitResult(
            train_indices=jnp.array(list(range(size)), dtype=jnp.int32),
            valid_indices=jnp.array([], dtype=jnp.int32),
            test_indices=jnp.array([], dtype=jnp.int32),
        )

    def _cluster_fingerprints(self, fp_list: list[Any]) -> list[tuple[int, ...]]:
        """Cluster fingerprints using Butina on condensed Tanimoto distances."""
        n_valid = len(fp_list)
        dists = []
        for i in range(1, n_valid):
            sims = self._DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
            dists.extend([1 - s for s in sims])

        dist_threshold = 1 - self.config.similarity_cutoff
        clusters = self._Butina.ClusterData(dists, n_valid, dist_threshold, isDistData=True)
        return sorted(clusters, key=len, reverse=True)

    def _assign_clusters_to_splits(
        self,
        sorted_clusters: list[tuple[int, ...]],
        indices: list[int],
        total_size: int,
    ) -> tuple[list[int], list[int], list[int]]:
        """Assign clusters to train/valid/test while preserving cluster membership."""
        remapped_clusters = ([indices[i] for i in cluster] for cluster in sorted_clusters)
        split_result = self.assign_groups_to_splits(remapped_clusters, total_size)
        return (
            split_result.train_indices.tolist(),
            split_result.valid_indices.tolist(),
            split_result.test_indices.tolist(),
        )

    def split(self, data_source: DataSourceModule) -> SplitResult:
        """Cluster by Tanimoto similarity and split.

        Args:
            data_source: Datarax DataSourceModule to split

        Returns:
            SplitResult with cluster-based train/valid/test indices
        """
        total_size = len(data_source)
        smiles_list = [data_source[i].data[self.config.smiles_key] for i in range(total_size)]

        # Compute fingerprints
        valid_fps = self._compute_fingerprints(smiles_list)

        # Track invalid molecules (those without valid fingerprints)
        valid_indices = {idx for idx, _ in valid_fps}
        invalid_indices = [i for i in range(total_size) if i not in valid_indices]

        if len(valid_fps) == 0:
            return self._all_train_result(total_size)

        # Unpack indices and fingerprints
        indices, fp_list = zip(*valid_fps)
        indices = list(indices)
        fp_list = list(fp_list)
        sorted_clusters = self._cluster_fingerprints(fp_list)
        train_inds, valid_inds, test_inds = self._assign_clusters_to_splits(
            sorted_clusters, indices, total_size
        )

        # Add invalid molecules to train (they couldn't be clustered)
        train_inds.extend(invalid_indices)

        return SplitResult(
            train_indices=jnp.array(train_inds, dtype=jnp.int32),
            valid_indices=jnp.array(valid_inds, dtype=jnp.int32),
            test_indices=jnp.array(test_inds, dtype=jnp.int32),
        )
