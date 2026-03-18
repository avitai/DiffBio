"""Differentiable MACCS structural keys fingerprint operator.

This module implements a differentiable version of the 166 MACCS
(Molecular ACCess System) structural keys fingerprint.

MACCS keys are predefined structural patterns (SMARTS) that encode
the presence/absence of specific molecular substructures. This
implementation provides a differentiable approximation using
learned pattern matching networks.

References:
    - https://rdkit.org/docs/source/rdkit.Chem.MACCSkeys.html
    - Durant et al. "Reoptimization of MDL Keys for Use in Drug Discovery" JCIM 2002
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx

from diffbio.operators.drug_discovery._graph_utils import (
    attach_fingerprint,
    ensure_rngs,
    initialize_graph_encoder,
    stabilize_operator_id,
    unpack_graph_inputs,
)


@dataclass
class MACCSKeysConfig(OperatorConfig):
    """Configuration for MACCS keys fingerprint operator.

    Attributes:
        n_bits: Number of fingerprint bits (default: 166 for standard MACCS).
        differentiable: Use learned pattern matching (default: True).
        temperature: Temperature for soft bit assignment (default: 1.0).
        hidden_dim: Hidden dimension for pattern networks (default: 64).
        num_layers: Number of message passing layers (default: 2).
        in_features: Number of input node features (default: 4).
        stochastic: Whether operator uses random sampling.
        stream_name: Optional stream name for data routing.
    """

    n_bits: int = 166
    differentiable: bool = True
    temperature: float = 1.0
    hidden_dim: int = 64
    num_layers: int = 2
    in_features: int = 4
    stochastic: bool = False
    stream_name: str | None = None


class MACCSKeysOperator(OperatorModule):
    """Differentiable MACCS structural keys fingerprint operator.

    For differentiable=True:
        Uses message passing and learned pattern detectors to approximate
        MACCS key detection. Each of the 166 keys is represented by a
        learned pattern matching network that outputs a soft presence score.

    For differentiable=False:
        Would use RDKit's exact MACCS implementation (not differentiable).

    The differentiable version enables gradient flow for end-to-end
    optimization while approximating the structural pattern detection
    of traditional MACCS keys.

    MACCS keys encode various structural features:

        - Atom types (C, N, O, S, halides, etc.)
        - Functional groups (carbonyl, hydroxyl, amine, etc.)
        - Ring systems (aromatic, aliphatic)
        - Bond patterns and connectivity

    Example:
        ```python
        config = MACCSKeysConfig(temperature=1.0)
        op = MACCSKeysOperator(config, rngs=nnx.Rngs(42))
        data = {"node_features": nodes, "adjacency": adj}
        result, _, _ = op.apply(data, {}, None)
        fingerprint = result["fingerprint"]  # shape: (166,)
        ```

    References:
        - Durant et al. "Reoptimization of MDL Keys" JCIM 2002
    """

    def __init__(
        self,
        config: MACCSKeysConfig,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize MACCS keys operator.

        Args:
            config: MACCS keys configuration.
            rngs: Flax NNX random number generators.
        """
        super().__init__(config, rngs=rngs)

        rngs = ensure_rngs(rngs)

        if config.differentiable:
            # Message passing for local structure aggregation
            rngs = initialize_graph_encoder(
                self,
                rngs=rngs,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                in_features=config.in_features,
                attr="encoder",
            )

            # Pattern detectors: one network per MACCS key
            # Each outputs a score indicating pattern presence
            self.pattern_detectors = nnx.Sequential(
                nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs),
                nnx.relu,
                nnx.Linear(config.hidden_dim, config.n_bits, rngs=rngs),
            )
        else:
            # RDKit mode
            stabilize_operator_id(self)
            try:
                from rdkit import Chem
                from rdkit.Chem import MACCSkeys as RDKitMACCS

                self._Chem = Chem
                self._MACCSkeys = RDKitMACCS
            except ImportError as e:
                raise ImportError(
                    "MACCSKeysOperator with differentiable=False requires RDKit: pip install rdkit"
                ) from e

    def _compute_differentiable_fp(
        self,
        node_features: jnp.ndarray,
        adjacency: jnp.ndarray,
        edge_features: jnp.ndarray | None = None,
        node_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute differentiable MACCS-style fingerprint.

        Uses message passing to aggregate local structure information,
        then applies learned pattern detectors with soft thresholding.

        Args:
            node_features: (num_nodes, num_features) atom features
            adjacency: (num_nodes, num_nodes) adjacency matrix
            edge_features: Optional edge features
            node_mask: Optional mask for valid nodes

        Returns:
            Fingerprint vector of shape (n_bits,)
        """
        # Message passing to capture local structure
        node_hidden = self.encoder(node_features, adjacency, edge_features)

        # Apply node mask
        if node_mask is not None:
            node_hidden = node_hidden * node_mask[:, None]

        # Pattern detection at each atom
        # pattern_logits: (num_nodes, n_bits)
        pattern_logits = self.pattern_detectors(node_hidden)

        # Apply temperature-scaled sigmoid for soft pattern matching
        soft_patterns = jax.nn.sigmoid(pattern_logits / self.config.temperature)

        # Aggregate across atoms using max (OR-like)
        # If any atom matches a pattern, the bit is set
        fingerprint = jnp.max(soft_patterns, axis=0)

        return fingerprint

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,  # noqa: ARG002
        stats: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        """Compute MACCS keys fingerprint.

        Args:
            data: Input data containing:
                For differentiable=True:
                    - node_features: (num_nodes, num_features) atom features
                    - adjacency: (num_nodes, num_nodes) adjacency matrix
                    - node_mask: (num_nodes,) optional mask for valid nodes
                For differentiable=False:
                    - smiles: SMILES string
            state: Per-element state (passed through).
            metadata: Optional metadata.
            random_params: Unused random parameters.
            stats: Optional statistics dictionary.

        Returns:
            Tuple of:
                - data with added "fingerprint" key
                - unchanged state
                - unchanged metadata
        """
        del random_params, stats  # Unused

        if self.config.differentiable:
            node_features, adjacency, edge_features, node_mask = unpack_graph_inputs(data)
            fp = self._compute_differentiable_fp(node_features, adjacency, edge_features, node_mask)
        else:
            smiles = data["smiles"]
            fp = self._compute_rdkit_fp(smiles)

        return attach_fingerprint(data, fp), state, metadata

    def _compute_rdkit_fp(self, smiles: str) -> jnp.ndarray:
        """Compute exact MACCS keys using RDKit.

        Args:
            smiles: SMILES string

        Returns:
            Binary fingerprint vector of shape (166,)
        """
        import numpy as np

        mol = self._Chem.MolFromSmiles(smiles)
        if mol is None:
            return jnp.zeros(self.config.n_bits, dtype=jnp.float32)

        fp = self._MACCSkeys.GenMACCSKeys(mol)

        # Convert to numpy (RDKit returns 167 bits, we use 1-166)
        arr = np.zeros(167, dtype=np.float32)
        for i in range(167):
            arr[i] = fp.GetBit(i)

        # Return bits 1-166 (index 0 is unused in standard MACCS)
        return jnp.asarray(arr[1:167])


def create_maccs_operator(
    differentiable: bool = True,
    temperature: float = 1.0,
    seed: int = 42,
) -> MACCSKeysOperator:
    """Create a MACCS keys fingerprint operator.

    Args:
        differentiable: Use learned pattern matching.
        temperature: Temperature for soft matching.
        seed: Random seed.

    Returns:
        Configured MACCSKeysOperator.
    """
    config = MACCSKeysConfig(
        differentiable=differentiable,
        temperature=temperature,
    )
    return MACCSKeysOperator(config, rngs=nnx.Rngs(seed))
