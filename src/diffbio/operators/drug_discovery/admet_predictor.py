"""Multi-task ADMET property prediction operator.

This module implements a ChemProp-style multi-task ADMET predictor
for predicting Absorption, Distribution, Metabolism, Excretion, and
Toxicity properties of drug candidates.

The implementation follows the TDC ADMET Benchmark with 22 standard endpoints.

References:
    - https://tdcommons.ai/benchmark/admet_group/overview/
    - https://github.com/chemprop/chemprop
    - Swanson et al. "ADMET-AI" Bioinformatics 2024
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from artifex.generative_models.core.base import MLP
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx

from diffbio.operators.drug_discovery._graph_utils import (
    build_optional_dropout,
    graph_sum_readout,
    initialize_graph_encoder_from_config,
)

logger = logging.getLogger(__name__)

# Standard TDC ADMET benchmark task names (22 tasks)
ADMET_TASK_NAMES: list[str] = [
    # Absorption (6)
    "Caco2_Wang",
    "HIA_Hou",
    "Pgp_Broccatelli",
    "Bioavailability_Ma",
    "Lipophilicity_AstraZeneca",
    "Solubility_AqSolDB",
    # Distribution (3)
    "BBB_Martins",
    "PPBR_AZ",
    "VDss_Lombardo",
    # Metabolism (6)
    "CYP2C9_Veith",
    "CYP2D6_Veith",
    "CYP3A4_Veith",
    "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels",
    # Excretion (3)
    "Half_Life_Obach",
    "Clearance_Hepatocyte_AZ",
    "Clearance_Microsome_AZ",
    # Toxicity (4)
    "LD50_Zhu",
    "hERG",
    "AMES",
    "DILI",
]

# Task types: classification or regression
ADMET_TASK_TYPES: dict[str, str] = {
    # Absorption
    "Caco2_Wang": "regression",
    "HIA_Hou": "classification",
    "Pgp_Broccatelli": "classification",
    "Bioavailability_Ma": "classification",
    "Lipophilicity_AstraZeneca": "regression",
    "Solubility_AqSolDB": "regression",
    # Distribution
    "BBB_Martins": "classification",
    "PPBR_AZ": "regression",
    "VDss_Lombardo": "regression",
    # Metabolism
    "CYP2C9_Veith": "classification",
    "CYP2D6_Veith": "classification",
    "CYP3A4_Veith": "classification",
    "CYP2C9_Substrate_CarbonMangels": "classification",
    "CYP2D6_Substrate_CarbonMangels": "classification",
    "CYP3A4_Substrate_CarbonMangels": "classification",
    # Excretion
    "Half_Life_Obach": "regression",
    "Clearance_Hepatocyte_AZ": "regression",
    "Clearance_Microsome_AZ": "regression",
    # Toxicity
    "LD50_Zhu": "regression",
    "hERG": "classification",
    "AMES": "classification",
    "DILI": "classification",
}


@dataclass(frozen=True)
class ADMETConfig(OperatorConfig):
    # pylint: disable=too-many-instance-attributes
    """Configuration for ADMET property predictor.

    Attributes:
        hidden_dim: Hidden dimension for message passing (default: 300).
        num_message_passing_steps: Number of D-MPNN iterations (default: 3).
        num_tasks: Number of ADMET prediction tasks (default: 22).
        dropout_rate: Dropout rate for regularization (default: 0.0).
        in_features: Number of input node features (default: 4).
        num_edge_features: Number of edge features (default: 4).
        ffn_hidden_dim: FFN hidden dimension (default: same as hidden_dim).
        ffn_num_layers: Number of FFN layers (default: 2).
        apply_task_activations: Apply sigmoid for classification tasks (default: False).
    """

    hidden_dim: int = 300
    num_message_passing_steps: int = 3
    num_tasks: int = 22
    dropout_rate: float = 0.0
    in_features: int = 4
    num_edge_features: int = 4
    ffn_hidden_dim: int | None = None
    ffn_num_layers: int = 2
    apply_task_activations: bool = False


class ADMETPredictor(OperatorModule):
    """Multi-task ADMET property predictor.

    Implements a ChemProp-style directed message passing neural network
    for predicting multiple ADMET properties simultaneously. The architecture
    uses a shared molecular encoder with task-specific prediction heads.

    Architecture:
        1. Message passing encoder (D-MPNN style)
        2. Graph-level readout via sum pooling
        3. Shared feed-forward layers
        4. Task-specific output heads

    The 22 standard TDC ADMET endpoints cover:
        - Absorption: Caco2, HIA, Pgp, Bioavailability, Lipophilicity, Solubility
        - Distribution: BBB, PPBR, VDss
        - Metabolism: CYP enzymes (2C9, 2D6, 3A4) inhibition and substrate
        - Excretion: Half-life, Hepatocyte clearance, Microsome clearance
        - Toxicity: LD50, hERG, AMES, DILI

    Example:
        ```python
        config = ADMETConfig(hidden_dim=256, num_tasks=22)
        predictor = ADMETPredictor(config, rngs=nnx.Rngs(42))
        data = {"node_features": nodes, "adjacency": adj, "node_mask": mask}
        result, _, _ = predictor.apply(data, {}, None)
        predictions = result["predictions"]  # shape: (22,)
        ```

    References:
        - https://tdcommons.ai/benchmark/admet_group/overview/
        - Yang et al. "Analyzing Learned Molecular Representations" JCIM 2019
    """

    def __init__(
        self,
        config: ADMETConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize ADMET predictor.

        Args:
            config: ADMET configuration.
            rngs: Flax NNX random number generators.
            name: Optional name for the operator.
        """
        super().__init__(config, rngs=rngs, name=name)

        rngs = initialize_graph_encoder_from_config(self, config, rngs=rngs)

        # FFN hidden dim defaults to hidden_dim
        ffn_hidden = config.ffn_hidden_dim or config.hidden_dim

        if config.ffn_num_layers > 1:
            self.ffn_backbone = MLP(
                hidden_dims=[ffn_hidden] * (config.ffn_num_layers - 1),
                in_features=config.hidden_dim,
                activation="relu",
                dropout_rate=config.dropout_rate,
                output_activation="relu",
                use_batch_norm=False,
                rngs=rngs,
            )
        else:
            self.ffn_backbone = None

        # Task-specific output heads (one per ADMET task)
        last_hidden = ffn_hidden if config.ffn_num_layers > 1 else config.hidden_dim
        task_heads = [nnx.Linear(last_hidden, 1, rngs=rngs) for _ in range(config.num_tasks)]
        self.task_heads = nnx.List(task_heads)

        # Dropout
        self.dropout = build_optional_dropout(config.dropout_rate, rngs=rngs)

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,  # noqa: ARG002
        stats: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        """Predict ADMET properties from molecular graph.

        Args:
            data: Input data containing:
                - node_features: (num_nodes, num_features) atom features
                - adjacency: (num_nodes, num_nodes) adjacency matrix
                - edge_features: Optional (num_nodes, num_nodes, num_edge_features)
                - node_mask: (num_nodes,) mask for valid nodes
            state: Per-element state (passed through).
            metadata: Optional metadata.
            random_params: Unused random parameters.
            stats: Optional statistics dictionary.

        Returns:
            Tuple of:
                - data with added "predictions" and "task_predictions" keys
                - unchanged state
                - unchanged metadata
        """
        del random_params, stats  # Unused

        graph_repr = graph_sum_readout(data, self.encoder, dropout=self.dropout)

        h = graph_repr
        if self.ffn_backbone is not None:
            ffn_output = self.ffn_backbone(h)
            if isinstance(ffn_output, tuple):
                raise TypeError("ADMETPredictor shared FFN must return a single tensor output.")
            h = ffn_output

        # Task-specific predictions
        task_predictions = []
        for i, head in enumerate(self.task_heads):
            pred = head(h).squeeze(-1)

            # Apply activation for classification tasks if configured
            if self.config.apply_task_activations:
                task_name = ADMET_TASK_NAMES[i] if i < len(ADMET_TASK_NAMES) else f"task_{i}"
                if ADMET_TASK_TYPES.get(task_name) == "classification":
                    pred = nnx.sigmoid(pred)

            task_predictions.append(pred)

        # Stack all predictions
        predictions = jnp.stack(task_predictions)

        result = {
            **data,
            "predictions": predictions,
            "task_predictions": task_predictions,
            "graph_representation": graph_repr,
        }

        return result, state, metadata


def create_admet_predictor(
    hidden_dim: int = 300,
    num_layers: int = 3,
    dropout_rate: float = 0.0,
    seed: int = 42,
) -> ADMETPredictor:
    """Create an ADMET predictor with standard configuration.

    Args:
        hidden_dim: Hidden dimension for message passing.
        num_layers: Number of message passing steps.
        dropout_rate: Dropout rate.
        seed: Random seed.

    Returns:
        Configured ADMETPredictor.
    """
    config = ADMETConfig(
        hidden_dim=hidden_dim,
        num_message_passing_steps=num_layers,
        dropout_rate=dropout_rate,
    )
    return ADMETPredictor(config, rngs=nnx.Rngs(seed))
