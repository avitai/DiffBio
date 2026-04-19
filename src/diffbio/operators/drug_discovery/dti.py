"""Differentiable drug-target interaction pipeline.

The pipeline combines existing DiffBio building blocks:

- ``TransformerSequenceEncoder`` for protein sequence embeddings.
- ``DifferentiableMolecularFingerprint`` for molecular graph fingerprints.

This module centralizes DTI input preparation so benchmarks do not maintain
bespoke protein/drug feature handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from datarax.core.config import OperatorConfig
from flax import nnx

from diffbio.operators.alignment import PROTEIN_ALPHABET
from diffbio.operators.drug_discovery.fingerprint import (
    DifferentiableMolecularFingerprint,
    MolecularFingerprintConfig,
)
from diffbio.operators.drug_discovery.primitives import (
    DEFAULT_ATOM_FEATURES,
    batch_smiles_to_graphs,
)
from diffbio.operators.foundation_models.contracts import AdapterMode, FoundationModelKind
from diffbio.operators.foundation_models.transformer_encoder import (
    TransformerSequenceEncoder,
    TransformerSequenceEncoderConfig,
)
from diffbio.sources.dti import validate_dti_dataset

PROTEIN_ONE_HOT_PREPROCESSING_VERSION = "protein_one_hot_v1"
DTI_PIPELINE_INTEGRATION_LAYER = "shared_dti_pipeline_v1"
_PROTEIN_ALPHABET_INDEX = {residue: index for index, residue in enumerate(PROTEIN_ALPHABET)}


@dataclass(frozen=True)
class _DTIProteinEncoderConfig:
    """Protein encoder configuration for the DTI pipeline."""

    protein_hidden_dim: int = 16
    protein_num_layers: int = 1
    protein_num_heads: int = 2
    protein_intermediate_dim: int = 32
    max_protein_length: int = 32


@dataclass(frozen=True)
class _DTIDrugEncoderConfig:
    """Drug encoder configuration for the DTI pipeline."""

    drug_fingerprint_dim: int = 16
    drug_hidden_dim: int = 16
    drug_num_layers: int = 2


@dataclass(frozen=True)
class _DTIPairScorerConfig:
    """Pair scorer and artifact configuration for the DTI pipeline."""

    pair_hidden_dim: int = 16
    foundation_artifact_id: str = "diffbio.dti_protein_encoder"
    foundation_preprocessing_version: str = PROTEIN_ONE_HOT_PREPROCESSING_VERSION


@dataclass(frozen=True)
class DTIPipelineConfig(
    _DTIProteinEncoderConfig,
    _DTIDrugEncoderConfig,
    _DTIPairScorerConfig,
    OperatorConfig,
):
    """Configuration for the shared differentiable DTI pipeline."""

    def __post_init__(self) -> None:
        """Validate the pipeline configuration."""
        super().__post_init__()
        positive_fields = {
            "protein_hidden_dim": self.protein_hidden_dim,
            "protein_num_layers": self.protein_num_layers,
            "protein_num_heads": self.protein_num_heads,
            "protein_intermediate_dim": self.protein_intermediate_dim,
            "max_protein_length": self.max_protein_length,
            "drug_fingerprint_dim": self.drug_fingerprint_dim,
            "drug_hidden_dim": self.drug_hidden_dim,
            "drug_num_layers": self.drug_num_layers,
            "pair_hidden_dim": self.pair_hidden_dim,
        }
        for field_name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{field_name} must be positive.")


class DifferentiableDTIPipeline(nnx.Module):
    """ConPLex-style DTI scorer with differentiable drug and protein encoders."""

    def __init__(
        self,
        config: DTIPipelineConfig,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """Initialize protein encoder, drug encoder, and pair scorer."""
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.config = nnx.static(config)
        self.protein_encoder = TransformerSequenceEncoder(
            TransformerSequenceEncoderConfig(
                hidden_dim=config.protein_hidden_dim,
                num_layers=config.protein_num_layers,
                num_heads=config.protein_num_heads,
                intermediate_dim=config.protein_intermediate_dim,
                max_length=config.max_protein_length,
                alphabet_size=len(PROTEIN_ALPHABET),
                dropout_rate=0.0,
                pooling="mean",
                artifact_id=config.foundation_artifact_id,
                preprocessing_version=config.foundation_preprocessing_version,
                adapter_mode=AdapterMode.NATIVE_TRAINABLE,
            ),
            rngs=rngs,
        )
        self.drug_encoder = DifferentiableMolecularFingerprint(
            MolecularFingerprintConfig(
                fingerprint_dim=config.drug_fingerprint_dim,
                hidden_dim=config.drug_hidden_dim,
                num_layers=config.drug_num_layers,
                in_features=DEFAULT_ATOM_FEATURES,
                normalize=True,
            ),
            rngs=rngs,
        )
        self.pair_hidden = nnx.Linear(
            config.protein_hidden_dim + config.drug_fingerprint_dim,
            config.pair_hidden_dim,
            rngs=rngs,
        )
        self.output = nnx.Linear(config.pair_hidden_dim, 1, rngs=rngs)

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        """Score a prepared paired protein/drug DTI batch."""
        del random_params, stats

        protein_result, _, _ = self.protein_encoder.apply(
            {
                "sequence": data["protein_one_hot"],
                "attention_mask": data["protein_attention_mask"],
            },
            {},
            None,
        )
        protein_embeddings = jnp.asarray(protein_result["embeddings"], dtype=jnp.float32)
        drug_fingerprints = self._encode_drug_graphs(data["drug_graphs"])
        pair_embeddings = jnp.concatenate([protein_embeddings, drug_fingerprints], axis=-1)
        hidden = nnx.gelu(self.pair_hidden(pair_embeddings))
        scores = self.output(hidden).squeeze(-1)

        return (
            {
                **data,
                "scores": scores,
                "protein_embeddings": protein_embeddings,
                "drug_fingerprints": drug_fingerprints,
                "pair_embeddings": pair_embeddings,
                "foundation_model": protein_result["foundation_model"],
                "dti_pipeline": self.pipeline_metadata(),
            },
            state,
            metadata,
        )

    def pipeline_metadata(self) -> dict[str, Any]:
        """Return benchmark-facing metadata for the integrated DTI path."""
        return {
            "integration_layer": DTI_PIPELINE_INTEGRATION_LAYER,
            "pipeline_name": type(self).__name__,
            "protein_encoder": {
                "operator": "TransformerSequenceEncoder",
                "model_family": FoundationModelKind.SEQUENCE_TRANSFORMER.value,
                "adapter_mode": AdapterMode.NATIVE_TRAINABLE.value,
                "preprocessing_version": self.config.foundation_preprocessing_version,
            },
            "drug_encoder": {
                "operator": "DifferentiableMolecularFingerprint",
                "differentiable": True,
            },
        }

    def _encode_drug_graphs(self, drug_graphs: dict[str, Any]) -> jnp.ndarray:
        """Encode a padded molecular graph batch with the shared fingerprint operator."""
        fingerprints = []
        batch_size = int(drug_graphs["node_features"].shape[0])
        for index in range(batch_size):
            graph = {
                "node_features": drug_graphs["node_features"][index],
                "adjacency": drug_graphs["adjacency"][index],
                "edge_features": drug_graphs["edge_features"][index],
                "node_mask": drug_graphs["node_mask"][index],
            }
            result, _, _ = self.drug_encoder.apply(graph, {}, None)
            fingerprints.append(result["fingerprint"])
        return jnp.stack(fingerprints)


def build_dti_pipeline_inputs(
    data: dict[str, Any],
    *,
    config: DTIPipelineConfig | None = None,
) -> dict[str, Any]:
    """Build one encoded protein/graph batch from a validated DTI payload."""
    validate_dti_dataset(data)
    resolved_config = config or DTIPipelineConfig()
    protein_one_hot, protein_attention_mask = encode_protein_sequences(
        data["protein_sequences"],
        max_length=resolved_config.max_protein_length,
    )
    return {
        "protein_one_hot": protein_one_hot,
        "protein_attention_mask": protein_attention_mask,
        "drug_graphs": batch_smiles_to_graphs(list(data["drug_smiles"])),
        "targets": jnp.asarray(data["targets"], dtype=jnp.float32),
    }


def encode_protein_sequences(
    sequences: list[str],
    *,
    max_length: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One-hot encode protein strings using the shared alignment alphabet."""
    if max_length <= 0:
        raise ValueError("max_length must be positive.")

    encoded = np.zeros((len(sequences), max_length, len(PROTEIN_ALPHABET)), dtype=np.float32)
    attention_mask = np.zeros((len(sequences), max_length), dtype=np.float32)

    for sequence_index, sequence in enumerate(sequences):
        for residue_index, residue in enumerate(sequence[:max_length]):
            amino_acid_index = _PROTEIN_ALPHABET_INDEX.get(residue.upper())
            if amino_acid_index is None:
                encoded[sequence_index, residue_index, :] = 1.0 / len(PROTEIN_ALPHABET)
            else:
                encoded[sequence_index, residue_index, amino_acid_index] = 1.0
            attention_mask[sequence_index, residue_index] = 1.0

    return jnp.asarray(encoded), jnp.asarray(attention_mask)
