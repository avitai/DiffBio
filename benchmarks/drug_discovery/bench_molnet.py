#!/usr/bin/env python3
"""MolNet molecular property prediction benchmark.

Evaluates DiffBio's CircularFingerprintOperator (ECFP4, radius=2)
on the BBBP (Blood-Brain Barrier Penetration) classification task
from the MoleculeNet benchmark suite.

Pipeline:
    1. Load BBBP via MolNetSource
    2. Convert SMILES to graphs, featurize with CircularFingerprintOperator
    3. Split 80/10/10 via RandomSplitter
    4. Train MLP classifier (~50 epochs, 20 in quick mode)
    5. Evaluate ROC-AUC on test set

Results are compared against published baselines:
GCN (0.877), AttentiveFP (0.858), D-MPNN (0.910).

Usage:
    python benchmarks/drug_discovery/bench_molnet.py
    python benchmarks/drug_discovery/bench_molnet.py --quick
"""

from __future__ import annotations

import logging
import math
from typing import Any

import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from benchmarks._base import DiffBioBenchmark, DiffBioBenchmarkConfig
from benchmarks._baselines.molnet import MOLNET_BASELINES
from diffbio.operators.drug_discovery import (
    DEFAULT_ATOM_FEATURES,
    CircularFingerprintConfig,
    CircularFingerprintOperator,
    smiles_to_graph,
)
from diffbio.sources.molnet import MolNetSource, MolNetSourceConfig
from diffbio.splitters.random import RandomSplitter, RandomSplitterConfig

logger = logging.getLogger(__name__)

_CONFIG = DiffBioBenchmarkConfig(
    name="drug_discovery/molnet",
    domain="drug_discovery",
    quick_subsample=200,
)


class _MLPClassifier(nnx.Module):
    """Two-layer MLP for binary classification on fingerprints."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.linear1 = nnx.Linear(in_features, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass returning logits of shape (batch, 1)."""
        h = nnx.relu(self.linear1(x))
        return self.linear2(h)


def _featurize_molecules(
    source: MolNetSource,
    fp_operator: CircularFingerprintOperator,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Convert all source elements to fingerprints and labels.

    Args:
        source: MolNet data source with SMILES and labels.
        fp_operator: Fingerprint operator for featurization.

    Returns:
        Tuple of (fingerprints, labels, n_valid) where n_valid
        is the count of successfully featurized molecules.
    """
    fingerprints: list[jnp.ndarray] = []
    labels: list[float] = []

    for element in source:
        smiles = element.data["smiles"]
        label = element.data["y"]
        if math.isnan(float(label)):
            continue
        try:
            graph = smiles_to_graph(smiles)
        except ValueError:
            continue

        result, _, _ = fp_operator.apply(graph, {}, None)
        fingerprints.append(result["fingerprint"])
        labels.append(float(label))

    fp_array = jnp.stack(fingerprints)
    label_array = jnp.array(labels)
    return fp_array, label_array, len(labels)


def _compute_roc_auc(
    labels: jnp.ndarray,
    scores: jnp.ndarray,
) -> float:
    """Compute ROC-AUC from binary labels and predicted scores.

    Uses a trapezoidal approximation over sorted thresholds.

    Args:
        labels: Binary ground-truth labels, shape (n,).
        scores: Predicted scores (logits or probabilities), shape (n,).

    Returns:
        ROC-AUC value in [0, 1].
    """
    labels_np = np.asarray(labels).ravel()
    scores_np = np.asarray(scores).ravel()

    n_pos = int(labels_np.sum())
    n_neg = len(labels_np) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(-scores_np)
    sorted_labels = labels_np[order]

    tpr_values: list[float] = [0.0]
    fpr_values: list[float] = [0.0]
    tp = 0
    fp = 0
    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_values.append(tp / n_pos)
        fpr_values.append(fp / n_neg)

    auc = 0.0
    for i in range(1, len(fpr_values)):
        auc += (fpr_values[i] - fpr_values[i - 1]) * (
            tpr_values[i] + tpr_values[i - 1]
        ) / 2.0
    return float(auc)


def _train_and_evaluate(
    fingerprints: jnp.ndarray,
    labels: jnp.ndarray,
    source: MolNetSource,
    *,
    n_epochs: int,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> dict[str, float]:
    """Train MLP on fingerprints and evaluate ROC-AUC.

    Args:
        fingerprints: Fingerprint matrix, shape (n, n_bits).
        labels: Binary labels, shape (n,).
        source: MolNet data source (used for splitting).
        n_epochs: Number of training epochs.
        learning_rate: Adam learning rate.
        seed: Random seed for model initialization.

    Returns:
        Dict with test_roc_auc, train_roc_auc, n_molecules.
    """
    n_bits = fingerprints.shape[1]
    splitter = RandomSplitter(
        RandomSplitterConfig(
            train_frac=0.8,
            valid_frac=0.1,
            test_frac=0.1,
            seed=42,
        )
    )
    split = splitter.split(source)

    # Clamp indices to valid range (some molecules were skipped)
    n_valid = fingerprints.shape[0]
    train_idx = np.asarray(split.train_indices)
    test_idx = np.asarray(split.test_indices)
    train_idx = train_idx[train_idx < n_valid]
    test_idx = test_idx[test_idx < n_valid]

    x_train = fingerprints[train_idx]
    y_train = labels[train_idx]
    x_test = fingerprints[test_idx]
    y_test = labels[test_idx]

    rngs = nnx.Rngs(seed)
    model = _MLPClassifier(n_bits, hidden_dim=128, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    @nnx.jit
    def train_step(
        mdl: _MLPClassifier,
        opt: nnx.Optimizer,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        """Single training step returning the loss."""
        def loss_fn(m: _MLPClassifier) -> jnp.ndarray:
            logits = m(x).squeeze(-1)
            return jnp.mean(
                optax.sigmoid_binary_cross_entropy(logits, y)
            )
        loss, grads = nnx.value_and_grad(loss_fn)(mdl)
        opt.update(mdl, grads)
        return loss

    for epoch in range(n_epochs):
        loss = train_step(model, optimizer, x_train, y_train)
        if (epoch + 1) % 10 == 0:
            logger.info("  Epoch %d/%d  loss=%.4f", epoch + 1, n_epochs, float(loss))

    train_logits = model(x_train).squeeze(-1)
    test_logits = model(x_test).squeeze(-1)

    train_auc = _compute_roc_auc(y_train, train_logits)
    test_auc = _compute_roc_auc(y_test, test_logits)

    return {
        "test_roc_auc": test_auc,
        "train_roc_auc": train_auc,
        "n_molecules": n_valid,
    }


class MolNetBenchmark(DiffBioBenchmark):
    """MoleculeNet BBBP property prediction benchmark.

    Evaluates DiffBio's differentiable circular fingerprints
    (ECFP4) coupled with a simple MLP classifier on the BBBP
    binary classification task, reporting ROC-AUC.
    """

    def __init__(
        self,
        config: DiffBioBenchmarkConfig = _CONFIG,
        *,
        quick: bool = False,
        data_dir: str = "",
    ) -> None:
        super().__init__(config, quick=quick, data_dir=data_dir)

    def _run_core(self) -> dict[str, Any]:
        """Load BBBP, featurize, train MLP, evaluate ROC-AUC."""
        # 1. Load dataset
        logger.info("Loading BBBP dataset...")
        source = MolNetSource(
            MolNetSourceConfig(
                dataset_name="bbbp",
                split="train",
                download=True,
            )
        )
        n_total = len(source)
        logger.info("  %d molecules loaded", n_total)

        # 2. Create fingerprint operator (ECFP4, radius=2)
        n_bits = 1024
        fp_config = CircularFingerprintConfig(
            radius=2,
            n_bits=n_bits,
            differentiable=True,
            in_features=DEFAULT_ATOM_FEATURES,
        )
        rngs = nnx.Rngs(42)
        fp_operator = CircularFingerprintOperator(fp_config, rngs=rngs)

        # 3. Featurize molecules
        logger.info("Featurizing molecules...")
        fingerprints, labels, n_valid = _featurize_molecules(
            source, fp_operator
        )
        logger.info(
            "  %d/%d molecules featurized successfully",
            n_valid, n_total,
        )

        # Subsample in quick mode
        if self.quick and n_valid > self.config.quick_subsample:
            fingerprints = fingerprints[: self.config.quick_subsample]
            labels = labels[: self.config.quick_subsample]
            n_valid = self.config.quick_subsample

        # 4. Train and evaluate
        n_epochs = 20 if self.quick else 50
        logger.info(
            "Training MLP classifier for %d epochs...", n_epochs
        )
        metrics = _train_and_evaluate(
            fingerprints, labels, source,
            n_epochs=n_epochs,
        )
        for key, value in sorted(metrics.items()):
            logger.info("  %s: %.4f", key, value)

        # 5. Build gradient-check inputs (single molecule)
        first_elem = source[0]
        if first_elem is not None:
            try:
                single_graph = smiles_to_graph(
                    first_elem.data["smiles"]
                )
            except ValueError:
                single_graph = {
                    "node_features": jnp.ones((3, DEFAULT_ATOM_FEATURES)),
                    "adjacency": jnp.eye(3),
                }
        else:
            single_graph = {
                "node_features": jnp.ones((3, DEFAULT_ATOM_FEATURES)),
                "adjacency": jnp.eye(3),
            }

        def loss_fn(
            model: CircularFingerprintOperator,
            data: dict[str, Any],
        ) -> jnp.ndarray:
            res, _, _ = model.apply(data, {}, None)
            return jnp.sum(res["fingerprint"])

        baselines = MOLNET_BASELINES.get("bbbp", {})

        return {
            "metrics": metrics,
            "operator": fp_operator,
            "input_data": single_graph,
            "loss_fn": loss_fn,
            "n_items": n_valid,
            "iterate_fn": lambda: fp_operator.apply(
                single_graph, {}, None
            ),
            "baselines": baselines,
            "dataset_info": {
                "name": "bbbp",
                "n_total": n_total,
                "n_valid": n_valid,
                "task_type": "classification",
            },
            "operator_config": {
                "radius": fp_config.radius,
                "n_bits": n_bits,
                "n_epochs": n_epochs,
            },
            "operator_name": "CircularFingerprintOperator",
            "dataset_name": "bbbp",
        }


def main() -> None:
    """CLI entry point."""
    DiffBioBenchmark.cli_main(MolNetBenchmark, _CONFIG)


if __name__ == "__main__":
    main()
