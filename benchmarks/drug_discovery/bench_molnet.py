#!/usr/bin/env python3
"""MolNet molecular property prediction benchmark.

Evaluates DiffBio's CircularFingerprintOperator (ECFP4, radius=2)
on the BBBP (Blood-Brain Barrier Penetration) classification task
from the MoleculeNet benchmark suite.

Pipeline:
    1. Load BBBP via MolNetSource
    2. Convert SMILES to graphs, featurize with CircularFingerprintOperator
    3. Split 80/10/10 via RandomSplitter
    4. Train MLP classifier on fingerprints (~50 epochs)
    5. Evaluate ROC-AUC on test set

Note: The CircularFingerprintOperator uses a hash-based message-passing
architecture that approximates Morgan fingerprints.  Its GNN encoder
produces structured features by design; end-to-end training requires
graph batching/padding infrastructure not yet implemented.

Results are compared against published baselines:
GCN (0.877), AttentiveFP (0.858), D-MPNN (0.910).

Usage:
    python benchmarks/drug_discovery/bench_molnet.py
    python benchmarks/drug_discovery/bench_molnet.py --quick
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Any

import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from benchmarks._base import DiffBioBenchmark, DiffBioBenchmarkConfig
from benchmarks._baselines.molnet import MOLNET_BASELINES
from benchmarks._optimizers import create_benchmark_optimizer
from diffbio.operators.drug_discovery import (
    CircularFingerprintConfig,
    CircularFingerprintOperator,
    DEFAULT_ATOM_FEATURES,
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


def featurize_molecules(
    source: Iterable[Any],
    fp_operator: CircularFingerprintOperator,
    *,
    limit: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, np.ndarray]:
    """Fingerprint the source's usable molecules, in source order.

    A molecule with a NaN label or a SMILES string RDKit cannot parse is skipped.

    Args:
        source: Elements whose ``data`` holds ``smiles`` and ``y``, such as a MolNetSource.
        fp_operator: Fingerprint operator for featurization.
        limit: Stop after this many usable molecules; all of them when None.

    Returns:
        Tuple of (fingerprints, labels, positions): row ``r`` holds the molecule at
        source position ``positions[r]``.
    """
    fingerprints: list[jnp.ndarray] = []
    labels: list[float] = []
    positions: list[int] = []

    for position, element in enumerate(source):
        if limit is not None and len(labels) >= limit:
            break
        label = float(element.data["y"])
        if math.isnan(label):
            continue
        try:
            graph = smiles_to_graph(element.data["smiles"])
        except ValueError:
            continue

        result, _, _ = fp_operator.apply(graph, {}, None)
        fingerprints.append(result["fingerprint"])
        labels.append(label)
        positions.append(position)

    return jnp.stack(fingerprints), jnp.array(labels), np.asarray(positions, dtype=np.int64)


def rows_for_positions(split_positions: Any, featurized_positions: np.ndarray) -> np.ndarray:
    """Return the fingerprint rows of the split's molecules, in the split's order.

    A splitter names molecules by source position; fingerprint rows skip the molecules that
    were not featurized, so the two differ after the first skipped one. Positions with no row
    are left out.

    Args:
        split_positions: Source positions from a splitter.
        featurized_positions: Source position of each fingerprint row, increasing.

    Returns:
        Row indices into the fingerprint and label arrays.
    """
    row_of = {int(position): row for row, position in enumerate(featurized_positions)}
    rows = [row_of[int(p)] for p in np.asarray(split_positions) if int(p) in row_of]
    return np.asarray(rows, dtype=np.int64)


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
        auc += (fpr_values[i] - fpr_values[i - 1]) * (tpr_values[i] + tpr_values[i - 1]) / 2.0
    return float(auc)


def _train_and_evaluate(
    fingerprints: jnp.ndarray,
    labels: jnp.ndarray,
    positions: np.ndarray,
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
        positions: Source position of each fingerprint row.
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

    n_valid = fingerprints.shape[0]
    train_idx = rows_for_positions(split.train_indices, positions)
    test_idx = rows_for_positions(split.test_indices, positions)

    x_train = fingerprints[train_idx]
    y_train = labels[train_idx]
    x_test = fingerprints[test_idx]
    y_test = labels[test_idx]

    rngs = nnx.Rngs(seed)
    model = _MLPClassifier(n_bits, hidden_dim=128, rngs=rngs)
    optimizer = nnx.Optimizer(
        model,
        create_benchmark_optimizer(model, learning_rate=learning_rate),
        wrt=nnx.Param,
    )

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
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y))

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
        limit = self.config.quick_subsample if self.quick else None
        fingerprints, labels, positions = featurize_molecules(source, fp_operator, limit=limit)
        n_valid = int(labels.shape[0])
        logger.info("  %d of %d molecules featurized", n_valid, n_total)

        # 4. Train MLP classifier on fingerprints
        n_epochs = 20 if self.quick else 50
        logger.info("Training MLP classifier for %d epochs...", n_epochs)
        metrics = _train_and_evaluate(
            fingerprints,
            labels,
            positions,
            source,
            n_epochs=n_epochs,
        )
        for key, value in sorted(metrics.items()):
            logger.info("  %s: %.4f", key, value)

        # 5. Build gradient-check inputs (single molecule)
        first_elem = source[0]
        if first_elem is not None:
            try:
                single_graph = smiles_to_graph(first_elem.data["smiles"])
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
            "iterate_fn": lambda: fp_operator.apply(single_graph, {}, None),
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
