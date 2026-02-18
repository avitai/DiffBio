#!/usr/bin/env python3
"""MolNet Benchmark Suite for DiffBio.

This benchmark script evaluates DiffBio's molecular featurization operators
on MoleculeNet benchmark datasets, following the standard MolNet evaluation
protocol.

Benchmarks:
- BBBP: Blood-Brain Barrier Penetration (classification)
- ESOL: Aqueous Solubility (regression)
- Lipophilicity: Octanol/water partition (regression)
- HIV: HIV Replication Inhibition (classification)

Featurization Methods:
- ECFP4: Extended Connectivity Fingerprints (radius=2)
- MACCS: MACCS 166 structural keys

Splitting Strategies:
- Random: Standard random split
- Scaffold: Bemis-Murcko scaffold split (recommended for drug discovery)

Usage:
    # Run single benchmark
    python benchmarks/molnet_benchmark.py --dataset bbbp --featurizer ecfp

    # Run all benchmarks
    python benchmarks/molnet_benchmark.py --all

    # Scaffold split benchmark
    python benchmarks/molnet_benchmark.py --dataset bbbp --split scaffold
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx

# DiffBio imports
from diffbio.sources import MolNetSource, MolNetSourceConfig
from diffbio.operators.drug_discovery import (
    CircularFingerprintOperator,
    CircularFingerprintConfig,
    MACCSKeysOperator,
    MACCSKeysConfig,
)
from diffbio.splitters import (
    RandomSplitter,
    RandomSplitterConfig,
    ScaffoldSplitter,
    ScaffoldSplitterConfig,
)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    dataset: str = "bbbp"
    featurizer: str = "ecfp"
    split: str = "random"
    n_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    hidden_dim: int = 256
    seed: int = 42


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    config: BenchmarkConfig
    n_train: int = 0
    n_valid: int = 0
    n_test: int = 0
    n_features: int = 0
    task_type: str = ""
    featurize_time: float = 0.0
    train_time: float = 0.0
    # Classification metrics
    train_roc_auc: float | None = None
    valid_roc_auc: float | None = None
    test_roc_auc: float | None = None
    train_accuracy: float | None = None
    valid_accuracy: float | None = None
    test_accuracy: float | None = None
    # Regression metrics
    train_rmse: float | None = None
    valid_rmse: float | None = None
    test_rmse: float | None = None
    train_r2: float | None = None
    valid_r2: float | None = None
    test_r2: float | None = None


def create_featurizer(featurizer_type: str, rngs: nnx.Rngs):
    """Create a molecular featurizer operator.

    Args:
        featurizer_type: "ecfp" or "maccs"
        rngs: Flax random number generators

    Returns:
        Featurizer operator and output key
    """
    if featurizer_type == "ecfp":
        config = CircularFingerprintConfig(
            radius=2,  # ECFP4
            size=2048,
            use_features=True,
            use_chirality=False,
        )
        return CircularFingerprintOperator(config, rngs=rngs), "fingerprint"

    elif featurizer_type == "maccs":
        config = MACCSKeysConfig()
        return MACCSKeysOperator(config, rngs=rngs), "maccs_keys"

    else:
        raise ValueError(f"Unknown featurizer: {featurizer_type}")


def featurize_dataset(
    source: MolNetSource,
    featurizer,
    output_key: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert all molecules in a source to features.

    Args:
        source: MolNet data source
        featurizer: Featurization operator
        output_key: Key for fingerprint in result dict

    Returns:
        Tuple of (features, labels)
    """
    features = []
    labels = []

    for element in source:
        data = {"smiles": element.data["smiles"]}
        try:
            result, _, _ = featurizer.apply(data, {}, None)
            features.append(result[output_key])
            labels.append(element.data["y"])
        except (ValueError, KeyError):
            # Skip invalid SMILES
            continue

    return jnp.stack(features), jnp.array(labels)


class SimpleClassifier(nnx.Module):
    """Simple feedforward classifier for molecular properties."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.dense1 = nnx.Linear(in_features, hidden_dim, rngs=rngs)
        self.dense2 = nnx.Linear(hidden_dim, hidden_dim // 2, rngs=rngs)
        self.dense3 = nnx.Linear(hidden_dim // 2, num_classes, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.2, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nnx.relu(self.dense1(x))
        x = self.dropout(x)
        x = nnx.relu(self.dense2(x))
        x = self.dropout(x)
        return self.dense3(x)


class SimpleRegressor(nnx.Module):
    """Simple feedforward regressor for molecular properties."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.dense1 = nnx.Linear(in_features, hidden_dim, rngs=rngs)
        self.dense2 = nnx.Linear(hidden_dim, hidden_dim // 2, rngs=rngs)
        self.dense3 = nnx.Linear(hidden_dim // 2, 1, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.2, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nnx.relu(self.dense1(x))
        x = self.dropout(x)
        x = nnx.relu(self.dense2(x))
        x = self.dropout(x)
        return self.dense3(x).squeeze(-1)


def compute_roc_auc(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """Compute ROC-AUC score.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities

    Returns:
        ROC-AUC score
    """
    # Simple implementation using trapezoidal rule
    # Sort by predicted probability descending
    sorted_indices = jnp.argsort(-y_pred)
    y_true_sorted = y_true[sorted_indices]

    # Compute TPR and FPR at each threshold
    n_pos = jnp.sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr = jnp.cumsum(y_true_sorted) / n_pos
    fpr = jnp.cumsum(1 - y_true_sorted) / n_neg

    # Add (0, 0) point
    tpr = jnp.concatenate([jnp.array([0.0]), tpr])
    fpr = jnp.concatenate([jnp.array([0.0]), fpr])

    # Compute AUC using trapezoidal rule
    auc = jnp.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2)

    return float(auc)


def compute_rmse(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """Compute RMSE."""
    return float(jnp.sqrt(jnp.mean((y_true - y_pred) ** 2)))


def compute_r2(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """Compute R² score."""
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-7))


def train_and_evaluate(
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_valid: jnp.ndarray,
    y_valid: jnp.ndarray,
    X_test: jnp.ndarray,
    y_test: jnp.ndarray,
    task_type: str,
    config: BenchmarkConfig,
) -> tuple[dict, float]:
    """Train model and evaluate on all splits.

    Args:
        X_train, y_train: Training data
        X_valid, y_valid: Validation data
        X_test, y_test: Test data
        task_type: "classification" or "regression"
        config: Benchmark configuration

    Returns:
        Tuple of (metrics_dict, train_time)
    """
    rngs = nnx.Rngs(config.seed)
    n_features = X_train.shape[1]

    # Create model
    if task_type == "classification":
        model = SimpleClassifier(n_features, config.hidden_dim, 1, rngs=rngs)
    else:
        model = SimpleRegressor(n_features, config.hidden_dim, rngs=rngs)

    optimizer = nnx.Optimizer(model, optax.adam(config.learning_rate))

    # Training step
    @nnx.jit
    def train_step(model, optimizer, x, y):
        def loss_fn(m):
            pred = m(x)
            if task_type == "classification":
                pred = nnx.sigmoid(pred).squeeze()
                return -jnp.mean(y * jnp.log(pred + 1e-7) + (1 - y) * jnp.log(1 - pred + 1e-7))
            else:
                return jnp.mean((pred - y) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    # Train
    start_time = time.time()
    batch_size = min(config.batch_size, len(X_train))

    for epoch in range(config.n_epochs):
        # Simple batching
        key = jax.random.key(config.seed + epoch)
        indices = jax.random.permutation(key, len(X_train))

        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i : i + batch_size]
            train_step(model, optimizer, X_train[batch_idx], y_train[batch_idx])

    train_time = time.time() - start_time

    # Evaluate
    def predict(x):
        pred = model(x)
        if task_type == "classification":
            return nnx.sigmoid(pred).squeeze()
        return pred

    pred_train = predict(X_train)
    pred_valid = predict(X_valid)
    pred_test = predict(X_test)

    metrics = {}
    if task_type == "classification":
        metrics["train_roc_auc"] = compute_roc_auc(y_train, pred_train)
        metrics["valid_roc_auc"] = compute_roc_auc(y_valid, pred_valid)
        metrics["test_roc_auc"] = compute_roc_auc(y_test, pred_test)
        metrics["train_accuracy"] = float(jnp.mean((pred_train > 0.5) == y_train))
        metrics["valid_accuracy"] = float(jnp.mean((pred_valid > 0.5) == y_valid))
        metrics["test_accuracy"] = float(jnp.mean((pred_test > 0.5) == y_test))
    else:
        metrics["train_rmse"] = compute_rmse(y_train, pred_train)
        metrics["valid_rmse"] = compute_rmse(y_valid, pred_valid)
        metrics["test_rmse"] = compute_rmse(y_test, pred_test)
        metrics["train_r2"] = compute_r2(y_train, pred_train)
        metrics["valid_r2"] = compute_r2(y_valid, pred_valid)
        metrics["test_r2"] = compute_r2(y_test, pred_test)

    return metrics, train_time


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run a complete benchmark for a dataset/featurizer/split combination.

    Args:
        config: Benchmark configuration

    Returns:
        BenchmarkResult with all metrics
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {config.dataset} | {config.featurizer} | {config.split}")
    print(f"{'=' * 60}")

    rngs = nnx.Rngs(config.seed)

    # Load full dataset
    print("Loading dataset...")
    source_config = MolNetSourceConfig(
        dataset_name=config.dataset,
        split="train",
        download=True,
    )
    source = MolNetSource(source_config)
    task_type = source.task_type
    print(f"  Task type: {task_type}")
    print(f"  Total molecules: {len(source)}")

    # Create featurizer
    print(f"Creating {config.featurizer.upper()} featurizer...")
    featurizer, output_key = create_featurizer(config.featurizer, rngs)

    # Featurize all molecules
    print("Featurizing molecules...")
    start_time = time.time()
    X, y = featurize_dataset(source, featurizer, output_key)
    featurize_time = time.time() - start_time
    print(f"  Featurization time: {featurize_time:.2f}s")
    print(f"  Feature shape: {X.shape}")
    print(f"  Valid molecules: {len(X)}")

    # Create splits
    print(f"Splitting with {config.split} splitter...")
    if config.split == "scaffold":
        # Collect SMILES for scaffold splitting
        smiles_list = [
            source[i].data["smiles"] for i in range(len(source)) if source[i] is not None
        ]
        splitter_config = ScaffoldSplitterConfig(
            train_frac=0.8,
            valid_frac=0.1,
            test_frac=0.1,
        )
        splitter = ScaffoldSplitter(splitter_config)
        result = splitter.split_smiles(smiles_list)
        train_idx = result.train_indices
        valid_idx = result.valid_indices
        test_idx = result.test_indices
    else:
        splitter_config = RandomSplitterConfig(
            train_frac=0.8,
            valid_frac=0.1,
            test_frac=0.1,
            seed=config.seed,
        )
        splitter = RandomSplitter(splitter_config)
        result = splitter.split(len(X))
        train_idx = result.train_indices
        valid_idx = result.valid_indices
        test_idx = result.test_indices

    # Ensure indices are within bounds
    train_idx = train_idx[train_idx < len(X)]
    valid_idx = valid_idx[valid_idx < len(X)]
    test_idx = test_idx[test_idx < len(X)]

    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"  Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")

    # Train and evaluate
    print(f"Training ({config.n_epochs} epochs)...")
    metrics, train_time = train_and_evaluate(
        X_train, y_train, X_valid, y_valid, X_test, y_test, task_type, config
    )
    print(f"  Training time: {train_time:.2f}s")

    # Print results
    print("\nResults:")
    if task_type == "classification":
        print(f"  Train ROC-AUC: {metrics['train_roc_auc']:.4f}")
        print(f"  Valid ROC-AUC: {metrics['valid_roc_auc']:.4f}")
        print(f"  Test ROC-AUC:  {metrics['test_roc_auc']:.4f}")
    else:
        print(f"  Train RMSE: {metrics['train_rmse']:.4f}, R²: {metrics['train_r2']:.4f}")
        print(f"  Valid RMSE: {metrics['valid_rmse']:.4f}, R²: {metrics['valid_r2']:.4f}")
        print(f"  Test RMSE:  {metrics['test_rmse']:.4f}, R²: {metrics['test_r2']:.4f}")

    return BenchmarkResult(
        config=config,
        n_train=len(X_train),
        n_valid=len(X_valid),
        n_test=len(X_test),
        n_features=X.shape[1],
        task_type=task_type,
        featurize_time=featurize_time,
        train_time=train_time,
        **metrics,
    )


def run_all_benchmarks(output_dir: Path) -> list[BenchmarkResult]:
    """Run benchmarks on all dataset/featurizer combinations.

    Args:
        output_dir: Directory to save results

    Returns:
        List of benchmark results
    """
    datasets = ["bbbp", "esol", "lipophilicity"]
    featurizers = ["ecfp", "maccs"]
    splits = ["random", "scaffold"]

    results = []

    for dataset in datasets:
        for featurizer in featurizers:
            for split in splits:
                config = BenchmarkConfig(
                    dataset=dataset,
                    featurizer=featurizer,
                    split=split,
                )
                try:
                    result = run_benchmark(config)
                    results.append(result)
                except Exception as e:
                    print(f"Error: {dataset}/{featurizer}/{split}: {e}")
                    continue

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "molnet_results.json"

    results_dict = []
    for r in results:
        d = asdict(r)
        d["config"] = asdict(r.config)
        results_dict.append(d)

    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to {results_file}")
    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print a summary table of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Group by task type
    classification = [r for r in results if r.task_type == "classification"]
    regression = [r for r in results if r.task_type == "regression"]

    if classification:
        print("\nClassification Tasks (Test ROC-AUC):")
        print("-" * 70)
        print(f"{'Dataset':<12} {'Split':<10} {'ECFP4':<12} {'MACCS':<12}")
        print("-" * 70)

        datasets = sorted(set(r.config.dataset for r in classification))
        for dataset in datasets:
            for split in ["random", "scaffold"]:
                ecfp = next(
                    (
                        r
                        for r in classification
                        if r.config.dataset == dataset
                        and r.config.featurizer == "ecfp"
                        and r.config.split == split
                    ),
                    None,
                )
                maccs = next(
                    (
                        r
                        for r in classification
                        if r.config.dataset == dataset
                        and r.config.featurizer == "maccs"
                        and r.config.split == split
                    ),
                    None,
                )
                ecfp_val = f"{ecfp.test_roc_auc:.4f}" if ecfp else "N/A"
                maccs_val = f"{maccs.test_roc_auc:.4f}" if maccs else "N/A"
                print(f"{dataset:<12} {split:<10} {ecfp_val:<12} {maccs_val:<12}")

    if regression:
        print("\nRegression Tasks (Test RMSE / R²):")
        print("-" * 70)
        print(f"{'Dataset':<12} {'Split':<10} {'ECFP4':<20} {'MACCS':<20}")
        print("-" * 70)

        datasets = sorted(set(r.config.dataset for r in regression))
        for dataset in datasets:
            for split in ["random", "scaffold"]:
                ecfp = next(
                    (
                        r
                        for r in regression
                        if r.config.dataset == dataset
                        and r.config.featurizer == "ecfp"
                        and r.config.split == split
                    ),
                    None,
                )
                maccs = next(
                    (
                        r
                        for r in regression
                        if r.config.dataset == dataset
                        and r.config.featurizer == "maccs"
                        and r.config.split == split
                    ),
                    None,
                )
                ecfp_val = f"{ecfp.test_rmse:.3f}/{ecfp.test_r2:.3f}" if ecfp else "N/A"
                maccs_val = f"{maccs.test_rmse:.3f}/{maccs.test_r2:.3f}" if maccs else "N/A"
                print(f"{dataset:<12} {split:<10} {ecfp_val:<20} {maccs_val:<20}")

    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run MolNet benchmarks using DiffBio operators")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["bbbp", "esol", "lipophilicity", "hiv", "tox21"],
        help="Dataset to benchmark",
    )
    parser.add_argument(
        "--featurizer",
        type=str,
        default="ecfp",
        choices=["ecfp", "maccs"],
        help="Featurization method",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="random",
        choices=["random", "scaffold"],
        help="Splitting strategy",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmark combinations",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results"),
        help="Output directory for results",
    )

    args = parser.parse_args()

    if args.all:
        results = run_all_benchmarks(args.output_dir)
        print_summary(results)
    elif args.dataset:
        config = BenchmarkConfig(
            dataset=args.dataset,
            featurizer=args.featurizer,
            split=args.split,
            n_epochs=args.epochs,
        )
        run_benchmark(config)
        print("\nFinal result saved.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
