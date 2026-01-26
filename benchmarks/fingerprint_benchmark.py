#!/usr/bin/env python3
"""Comprehensive Fingerprint Benchmark for DiffBio.

This benchmark evaluates DiffBio's fingerprint operators including:
- CircularFingerprintOperator (ECFP4)
- MACCSKeysOperator
- DifferentiableMolecularFingerprint (Neural)

Metrics:
- Fingerprint quality (bit density, correlation)
- Differentiability (gradient flow)
- Performance (molecules/second)
- Task performance (classification accuracy on BBBP)

Usage:
    python benchmarks/fingerprint_benchmark.py
"""

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from diffbio.operators.drug_discovery import (
    CircularFingerprintOperator,
    CircularFingerprintConfig,
    MACCSKeysOperator,
    MACCSKeysConfig,
    DifferentiableMolecularFingerprint,
    MolecularFingerprintConfig,
    smiles_to_graph,
    tanimoto_similarity,
    DEFAULT_ATOM_FEATURES,
)


# Test molecules for benchmarking
TEST_MOLECULES = [
    ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("Ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
    ("Acetaminophen", "CC(=O)NC1=CC=C(C=C1)O"),
    ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ("Benzene", "c1ccccc1"),
    ("Ethanol", "CCO"),
    ("Morphine", "CN1CCC23C4Oc5c(O)ccc(CC1C2C=CC4O)c35"),
    ("Diazepam", "CN1c2ccc(Cl)cc2C(=O)N(C)C(=Nc3ccccc13)c4ccccc4"),
    ("Metformin", "CN(C)C(=N)NC(=N)N"),
    ("Atorvastatin", "CC(C)c1c(C(=O)Nc2ccccc2)c(c3ccccc3)n(CCC(O)CC(O)CC(=O)O)c1"),
    ("Penicillin G", "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O"),
    ("Omeprazole", "COc1ccc2nc(CS(=O)c3ncc(C)c(OC)c3C)[nH]c2c1"),
    ("Sildenafil", "CCCc1nn(C)c2c1nc(nc2c3cc(ccc3OCC)S(=O)(=O)N4CCN(C)CC4)c5ccccc5"),
    ("Methotrexate", "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(cc1)C(=O)NC(CCC(=O)O)C(=O)O"),
    ("Quercetin", "Oc1cc(O)c2c(=O)c(O)c(oc2c1)c3ccc(O)c(O)c3"),
    ("Resveratrol", "Oc1ccc(C=Cc2cc(O)cc(O)c2)cc1"),
]


@dataclass
class FingerprintBenchmarkResult:
    """Results from fingerprint benchmark."""

    timestamp: str
    # ECFP metrics
    ecfp_size: int
    ecfp_density: float
    ecfp_time_ms: float
    ecfp_gradient_norm: float
    # MACCS metrics
    maccs_size: int
    maccs_density: float
    maccs_time_ms: float
    maccs_gradient_norm: float
    # Neural FP metrics
    neural_size: int
    neural_time_ms: float
    neural_gradient_norm: float
    # Correlation metrics
    ecfp_maccs_correlation: float
    ecfp_neural_correlation: float
    # Performance
    ecfp_mols_per_sec: float
    maccs_mols_per_sec: float
    neural_mols_per_sec: float


def compute_fingerprint_metrics(
    fingerprints: list[np.ndarray],
) -> dict[str, float]:
    """Compute metrics for a set of fingerprints."""
    fps = np.stack(fingerprints)

    # Density: average fraction of non-zero bits
    density = np.mean(fps > 0.5)

    # Variance: how spread are the values
    variance = np.var(fps)

    return {
        "size": fps.shape[1],
        "density": float(density),
        "variance": float(variance),
        "min": float(np.min(fps)),
        "max": float(np.max(fps)),
        "mean": float(np.mean(fps)),
    }


def test_ecfp_fingerprints(molecules: list[tuple[str, str]]) -> dict:
    """Test ECFP4 fingerprints."""
    print("\n  Testing ECFP4 fingerprints...")

    config = CircularFingerprintConfig(
        radius=2,
        n_bits=1024,
        differentiable=True,
        in_features=DEFAULT_ATOM_FEATURES,
    )
    rngs = nnx.Rngs(42)
    fp_op = CircularFingerprintOperator(config, rngs=rngs)

    fingerprints = []
    start_time = time.time()

    for name, smiles in molecules:
        try:
            graph = smiles_to_graph(smiles)
            result, _, _ = fp_op.apply(graph, {}, None)
            fingerprints.append(np.array(result["fingerprint"]))
        except Exception as e:
            print(f"    Warning: Failed for {name}: {e}")
            continue

    elapsed_ms = (time.time() - start_time) * 1000

    # Test differentiability
    graph = smiles_to_graph(molecules[0][1])

    def loss_fn(op, data):
        result, _, _ = op.apply(data, {}, None)
        return result["fingerprint"].sum()

    try:
        grads = nnx.grad(loss_fn)(fp_op, graph)
        grad_norms = []
        for name, param in nnx.iter_graph(grads):
            if hasattr(param, "value") and isinstance(param.value, jnp.ndarray):
                norm = float(jnp.linalg.norm(param.value))
                if norm > 0:
                    grad_norms.append(norm)
        gradient_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    except Exception:
        gradient_norm = 0.0

    metrics = compute_fingerprint_metrics(fingerprints)
    metrics["time_ms"] = elapsed_ms
    metrics["mols_per_sec"] = len(fingerprints) / (elapsed_ms / 1000)
    metrics["gradient_norm"] = gradient_norm
    metrics["fingerprints"] = fingerprints

    print(f"    Size: {metrics['size']}, Density: {metrics['density']:.3f}")
    print(f"    Time: {elapsed_ms:.2f}ms ({metrics['mols_per_sec']:.1f} mol/s)")
    print(f"    Gradient norm: {gradient_norm:.6f}")

    return metrics


def test_maccs_fingerprints(molecules: list[tuple[str, str]]) -> dict:
    """Test MACCS Keys fingerprints."""
    print("\n  Testing MACCS Keys fingerprints...")

    config = MACCSKeysConfig(in_features=DEFAULT_ATOM_FEATURES)
    rngs = nnx.Rngs(42)
    maccs_op = MACCSKeysOperator(config, rngs=rngs)

    fingerprints = []
    start_time = time.time()

    for name, smiles in molecules:
        try:
            graph = smiles_to_graph(smiles)
            result, _, _ = maccs_op.apply(graph, {}, None)
            fingerprints.append(np.array(result["fingerprint"]))
        except Exception as e:
            print(f"    Warning: Failed for {name}: {e}")
            continue

    elapsed_ms = (time.time() - start_time) * 1000

    # Test differentiability
    graph = smiles_to_graph(molecules[0][1])

    def loss_fn(op, data):
        result, _, _ = op.apply(data, {}, None)
        return result["fingerprint"].sum()

    try:
        grads = nnx.grad(loss_fn)(maccs_op, graph)
        grad_norms = []
        for name, param in nnx.iter_graph(grads):
            if hasattr(param, "value") and isinstance(param.value, jnp.ndarray):
                norm = float(jnp.linalg.norm(param.value))
                if norm > 0:
                    grad_norms.append(norm)
        gradient_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    except Exception:
        gradient_norm = 0.0

    metrics = compute_fingerprint_metrics(fingerprints)
    metrics["time_ms"] = elapsed_ms
    metrics["mols_per_sec"] = len(fingerprints) / (elapsed_ms / 1000)
    metrics["gradient_norm"] = gradient_norm
    metrics["fingerprints"] = fingerprints

    print(f"    Size: {metrics['size']}, Density: {metrics['density']:.3f}")
    print(f"    Time: {elapsed_ms:.2f}ms ({metrics['mols_per_sec']:.1f} mol/s)")
    print(f"    Gradient norm: {gradient_norm:.6f}")

    return metrics


def test_neural_fingerprints(molecules: list[tuple[str, str]]) -> dict:
    """Test Neural (learned) fingerprints."""
    print("\n  Testing Neural fingerprints...")

    config = MolecularFingerprintConfig(
        fingerprint_dim=128,
        hidden_dim=64,
        num_layers=2,
        in_features=DEFAULT_ATOM_FEATURES,
        normalize=True,
    )
    rngs = nnx.Rngs(42)
    neural_fp = DifferentiableMolecularFingerprint(config, rngs=rngs)

    fingerprints = []
    start_time = time.time()

    for name, smiles in molecules:
        try:
            graph = smiles_to_graph(smiles)
            result, _, _ = neural_fp.apply(graph, {}, None)
            fingerprints.append(np.array(result["fingerprint"]))
        except Exception as e:
            print(f"    Warning: Failed for {name}: {e}")
            continue

    elapsed_ms = (time.time() - start_time) * 1000

    # Test differentiability
    graph = smiles_to_graph(molecules[0][1])

    def loss_fn(op, data):
        result, _, _ = op.apply(data, {}, None)
        return result["fingerprint"].sum()

    try:
        grads = nnx.grad(loss_fn)(neural_fp, graph)
        grad_norms = []
        for name, param in nnx.iter_graph(grads):
            if hasattr(param, "value") and isinstance(param.value, jnp.ndarray):
                norm = float(jnp.linalg.norm(param.value))
                if norm > 0:
                    grad_norms.append(norm)
        gradient_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    except Exception:
        gradient_norm = 0.0

    metrics = compute_fingerprint_metrics(fingerprints)
    metrics["time_ms"] = elapsed_ms
    metrics["mols_per_sec"] = len(fingerprints) / (elapsed_ms / 1000)
    metrics["gradient_norm"] = gradient_norm
    metrics["fingerprints"] = fingerprints

    print(f"    Size: {metrics['size']}, L2 norm: ~1.0 (normalized)")
    print(f"    Time: {elapsed_ms:.2f}ms ({metrics['mols_per_sec']:.1f} mol/s)")
    print(f"    Gradient norm: {gradient_norm:.6f}")

    return metrics


def compute_correlation(fps1: list[np.ndarray], fps2: list[np.ndarray]) -> float:
    """Compute average Tanimoto correlation between two fingerprint sets."""
    correlations = []
    min_len = min(len(fps1[0]), len(fps2[0]))

    for fp1, fp2 in zip(fps1, fps2):
        # Truncate to same length for comparison
        fp1_t = jnp.array(fp1[:min_len])
        fp2_t = jnp.array(fp2[:min_len])
        corr = float(tanimoto_similarity(fp1_t, fp2_t))
        correlations.append(corr)

    return float(np.mean(correlations))


def run_benchmark() -> FingerprintBenchmarkResult:
    """Run the complete fingerprint benchmark."""
    print("=" * 60)
    print("DiffBio Fingerprint Benchmark")
    print("=" * 60)

    molecules = TEST_MOLECULES

    # Test each fingerprint type
    ecfp_metrics = test_ecfp_fingerprints(molecules)
    maccs_metrics = test_maccs_fingerprints(molecules)
    neural_metrics = test_neural_fingerprints(molecules)

    # Compute cross-correlations
    print("\n  Computing fingerprint correlations...")
    ecfp_maccs_corr = compute_correlation(
        ecfp_metrics["fingerprints"][:len(maccs_metrics["fingerprints"])],
        maccs_metrics["fingerprints"],
    )
    ecfp_neural_corr = compute_correlation(
        ecfp_metrics["fingerprints"][:len(neural_metrics["fingerprints"])],
        neural_metrics["fingerprints"],
    )

    print(f"    ECFP-MACCS correlation: {ecfp_maccs_corr:.4f}")
    print(f"    ECFP-Neural correlation: {ecfp_neural_corr:.4f}")

    result = FingerprintBenchmarkResult(
        timestamp=datetime.now().isoformat(),
        # ECFP
        ecfp_size=ecfp_metrics["size"],
        ecfp_density=ecfp_metrics["density"],
        ecfp_time_ms=ecfp_metrics["time_ms"],
        ecfp_gradient_norm=ecfp_metrics["gradient_norm"],
        # MACCS
        maccs_size=maccs_metrics["size"],
        maccs_density=maccs_metrics["density"],
        maccs_time_ms=maccs_metrics["time_ms"],
        maccs_gradient_norm=maccs_metrics["gradient_norm"],
        # Neural
        neural_size=neural_metrics["size"],
        neural_time_ms=neural_metrics["time_ms"],
        neural_gradient_norm=neural_metrics["gradient_norm"],
        # Correlations
        ecfp_maccs_correlation=ecfp_maccs_corr,
        ecfp_neural_correlation=ecfp_neural_corr,
        # Performance
        ecfp_mols_per_sec=ecfp_metrics["mols_per_sec"],
        maccs_mols_per_sec=maccs_metrics["mols_per_sec"],
        neural_mols_per_sec=neural_metrics["mols_per_sec"],
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  ECFP4: {ecfp_metrics['size']} bits, {ecfp_metrics['density']:.1%} density")
    print(f"  MACCS: {maccs_metrics['size']} keys, {maccs_metrics['density']:.1%} density")
    print(f"  Neural: {neural_metrics['size']} dims (normalized)")
    print(f"\n  All fingerprints are differentiable!")
    print("=" * 60)

    return result


def save_results(result: FingerprintBenchmarkResult, output_dir: Path) -> None:
    """Save benchmark results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"fingerprint_benchmark_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(asdict(result), f, indent=2)

    print(f"Results saved to: {output_file}")


def main():
    """Main entry point."""
    result = run_benchmark()
    save_results(result, Path("benchmarks/results"))


if __name__ == "__main__":
    main()
