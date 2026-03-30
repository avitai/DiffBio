#!/usr/bin/env python
"""Benchmark: DiffBio CircularFingerprintOperator vs DeepChem CircularFingerprint.

This script compares DiffBio's CircularFingerprintOperator (in RDKit mode) against
DeepChem's CircularFingerprint featurizer to validate correctness and measure
performance differences.

Usage:
    # Install benchmark dependencies first
    uv pip install diffbio[benchmark]

    # Run the benchmark
    python benchmarks/circular_fingerprint_benchmark.py

Metrics compared:
    1. Fingerprint correlation (Tanimoto similarity)
    2. Bit-level agreement (exact match percentage)
    3. Computation speed (molecules/second)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np


# DiffBio imports
from diffbio.operators.drug_discovery import (
    CircularFingerprintConfig,
    CircularFingerprintOperator,
    tanimoto_similarity,
)


# Test molecules - diverse drug-like compounds from various sources
# Includes FDA-approved drugs, natural products, and common pharmaceutical scaffolds
TEST_MOLECULES = [
    # Simple organic molecules
    ("Ethanol", "CCO"),
    ("Benzene", "c1ccccc1"),
    ("Acetic acid", "CC(=O)O"),
    ("Phenol", "c1ccc(O)cc1"),
    ("Aniline", "c1ccc(N)cc1"),
    ("Toluene", "Cc1ccccc1"),
    ("Acetone", "CC(=O)C"),
    ("Glycine", "NCC(=O)O"),
    ("Urea", "NC(=O)N"),
    ("Formaldehyde", "C=O"),
    # Common heterocycles
    ("Pyridine", "c1ccncc1"),
    ("Furan", "c1ccoc1"),
    ("Thiophene", "c1ccsc1"),
    ("Imidazole", "c1cnc[nH]1"),
    ("Indole", "c1ccc2[nH]ccc2c1"),
    ("Quinoline", "c1ccc2ncccc2c1"),
    ("Pyrrole", "c1cc[nH]c1"),
    ("Oxazole", "c1coc[nH]1"),
    ("Thiazole", "c1cscn1"),
    ("Purine", "c1ncc2[nH]cnc2n1"),
    # NSAIDs (Non-steroidal anti-inflammatory drugs)
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("Naproxen", "COc1ccc2cc(ccc2c1)C(C)C(=O)O"),
    ("Diclofenac", "OC(=O)Cc1ccccc1Nc2c(Cl)cccc2Cl"),
    ("Indomethacin", "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c3ccc(Cl)cc3"),
    ("Celecoxib", "Cc1ccc(c(c1)c2cc(nn2c3ccc(cc3)S(=O)(=O)N)C(F)(F)F)C"),
    # Analgesics
    ("Acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
    ("Morphine", "CN1CCC23C4Oc5c(O)ccc(CC1C2C=CC4O)c35"),
    ("Codeine", "COc1ccc2CC3N(C)CCC4C5Oc1c2C45C=CC3O"),
    ("Tramadol", "COc1ccccc1C2(O)CCCCC2CN(C)C"),
    # Antibiotics
    ("Penicillin G", "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O"),
    ("Amoxicillin", "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O"),
    ("Ciprofloxacin", "OC(=O)c1cn(C2CC2)c3cc(N4CCNCC4)c(F)cc3c1=O"),
    # fmt: off
    (
        "Azithromycin",
        "CC1C(O)C(C)C(=O)OC(CC(C)OC2OC(C)C(N(C)C)C(O)C2O)C(C)C(OC3OC(C)C(O)(C(C)C3)C)C(C)C(C)(O)CC(C)C1N(C)C",
    ),  # noqa: E501
    # fmt: on
    ("Metronidazole", "Cc1ncc(n1CCO)[N+](=O)[O-]"),
    # Cardiovascular drugs
    ("Atorvastatin", "CC(C)c1c(C(=O)Nc2ccccc2)c(c3ccccc3)n(CCC(O)CC(O)CC(=O)O)c1"),
    ("Lisinopril", "NCCCC(NC(CCc1ccccc1)C(=O)O)C(=O)N2CCCC2C(=O)O"),
    ("Amlodipine", "CCOC(=O)C1=C(COCCN)NC(C)=C(C1c2ccccc2Cl)C(=O)OC"),
    ("Metoprolol", "CC(C)NCC(O)COc1ccc(CCOC)cc1"),
    ("Warfarin", "CC(=O)CC(c1ccccc1)c2c(O)c3ccccc3oc2=O"),
    # CNS drugs
    ("Caffeine", "Cn1cnc2c1c(=O)n(c(=O)n2C)C"),
    ("Diazepam", "CN1c2ccc(Cl)cc2C(=O)N(C)C(=Nc3ccccc13)c4ccccc4"),
    ("Fluoxetine", "CNCCC(c1ccccc1)Oc2ccc(cc2)C(F)(F)F"),
    ("Sertraline", "CNC1CCC(c2ccc(Cl)c(Cl)c2)c3ccccc13"),
    ("Risperidone", "Cc1cccc(c1)c2nc3n(CCC4CCN(CC4)c5ncc(C)c(=O)n5C)cccc3n2"),
    # Antidiabetic drugs
    ("Metformin", "CN(C)C(=N)NC(=N)N"),
    ("Glibenclamide", "COc1ccc(Cl)cc1C(=O)NCCc2ccc(cc2)S(=O)(=O)NC(=O)NC3CCCCC3"),
    ("Pioglitazone", "CCc1ccc(CCOc2ccc(cc2)CC3SC(=O)NC3=O)nc1"),
    # Anticancer drugs
    ("Doxorubicin", "COc1cccc2c1C(=O)c3c(O)c4CC(O)(CC(OC5CC(N)C(O)C(C)O5)c4c(O)c3C2=O)C(=O)CO"),
    ("Methotrexate", "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(cc1)C(=O)NC(CCC(=O)O)C(=O)O"),
    ("Tamoxifen", "CCC(=C(c1ccccc1)c2ccc(OCCN(C)C)cc2)c3ccccc3"),
    ("Imatinib", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc4nccc(n4)c5cccnc5"),
    # Proton pump inhibitors
    ("Omeprazole", "COc1ccc2nc(CS(=O)c3ncc(C)c(OC)c3C)[nH]c2c1"),
    ("Lansoprazole", "Cc1c(OCC(F)(F)F)ccnc1CS(=O)c2nc3ccccc3[nH]2"),
    ("Pantoprazole", "COc1ccnc(CS(=O)c2nc3ccc(OC(F)F)cc3[nH]2)c1OC"),
    # Antihistamines
    ("Diphenhydramine", "CN(C)CCOC(c1ccccc1)c2ccccc2"),
    ("Cetirizine", "OC(=O)COCCN1CCN(CC1)C(c2ccccc2)c3ccc(Cl)cc3"),
    ("Loratadine", "CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc4ncccc24)CC1"),
    # Antiviral drugs
    ("Acyclovir", "Nc1nc2c(ncn2COCCO)c(=O)[nH]1"),
    ("Oseltamivir", "CCOC(=O)C1=CC(OC(CC)CC)C(NC(C)=O)C(N)C1"),
    ("Remdesivir", "CCC(CC)COC(=O)C(C)NP(=O)(OCC1OC(C#N)(c2ccc3c(N)ncnn23)C(O)C1O)Oc4ccccc4"),
    # Natural products
    ("Quercetin", "Oc1cc(O)c2c(=O)c(O)c(oc2c1)c3ccc(O)c(O)c3"),
    ("Curcumin", "COc1cc(C=CC(=O)CC(=O)C=Cc2ccc(O)c(OC)c2)ccc1O"),
    ("Resveratrol", "Oc1ccc(C=Cc2cc(O)cc(O)c2)cc1"),
    ("Capsaicin", "COc1cc(CNC(=O)CCCC\\C=C\\C(C)C)ccc1O"),
    # Steroids
    ("Cortisol", "CC12CCC(=O)C=C1CCC3C2C(O)CC4(C)C3CCC4(O)C(=O)CO"),
    ("Prednisone", "CC12CC(=O)C3C(CCC4=CC(=O)C=CC34C)C1CCC2(O)C(=O)CO"),
    ("Testosterone", "CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O"),
    # Vitamins
    ("Vitamin C", "OCC(O)C1OC(=O)C(O)=C1O"),
    ("Vitamin B6", "Cc1ncc(CO)c(CO)c1O"),
    ("Nicotinamide", "NC(=O)c1cccnc1"),
    # Additional FDA-approved drugs for diversity
    ("Sildenafil", "CCCc1nn(C)c2c1nc(nc2c3cc(ccc3OCC)S(=O)(=O)N4CCN(C)CC4)c5ccccc5"),
    ("Tadalafil", "CN1CC(=O)N2C(Cc3c([nH]c4ccccc34)C2c5ccc6OCOc6c5)C1=O"),
    ("Gabapentin", "NCC1(CCCCC1)CC(=O)O"),
    ("Pregabalin", "CC(C)CC(CN)CC(=O)O"),
    ("Duloxetine", "CNCCC(Oc1cccc2ccccc12)c3cccs3"),
    ("Venlafaxine", "COc1ccc(C(CN(C)C)C2(O)CCCCC2)cc1"),
    ("Escitalopram", "CN(C)CCCC1(OCc2cc(C#N)ccc12)c3ccc(F)cc3"),
    ("Aripiprazole", "Clc1cccc(N2CCN(CCCCOc3ccc4CCC(=O)Nc4c3)CC2)c1Cl"),
    ("Quetiapine", "OCCOCCN1CCN(CC1)c2nc3ccccc3Sc4ccccc24"),
    ("Olanzapine", "Cc1cc2Nc3ccccc3N=C(N4CCN(C)CC4)c2s1"),
    ("Clopidogrel", "COC(=O)C(c1ccccc1Cl)N2CCc3sccc3C2"),
    ("Montelukast", "CC(C)(O)c1ccccc1CCC(SCC2(CC(=O)O)CC2)c3ccc(cc3)C=Cc4ccc5ccc(Cl)cc5n4"),
    ("Losartan", "CCCCc1nc(Cl)c(CO)n1Cc2ccc(cc2)c3ccccc3c4nnn[nH]4"),
    ("Valsartan", "CCCCC(=O)N(Cc1ccc(cc1)c2ccccc2c3nnn[nH]3)C(C(C)C)C(=O)O"),
    # Kinase inhibitors
    ("Erlotinib", "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"),
    ("Gefitinib", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN4CCOCC4"),
    ("Sorafenib", "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1"),
]


def validate_smiles(smiles_list: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Validate SMILES strings and return only valid ones.

    Uses RDKit to validate that each SMILES can be parsed.
    """
    try:
        from rdkit import Chem
    except ImportError as err:
        raise ImportError("RDKit is required for validation") from err

    valid = []
    invalid_count = 0
    for name, smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid.append((name, smiles))
        else:
            invalid_count += 1
            print(f"  Warning: Invalid SMILES for {name}: {smiles}")

    if invalid_count > 0:
        print(f"  Filtered out {invalid_count} invalid molecules")

    return valid


@dataclass(frozen=True, kw_only=True)
class CircularFingerprintBenchmarkResult:
    """Results from a single circular fingerprint benchmark run."""

    name: str
    n_molecules: int
    # Accuracy metrics
    mean_tanimoto: float
    min_tanimoto: float
    max_tanimoto: float
    exact_match_rate: float
    mean_bit_agreement: float
    # Performance metrics
    diffbio_time_ms: float
    deepchem_time_ms: float
    diffbio_mols_per_sec: float
    deepchem_mols_per_sec: float
    speedup: float


def compute_bit_agreement(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute percentage of bits that match exactly."""
    # Convert to binary
    fp1_binary = (fp1 > 0).astype(int)
    fp2_binary = (fp2 > 0).astype(int)
    return np.mean(fp1_binary == fp2_binary)


def run_diffbio_fingerprints(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[np.ndarray, float]:
    """Compute fingerprints using DiffBio (RDKit mode)."""
    config = CircularFingerprintConfig(
        radius=radius,
        n_bits=n_bits,
        differentiable=False,  # Use RDKit for exact comparison
    )
    fp_op = CircularFingerprintOperator(config)

    fingerprints = []
    start_time = time.perf_counter()

    for smiles in smiles_list:
        data = {"smiles": smiles}
        result, _, _ = fp_op.apply(data, {}, None)
        fingerprints.append(np.array(result["fingerprint"]))

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return np.array(fingerprints), elapsed_ms


def run_deepchem_fingerprints(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[np.ndarray, float]:
    """Compute fingerprints using DeepChem."""
    try:
        from deepchem.feat import CircularFingerprint  # pyright: ignore[reportMissingImports]
    except ImportError as err:
        raise ImportError(
            "DeepChem not installed. Install with: uv pip install diffbio[benchmark]"
        ) from err

    featurizer = CircularFingerprint(radius=radius, size=n_bits)

    # DeepChem can return ragged arrays if some molecules fail
    # Process one at a time to handle failures gracefully
    fingerprints = []
    start_time = time.perf_counter()
    for smiles in smiles_list:
        fp = featurizer.featurize([smiles])
        if fp is not None and len(fp) > 0 and hasattr(fp[0], "__len__") and len(fp[0]) == n_bits:
            fingerprints.append(fp[0])
        else:
            # Return zero fingerprint for failed molecules
            fingerprints.append(np.zeros(n_bits, dtype=np.float32))
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return np.array(fingerprints), elapsed_ms


def compare_fingerprints(
    diffbio_fps: np.ndarray,
    deepchem_fps: np.ndarray,
) -> dict:
    """Compare two sets of fingerprints."""
    n_mols = len(diffbio_fps)

    # Compute Tanimoto similarities between corresponding fingerprints
    tanimotos = []
    bit_agreements = []
    exact_matches = 0

    for i in range(n_mols):
        fp1 = jnp.array(diffbio_fps[i])
        fp2 = jnp.array(deepchem_fps[i])

        # Tanimoto similarity
        sim = float(tanimoto_similarity(fp1, fp2))
        tanimotos.append(sim)

        # Bit agreement
        agreement = compute_bit_agreement(diffbio_fps[i], deepchem_fps[i])
        bit_agreements.append(agreement)

        # Exact match
        if np.allclose(diffbio_fps[i], deepchem_fps[i]):
            exact_matches += 1

    return {
        "mean_tanimoto": np.mean(tanimotos),
        "min_tanimoto": np.min(tanimotos),
        "max_tanimoto": np.max(tanimotos),
        "std_tanimoto": np.std(tanimotos),
        "exact_match_rate": exact_matches / n_mols,
        "mean_bit_agreement": np.mean(bit_agreements),
        "tanimotos": tanimotos,
    }


def _run_single_comparison(
    name: str,
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
    n_warmup: int = 5,
    n_runs: int = 20,
) -> CircularFingerprintBenchmarkResult:
    """Run a single comparison between DiffBio and DeepChem."""
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {name}")
    print(f"  Molecules: {len(smiles_list)}")
    print(f"  Radius: {radius}, Bits: {n_bits}")
    print(f"{'=' * 60}")

    # Warmup runs - use full dataset to ensure JIT compilation is complete
    # This ensures compilation overhead is not included in timed runs
    print("  Warming up (full dataset to trigger JIT compilation)...")
    for _ in range(n_warmup):
        run_diffbio_fingerprints(smiles_list, radius, n_bits)
        run_deepchem_fingerprints(smiles_list, radius, n_bits)

    # Timed runs
    diffbio_times = []
    deepchem_times = []

    print(f"  Running {n_runs} timed iterations...")
    for _ in range(n_runs):
        _, diffbio_time = run_diffbio_fingerprints(smiles_list, radius, n_bits)
        _, deepchem_time = run_deepchem_fingerprints(smiles_list, radius, n_bits)
        diffbio_times.append(diffbio_time)
        deepchem_times.append(deepchem_time)

    # Get fingerprints for accuracy comparison
    diffbio_fps, _ = run_diffbio_fingerprints(smiles_list, radius, n_bits)
    deepchem_fps, _ = run_deepchem_fingerprints(smiles_list, radius, n_bits)

    # Compare fingerprints
    comparison = compare_fingerprints(diffbio_fps, deepchem_fps)

    # Compute statistics
    avg_diffbio_time = np.mean(diffbio_times)
    avg_deepchem_time = np.mean(deepchem_times)
    n_mols = len(smiles_list)

    result = CircularFingerprintBenchmarkResult(
        name=name,
        n_molecules=n_mols,
        mean_tanimoto=float(comparison["mean_tanimoto"]),
        min_tanimoto=float(comparison["min_tanimoto"]),
        max_tanimoto=float(comparison["max_tanimoto"]),
        exact_match_rate=float(comparison["exact_match_rate"]),
        mean_bit_agreement=float(
            comparison["mean_bit_agreement"],
        ),
        diffbio_time_ms=float(avg_diffbio_time),
        deepchem_time_ms=float(avg_deepchem_time),
        diffbio_mols_per_sec=float(
            (n_mols / avg_diffbio_time) * 1000,
        ),
        deepchem_mols_per_sec=float(
            (n_mols / avg_deepchem_time) * 1000,
        ),
        speedup=float(avg_deepchem_time / avg_diffbio_time),
    )

    return result


def print_results(
    results: list[CircularFingerprintBenchmarkResult],
) -> None:
    """Print benchmark results in a formatted table."""
    try:
        from tabulate import tabulate
    except ImportError:
        print("tabulate not installed, using simple output")
        for r in results:
            print(f"\n{r.name}:")
            print(f"  Tanimoto: {r.mean_tanimoto:.4f} (min: {r.min_tanimoto:.4f})")
            print(f"  Bit Agreement: {r.mean_bit_agreement:.2%}")
            print(f"  Exact Match: {r.exact_match_rate:.2%}")
            print(f"  DiffBio: {r.diffbio_mols_per_sec:.1f} mol/s")
            print(f"  DeepChem: {r.deepchem_mols_per_sec:.1f} mol/s")
        return

    print("\n" + "=" * 80)
    print("ACCURACY COMPARISON")
    print("=" * 80)

    accuracy_table = [
        [
            r.name,
            f"{r.mean_tanimoto:.4f}",
            f"{r.min_tanimoto:.4f}",
            f"{r.mean_bit_agreement:.2%}",
            f"{r.exact_match_rate:.2%}",
        ]
        for r in results
    ]
    print(
        tabulate(
            accuracy_table,
            headers=[
                "Benchmark",
                "Mean Tanimoto",
                "Min Tanimoto",
                "Bit Agreement",
                "Exact Match",
            ],
            tablefmt="grid",
        )
    )

    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    perf_table = [
        [
            r.name,
            r.n_molecules,
            f"{r.diffbio_time_ms:.2f}",
            f"{r.deepchem_time_ms:.2f}",
            f"{r.diffbio_mols_per_sec:.1f}",
            f"{r.deepchem_mols_per_sec:.1f}",
            f"{r.speedup:.2f}x",
        ]
        for r in results
    ]
    print(
        tabulate(
            perf_table,
            headers=[
                "Benchmark",
                "N Mols",
                "DiffBio (ms)",
                "DeepChem (ms)",
                "DiffBio (mol/s)",
                "DeepChem (mol/s)",
                "Speedup",
            ],
            tablefmt="grid",
        )
    )


def _save_results(
    results: list[CircularFingerprintBenchmarkResult],
    output_dir: Path,
) -> None:
    """Save benchmark results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = (
        output_dir
        / f"circular_fingerprint_benchmark_{timestamp}.json"
    )

    data = {
        "timestamp": timestamp,
        "benchmarks": [asdict(r) for r in results],
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def run_benchmark(
    *,
    quick: bool = False,
) -> CircularFingerprintBenchmarkResult:
    """Run a representative circular fingerprint benchmark.

    When ``quick=True``, uses only the first 10 molecules and
    fewer timed runs. Used by ``run_all.py``.

    Args:
        quick: If True, use reduced molecule set and fewer runs.

    Returns:
        CircularFingerprintBenchmarkResult for ECFP4 comparison.
    """
    print("=" * 80)
    print("DiffBio vs DeepChem: Circular Fingerprint Benchmark")
    print("=" * 80)

    print("\nValidating molecules...")
    if quick:
        valid_molecules = validate_smiles(TEST_MOLECULES[:10])
    else:
        valid_molecules = validate_smiles(TEST_MOLECULES)
    print(
        f"Using {len(valid_molecules)} valid molecules "
        f"out of {len(TEST_MOLECULES)}"
    )

    smiles_list = [smiles for _, smiles in valid_molecules]
    n_warmup = 2 if quick else 5
    n_runs = 5 if quick else 20

    # Always run ECFP4 as the representative benchmark
    result = _run_single_comparison(
        name="ECFP4 (radius=2, 2048 bits)",
        smiles_list=smiles_list,
        radius=2,
        n_bits=2048,
        n_warmup=n_warmup,
        n_runs=n_runs,
    )

    save_benchmark_result(
        result=asdict(result),
        domain="drug_discovery",
        benchmark_name="circular_fingerprint_benchmark",
    )
    return result


def run_all_comparisons() -> None:
    """Run the full comparison suite (all radii and bit sizes).

    Called by ``main()`` for the CLI entry point.
    """
    print("=" * 80)
    print("DiffBio vs DeepChem: Circular Fingerprint Benchmark")
    print("=" * 80)

    print("\nValidating molecules...")
    valid_molecules = validate_smiles(TEST_MOLECULES)
    print(
        f"Using {len(valid_molecules)} valid molecules "
        f"out of {len(TEST_MOLECULES)}"
    )

    smiles_list = [smiles for _, smiles in valid_molecules]

    results: list[CircularFingerprintBenchmarkResult] = []

    # Benchmark 1: ECFP4 (radius=2)
    results.append(
        _run_single_comparison(
            name="ECFP4 (radius=2, 2048 bits)",
            smiles_list=smiles_list,
            radius=2,
            n_bits=2048,
        )
    )

    # Benchmark 2: ECFP6 (radius=3)
    results.append(
        _run_single_comparison(
            name="ECFP6 (radius=3, 2048 bits)",
            smiles_list=smiles_list,
            radius=3,
            n_bits=2048,
        )
    )

    # Benchmark 3: Different bit sizes
    results.append(
        _run_single_comparison(
            name="ECFP4 (radius=2, 1024 bits)",
            smiles_list=smiles_list,
            radius=2,
            n_bits=1024,
        )
    )

    results.append(
        _run_single_comparison(
            name="ECFP4 (radius=2, 4096 bits)",
            smiles_list=smiles_list,
            radius=2,
            n_bits=4096,
        )
    )

    # Print results
    print_results(results)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    avg_tanimoto = np.mean([r.mean_tanimoto for r in results])
    avg_bit_agreement = np.mean(
        [r.mean_bit_agreement for r in results],
    )
    avg_speedup = np.mean([r.speedup for r in results])

    print(f"Average Tanimoto Similarity: {avg_tanimoto:.4f}")
    print(f"Average Bit Agreement: {avg_bit_agreement:.2%}")
    print(
        f"Average Speedup (DiffBio vs DeepChem): "
        f"{avg_speedup:.2f}x"
    )

    if avg_tanimoto >= 0.99:
        print(
            "\n[PASS] DiffBio fingerprints are highly "
            "consistent with DeepChem"
        )
    elif avg_tanimoto >= 0.95:
        print(
            "\n[PASS] DiffBio fingerprints are consistent "
            "with DeepChem (minor differences)"
        )
    else:
        print(
            "\n[WARN] Significant differences detected "
            "between implementations"
        )

    # Save results
    output_dir = Path(__file__).parent / "results"
    _save_results(results, output_dir)


def main() -> None:
    """Run the full benchmark suite."""
    run_all_comparisons()


if __name__ == "__main__":
    main()
