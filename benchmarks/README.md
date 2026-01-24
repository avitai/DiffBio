# DiffBio Benchmarks

This directory contains benchmarks comparing DiffBio implementations against reference libraries.

## Available Benchmarks

### Circular Fingerprint Benchmark

Compares DiffBio's `CircularFingerprintOperator` against DeepChem's `CircularFingerprint` featurizer.

**Metrics:**
- Fingerprint correlation (Tanimoto similarity)
- Bit-level agreement (exact match percentage)
- Computation speed (molecules/second)

## Running Benchmarks

### Prerequisites

Install benchmark dependencies:

```bash
uv pip install -e ".[benchmark]"
```

### Run Circular Fingerprint Benchmark

```bash
python benchmarks/circular_fingerprint_benchmark.py
```

Results are saved to `benchmarks/results/` as JSON files.

## Results

Benchmark results are stored in the `results/` subdirectory with timestamps.

### Expected Results

For the Circular Fingerprint benchmark:
- **Tanimoto Similarity**: Should be ~1.0 (identical fingerprints)
- **Bit Agreement**: Should be ~100% when using RDKit mode
- **Performance**: DiffBio and DeepChem should have similar performance in RDKit mode

## Adding New Benchmarks

1. Create a new Python script in this directory
2. Follow the pattern in `circular_fingerprint_benchmark.py`
3. Save results to the `results/` subdirectory
4. Update this README with the new benchmark description
