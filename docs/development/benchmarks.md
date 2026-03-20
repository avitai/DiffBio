# Benchmarks

DiffBio includes benchmark scripts that validate operator correctness against
reference implementations and measure performance. Benchmarks live in the
`benchmarks/` directory and write results to `benchmarks/results/` (gitignored).

---

## Running Benchmarks

```bash
source ./activate.sh
uv run python benchmarks/<script_name>.py
```

Each benchmark prints a summary to stdout and saves detailed results as JSON.

---

## Available Benchmarks

### Circular Fingerprint Benchmark

Compares `CircularFingerprintOperator` against DeepChem's `CircularFingerprint`
featurizer on 87 FDA-approved drug-like compounds.

```bash
uv run python benchmarks/circular_fingerprint_benchmark.py
```

**Metrics**: Tanimoto similarity, bit agreement, exact match rate, throughput.

| Configuration | Tanimoto | Bit Agreement | Speedup |
|---|---|---|---|
| ECFP4 (radius=2, 2048 bits) | 1.000 | 100.00% | 1.81x |
| ECFP6 (radius=3, 2048 bits) | 1.000 | 100.00% | 2.03x |
| ECFP4 (radius=2, 4096 bits) | 1.000 | 100.00% | 2.67x |

DiffBio fingerprints are 100% identical to DeepChem and 1.9x faster on
average, with better scaling for larger fingerprints.

### Fingerprint Benchmark (Extended)

Evaluates all fingerprint operators (circular, MACCS, neural) with
differentiability and JIT verification.

```bash
uv run python benchmarks/fingerprint_benchmark.py
```

**Metrics**: Fingerprint accuracy, gradient flow, JIT compilation, throughput
comparison across fingerprint types.

### Alignment Benchmark

Evaluates `SmoothSmithWaterman` for correctness (vs standard SW), temperature
sensitivity, differentiability verification, and JIT performance.

```bash
uv run python benchmarks/alignment_benchmark.py
```

**Metrics**: Score accuracy across temperature values, gradient norm, JIT
speedup ratio, memory usage.

### MolNet Benchmark

Evaluates molecular featurization operators on MoleculeNet benchmark datasets,
following the standard MolNet evaluation protocol.

```bash
uv run python benchmarks/molnet_benchmark.py
```

**Metrics**: AUROC/RMSE on standard MolNet tasks (BACE, HIV, Tox21, etc.),
featurization throughput, comparison against published baselines.

### scVI Benchmark

Evaluates `VAENormalizer` with ZINB likelihood against scVI-style metrics on
synthetic PBMC-like data.

```bash
uv run python benchmarks/scvi_benchmark.py
```

**Metrics**: ELBO convergence, reconstruction MSE, latent silhouette score,
batch ASW, ARI, NMI (via calibrax). See the
[scVI Benchmark Example](../examples/advanced/scvi-benchmark.md) for a
walkthrough with figures.

### Single-Cell Benchmark

Evaluates single-cell operators (`DifferentiableHarmony`, `SoftKMeansClustering`,
etc.) on synthetic multi-batch datasets.

```bash
uv run python benchmarks/singlecell_benchmark.py
```

**Metrics**: Batch mixing (entropy), clustering ARI/NMI, silhouette score,
training convergence rate.

---

## Benchmark Methodology

All benchmarks follow a consistent protocol:

1. **Warmup**: 5 full runs to ensure JIT compilation before timing
2. **Iterations**: 20 timed runs per configuration
3. **Timing**: `time.perf_counter()` for high-resolution measurement
4. **Validation**: Input molecules/sequences validated before benchmarking
5. **Reproducibility**: Fixed random seeds, deterministic data generation

### Test Environment

| Component | Value |
|---|---|
| Platform | Linux 6.8.0 (Ubuntu) |
| Python | 3.12.6 |
| JAX | 0.9.0.1 |
| GPU | NVIDIA GeForce RTX 4090 (24 GB) |
| Backend | CUDA (gpu) |

---

## Adding a New Benchmark

1. Create `benchmarks/<name>_benchmark.py` following the existing pattern
2. Use `time.perf_counter()` for timing, fixed seeds for reproducibility
3. Save results to `benchmarks/results/<name>/` as JSON
4. Add a section to this page documenting metrics and results
5. Include a `if __name__ == "__main__"` guard so the script runs standalone

---

## Results Storage

Benchmark results are saved to `benchmarks/results/` which is gitignored.
To regenerate results:

```bash
source ./activate.sh
for f in benchmarks/*_benchmark.py; do
    echo "=== Running $(basename $f) ==="
    uv run python "$f"
done
```
