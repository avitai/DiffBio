# Benchmarks

This page documents the benchmarking methodology and results for DiffBio operators compared to reference implementations.

## Circular Fingerprint Benchmark

Compares DiffBio's `CircularFingerprintOperator` against DeepChem's `CircularFingerprint` featurizer to validate correctness and measure performance.

### Metrics

The benchmark evaluates:

1. **Fingerprint Accuracy**
   - **Tanimoto Similarity**: Measures similarity between DiffBio and DeepChem fingerprints (1.0 = identical)
   - **Bit Agreement**: Percentage of bits that match exactly
   - **Exact Match Rate**: Percentage of molecules with identical fingerprints

2. **Performance**
   - **Computation Time**: Milliseconds to compute fingerprints
   - **Throughput**: Molecules processed per second
   - **Speedup**: Ratio of DeepChem time to DiffBio time

### Test Dataset

The benchmark uses **87 diverse drug-like compounds** covering:

| Category | Examples | Count |
|----------|----------|-------|
| Simple organic molecules | Ethanol, Benzene, Phenol, Glycine | 10 |
| Common heterocycles | Pyridine, Thiophene, Imidazole, Indole, Purine | 9 |
| NSAIDs | Aspirin, Ibuprofen, Naproxen, Diclofenac, Celecoxib | 6 |
| Analgesics | Acetaminophen, Morphine, Codeine, Tramadol | 4 |
| Antibiotics | Penicillin G, Amoxicillin, Ciprofloxacin, Azithromycin | 5 |
| Cardiovascular drugs | Atorvastatin, Lisinopril, Amlodipine, Warfarin | 5 |
| CNS drugs | Caffeine, Diazepam, Fluoxetine, Sertraline, Risperidone | 5 |
| Antidiabetic drugs | Metformin, Glibenclamide, Pioglitazone | 3 |
| Anticancer drugs | Doxorubicin, Methotrexate, Tamoxifen, Imatinib | 4 |
| Proton pump inhibitors | Omeprazole, Lansoprazole, Pantoprazole | 3 |
| Antihistamines | Diphenhydramine, Cetirizine, Loratadine | 3 |
| Antiviral drugs | Acyclovir, Oseltamivir, Remdesivir | 3 |
| Natural products | Quercetin, Curcumin, Resveratrol, Capsaicin | 4 |
| Steroids | Cortisol, Prednisone, Testosterone | 3 |
| Vitamins | Vitamin C, Vitamin B6, Nicotinamide | 3 |
| Additional FDA-approved | Sildenafil, Gabapentin, Escitalopram, Clopidogrel, etc. | 17 |
| Kinase inhibitors | Erlotinib, Gefitinib, Sorafenib | 3 |

### Running the Benchmark

#### Prerequisites

Install benchmark dependencies:

```bash
uv pip install -e ".[benchmark]"
```

#### Execute

```bash
source activate.sh
python benchmarks/circular_fingerprint_benchmark.py
```

### Results

#### Accuracy Comparison

| Benchmark | Mean Tanimoto | Min Tanimoto | Bit Agreement | Exact Match |
|-----------|---------------|--------------|---------------|-------------|
| ECFP4 (radius=2, 2048 bits) | 1.0 | 1.0 | 100.00% | 100.00% |
| ECFP6 (radius=3, 2048 bits) | 1.0 | 1.0 | 100.00% | 100.00% |
| ECFP4 (radius=2, 1024 bits) | 1.0 | 1.0 | 100.00% | 100.00% |
| ECFP4 (radius=2, 4096 bits) | 1.0 | 1.0 | 100.00% | 100.00% |

**Result**: DiffBio fingerprints are **100% identical** to DeepChem reference implementation.

#### Performance Comparison

| Benchmark | N Mols | DiffBio (ms) | DeepChem (ms) | DiffBio (mol/s) | DeepChem (mol/s) | Speedup |
|-----------|--------|--------------|---------------|-----------------|------------------|---------|
| ECFP4 (radius=2, 2048 bits) | 87 | 35.31 | 63.99 | 2,464 | 1,360 | 1.81x |
| ECFP6 (radius=3, 2048 bits) | 87 | 34.34 | 69.89 | 2,533 | 1,245 | 2.03x |
| ECFP4 (radius=2, 1024 bits) | 87 | 40.75 | 50.23 | 2,135 | 1,732 | 1.23x |
| ECFP4 (radius=2, 4096 bits) | 87 | 39.58 | 105.77 | 2,198 | 823 | 2.67x |

**Result**: DiffBio is **1.94x faster** on average, with up to **2.67x speedup** for larger fingerprints.

### Summary

| Metric | Value |
|--------|-------|
| Test molecules | 87 FDA-approved drugs and drug-like compounds |
| Average Tanimoto Similarity | 1.0000 |
| Average Bit Agreement | 100.00% |
| Average Speedup | 1.94x |

DiffBio's `CircularFingerprintOperator` achieves:

- **Perfect accuracy**: 100% match with DeepChem reference
- **Superior performance**: ~2x faster than DeepChem
- **Better scaling**: Larger fingerprints show greater speedup (up to 2.67x for 4096 bits)

### Implementation Details

The performance advantage comes from:

1. **Optimized RDKit integration**: Using `ConvertToNumpyArray` for efficient bit vector conversion
2. **Minimal memory allocation**: Direct buffer writes instead of intermediate objects
3. **Efficient JAX conversion**: Zero-copy `jnp.asarray()` wrapping of numpy arrays

### Benchmark Methodology

- **Validation**: Molecules are validated with RDKit before benchmarking
- **Warmup**: 5 full-dataset warmup runs to ensure JIT compilation is complete before timing
- **Iterations**: 20 timed iterations per configuration
- **Timing**: `time.perf_counter()` for high-resolution timing

### Test Environment

| Component | Specification |
|-----------|---------------|
| Platform | Linux 6.8.0 (Ubuntu) |
| Python | 3.12.6 |
| JAX | 0.8.0 |
| GPU | NVIDIA GeForce RTX 4090 (24 GB) |
| Backend | gpu (CUDA) |

The benchmark uses `source activate.sh` to properly configure the GPU environment before running.

### Adding New Benchmarks

1. Create a new Python script in `benchmarks/`
2. Follow the pattern in `circular_fingerprint_benchmark.py`
3. Results are saved to `benchmarks/results/` (gitignored)
4. Update this documentation with the new benchmark description
