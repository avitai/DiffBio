# Dataset Splitters

DiffBio provides dataset splitting utilities that ensure proper train/validation/test separation for bioinformatics and drug discovery applications.

<span class="operator-md">Data Preparation</span> <span class="diff-structural">Structural Module</span>

## Overview

Proper dataset splitting is critical for unbiased model evaluation. DiffBio splitters address domain-specific challenges:

- **Drug Discovery**: Structurally similar molecules should not appear in both train and test sets
- **Bioinformatics**: Homologous sequences should be grouped together to prevent data leakage

## Splitter Hierarchy

```
SplitterModule (StructuralModule)
├── RandomSplitter          # Simple random splitting
├── StratifiedSplitter      # Preserve class distribution
├── ScaffoldSplitter        # Molecular scaffold-based (drug discovery)
├── TanimotoClusterSplitter # Fingerprint similarity (drug discovery)
└── SequenceIdentitySplitter # Sequence identity (bioinformatics)
```

## Why Domain-Specific Splitting Matters

### The Data Leakage Problem

Standard random splitting can lead to overly optimistic performance estimates:

```python
# BAD: Random splitting allows similar molecules in train and test
# Molecule A (train): Aspirin with methyl group
# Molecule B (test):  Aspirin with ethyl group
# Model memorizes scaffold, appears to "predict" well
```

### The Solution: Structure-Aware Splitting

```python
# GOOD: Scaffold splitting keeps similar molecules together
# All aspirin analogs in train OR test, not both
# Model must generalize to unseen scaffolds
```

## Random Splitters

### RandomSplitter

Simple random splitting for general-purpose use.

```python
from diffbio.splitters import RandomSplitter, RandomSplitterConfig
from diffbio.sources import MolNetSource, MolNetSourceConfig
from flax import nnx

# Load data
source = MolNetSource(MolNetSourceConfig(dataset_name="esol"))

# Configure splitter
config = RandomSplitterConfig(
    train_frac=0.8,
    valid_frac=0.1,
    test_frac=0.1,
    seed=42,  # For reproducibility
)
splitter = RandomSplitter(config, rngs=nnx.Rngs(42))

# Get split indices
result = splitter.split(source)

print(f"Train samples: {result.train_size}")
print(f"Valid samples: {result.valid_size}")
print(f"Test samples: {result.test_size}")
```

### StratifiedSplitter

Preserves class distribution in each split - essential for imbalanced datasets.

```python
from diffbio.splitters import StratifiedSplitter, StratifiedSplitterConfig

config = StratifiedSplitterConfig(
    train_frac=0.8,
    valid_frac=0.1,
    test_frac=0.1,
    label_key="y",  # Key containing class labels
    seed=42,
)
splitter = StratifiedSplitter(config, rngs=nnx.Rngs(42))
result = splitter.split(source)

# Each split maintains similar class proportions
```

## Molecular Splitters (Drug Discovery)

### ScaffoldSplitter

Splits by Bemis-Murcko molecular scaffold - the industry standard for drug discovery benchmarks.

```
Molecule Examples:
├── Aspirin (salicylate scaffold) → Train
│   ├── Aspirin
│   ├── Methyl salicylate
│   └── Salicylic acid
├── Ibuprofen (phenylpropanoic scaffold) → Valid
│   ├── Ibuprofen
│   └── Naproxen
└── Caffeine (xanthine scaffold) → Test
    ├── Caffeine
    └── Theobromine
```

```python
from diffbio.splitters import ScaffoldSplitter, ScaffoldSplitterConfig

# Requires RDKit: pip install rdkit
config = ScaffoldSplitterConfig(
    smiles_key="smiles",  # Key in data elements
    train_frac=0.8,
    valid_frac=0.1,
    test_frac=0.1,
)
splitter = ScaffoldSplitter(config)

result = splitter.split(molecule_source)
# Similar scaffolds grouped together
```

**How it works:**

1. Extract Bemis-Murcko scaffold from each molecule
2. Group molecules by scaffold
3. Assign scaffold groups to splits (largest groups first)
4. All molecules with same scaffold end up in same split

### TanimotoClusterSplitter

Clusters molecules by fingerprint similarity using the Butina algorithm.

```python
from diffbio.splitters import TanimotoClusterSplitter, TanimotoClusterSplitterConfig

config = TanimotoClusterSplitterConfig(
    smiles_key="smiles",
    fingerprint_type="morgan",  # "morgan", "rdkit", or "maccs"
    fingerprint_radius=2,       # Radius for Morgan fingerprints
    fingerprint_bits=2048,      # Number of bits
    similarity_cutoff=0.6,      # Tanimoto similarity threshold
    train_frac=0.8,
)
splitter = TanimotoClusterSplitter(config)
result = splitter.split(molecule_source)
```

**Fingerprint Types:**

| Type | Description | Best For |
|------|-------------|----------|
| `morgan` | Circular fingerprints (ECFP) | General similarity |
| `rdkit` | RDKit topological fingerprints | Substructure patterns |
| `maccs` | 166 structural keys | Quick screening |

**How it works:**

1. Compute fingerprints for all molecules
2. Calculate pairwise Tanimoto similarities
3. Cluster using Butina algorithm (similar molecules grouped)
4. Assign clusters to splits

## Sequence Splitters (Bioinformatics)

### SequenceIdentitySplitter

Clusters sequences by identity threshold - essential for genomics and proteomics.

```python
from diffbio.splitters import SequenceIdentitySplitter, SequenceIdentitySplitterConfig

config = SequenceIdentitySplitterConfig(
    sequence_key="sequence",     # Key containing sequence
    identity_threshold=0.3,      # Max identity between train/test
    alignment_method="simple",   # "simple" or "mmseqs2"
    train_frac=0.8,
)
splitter = SequenceIdentitySplitter(config)
result = splitter.split(sequence_source)
```

**Identity Threshold Guidelines:**

| Task | Threshold | Rationale |
|------|-----------|-----------|
| Protein function | 0.3 | Homologs share function |
| Secondary structure | 0.25 | Similar structure at low identity |
| Binding site prediction | 0.4 | Higher similarity needed |
| DNA regulatory motifs | 0.7 | Conserved regions |

**How it works:**

1. Greedy clustering by sequence identity
2. First sequence becomes cluster representative
3. New sequences join cluster if identity > threshold
4. Assign clusters to splits

## Creating Split Data Sources

After splitting, create separate data sources for training:

```python
from diffbio.splitters import ScaffoldSplitter, ScaffoldSplitterConfig
from diffbio.sources import MolNetSource, MolNetSourceConfig

# Load full dataset
source = MolNetSource(MolNetSourceConfig(dataset_name="bbbp"))

# Create splitter
splitter = ScaffoldSplitter(ScaffoldSplitterConfig(smiles_key="smiles"))

# Create separate sources for each split
train_source, valid_source, test_source = splitter.create_split_sources(
    source,
    lazy=True,  # Memory-efficient: load on demand
)

# Use directly or with Datarax samplers
print(f"Train: {len(train_source)} samples")
print(f"Valid: {len(valid_source)} samples")
print(f"Test: {len(test_source)} samples")
```

### Lazy vs Eager Loading

```python
# LAZY (recommended for large datasets)
train, valid, test = splitter.create_split_sources(source, lazy=True)
# Uses IndexedViewSource - elements loaded on demand
# Lower memory, slightly slower iteration

# EAGER (for small datasets or repeated iteration)
train, valid, test = splitter.create_split_sources(source, lazy=False)
# Uses MemorySource - all elements in memory
# Higher memory, faster iteration
```

## K-Fold Cross-Validation

```python
from diffbio.splitters import RandomSplitter, RandomSplitterConfig

config = RandomSplitterConfig(seed=42)
splitter = RandomSplitter(config, rngs=nnx.Rngs(42))

# Get k-fold splits
k = 5
folds = splitter.k_fold_split(source, k=k)

for fold_idx, (train_indices, val_indices) in enumerate(folds):
    print(f"Fold {fold_idx + 1}:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Valid: {len(val_indices)} samples")
```

## Integration with Datarax

```python
from diffbio.sources import MolNetSource, MolNetSourceConfig
from diffbio.splitters import ScaffoldSplitter, ScaffoldSplitterConfig
from datarax.samplers import ShuffleSampler, ShuffleSamplerConfig
from flax import nnx

# 1. Load dataset
source = MolNetSource(MolNetSourceConfig(dataset_name="tox21"))

# 2. Split by scaffold
splitter = ScaffoldSplitter(ScaffoldSplitterConfig(smiles_key="smiles"))
train_source, valid_source, test_source = splitter.create_split_sources(
    source, lazy=True
)

# 3. Create samplers
train_sampler = ShuffleSampler(
    ShuffleSamplerConfig(batch_size=32),
    data_source=train_source,
    rngs=nnx.Rngs(42),
)

valid_sampler = ShuffleSampler(
    ShuffleSamplerConfig(batch_size=32, shuffle=False),
    data_source=valid_source,
)

# 4. Training loop
for epoch in range(10):
    for batch in train_sampler:
        # Train on batch
        pass

    for batch in valid_sampler:
        # Validate
        pass
```

## Best Practices

### Drug Discovery

1. **Always use scaffold or similarity splitting** for fair evaluation
2. **ScaffoldSplitter** for diverse compound libraries
3. **TanimotoClusterSplitter** for congeneric series

### Bioinformatics

1. **Use SequenceIdentitySplitter** for protein/gene tasks
2. **Choose threshold based on task** (lower = stricter)
3. **Consider MMseqs2** for large datasets (faster clustering)

### General

1. **Set random seed** for reproducibility
2. **Use stratified splitting** for imbalanced classification
3. **Check split sizes** match expected fractions
4. **Verify no overlap** between splits

## Splitter Selection Guide

| Use Case | Recommended Splitter |
|----------|---------------------|
| General ML | RandomSplitter |
| Imbalanced classes | StratifiedSplitter |
| Drug discovery benchmark | ScaffoldSplitter |
| Lead optimization | TanimotoClusterSplitter |
| Protein function prediction | SequenceIdentitySplitter |
| Genomic sequence analysis | SequenceIdentitySplitter |
