# Splitters API

Dataset splitting utilities for train/validation/test splitting in bioinformatics and drug discovery applications.

## Base Classes

### SplitterModule

::: diffbio.splitters.base.SplitterModule
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - split
        - process
        - k_fold_split
        - create_split_sources

### SplitterConfig

::: diffbio.splitters.base.SplitterConfig
    options:
      show_root_heading: true
      members: []

### SplitResult

::: diffbio.splitters.base.SplitResult
    options:
      show_root_heading: true
      members:
        - train_indices
        - valid_indices
        - test_indices
        - train_size
        - valid_size
        - test_size

## Random Splitters

### RandomSplitter

::: diffbio.splitters.random.RandomSplitter
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - split
        - k_fold_split

### RandomSplitterConfig

::: diffbio.splitters.random.RandomSplitterConfig
    options:
      show_root_heading: true
      members: []

### StratifiedSplitter

::: diffbio.splitters.random.StratifiedSplitter
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - split

### StratifiedSplitterConfig

::: diffbio.splitters.random.StratifiedSplitterConfig
    options:
      show_root_heading: true
      members: []

## Molecular Splitters

### ScaffoldSplitter

::: diffbio.splitters.molecular.ScaffoldSplitter
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - split

### ScaffoldSplitterConfig

::: diffbio.splitters.molecular.ScaffoldSplitterConfig
    options:
      show_root_heading: true
      members: []

### TanimotoClusterSplitter

::: diffbio.splitters.molecular.TanimotoClusterSplitter
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - split

### TanimotoClusterSplitterConfig

::: diffbio.splitters.molecular.TanimotoClusterSplitterConfig
    options:
      show_root_heading: true
      members: []

## Sequence Splitters

### SequenceIdentitySplitter

::: diffbio.splitters.sequence.SequenceIdentitySplitter
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - split

### SequenceIdentitySplitterConfig

::: diffbio.splitters.sequence.SequenceIdentitySplitterConfig
    options:
      show_root_heading: true
      members: []

## Usage Examples

### Basic Random Splitting

```python
from diffbio.splitters import RandomSplitter, RandomSplitterConfig
from flax import nnx

# Create splitter with 80/10/10 split
config = RandomSplitterConfig(
    train_frac=0.8,
    valid_frac=0.1,
    test_frac=0.1,
    seed=42,
)
splitter = RandomSplitter(config, rngs=nnx.Rngs(42))

# Split a data source
result = splitter.split(data_source)
print(f"Train: {result.train_size}")
print(f"Valid: {result.valid_size}")
print(f"Test: {result.test_size}")
```

### Stratified Splitting

```python
from diffbio.splitters import StratifiedSplitter, StratifiedSplitterConfig

# Preserve class distribution in splits
config = StratifiedSplitterConfig(
    train_frac=0.8,
    valid_frac=0.1,
    test_frac=0.1,
    label_key="label",  # Key in data element containing class label
    seed=42,
)
splitter = StratifiedSplitter(config, rngs=nnx.Rngs(42))
result = splitter.split(data_source)
```

### Scaffold Splitting (Drug Discovery)

```python
from diffbio.splitters import ScaffoldSplitter, ScaffoldSplitterConfig

# Split by Bemis-Murcko scaffold (requires RDKit)
config = ScaffoldSplitterConfig(
    smiles_key="smiles",
    train_frac=0.8,
    valid_frac=0.1,
    test_frac=0.1,
)
splitter = ScaffoldSplitter(config)
result = splitter.split(molecule_source)
# Similar molecules (same scaffold) end up in same split
```

### Tanimoto Cluster Splitting

```python
from diffbio.splitters import TanimotoClusterSplitter, TanimotoClusterSplitterConfig

# Split by fingerprint similarity clustering (requires RDKit)
config = TanimotoClusterSplitterConfig(
    smiles_key="smiles",
    fingerprint_type="morgan",  # or "rdkit", "maccs"
    fingerprint_radius=2,
    similarity_cutoff=0.6,
)
splitter = TanimotoClusterSplitter(config)
result = splitter.split(molecule_source)
```

### Sequence Identity Splitting (Bioinformatics)

```python
from diffbio.splitters import SequenceIdentitySplitter, SequenceIdentitySplitterConfig

# Split by sequence identity clustering
config = SequenceIdentitySplitterConfig(
    sequence_key="sequence",
    identity_threshold=0.3,  # Max identity between train/test
    alignment_method="simple",  # or "mmseqs2"
)
splitter = SequenceIdentitySplitter(config)
result = splitter.split(sequence_source)
# Similar sequences end up in same split
```

### Creating Split Data Sources

```python
from diffbio.splitters import ScaffoldSplitter, ScaffoldSplitterConfig

# Create splitter
config = ScaffoldSplitterConfig(smiles_key="smiles")
splitter = ScaffoldSplitter(config)

# Create separate data sources for each split
train_source, valid_source, test_source = splitter.create_split_sources(
    data_source,
    lazy=True,  # Use IndexedViewSource for memory efficiency
)

# Use with Datarax samplers
from datarax.samplers import ShuffleSampler, ShuffleSamplerConfig
sampler_config = ShuffleSamplerConfig(batch_size=32)
train_sampler = ShuffleSampler(sampler_config, data_source=train_source)
```

### K-Fold Cross-Validation

```python
from diffbio.splitters import RandomSplitter, RandomSplitterConfig

config = RandomSplitterConfig(seed=42)
splitter = RandomSplitter(config, rngs=nnx.Rngs(42))

# Get 5-fold splits
folds = splitter.k_fold_split(data_source, k=5)

for fold_idx, (train_indices, val_indices) in enumerate(folds):
    print(f"Fold {fold_idx}: {len(train_indices)} train, {len(val_indices)} val")
```

## Input Specifications

### All Splitters

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_source` | DataSourceModule | Datarax data source to split |

### ScaffoldSplitter / TanimotoClusterSplitter

Data elements must contain:

| Key | Type | Description |
|-----|------|-------------|
| `smiles_key` | str | SMILES string for the molecule |

### SequenceIdentitySplitter

Data elements must contain:

| Key | Type | Description |
|-----|------|-------------|
| `sequence_key` | str | DNA/RNA/protein sequence |

### StratifiedSplitter

Data elements must contain:

| Key | Type | Description |
|-----|------|-------------|
| `label_key` | int/str | Class label for stratification |

## Output Specifications

### SplitResult

| Field | Type | Description |
|-------|------|-------------|
| `train_indices` | jnp.ndarray | Indices for training set |
| `valid_indices` | jnp.ndarray | Indices for validation set |
| `test_indices` | jnp.ndarray | Indices for test set |

## Splitter Comparison

| Splitter | Use Case | Domain | Dependencies |
|----------|----------|--------|--------------|
| RandomSplitter | General purpose | Any | None |
| StratifiedSplitter | Class-imbalanced data | Any | None |
| ScaffoldSplitter | Drug discovery | Chemistry | RDKit |
| TanimotoClusterSplitter | Molecular similarity | Chemistry | RDKit |
| SequenceIdentitySplitter | Genomics/Proteomics | Biology | None |
