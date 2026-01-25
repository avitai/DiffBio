# Data Sources

DiffBio provides data source modules that integrate seamlessly with the Datarax framework for loading bioinformatics and drug discovery datasets.

<span class="operator-md">Data Loading</span> <span class="diff-structural">Structural Module</span>

## Overview

Data sources in DiffBio extend `datarax.core.data_source.DataSourceModule`, providing:

- **Consistent Interface**: All sources implement `__len__`, `__getitem__`, and `__iter__`
- **Lazy Loading**: Data is loaded on-demand to minimize memory usage
- **Metadata Support**: Each element includes source-specific metadata
- **Sampler Integration**: Direct compatibility with Datarax samplers

## Source Hierarchy

```
DataSourceModule (Datarax)
├── BAMSource             # BAM/CRAM file reading
├── FastaSource           # FASTA file reading
├── MolNetSource          # MoleculeNet benchmark datasets
└── IndexedViewSource     # Lazy view over subset of another source
```

## Genomics Sources

DiffBio provides optimized data sources for common genomics file formats, built on state-of-the-art libraries.

### BAMSource - Aligned Reads

BAMSource reads aligned sequencing reads from BAM/CRAM files using [pysam](https://pysam.readthedocs.io/), a lightweight wrapper around HTSlib.

#### Quick Start

```python
from pathlib import Path
from diffbio.sources import BAMSource, BAMSourceConfig

# Load aligned reads
config = BAMSourceConfig(
    file_path=Path("sample.bam"),
    min_mapping_quality=20,
)
source = BAMSource(config)

print(f"Reads: {len(source)}")

for element in source:
    seq = element.data["sequence"]       # One-hot encoded (length, 4)
    qual = element.data["quality_scores"]  # Phred scores (length,)
    name = element.data["read_name"]
```

#### Configuration Options

```python
from pathlib import Path
from diffbio.sources import BAMSourceConfig

config = BAMSourceConfig(
    # Required: path to BAM/CRAM file
    file_path=Path("sample.bam"),

    # Optional: reference FASTA (required for CRAM)
    reference_path=Path("reference.fa"),

    # Filter reads by mapping quality
    min_mapping_quality=20,

    # Include unmapped reads
    include_unmapped=False,

    # Query specific region only
    region="chr1:1000000-2000000",

    # How to handle N nucleotides
    handle_n="uniform",  # or "zero"
)
```

#### Data Element Format

```python
element = source[0]

# Read sequence as one-hot encoding
sequence = element.data["sequence"]       # shape: (read_length, 4)
quality = element.data["quality_scores"]  # shape: (read_length,)
name = element.data["read_name"]          # str

# Metadata
idx = element.metadata["idx"]
reference = element.metadata["reference_name"]  # e.g., "chr1"
position = element.metadata["reference_start"]  # 0-based
mapq = element.metadata["mapping_quality"]
```

#### Performance Tips

1. **Use indexed BAM files**: Ensure `.bai` index exists for random access
2. **Filter by region**: Use `region` parameter to load only needed reads
3. **Filter at load time**: Use `min_mapping_quality` instead of post-filtering
4. **Use iterators**: Process reads one at a time, don't load all into memory

### FastaSource - Reference Sequences

FastaSource reads DNA/RNA sequences from FASTA files using [pyfaidx](https://github.com/mdshw5/pyfaidx), providing samtools-compatible indexed access.

#### Quick Start

```python
from pathlib import Path
from diffbio.sources import FastaSource, FastaSourceConfig

# Load reference genome
config = FastaSourceConfig(
    file_path=Path("genome.fasta"),
)
source = FastaSource(config)

print(f"Chromosomes: {source.sequence_names}")

# Access by name
chr1 = source.get_by_name("chr1")
sequence = chr1.data["sequence"]  # One-hot encoded
```

#### Configuration Options

```python
from pathlib import Path
from diffbio.sources import FastaSourceConfig

config = FastaSourceConfig(
    # Required: path to FASTA file
    file_path=Path("genome.fasta"),

    # How to handle N nucleotides
    handle_n="uniform",  # [0.25, 0.25, 0.25, 0.25]
    # handle_n="zero",   # [0, 0, 0, 0]

    # Create .fai index if missing
    create_index=True,
)
```

#### Data Element Format

```python
element = source[0]

# Sequence as one-hot encoding
sequence = element.data["sequence"]     # shape: (seq_length, 4)
seq_id = element.data["sequence_id"]    # str: "chr1"
description = element.data["description"]  # str: full header

# Metadata
idx = element.metadata["idx"]
length = element.metadata["length"]
```

#### Access Patterns

```python
from diffbio.sources import FastaSource, FastaSourceConfig

config = FastaSourceConfig(file_path=Path("genome.fasta"))
source = FastaSource(config)

# 1. Iterate over all sequences
for element in source:
    print(f"{element.data['sequence_id']}: {element.metadata['length']} bp")

# 2. Access by index
first_seq = source[0]

# 3. Access by name
chr1 = source.get_by_name("chr1")

# 4. List all sequence names
names = source.sequence_names  # ["chr1", "chr2", ...]

# 5. Batch access
batch = source.get_batch(10)
```

#### Performance Tips

1. **Use indexed FASTA**: `.fai` index enables O(1) random access
2. **Lazy loading**: Sequences loaded only when accessed
3. **Access by name**: Use `get_by_name()` for specific chromosomes
4. **BGZF compression**: Works with bgzip-compressed files

### Integration with DiffBio Operators

Genomics sources output one-hot encoded sequences compatible with DiffBio operators:

```python
from diffbio.sources import FastaSource, FastaSourceConfig
from diffbio.operators.alignment import SmoothSmithWaterman, SmithWatermanConfig

# Load sequences
fasta = FastaSource(FastaSourceConfig(file_path=Path("genome.fasta")))
seq1 = fasta.get_by_name("seq1").data["sequence"]
seq2 = fasta.get_by_name("seq2").data["sequence"]

# Align with differentiable Smith-Waterman
aligner = SmoothSmithWaterman(SmithWatermanConfig())
result, _, _ = aligner.apply(
    {"query": seq1, "reference": seq2},
    {},
    None,
)
score = result["score"]
```

### Installation

Genomics sources require optional dependencies:

```bash
# Install genomics dependencies
uv pip install -e ".[genomics]"

# Or install individually
pip install pysam pyfaidx
```

## MolNet Benchmark Datasets

### Overview

MolNetSource provides access to the [MoleculeNet](https://moleculenet.org/) benchmark suite - the standard benchmark collection for molecular machine learning.

### Quick Start

```python
from diffbio.sources import MolNetSource, MolNetSourceConfig

# Load the BBBP (Blood-Brain Barrier Penetration) dataset
config = MolNetSourceConfig(
    dataset_name="bbbp",
    split="train",
    download=True,
)
source = MolNetSource(config)

print(f"Dataset: {len(source)} molecules")
print(f"Task type: {source.task_type}")
print(f"Number of tasks: {source.n_tasks}")

# Access individual molecules
element = source[0]
print(f"SMILES: {element.data['smiles']}")
print(f"Label: {element.data['y']}")
```

### Available Datasets

#### Classification Benchmarks

| Dataset | Tasks | Description | Molecules |
|---------|-------|-------------|-----------|
| `bbbp` | 1 | Blood-brain barrier penetration | ~2,000 |
| `tox21` | 12 | Toxicity across 12 assays | ~8,000 |
| `hiv` | 1 | HIV replication inhibition | ~40,000 |
| `bace` | 1 | BACE-1 inhibitor activity | ~1,500 |
| `clintox` | 2 | Clinical trial toxicity | ~1,500 |
| `sider` | 27 | Drug side effects | ~1,400 |

#### Regression Benchmarks

| Dataset | Tasks | Description | Molecules |
|---------|-------|-------------|-----------|
| `esol` | 1 | Aqueous solubility (log mol/L) | ~1,100 |
| `freesolv` | 1 | Hydration free energy (kcal/mol) | ~640 |
| `lipophilicity` | 1 | Octanol/water partition coefficient | ~4,200 |

### Configuration Options

```python
from pathlib import Path
from diffbio.sources import MolNetSourceConfig

config = MolNetSourceConfig(
    # Required: dataset name
    dataset_name="tox21",

    # Which split to load
    split="train",  # "train", "valid", or "test"

    # Custom data directory (default: ~/.diffbio/molnet)
    data_dir=Path("/path/to/data"),

    # Auto-download if not found
    download=True,
)
```

### Data Element Format

Each element from MolNetSource is a `DataElement` with:

```python
element = source[0]

# Molecular data
smiles = element.data["smiles"]  # str: SMILES representation
labels = element.data["y"]       # float or array: task labels

# Metadata
idx = element.metadata["idx"]          # int: index in split
dataset = element.metadata["dataset"]  # str: dataset name

# State (for stateful processing)
state = element.state  # dict: empty by default
```

### Multi-Task Datasets

Some datasets (tox21, sider, clintox) have multiple prediction tasks:

```python
from diffbio.sources import MolNetSource, MolNetSourceConfig

# Load Tox21 with 12 toxicity tasks
config = MolNetSourceConfig(dataset_name="tox21", split="train")
source = MolNetSource(config)

print(f"Number of tasks: {source.n_tasks}")  # 12

element = source[0]
labels = element.data["y"]  # Array of shape (12,)
# NaN values indicate missing labels for that task
```

## IndexedViewSource

### Overview

IndexedViewSource provides a lazy view over a subset of another data source. This is particularly useful for:

- **Memory Efficiency**: Don't duplicate data when splitting
- **Split Views**: Access train/valid/test splits as separate sources
- **Shuffled Access**: Optionally shuffle iteration order

### Basic Usage

```python
from diffbio.sources import IndexedViewSource, IndexedViewSourceConfig
import jax.numpy as jnp

# Create a view of specific indices
indices = jnp.array([0, 5, 10, 15, 20])

config = IndexedViewSourceConfig(
    shuffle=False,  # Preserve index order
)
view = IndexedViewSource(config, parent_source, indices)

print(f"View size: {len(view)}")  # 5
element = view[0]  # Loads from parent_source[0]
```

### Shuffled Iteration

```python
from diffbio.sources import IndexedViewSource, IndexedViewSourceConfig
from flax import nnx

# Shuffle for training
config = IndexedViewSourceConfig(
    shuffle=True,
    seed=42,  # Reproducible shuffling
)
view = IndexedViewSource(config, parent_source, train_indices, rngs=nnx.Rngs(42))

# Each epoch sees data in different order
for element in view:
    # Process shuffled elements
    pass
```

### Integration with Splitters

IndexedViewSource is typically created automatically by splitters:

```python
from diffbio.sources import MolNetSource, MolNetSourceConfig
from diffbio.splitters import ScaffoldSplitter, ScaffoldSplitterConfig

# Load full dataset
source = MolNetSource(MolNetSourceConfig(dataset_name="bbbp"))

# Split creates IndexedViewSource instances
splitter = ScaffoldSplitter(ScaffoldSplitterConfig(smiles_key="smiles"))
train_source, valid_source, test_source = splitter.create_split_sources(
    source,
    lazy=True,  # Returns IndexedViewSource (memory efficient)
)

# Use as regular data sources
print(f"Train: {len(train_source)} molecules")
print(f"Valid: {len(valid_source)} molecules")
print(f"Test: {len(test_source)} molecules")
```

## Integration with Datarax Samplers

### Batch Training Setup

```python
from diffbio.sources import MolNetSource, MolNetSourceConfig
from diffbio.splitters import ScaffoldSplitter, ScaffoldSplitterConfig
from datarax.samplers import ShuffleSampler, ShuffleSamplerConfig
from flax import nnx

# 1. Load dataset
source = MolNetSource(MolNetSourceConfig(dataset_name="bbbp"))

# 2. Split by molecular scaffold
splitter = ScaffoldSplitter(ScaffoldSplitterConfig(smiles_key="smiles"))
train_source, valid_source, test_source = splitter.create_split_sources(
    source, lazy=True
)

# 3. Create samplers
train_sampler = ShuffleSampler(
    ShuffleSamplerConfig(batch_size=32, shuffle=True),
    data_source=train_source,
    rngs=nnx.Rngs(42),
)

valid_sampler = ShuffleSampler(
    ShuffleSamplerConfig(batch_size=32, shuffle=False),
    data_source=valid_source,
)

# 4. Training loop
for epoch in range(100):
    # Training
    for batch in train_sampler:
        smiles_batch = [elem.data["smiles"] for elem in batch]
        labels_batch = [elem.data["y"] for elem in batch]
        # Train on batch

    # Validation
    for batch in valid_sampler:
        # Evaluate on batch
        pass
```

### Custom Collation

```python
import jax.numpy as jnp
from diffbio.operators.drug_discovery import batch_smiles_to_graphs

def collate_molecules(batch):
    """Custom collation for molecular graphs."""
    smiles_list = [elem.data["smiles"] for elem in batch]
    labels = jnp.array([elem.data["y"] for elem in batch])

    # Convert SMILES to molecular graphs
    graphs = batch_smiles_to_graphs(smiles_list)

    return {
        "graphs": graphs,
        "labels": labels,
    }

# Use with sampler
for batch_elements in train_sampler:
    batch = collate_molecules(batch_elements)
    # batch["graphs"] contains padded molecular graphs
    # batch["labels"] contains label array
```

## Best Practices

### 1. Use Lazy Loading for Large Datasets

```python
# GOOD: Lazy loading with IndexedViewSource
train, valid, test = splitter.create_split_sources(source, lazy=True)

# LESS OPTIMAL: Eager loading copies all data
train, valid, test = splitter.create_split_sources(source, lazy=False)
```

### 2. Set Seeds for Reproducibility

```python
from diffbio.sources import MolNetSourceConfig
from diffbio.splitters import RandomSplitterConfig

# Consistent splits across runs
splitter_config = RandomSplitterConfig(seed=42)

# Consistent shuffling
view_config = IndexedViewSourceConfig(shuffle=True, seed=42)
```

### 3. Check Dataset Properties

```python
source = MolNetSource(MolNetSourceConfig(dataset_name="tox21"))

# Understand the task
print(f"Task type: {source.task_type}")  # "classification"
print(f"Number of tasks: {source.n_tasks}")  # 12

# Handle missing labels in multi-task datasets
import jax.numpy as jnp
for elem in source:
    labels = elem.data["y"]
    valid_mask = ~jnp.isnan(labels)
    # Only compute loss for valid labels
```

### 4. Use Domain-Appropriate Splitting

```python
# Drug discovery: Split by molecular structure
from diffbio.splitters import ScaffoldSplitter
splitter = ScaffoldSplitter(ScaffoldSplitterConfig(smiles_key="smiles"))

# Bioinformatics: Split by sequence identity
from diffbio.splitters import SequenceIdentitySplitter
splitter = SequenceIdentitySplitter(SequenceIdentitySplitterConfig(
    sequence_key="sequence",
    identity_threshold=0.3,
))
```

## Troubleshooting

### Dataset Download Issues

```python
# Specify custom directory if default fails
config = MolNetSourceConfig(
    dataset_name="bbbp",
    data_dir=Path("/writable/directory"),
    download=True,
)
```

### Missing Labels

```python
import jax.numpy as jnp

# Multi-task datasets may have missing labels
element = source[0]
labels = element.data["y"]

# Check for NaN values
valid_mask = ~jnp.isnan(labels)
valid_labels = labels[valid_mask]
```

### Memory Issues with Large Datasets

```python
# Use lazy loading
train, valid, test = splitter.create_split_sources(source, lazy=True)

# Process in batches rather than loading all at once
for batch in sampler:
    # Process batch
    pass
```

## Source Selection Guide

| Use Case | Recommended Source |
|----------|-------------------|
| BAM/CRAM aligned reads | BAMSource |
| FASTA reference sequences | FastaSource |
| MolNet benchmarks | MolNetSource |
| Split views | IndexedViewSource (via splitter) |
| Custom datasets | Extend DataSourceModule |

## Related Resources

### Dataset Splitting

- **[Dataset Splitters](splitters.md)**: Domain-aware splitting for unbiased evaluation
- **[ScaffoldSplitter](splitters.md#scaffoldsplitter)**: Drug discovery scaffold-based splitting
- **[SequenceIdentitySplitter](splitters.md#sequenceidentitysplitter)**: Bioinformatics sequence-based splitting

### Operators

- **[Drug Discovery Operators](operators/drug-discovery.md)**: Use molecular graphs from MolNetSource
- **[Alignment Operators](operators/alignment.md)**: Align sequences from FastaSource
- **[Variant Operators](operators/variant.md)**: Process reads from BAMSource

### API Reference

- **[Data Sources API](../api/sources.md)**: Complete API documentation for all data sources
- **[Dataset Splitters API](../api/splitters.md)**: Complete API documentation for all splitters
