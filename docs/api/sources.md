# Data Sources API

Data source modules for loading bioinformatics and drug discovery datasets, extending Datarax's DataSourceModule.

## Genomics Sources

### BAMSource

::: diffbio.sources.bam.BAMSource
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - __len__
        - __getitem__
        - __iter__
        - reset
        - get_batch

### BAMSourceConfig

::: diffbio.sources.bam.BAMSourceConfig
    options:
      show_root_heading: true
      members: []

### FastaSource

::: diffbio.sources.fasta.FastaSource
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - __len__
        - __getitem__
        - __iter__
        - reset
        - get_batch
        - get_by_name
        - sequence_names

### FastaSourceConfig

::: diffbio.sources.fasta.FastaSourceConfig
    options:
      show_root_heading: true
      members: []

## MolNet Benchmark Source

### MolNetSource

::: diffbio.sources.molnet.MolNetSource
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - __len__
        - __getitem__
        - __iter__
        - task_type
        - n_tasks

### MolNetSourceConfig

::: diffbio.sources.molnet.MolNetSourceConfig
    options:
      show_root_heading: true
      members: []

## Indexed View Source

### IndexedViewSource

::: diffbio.sources.indexed_view.IndexedViewSource
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - __len__
        - __getitem__
        - __iter__

### IndexedViewSourceConfig

::: diffbio.sources.indexed_view.IndexedViewSourceConfig
    options:
      show_root_heading: true
      members: []

## AnnData Source

### AnnDataSource

::: diffbio.sources.anndata_source.AnnDataSource
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - __len__
        - __getitem__
        - __iter__

### AnnDataSourceConfig

::: diffbio.sources.anndata_source.AnnDataSourceConfig
    options:
      show_root_heading: true
      members: []

## AnnData Interop

### to_anndata

::: diffbio.sources.anndata_interop.to_anndata
    options:
      show_root_heading: true
      show_source: false

### from_anndata

::: diffbio.sources.anndata_interop.from_anndata
    options:
      show_root_heading: true
      show_source: false

## Usage Examples

### Reading BAM/CRAM Files

```python
from pathlib import Path
from diffbio.sources import BAMSource, BAMSourceConfig

# Load aligned reads from a BAM file
config = BAMSourceConfig(
    file_path=Path("sample.bam"),
    min_mapping_quality=20,  # Filter low-quality alignments
    include_unmapped=False,  # Skip unmapped reads
)
source = BAMSource(config)

print(f"Number of reads: {len(source)}")

# Access reads
for element in source:
    # One-hot encoded sequence (length, 4)
    sequence = element.data["sequence"]
    # Phred quality scores (length,)
    quality = element.data["quality_scores"]
    # Read name
    name = element.data["read_name"]

    print(f"Read {name}: {sequence.shape}, avg quality: {quality.mean():.1f}")
```

### Reading FASTA Files

```python
from pathlib import Path
from diffbio.sources import FastaSource, FastaSourceConfig

# Load sequences from a FASTA file
config = FastaSourceConfig(
    file_path=Path("genome.fasta"),
    handle_n="uniform",  # or "zero" for N nucleotides
    create_index=True,   # Create .fai index for random access
)
source = FastaSource(config)

print(f"Number of sequences: {len(source)}")
print(f"Sequence names: {source.sequence_names}")

# Access by index
for element in source:
    seq_id = element.data["sequence_id"]
    sequence = element.data["sequence"]  # One-hot encoded
    print(f"{seq_id}: {sequence.shape[0]} bp")

# Access by name
chr1 = source.get_by_name("chr1")
if chr1 is not None:
    print(f"Chromosome 1 length: {chr1.data['sequence'].shape[0]}")
```

### Region-Based BAM Access

```python
from pathlib import Path
from diffbio.sources import BAMSource, BAMSourceConfig

# Load only reads from a specific region
config = BAMSourceConfig(
    file_path=Path("sample.bam"),
    region="chr1:1000000-2000000",  # 1Mb region on chr1
)
source = BAMSource(config)

print(f"Reads in region: {len(source)}")
```

### Loading MolNet Benchmarks

```python
from diffbio.sources import MolNetSource, MolNetSourceConfig

# Load BBBP (Blood-Brain Barrier Penetration) dataset
config = MolNetSourceConfig(
    dataset_name="bbbp",
    split="train",  # "train", "valid", or "test"
    download=True,  # Auto-download if not found
)
source = MolNetSource(config)

print(f"Dataset size: {len(source)}")
print(f"Task type: {source.task_type}")  # "classification"
print(f"Number of tasks: {source.n_tasks}")  # 1

# Iterate over elements
for element in source:
    smiles = element.data["smiles"]
    label = element.data["y"]
    print(f"{smiles}: {label}")
```

### Available MolNet Datasets

| Dataset | Task Type | Tasks | Description |
|---------|-----------|-------|-------------|
| `bbbp` | classification | 1 | Blood-brain barrier penetration |
| `tox21` | classification | 12 | Toxicity across 12 assays |
| `hiv` | classification | 1 | HIV replication inhibition |
| `bace` | classification | 1 | BACE-1 inhibitor activity |
| `clintox` | classification | 2 | Clinical trial toxicity |
| `sider` | classification | 27 | Drug side effects |
| `esol` | regression | 1 | Aqueous solubility |
| `freesolv` | regression | 1 | Hydration free energy |
| `lipophilicity` | regression | 1 | Octanol/water partition |

### Using IndexedViewSource for Lazy Loading

```python
from diffbio.sources import IndexedViewSource, IndexedViewSourceConfig
from diffbio.splitters import ScaffoldSplitter, ScaffoldSplitterConfig
import jax.numpy as jnp

# Create a splitter
splitter_config = ScaffoldSplitterConfig(smiles_key="smiles")
splitter = ScaffoldSplitter(splitter_config)

# Get split indices
result = splitter.split(data_source)

# Create lazy view for training data
view_config = IndexedViewSourceConfig(shuffle=True, seed=42)
train_view = IndexedViewSource(
    view_config,
    data_source,
    result.train_indices,
)

# Iterate - elements loaded on demand
for element in train_view:
    print(element.data["smiles"])
```

### Integration with Datarax Samplers

```python
from diffbio.sources import MolNetSource, MolNetSourceConfig
from diffbio.splitters import ScaffoldSplitter, ScaffoldSplitterConfig
from datarax.samplers import ShuffleSampler, ShuffleSamplerConfig

# Load dataset
source_config = MolNetSourceConfig(dataset_name="bbbp", split="train")
source = MolNetSource(source_config)

# Split by scaffold
splitter_config = ScaffoldSplitterConfig(smiles_key="smiles")
splitter = ScaffoldSplitter(splitter_config)

# Create split sources (lazy loading)
train_source, valid_source, test_source = splitter.create_split_sources(
    source,
    lazy=True,
)

# Use with Datarax sampler
sampler_config = ShuffleSamplerConfig(batch_size=32)
train_sampler = ShuffleSampler(sampler_config, data_source=train_source)

# Training loop
for batch in train_sampler:
    # Process batch
    pass
```

### Custom Data Directory

```python
from pathlib import Path
from diffbio.sources import MolNetSource, MolNetSourceConfig

# Specify custom data directory
config = MolNetSourceConfig(
    dataset_name="tox21",
    split="train",
    data_dir=Path("/custom/path/to/data"),
    download=True,
)
source = MolNetSource(config)
```

### Accessing Individual Elements

```python
from diffbio.sources import MolNetSource, MolNetSourceConfig

config = MolNetSourceConfig(dataset_name="esol", split="train")
source = MolNetSource(config)

# Access by index
element = source[0]
if element is not None:
    smiles = element.data["smiles"]
    solubility = element.data["y"]
    metadata = element.metadata  # {"idx": 0, "dataset": "esol"}
```

## Data Element Format

### MolNetSource Elements

Each element from MolNetSource contains:

| Field | Type | Description |
|-------|------|-------------|
| `data["smiles"]` | str | SMILES representation of molecule |
| `data["y"]` | float or jnp.ndarray | Label(s) for the molecule |
| `state` | dict | Empty state dictionary |
| `metadata["idx"]` | int | Index within the split |
| `metadata["dataset"]` | str | Dataset name |

### IndexedViewSource Elements

Elements from IndexedViewSource are passed through from the underlying source, with indices remapped to the view's subset.

## Configuration Options

### MolNetSourceConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_name` | str | required | Name of MolNet dataset |
| `split` | str | "train" | Which split: "train", "valid", "test" |
| `data_dir` | Path | ~/.diffbio/molnet | Data storage directory |
| `download` | bool | True | Auto-download if missing |

### IndexedViewSourceConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shuffle` | bool | False | Shuffle indices on iteration |
| `seed` | int | None | Random seed for shuffling |
