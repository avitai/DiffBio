# Data Sources API

Data source modules for loading bioinformatics and drug discovery datasets, extending Datarax's DataSourceModule.

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

## Usage Examples

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
