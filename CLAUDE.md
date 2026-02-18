# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DiffBio is an end-to-end differentiable bioinformatics library built on JAX/Flax and Datarax, providing differentiable implementations of sequence alignment, variant calling, pileup generation, and genomic analysis pipelines. The library enables gradient-based optimization of entire bioinformatics workflows.

## Non-Negotiable Technical Guidelines

### 1. Neural Network Framework: ALWAYS Use Flax NNX

- **ALWAYS USE Flax NNX** as the neural network backend
- **NEVER USE Flax Linen** or other JAX-based neural network frameworks
- **NEVER USE PyTorch or TensorFlow** for any implementation
- When JAX and Flax NNX offer similar functionality, always choose Flax NNX
- Reference: <https://flax.readthedocs.io/en/latest/> for API, guides, and glossary

### 2. JAX/Flax NNX Compatibility Constraints

- **NEVER use numpy-based packages inside nnx.Module classes** (no scipy, sklearn)
- Use `jax.numpy` instead of `numpy`
- Use `jax.scipy` instead of `scipy` (limited subset available)
- Keep numpy operations outside of module classes when necessary

### 3. Datarax OperatorModule Pattern

All DiffBio operators inherit from `datarax.core.operator.OperatorModule`:

```python
from datarax.core.operator import OperatorModule
from datarax.core.config import OperatorConfig
from dataclasses import dataclass

@dataclass
class MyOperatorConfig(OperatorConfig):
    """Configuration for MyOperator."""
    my_param: float = 1.0
    stochastic: bool = False
    stream_name: str | None = None

class MyOperator(OperatorModule):
    def __init__(self, config: MyOperatorConfig, *, rngs: nnx.Rngs = None):
        super().__init__(config, rngs=rngs)
        self.param = nnx.Param(jnp.array(config.my_param))

    def apply(self, data, state, metadata, random_params=None, stats=None):
        # Implementation here
        result = self._process(data)
        return {**data, "output": result}, state, metadata
```

### 4. Test-Driven Development is Mandatory

- Write tests first, then implement functionality
- Tests define expected behavior, not current implementation
- Never modify tests to accommodate flawed implementations
- Aim for minimum 80% coverage for new code

## Correct NNX Module Implementation Patterns

### Module Initialization Pattern

```python
class MyModule(nnx.Module):
    def __init__(
        self,
        # Required positional arguments first
        in_features: int,
        out_features: int,

        # Optional arguments with defaults
        dropout_rate: float = 0.0,

        # Always require rngs as keyword-only
        *,
        rngs: nnx.Rngs,  # Usually required, not optional
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()  # ALWAYS call this

        # Store configuration
        self.in_features = in_features

        # Initialize submodules with rngs
        self.dense = nnx.Linear(
            in_features=in_features,
            out_features=out_features,
            rngs=rngs,
            dtype=dtype,
        )

        # Initialize dropout if needed
        if dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout = None
```

### RNG Handling Patterns

```python
# CORRECT: Always check if key exists and provide fallback
if rngs is not None and "sample" in rngs:
    key = rngs.sample()  # Use the key method, not .key.value
else:
    key = jax.random.key(0)  # Fallback

# WRONG - Will cause errors
key = rngs.get("sample")  # AttributeError
key = rngs["sample"]      # KeyError if key doesn't exist
```

### Activation Functions

```python
# CORRECT - Use nnx activation functions
from flax import nnx

activation = nnx.gelu  # not jax.nn.gelu
activation = nnx.relu  # not jax.nn.relu
activation = nnx.silu  # not jax.nn.silu
```

## Key Development Commands

### Environment Setup

```bash
# Run setup script (creates venv, installs dependencies, detects GPU)
./setup.sh

# Activate the environment
source ./activate.sh
```

### Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/operators/test_alignment.py -xvs

# Run single test
uv run pytest tests/operators/test_alignment.py::TestSmithWaterman::test_basic -xvs

# Run tests with coverage
uv run pytest --cov=src/diffbio --cov-report=html

# Run GPU-specific tests (requires CUDA)
uv run pytest -m gpu
```

### Code Quality

```bash
# Run all pre-commit checks
uv run pre-commit run --all-files

# Run specific checks
uv run ruff check src/          # Linting
uv run ruff format src/         # Formatting
uv run pyright src/             # Type checking
```

### Documentation

```bash
# Serve docs locally
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

## Architecture Overview

### Project Structure

```
src/diffbio/
├── __init__.py
├── configs.py                  # Shared operator configurations
├── constants.py                # Global constants (alphabets, thresholds)
├── core/                       # Core abstractions and shared components
│   ├── base_operators.py       # Base operator classes
│   ├── data_types.py           # Common data type definitions
│   ├── differentiable_ops.py   # Reusable differentiable primitives
│   └── neural_components.py    # Shared neural network building blocks
├── operators/                  # Differentiable operators
│   ├── quality_filter.py       # DifferentiableQualityFilter
│   ├── alignment/              # Sequence alignment operators
│   │   ├── smith_waterman.py   # SmoothSmithWaterman
│   │   ├── scoring.py          # Scoring matrices
│   │   ├── profile_hmm.py      # Profile HMM alignment
│   │   └── soft_msa.py         # Soft multiple sequence alignment
│   ├── variant/                # Variant calling operators
│   │   ├── pileup.py           # DifferentiablePileup
│   │   ├── classifier.py       # VariantClassifier
│   │   ├── cnn_classifier.py   # CNN-based variant classifier
│   │   ├── cnv_segmentation.py # Copy number variation segmentation
│   │   ├── deepvariant_pileup.py # DeepVariant-style multi-channel pileup
│   │   └── quality_recalibration.py # Base quality recalibration
│   ├── drug_discovery/         # Molecular property prediction
│   │   ├── primitives.py       # Graph/molecular primitives
│   │   ├── message_passing.py  # GNN message passing layers
│   │   ├── fingerprint.py      # Molecular fingerprints
│   │   ├── maccs_keys.py       # MACCS structural keys
│   │   ├── attentive_fp.py     # Attentive fingerprint model
│   │   ├── admet_predictor.py  # ADMET property prediction
│   │   ├── property_predictor.py # General property prediction
│   │   └── similarity.py       # Molecular similarity operators
│   ├── singlecell/             # Single-cell analysis operators
│   │   ├── soft_clustering.py  # Differentiable clustering
│   │   ├── batch_correction.py # Batch effect correction
│   │   ├── ambient_removal.py  # Ambient RNA removal
│   │   └── velocity.py         # RNA velocity estimation
│   ├── epigenomics/            # Epigenomic analysis operators
│   │   ├── peak_calling.py     # Differentiable peak calling
│   │   └── chromatin_state.py  # Chromatin state annotation
│   ├── rnaseq/                 # RNA-seq operators
│   │   ├── splicing_psi.py     # Splicing PSI estimation
│   │   └── motif_discovery.py  # Sequence motif discovery
│   ├── normalization/          # Normalization and embedding operators
│   │   ├── vae_normalizer.py   # VAE-based normalization
│   │   ├── embedding.py        # Embedding operators
│   │   └── umap.py             # Differentiable UMAP
│   ├── statistical/            # Statistical model operators
│   │   ├── hmm.py              # Hidden Markov model
│   │   ├── em_quantification.py # EM-based quantification
│   │   └── nb_glm.py           # Negative binomial GLM
│   ├── language_models/        # Biological language models
│   │   └── transformer_encoder.py # Transformer encoder for sequences
│   ├── population/             # Population genetics operators
│   │   └── ancestry_estimation.py # Ancestry estimation
│   ├── assembly/               # Genome assembly operators
│   │   ├── gnn_assembly.py     # GNN-based assembly
│   │   └── metagenomic_binning.py # Metagenomic binning
│   ├── molecular_dynamics/     # Molecular dynamics operators
│   │   ├── primitives.py       # MD primitives
│   │   ├── force_field.py      # Differentiable force fields
│   │   └── integrator.py       # MD integrators
│   ├── mapping/                # Read mapping operators
│   │   └── neural_mapper.py    # Neural read mapper
│   ├── preprocessing/          # Read preprocessing operators
│   │   ├── adapter_removal.py  # Adapter trimming
│   │   ├── duplicate_filter.py # Duplicate filtering
│   │   └── error_correction.py # Sequencing error correction
│   ├── protein/                # Protein analysis operators
│   │   └── secondary_structure.py # Secondary structure prediction
│   ├── rna_structure/          # RNA structure operators
│   │   └── rna_folding.py      # RNA folding prediction
│   ├── multiomics/             # Multi-omics integration operators
│   │   ├── hic_contact.py      # Hi-C contact map analysis
│   │   ├── spatial_deconvolution.py # Spatial deconvolution
│   │   └── spatial_gene_detection.py # Spatial gene detection
│   ├── metabolomics/           # Metabolomics operators
│   │   └── spectral_similarity.py # Spectral similarity
│   └── crispr/                 # CRISPR analysis operators
│       └── guide_scoring.py    # Guide RNA scoring
├── pipelines/                  # End-to-end pipelines
│   ├── variant_calling.py      # VariantCallingPipeline
│   ├── enhanced_variant_calling.py # Enhanced variant calling
│   ├── differential_expression.py # Differential expression pipeline
│   ├── single_cell.py          # Single-cell analysis pipeline
│   └── preprocessing.py        # Read preprocessing pipeline
├── sources/                    # Data source loaders
│   ├── fasta.py                # FASTA file reader
│   ├── bam.py                  # BAM file reader
│   ├── molnet.py               # MoleculeNet dataset loader
│   └── indexed_view.py         # Indexed data view abstraction
├── splitters/                  # Dataset splitting strategies
│   ├── base.py                 # Base splitter interface
│   ├── random.py               # Random splitting
│   ├── molecular.py            # Scaffold-based molecular splitting
│   └── sequence.py             # Sequence-aware splitting
├── losses/                     # Loss functions
│   ├── alignment_losses.py     # Alignment-specific losses
│   ├── biological_regularization.py # Biological constraint losses
│   ├── singlecell_losses.py    # Single-cell analysis losses
│   └── statistical_losses.py   # Statistical model losses
├── sequences/                  # Sequence utilities
│   └── dna.py                  # DNA encoding and manipulation
└── utils/                      # Utilities
    ├── training.py             # Training utilities (Trainer, TrainingConfig)
    ├── nn_utils.py             # Neural network helper functions
    └── quality.py              # Quality score utilities
```

### Key Design Principles

1. **All operators inherit from OperatorModule**: Consistent `apply()` interface for composability
2. **Dictionary-based data flow**: Data passed as dictionaries with string keys
3. **One-hot sequence encoding**: Sequences are (length, alphabet_size) arrays for differentiability
4. **Temperature-controlled smoothing**: All smooth approximations use temperature parameter

### Differentiability Techniques

1. **Logsumexp Relaxation**: Replace discrete `max` with smooth `logsumexp` (Smith-Waterman)
2. **Sigmoid Thresholds**: Replace hard thresholds with sigmoid (Quality filtering)
3. **Segment Sum Aggregation**: Replace counting with weighted accumulation (Pileup)

## Testing Strategy

### Test Organization

```
tests/
├── conftest.py          # Shared fixtures
├── operators/           # Operator unit tests
│   ├── test_alignment.py
│   ├── test_pileup.py
│   └── test_quality_filter.py
├── pipelines/           # Pipeline tests
│   └── test_variant_calling.py
├── integration/         # Integration tests
│   └── test_operator_composition.py
└── utils/               # Utility tests
    └── test_training.py
```

### Test Patterns

```python
def test_differentiability(self):
    """Verify operator is differentiable."""
    config = MyOperatorConfig()
    operator = MyOperator(config)

    def loss_fn(op, data):
        result, _, _ = op.apply(data, {}, None)
        return result["output"].sum()

    data = {"input": jnp.ones((10,))}

    # Should not raise any errors
    grads = jax.grad(loss_fn)(operator, data)
    assert grads is not None
```

## Common Mistakes to Avoid

1. **Using `flax.linen` instead of `flax.nnx`**
2. **Not calling `super().__init__()` in module constructors**
3. **Using `rngs.get("key_name")` instead of checking with `"key_name" in rngs`**
4. **Not providing fallback RNG keys for `jax.random` operations**
5. **Using numpy-based packages inside nnx.Module classes**
6. **Missing type annotations on method signatures**
7. **Not inheriting from OperatorModule for new operators**
8. **Using integer sequence encoding instead of one-hot encoding**
9. **Forgetting to handle temperature parameter in smooth approximations**

## Critical Files to Know

- `src/diffbio/operators/alignment/smith_waterman.py`: Core differentiable alignment
- `src/diffbio/operators/variant/pileup.py`: Differentiable pileup generation
- `src/diffbio/pipelines/variant_calling.py`: End-to-end variant calling pipeline
- `src/diffbio/utils/training.py`: Training utilities (Trainer, TrainingConfig)
- `pyproject.toml`: Project configuration and dependencies
- `conftest.py`: Pytest fixtures and configuration

## Development Tips

- Always run tests before committing: `uv run pytest tests/ -x`
- Use type hints and protocols for all new code
- Follow existing patterns in neighboring files
- Check GPU availability with device fixture before GPU operations
- Use frozen dataclass configurations for operator configs
- Run pre-commit hooks: `uv run pre-commit run --all-files`
- Run `source activate.sh` before running pytest and python scripts
