# Operators Overview

DiffBio provides a collection of differentiable operators for bioinformatics analysis. Each operator inherits from Datarax's `OperatorModule` for consistent interfaces and composability.

## Available Operators

<div class="performance-metrics">
<div class="metric-card">
  <div class="metric-value">Alignment</div>
  <div class="metric-label">Smith-Waterman</div>
</div>
<div class="metric-card">
  <div class="metric-value">Pileup</div>
  <div class="metric-label">Read Aggregation</div>
</div>
<div class="metric-card">
  <div class="metric-value">Filter</div>
  <div class="metric-label">Quality Control</div>
</div>
</div>

### Core Operators

| Operator | Description | Status |
|----------|-------------|--------|
| [`DifferentiableQualityFilter`](quality-filter.md) | Sigmoid-based soft quality filtering | <span class="diff-high">Implemented</span> |
| [`DifferentiablePileup`](pileup.md) | Soft pileup generation | <span class="diff-high">Implemented</span> |
| [`SmoothSmithWaterman`](smith-waterman.md) | Differentiable local sequence alignment | <span class="diff-high">Implemented</span> |
| `VariantClassifier` | Neural variant classifier | <span class="diff-high">Implemented</span> |

### [Alignment Operators](alignment.md)

| Operator | Description | Status |
|----------|-------------|--------|
| `SoftProgressiveMSA` | Differentiable multiple sequence alignment with guide tree | <span class="diff-high">Implemented</span> |
| `ProfileHMM` | Profile Hidden Markov Model for sequence homology | <span class="diff-high">Implemented</span> |

### [Epigenomics Operators](epigenomics.md)

| Operator | Description | Status |
|----------|-------------|--------|
| `DifferentiablePeakCaller` | CNN-based peak calling for ChIP-seq/ATAC-seq | <span class="diff-high">Implemented</span> |
| `ChromatinStateAnnotator` | HMM-based chromatin state classification | <span class="diff-high">Implemented</span> |

### [RNA-seq Operators](rnaseq.md)

| Operator | Description | Status |
|----------|-------------|--------|
| `SplicingPSI` | Differentiable PSI calculation for alternative splicing | <span class="diff-high">Implemented</span> |
| `DifferentiableMotifDiscovery` | Learnable PWM-based motif discovery | <span class="diff-high">Implemented</span> |

### [Single-Cell Operators](singlecell.md)

| Operator | Description | Status |
|----------|-------------|--------|
| `SoftKMeansClustering` | Differentiable soft k-means with learnable centroids | <span class="diff-high">Implemented</span> |
| `DifferentiableHarmony` | Harmony-style batch correction | <span class="diff-high">Implemented</span> |
| `DifferentiableVelocity` | RNA velocity via neural ODEs | <span class="diff-high">Implemented</span> |
| `DifferentiableAmbientRemoval` | VAE-based ambient RNA decontamination | <span class="diff-high">Implemented</span> |

### [Preprocessing Operators](preprocessing.md)

| Operator | Description | Status |
|----------|-------------|--------|
| `SoftAdapterRemoval` | Differentiable adapter trimming with soft alignment | <span class="diff-high">Implemented</span> |
| `DifferentiableDuplicateWeighting` | Probabilistic duplicate weighting | <span class="diff-high">Implemented</span> |
| `SoftErrorCorrection` | Neural network-based error correction | <span class="diff-high">Implemented</span> |

### [Normalization Operators](normalization.md)

| Operator | Description | Status |
|----------|-------------|--------|
| `VAENormalizer` | scVI-style VAE for count normalization | <span class="diff-high">Implemented</span> |
| `DifferentiableUMAP` | Differentiable UMAP dimensionality reduction | <span class="diff-high">Implemented</span> |
| `SequenceEmbedding` | Learned sequence embeddings | <span class="diff-high">Implemented</span> |

### [Statistical Operators](statistical.md)

| Operator | Description | Status |
|----------|-------------|--------|
| `DifferentiableHMM` | Forward algorithm with logsumexp stability | <span class="diff-high">Implemented</span> |
| `DifferentiableNBGLM` | Negative binomial GLM for differential expression | <span class="diff-high">Implemented</span> |
| `DifferentiableEMQuantifier` | Unrolled EM for transcript quantification | <span class="diff-high">Implemented</span> |

### [Assembly & Mapping Operators](assembly-mapping.md)

| Operator | Description | Status |
|----------|-------------|--------|
| `GNNAssemblyNavigator` | GNN for assembly graph traversal | <span class="diff-high">Implemented</span> |
| `NeuralReadMapper` | Cross-attention based read mapping | <span class="diff-high">Implemented</span> |
| `DifferentiableMetagenomicBinner` | VAMB-style VAE for metagenomic binning | <span class="diff-high">Implemented</span> |

### [Multi-omics Operators](multiomics.md)

| Operator | Description | Status |
|----------|-------------|--------|
| `SpatialDeconvolution` | Cell type deconvolution for spatial transcriptomics | <span class="diff-high">Implemented</span> |
| `HiCContactAnalysis` | Chromatin contact analysis for Hi-C data | <span class="diff-high">Implemented</span> |
| `DifferentiableSpatialGeneDetector` | SpatialDE-style spatial gene detection | <span class="diff-high">Implemented</span> |

### [Variant Operators](variant.md)

| Operator | Description | Status |
|----------|-------------|--------|
| `CNNVariantClassifier` | CNN-based variant classification | <span class="diff-high">Implemented</span> |
| `CNVSegmentation` | Copy number variation segmentation | <span class="diff-high">Implemented</span> |
| `QualityRecalibration` | Base quality score recalibration | <span class="diff-high">Implemented</span> |
| `DeepVariantStylePileup` | Multi-channel pileup image generation for DeepVariant-style CNNs | <span class="diff-high">Implemented</span> |

### [Population Genetics Operators](population.md)

| Operator | Description | Status |
|----------|-------------|--------|
| `DifferentiableAncestryEstimator` | Neural ADMIXTURE-style ancestry estimation | <span class="diff-high">Implemented</span> |

### [CRISPR Operators](crispr.md)

| Operator | Description | Status |
|----------|-------------|--------|
| `DifferentiableCRISPRScorer` | DeepCRISPR-style guide RNA efficiency prediction | <span class="diff-high">Implemented</span> |

### [Metabolomics Operators](metabolomics.md)

| Operator | Description | Status |
|----------|-------------|--------|
| `DifferentiableSpectralSimilarity` | MS2DeepScore-style Siamese network for MS/MS similarity | <span class="diff-high">Implemented</span> |

### [Protein Structure Operators](protein.md)

| Operator | Description | Status |
|----------|-------------|--------|
| `DifferentiableSecondaryStructure` | PyDSSP-style DSSP with continuous H-bond matrix | <span class="diff-high">Implemented</span> |

### [Language Model Operators](language-models.md)

| Operator | Description | Status |
|----------|-------------|--------|
| `TransformerSequenceEncoder` | DNABERT/RNA-FM-style transformer for sequence embedding | <span class="diff-high">Implemented</span> |

### [RNA Structure Operators](rna-structure.md)

| Operator | Description | Status |
|----------|-------------|--------|
| `DifferentiableRNAFold` | McCaskill-style partition function for base pair probabilities | <span class="diff-high">Implemented</span> |

### [Molecular Dynamics Operators](molecular-dynamics.md)

| Operator | Description | Status |
|----------|-------------|--------|
| `ForceFieldOperator` | Differentiable force field (LJ, Morse, Soft Sphere) using JAX-MD | <span class="diff-high">Implemented</span> |
| `MDIntegratorOperator` | Time integration for MD (velocity Verlet, Langevin) using JAX-MD | <span class="diff-high">Implemented</span> |

### [Drug Discovery Operators](drug-discovery.md)

| Operator | Description | Status |
|----------|-------------|--------|
| `MolecularPropertyPredictor` | ChemProp-style D-MPNN for molecular property prediction | <span class="diff-high">Implemented</span> |
| `ADMETPredictor` | Multi-task ADMET prediction (22 TDC endpoints) | <span class="diff-high">Implemented</span> |
| `DifferentiableMolecularFingerprint` | Neural graph fingerprints as alternative to ECFP/Morgan | <span class="diff-high">Implemented</span> |
| `CircularFingerprintOperator` | Differentiable ECFP/Morgan circular fingerprints | <span class="diff-high">Implemented</span> |
| `MACCSKeysOperator` | Differentiable MACCS 166 structural keys fingerprint | <span class="diff-high">Implemented</span> |
| `AttentiveFP` | Attention-based graph fingerprint with GRU (Xiong et al. 2019) | <span class="diff-high">Implemented</span> |
| `MolecularSimilarityOperator` | Differentiable Tanimoto/cosine/Dice similarity | <span class="diff-high">Implemented</span> |

## Operator Interface

All DiffBio operators implement the Datarax `OperatorModule` interface:

```python
class OperatorModule:
    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict | None,
        random_params: Any = None,
        stats: dict | None = None,
    ) -> tuple[PyTree, PyTree, dict | None]:
        """Transform data through the operator.

        Args:
            data: Input data as a PyTree (typically dict)
            state: Per-element state (passed through or modified)
            metadata: Optional metadata
            random_params: Random parameters for stochastic operators
            stats: Optional statistics dictionary

        Returns:
            Tuple of (transformed_data, updated_state, updated_metadata)
        """
```

### Configuration Pattern

Each operator has a corresponding configuration dataclass:

```python
from dataclasses import dataclass
from datarax.core.config import OperatorConfig

@dataclass
class MyOperatorConfig(OperatorConfig):
    # Configuration fields with defaults
    temperature: float = 1.0
    stochastic: bool = False
    stream_name: str | None = None
```

### Example Usage

```python
from diffbio.operators import DifferentiableQualityFilter, QualityFilterConfig

# 1. Create configuration
config = QualityFilterConfig(initial_threshold=20.0)

# 2. Instantiate operator
operator = DifferentiableQualityFilter(config)

# 3. Prepare data
data = {
    "sequence": sequence_tensor,
    "quality_scores": quality_tensor,
}

# 4. Apply operator
result_data, state, metadata = operator.apply(data, {}, None)
```

## Composing Operators

Operators can be composed into pipelines using Datarax's composition utilities:

### Sequential Composition

```python
from datarax.core.operator import CompositeOperator

# Chain operators sequentially
pipeline = CompositeOperator([
    quality_filter,
    aligner,
    pileup_generator,
])

# Apply entire pipeline
result, state, meta = pipeline.apply(data, {}, None)
```

### Manual Composition

```python
def my_pipeline(data):
    # Step 1: Quality filtering
    data, state, meta = quality_filter.apply(data, {}, None)

    # Step 2: Alignment
    data, state, meta = aligner.apply(data, state, meta)

    # Step 3: Pileup
    data, state, meta = pileup_op.apply(data, state, meta)

    return data
```

## Learnable Parameters

DiffBio operators use Flax NNX for parameter management:

### Accessing Parameters

```python
from flax import nnx

# Get all parameters
params = nnx.state(operator, nnx.Param)
print(params)

# Access specific parameter
print(operator.threshold[...])  # Array value
```

### Updating Parameters

```python
# Manual update
operator.threshold[...] = new_value

# Gradient-based update
import optax

optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

def update_step(params, grads, opt_state):
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state
```

## JAX Transformations

All operators are compatible with JAX transformations:

### JIT Compilation

```python
import jax

@jax.jit
def apply_operator(data):
    result, _, _ = operator.apply(data, {}, None)
    return result

# Fast execution after first compile
result = apply_operator(data)
```

### Vectorization

```python
# Process batch of inputs
def single_apply(single_data):
    result, _, _ = operator.apply(single_data, {}, None)
    return result

batch_apply = jax.vmap(single_apply)
batch_results = batch_apply(batch_data)
```

### Gradient Computation

```python
def loss_fn(data):
    result, _, _ = operator.apply(data, {}, None)
    return result['score'].mean()

# Compute gradients w.r.t. operator parameters
grad_fn = jax.grad(loss_fn)
grads = grad_fn(data)
```

## Best Practices

### 1. Use Configuration Objects

```python
# Good: Use config dataclass
config = SmithWatermanConfig(
    temperature=1.0,
    gap_open=-10.0,
)
aligner = SmoothSmithWaterman(config, scoring_matrix=scoring)

# Avoid: Hardcoded values scattered in code
```

### 2. Preserve Input Keys

When implementing custom operators, preserve input data keys:

```python
def apply(self, data, state, metadata, ...):
    result = self.process(data['input'])

    # Good: Preserve input keys
    transformed_data = {
        **data,  # Keep original keys
        'output': result,
    }

    return transformed_data, state, metadata
```

### 3. Use Appropriate Temperature

| Use Case | Recommended Temperature |
|----------|------------------------|
| Training start | 5.0 - 10.0 |
| Training end | 0.1 - 1.0 |
| Inference (soft) | 1.0 |
| Inference (hard) | 0.01 |

### 4. JIT for Performance

Always JIT-compile hot paths:

```python
@jax.jit
def process_batch(operator, batch_data):
    results = []
    for data in batch_data:
        result, _, _ = operator.apply(data, {}, None)
        results.append(result)
    return results
```

## Next Steps

- Learn about the [Smith-Waterman](smith-waterman.md) alignment operator
- Explore the [Pileup](pileup.md) operator for variant calling
- See the [Quality Filter](quality-filter.md) for preprocessing

## Related Resources

### Data Loading

- **[Data Sources](../sources.md)**: Load genomics data (BAM, FASTA) and molecular datasets (MolNet)
- **[Dataset Splitters](../splitters.md)**: Domain-aware dataset splitting (scaffold, sequence identity)

### Pipelines

- **[Pipeline Overview](../pipelines/overview.md)**: End-to-end differentiable bioinformatics pipelines
- **[Variant Calling Pipeline](../pipelines/variant-calling.md)**: Complete variant calling workflow
- **[Single-Cell Pipeline](../pipelines/single-cell.md)**: Single-cell RNA-seq analysis

### Training

- **[Training Overview](../training/overview.md)**: Training DiffBio models
- **[Training Utilities](../training/utilities.md)**: Trainer class and configuration
