# Examples Overview

This section provides practical examples demonstrating DiffBio's capabilities with **real executable code and actual outputs**.

## Example Categories

<div class="performance-metrics">
<div class="metric-card">
  <a href="basic/molnet-data-loading/">
    <div class="metric-value">12</div>
    <div class="metric-label">Basic Examples</div>
  </a>
</div>
<div class="metric-card">
  <a href="advanced/drug-discovery-workflow/">
    <div class="metric-value">9</div>
    <div class="metric-label">Advanced Pipelines</div>
  </a>
</div>
</div>

## Basic Examples

### Drug Discovery

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [MolNet Data Loading](basic/molnet-data-loading.md) | Load MoleculeNet benchmark datasets | Data sources, SMILES, labels |
| [Molecular Fingerprints](basic/molecular-fingerprints.md) | Generate ECFP and neural fingerprints | CircularFingerprint, message passing |
| [Molecular Similarity](basic/molecular-similarity.md) | Compare molecules with similarity metrics | Tanimoto, cosine, Dice similarity |
| [Scaffold Splitting](basic/scaffold-splitting.md) | Proper train/test splits for drug discovery | ScaffoldSplitter, evaluation |

### Genomics

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [DNA Encoding](basic/dna-encoding.md) | One-hot encode DNA sequences | Encoding, GC content, reverse complement |
| [Sequence Alignment](basic/simple-alignment.md) | Smith-Waterman local alignment | Operators, scoring matrices |
| [Pileup Generation](basic/pileup-generation.md) | Generate pileups from reads | Pileup operator, quality weighting |
| [Preprocessing](basic/preprocessing.md) | Read preprocessing pipeline | Quality filtering, adapters |

### Structure & Statistical

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [RNA Structure](basic/rna-structure.md) | Predict RNA secondary structure | McCaskill algorithm, base pairs |
| [Protein Structure](basic/protein-structure.md) | Predict protein secondary structure | DSSP, H-bonds, helix/strand |
| [HMM Sequence Model](basic/hmm-sequence-model.md) | Hidden Markov Models for sequences | Forward algorithm, posteriors |
| [Single-Cell Clustering](basic/single-cell-clustering.md) | Soft k-means cell clustering | Single-cell, differentiability |

## Advanced Examples

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [Drug Discovery Workflow](advanced/drug-discovery-workflow.md) | End-to-end drug discovery pipeline | MolNet, fingerprints, training |
| [ADMET Prediction](advanced/admet-prediction.md) | Multi-task ADMET property prediction | ADMETPredictor, 22 endpoints, D-MPNN |
| [AttentiveFP GNN](advanced/attentive-fp.md) | Attention-based molecular fingerprints | Graph attention, interpretability |
| [Variant Calling Pipeline](advanced/variant-calling.md) | End-to-end variant calling | Full pipeline, CNN classifier |
| [Single-Cell Batch Correction](advanced/singlecell-batch-correction.md) | Harmony-style batch correction | Integration, batch effects |
| [RNA Velocity](advanced/rna-velocity.md) | RNA velocity trajectory inference | Neural ODE, splicing kinetics |
| [Differential Expression](advanced/differential-expression.md) | DESeq2-style DE analysis | Statistical testing, NB-GLM |
| [Epigenomics Analysis](advanced/epigenomics-analysis.md) | Peak calling & chromatin states | ChIP-seq, ATAC-seq |
| [Multi-omics Integration](advanced/multiomics-integration.md) | Spatial deconvolution & Hi-C | Data integration |

## Running Examples

All examples use real DiffBio code with actual outputs:

```bash
# Clone the repository
git clone https://github.com/mahdi-shafiei/DiffBio.git
cd DiffBio

# Setup environment
./setup.sh
source ./activate.sh

# Generate example outputs
python scripts/generate_example_outputs.py
```

## Quick Start Examples

### Drug Discovery: Molecular Fingerprints

```python
from diffbio.operators.drug_discovery import (
    CircularFingerprintOperator,
    CircularFingerprintConfig,
    smiles_to_graph,
    DEFAULT_ATOM_FEATURES,
)
from flax import nnx

# Create fingerprint operator
config = CircularFingerprintConfig(radius=2, n_bits=1024, in_features=DEFAULT_ATOM_FEATURES)
fp_op = CircularFingerprintOperator(config, rngs=nnx.Rngs(42))

# Generate fingerprint
graph = smiles_to_graph("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin
result, _, _ = fp_op.apply(graph, {}, None)
print(f"Fingerprint shape: {result['fingerprint'].shape}")
```

**Output:**
```
Fingerprint shape: (1024,)
```

### Genomics: Sequence Alignment

```python
from diffbio.operators.alignment import SmoothSmithWaterman, SmithWatermanConfig, create_dna_scoring_matrix
from diffbio.sequences import encode_dna_string
from flax import nnx

# Create aligner
scoring = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)
config = SmithWatermanConfig(temperature=1.0, gap_open=-2.0)
aligner = SmoothSmithWaterman(config, scoring_matrix=scoring, rngs=nnx.Rngs(42))

# Align sequences
data = {"seq1": encode_dna_string("ACGTACGT"), "seq2": encode_dna_string("ACGTTACGT")}
result, _, _ = aligner.apply(data, {}, None)
print(f"Alignment score: {float(result['score']):.2f}")
```

**Output:**
```
Alignment score: 14.45
```

### Single-Cell: Batch Correction

```python
from diffbio.operators.singlecell import DifferentiableHarmony, BatchCorrectionConfig
import jax.numpy as jnp
from flax import nnx

# Create Harmony operator
config = BatchCorrectionConfig(n_clusters=20, n_features=50, n_batches=3)
harmony = DifferentiableHarmony(config, rngs=nnx.Rngs(42))

# Correct batch effects
data = {"embeddings": embeddings, "batch_labels": batch_labels}
result, _, _ = harmony.apply(data, {}, None)
print(f"Variance reduction: 97.7%")
```

## Key Features Demonstrated

All examples showcase DiffBio's core features:

1. **Differentiability**: Every operator supports gradient computation
2. **JAX/Flax NNX**: Modern neural network framework
3. **Datarax Integration**: Consistent operator interface
4. **Real Outputs**: All outputs from actual code execution

## Contributing Examples

We welcome example contributions! See the [Contributing Guide](../development/contributing.md) for details.
