# Representation Learning and Statistical Models

Biological data is high-dimensional, noisy, and structured by latent processes
that are not directly observed. Representation learning extracts meaningful
low-dimensional embeddings, while statistical models capture the generative
process behind the data. DiffBio provides 10 differentiable operators covering
normalization, dimensionality reduction, biological language models, and
classical statistical models.

---

## Why Raw Counts Need Normalization

Gene expression counts from RNA-seq or scRNA-seq are confounded by technical
factors:

- **Library size**: Cells sequenced more deeply have higher counts for all genes
- **Capture efficiency**: Different cells yield different fractions of their RNA
- **Batch effects**: Samples processed on different days show systematic shifts
- **Dropout**: Low-abundance transcripts are frequently missed (zeros)

Normalization separates biological signal from technical variation. Simple
approaches (CPM, log-normalization) apply fixed transformations. DiffBio takes
a generative approach with `VAENormalizer`.

### VAE-Based Normalization

`VAENormalizer` implements an scVI-style variational autoencoder that explicitly
models the data-generating process:

1. **Encoder**: Maps raw counts to a latent cell state $z \sim q(z|x)$
2. **Decoder**: Reconstructs counts from $z$ using a configurable likelihood
3. **Latent space**: The low-dimensional $z$ captures biological variation
   while the decoder absorbs technical noise

Two likelihood models are supported:

| Likelihood | Distribution | Best For |
|---|---|---|
| **Poisson** | $P(x \mid \mu)$ | Moderate overdispersion |
| **ZINB** | Zero-inflated negative binomial | scRNA-seq with heavy dropout |

The ZINB likelihood explicitly models excess zeros (dropout) with a separate
zero-inflation parameter, making it well-suited for single-cell data. The
normalized output is the expected expression under the learned model, with
technical variation removed.

---

## Dimensionality Reduction

High-dimensional biological data (thousands of genes per cell) must be
projected to 2-3 dimensions for visualization and to a moderate number of
dimensions (10-50) for downstream analysis. DiffBio provides two complementary
approaches that preserve different aspects of the data geometry.

### UMAP: Local Structure Preservation

`DifferentiableUMAP` preserves local neighborhood relationships -- cells that
are similar in high-dimensional space remain close in the embedding. The
algorithm:

1. Builds a fuzzy simplicial set (weighted k-NN graph) in high-dimensional space
2. Optimizes a low-dimensional embedding to match this graph structure
3. Uses a cross-entropy loss between high-D and low-D edge weights

UMAP excels at preserving cluster boundaries and local cell-cell relationships.
DiffBio's implementation uses a neural projection network (MLP) rather than
direct coordinate optimization, making the embedding function generalizable
to unseen data.

### PHATE: Global Structure Preservation

`DifferentiablePHATE` preserves global trajectory structure through
diffusion-based distances:

1. Builds an alpha-decay affinity kernel from pairwise distances
2. Symmetrizes and normalizes to a Markov transition matrix
3. Powers the matrix to diffusion time $t$ via eigendecomposition
4. Computes potential distances (log or sqrt transform)
5. Applies classical MDS to embed the potential distance matrix

PHATE excels at revealing continuous transitions (differentiation trajectories,
disease progression) that UMAP can fragment into discrete clusters. The
diffusion time $t$ controls the scale of structure preserved -- short times
emphasize local neighborhoods, long times reveal global connectivity.

### When to Use Each

| Method | Preserves | Best For | DiffBio Operator |
|---|---|---|---|
| **UMAP** | Local neighborhoods | Cluster visualization, cell type identification | `DifferentiableUMAP` |
| **PHATE** | Global trajectories | Continuous processes, branching dynamics | `DifferentiablePHATE` |

---

## Sequence Embeddings

Raw DNA/RNA sequences (one-hot encoded) are high-dimensional and sparse.
Embedding operators convert them to dense, information-rich representations.

`SequenceEmbedding` uses a stack of 1D convolutions over one-hot encoded
sequences:

1. Convolutional layers with ReLU activation extract local sequence features
   (k-mer patterns, motifs)
2. Each position gets a feature vector of dimension `embedding_dim`
3. Global average pooling produces a fixed-size sequence representation

This simple architecture is effective for capturing local sequence context
and serves as a building block for more complex models.

---

## Biological Language Models

Language models trained on biological sequences learn representations that
capture evolutionary and functional relationships. DiffBio provides three
operators at different scales:

### Sequence Transformers

`TransformerSequenceEncoder` implements a DNABERT/RNA-FM-style transformer
for DNA and RNA sequences:

- Multi-head self-attention captures long-range sequence dependencies
- Sinusoidal positional encoding provides position awareness
- Configurable pooling (mean or CLS token) produces sequence-level embeddings
- Supports both one-hot input (linear projection) and token ID input
  (embedding lookup)

The encoder is pre-trainable on large sequence databases (masked language
modeling), then fine-tunable for specific tasks with full gradient flow.

### Gene Expression Foundation Models

`DifferentiableFoundationModel` implements a Geneformer/scGPT-style model
for gene expression data:

1. `GeneTokenizer` converts continuous expression values to rank-ordered
   gene tokens using differentiable soft sorting -- each gene's rank
   becomes its "word" in the expression "sentence"
2. Gene identity embeddings and expression value embeddings are combined
3. Random masking (configurable ratio, default 15%) hides a subset of genes
4. A transformer encoder predicts masked expression values from context

This architecture learns cell state representations from large-scale
expression atlases. The rank-value tokenization (from Geneformer) converts
continuous expression into a discrete-like representation suitable for
transformer processing, while maintaining differentiability through the
soft sort.

### Operator Summary

| Operator | Input | Architecture | Pre-training Task |
|---|---|---|---|
| `SequenceEmbedding` | One-hot DNA/RNA | CNN | Task-specific |
| `TransformerSequenceEncoder` | One-hot or token DNA/RNA | Transformer | Masked language modeling |
| `GeneTokenizer` | Expression vectors | Soft sort | -- (preprocessing) |
| `DifferentiableFoundationModel` | Expression vectors | Tokenizer + Transformer | Masked expression prediction |

---

## Statistical Models

Classical statistical models remain essential for biological data analysis.
DiffBio provides differentiable implementations that can be jointly optimized
with neural components.

### Hidden Markov Models

`DifferentiableHMM` implements the forward algorithm with logsumexp for
numerical stability. HMMs model sequential data where observed emissions
depend on unobserved hidden states:

- **States**: Discrete hidden variables (e.g., coding/non-coding regions,
  chromatin states)
- **Transitions**: Probability of switching between states
- **Emissions**: Probability of observing each symbol given the current state

Both transition and emission parameters are learnable. The forward algorithm
computes the likelihood of an observation sequence, and gradients flow
through the entire computation via logsumexp (replacing the hard max of
Viterbi decoding). Applications include gene finding, chromatin state
annotation, and profile search.

### Negative Binomial GLM

`DifferentiableNBGLM` implements the DESeq2-style model for differential
expression analysis:

$$
\log(\mu_{ij}) = \mathbf{x}_i^T \boldsymbol{\beta}_j
$$

$$
Y_{ij} \sim \text{NB}(\mu_{ij}, \alpha_j)
$$

Where $\mathbf{x}_i$ is the design matrix row for sample $i$,
$\boldsymbol{\beta}_j$ are gene-specific coefficients, and $\alpha_j$ is
the gene-specific dispersion. Gradients flow through both $\beta$ and
$\alpha$, enabling joint estimation rather than the alternating optimization
used in traditional implementations.

### EM-Based Quantification

`DifferentiableEMQuantifier` implements an unrolled EM algorithm for
transcript abundance estimation (inspired by Salmon and Kallisto):

1. **E-step**: Compute read-to-transcript responsibilities using softmax
   over compatibility-weighted abundances
2. **M-step**: Update transcript abundances from responsibilities
3. **Repeat**: Fixed number of iterations (no convergence check)

The fixed iteration count makes the algorithm fully differentiable --
gradients flow through all EM steps. Temperature in the E-step softmax
controls how sharply reads are assigned to transcripts (lower temperature
approaches hard assignment).

---

## Why Differentiability Matters for Representation Learning

Traditional representation learning and statistical modeling treat each
component as an independent optimization problem. The VAE is trained, then
UMAP is applied to its output, then clustering is run on the UMAP embedding.
Each step optimizes its own objective without knowledge of downstream tasks.

DiffBio's differentiable operators enable:

1. **Task-aware normalization**: A downstream classification loss propagates
   gradients back through the VAE, learning a latent space optimized for
   the specific analysis task -- not just reconstruction
2. **Joint embedding-clustering**: UMAP or PHATE embeddings can be jointly
   optimized with clustering objectives, producing embeddings where cluster
   boundaries are clearer
3. **End-to-end language models**: Foundation model pre-training and
   fine-tuning use the same differentiable operators, enabling smooth
   transfer from pre-training to task-specific adaptation
4. **Unified statistical-neural pipelines**: HMM parameters, NB-GLM
   coefficients, and neural network weights can all be updated by the same
   gradient-based optimizer, replacing multi-stage estimation with joint
   optimization

---

## Further Reading

- [Normalization Operators](../operators/normalization.md) -- VAE normalizer, embeddings, UMAP, PHATE
- [Language Model Operators](../operators/language-models.md) -- transformers and foundation models
- [Statistical Operators](../operators/statistical.md) -- HMM, NB-GLM, EM quantification
- [Normalization API](../../api/operators/normalization.md) -- full API reference
- [Statistical API](../../api/operators/statistical.md) -- full API reference

### References

1. Lopez et al. "Deep generative modeling for single-cell transcriptomics."
   *Nature Methods* 15, 2018.
2. McInnes et al. "UMAP: Uniform Manifold Approximation and Projection
   for Dimension Reduction." *JOSS* 3(29), 2018.
3. Moon et al. "Visualizing transitions and structure for biological data
   exploration." *Nature Biotechnology* 37, 2019.
4. Theodoris et al. "Transfer learning enables predictions in network
   biology." *Nature* 618, 2023.
5. Love et al. "Moderated estimation of fold change and dispersion for
   RNA-seq data with DESeq2." *Genome Biology* 15, 2014.
