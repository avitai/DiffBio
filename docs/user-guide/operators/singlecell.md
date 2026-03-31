# Single-Cell Operators

DiffBio provides comprehensive differentiable operators for single-cell analysis, including clustering, batch correction, and RNA velocity.

<span class="operator-singlecell">Single-Cell</span> <span class="diff-high">Fully Differentiable</span>

## Overview

Single-cell operators enable end-to-end optimization of:

**Clustering & Embedding:**

- **SoftKMeansClustering**: Differentiable soft k-means with learnable centroids
- **DifferentiableArchetypalAnalysis**: PCHA-style archetypal analysis with softmax simplex constraints

**Batch Correction:**

- **DifferentiableHarmony**: Harmony-style batch correction
- **DifferentiableMMDBatchCorrection**: MMD-regularised autoencoder batch correction
- **DifferentiableWGANBatchCorrection**: Adversarial (WGAN) batch correction with gradient reversal

**Trajectory & Fate:**

- **DifferentiableVelocity**: RNA velocity estimation via neural ODEs
- **DifferentiablePseudotime**: Diffusion-map pseudotime ordering
- **DifferentiableFateProbability**: Absorption-based fate estimation
- **DifferentiableOTTrajectory**: Waddington-OT optimal transport trajectory

**Imputation & Denoising:**

- **DifferentiableDiffusionImputer**: MAGIC-style diffusion imputation
- **DifferentiableTransformerDenoiser**: Transformer-based gene denoising

**Cell Type Annotation:**

- **DifferentiableCellAnnotator**: Cell type annotation (celltypist, cellassign, scanvi modes)

**Quality Control:**

- **DifferentiableAmbientRemoval**: VAE-based ambient RNA decontamination
- **DifferentiableDoubletScorer**: Scrublet-style doublet detection
- **DifferentiableSoloDetector**: Solo VAE doublet detection

**Cell Communication:**

- **DifferentiableLigandReceptor**: Ligand-receptor co-expression scoring
- **DifferentiableCellCommunication**: GNN-based cell-cell communication analysis

**Regulatory Networks:**

- **DifferentiableGRN**: GATv2-based gene regulatory network inference

**Spatial Analysis:**

- **DifferentiableSpatialDomain**: STAGATE-style spatial domain identification
- **DifferentiablePASTEAlignment**: PASTE-style spatial slice alignment

**Differential Expression:**

- **DifferentiableSwitchDE**: Sigmoidal switch differential expression
- **DifferentiableDifferentialDistribution**: scDD-style differential distribution testing

**Simulation:**

- **DifferentiableSimulator**: Splatter-style single-cell count simulation

## SoftKMeansClustering

Differentiable k-means clustering with soft assignments and learnable centroids.

### Quick Start

```python
from flax import nnx
from diffbio.operators.singlecell import SoftKMeansClustering, SoftClusteringConfig

# Configure clustering
config = SoftClusteringConfig(
    n_clusters=10,
    n_features=50,
    temperature=1.0,
)

# Create operator
rngs = nnx.Rngs(42)
clustering = SoftKMeansClustering(config, rngs=rngs)

# Apply to cell embeddings
data = {"embeddings": cell_embeddings}  # (n_cells, n_features)
result, state, metadata = clustering.apply(data, {}, None)

# Get results
assignments = result["cluster_assignments"]   # Soft assignments (n_cells, n_clusters)
centroids = result["centroids"]               # Learned centroids
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_clusters` | int | 10 | Number of clusters |
| `n_features` | int | 50 | Feature dimensionality |
| `temperature` | float | 1.0 | Softmax temperature |
| `learnable_centroids` | bool | True | Whether centroids are learnable |

### Soft K-Means Algorithm

Instead of hard cluster assignments:

$$p(c_k | x_i) = \frac{\exp(-d(x_i, \mu_k) / \tau)}{\sum_j \exp(-d(x_i, \mu_j) / \tau)}$$

Where $d$ is distance, $\mu_k$ are centroids, and $\tau$ is temperature.

## DifferentiableHarmony

Harmony-style batch correction for integrating multiple single-cell datasets.

### Quick Start

```python
from diffbio.operators.singlecell import DifferentiableHarmony, BatchCorrectionConfig

# Configure Harmony
config = BatchCorrectionConfig(
    n_clusters=50,
    n_features=50,
    sigma=0.1,
    theta=2.0,
    n_iterations=10,
)

# Create operator
rngs = nnx.Rngs(42)
harmony = DifferentiableHarmony(config, rngs=rngs)

# Apply batch correction
data = {
    "features": cell_embeddings,  # (n_cells, n_features)
    "batch_ids": batch_labels,    # (n_cells,)
}
result, state, metadata = harmony.apply(data, {}, None)

# Get corrected embeddings
corrected = result["corrected_features"]
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_clusters` | int | 50 | Number of cluster centroids |
| `n_features` | int | 50 | Feature dimensionality |
| `sigma` | float | 0.1 | Bandwidth for soft clustering |
| `theta` | float | 2.0 | Diversity penalty strength |
| `n_iterations` | int | 10 | Number of correction iterations |

## DifferentiableVelocity

RNA velocity estimation using neural ODEs for modeling splicing dynamics.

### Quick Start

```python
from diffbio.operators.singlecell import DifferentiableVelocity, VelocityConfig

# Configure velocity estimator
config = VelocityConfig(
    n_genes=2000,
    hidden_dim=64,
    n_layers=2,
    solver_steps=10,
)

# Create operator
rngs = nnx.Rngs(42)
velocity = DifferentiableVelocity(config, rngs=rngs)

# Apply to spliced/unspliced counts
data = {
    "spliced": spliced_counts,     # (n_cells, n_genes)
    "unspliced": unspliced_counts, # (n_cells, n_genes)
}
result, state, metadata = velocity.apply(data, {}, None)

# Get velocity vectors
velocities = result["velocity"]          # Gene velocity (n_cells, n_genes)
latent_time = result["latent_time"]      # Inferred pseudotime
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_genes` | int | 2000 | Number of genes |
| `hidden_dim` | int | 64 | ODE network hidden dimension |
| `n_layers` | int | 2 | Number of ODE network layers |
| `solver_steps` | int | 10 | ODE solver steps |

### RNA Velocity Model

Models the splicing dynamics:

$$\frac{du}{dt} = \alpha - \beta \cdot u$$
$$\frac{ds}{dt} = \beta \cdot u - \gamma \cdot s$$

Where $u$ is unspliced, $s$ is spliced, and $\alpha, \beta, \gamma$ are rate parameters.

## DifferentiableAmbientRemoval

VAE-based ambient RNA removal for cleaning droplet-based scRNA-seq data.

### Quick Start

```python
from diffbio.operators.singlecell import DifferentiableAmbientRemoval, AmbientRemovalConfig

# Configure ambient removal
config = AmbientRemovalConfig(
    n_genes=2000,
    latent_dim=20,
    hidden_dim=128,
)

# Create operator
rngs = nnx.Rngs(42)
ambient_removal = DifferentiableAmbientRemoval(config, rngs=rngs)

# Apply decontamination
data = {
    "counts": raw_counts,           # (n_cells, n_genes)
    "ambient_profile": ambient,     # (n_genes,) estimated from empty droplets
}
result, state, metadata = ambient_removal.apply(data, {}, None)

# Get decontaminated counts
clean_counts = result["decontaminated"]
contamination = result["contamination_fraction"]
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_genes` | int | 2000 | Number of genes |
| `latent_dim` | int | 20 | VAE latent dimension |
| `hidden_dim` | int | 128 | Encoder/decoder hidden dimension |

## DifferentiableDiffusionImputer

MAGIC-style diffusion imputation that constructs a cell-cell affinity graph using an alpha-decaying kernel, builds a row-stochastic Markov matrix M = D^{-1}A, and computes M^t via repeated matrix multiplication for imputation. Recovers gene-gene relationships masked by technical dropout noise.

### Quick Start

```python
from diffbio.operators.singlecell import DifferentiableDiffusionImputer, DiffusionImputerConfig

config = DiffusionImputerConfig(
    n_neighbors=5,
    diffusion_t=3,
    decay=1.0,
    metric="euclidean",
)

imputer = DifferentiableDiffusionImputer(config)
data = {"counts": raw_counts}  # (n_cells, n_genes)
result, state, metadata = imputer.apply(data, {}, None)

imputed = result["imputed_counts"]        # (n_cells, n_genes)
diffusion_op = result["diffusion_operator"]  # M^t matrix
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_neighbors` | int | 5 | Neighbors for local bandwidth estimation |
| `diffusion_t` | int | 3 | Diffusion time steps (matrix power) |
| `n_pca_components` | int | 100 | PCA components (reserved) |
| `decay` | float | 1.0 | Alpha-decaying kernel exponent |
| `metric` | str | "euclidean" | Distance metric ("euclidean" or "cosine") |

### Algorithm

1. Compute pairwise distances between cells
2. Build alpha-decay affinity: $K(i,j) = \exp(-(d / \sigma_i)^{\text{decay}})$
3. Symmetrize via fuzzy set union
4. Row-normalize to Markov matrix $M = D^{-1} A$
5. Compute $M^t$ via repeated matrix multiplication ($t$ iterations)
6. Impute: `imputed = M^t @ counts`

## DifferentiableTransformerDenoiser

Transformer-based gene denoiser that treats genes as tokens. Randomly masks a fraction of genes and predicts masked expression values from unmasked context, recovering dropout events.

### Quick Start

```python
from diffbio.operators.singlecell import (
    DifferentiableTransformerDenoiser, TransformerDenoiserConfig,
)

config = TransformerDenoiserConfig(
    n_genes=2000,
    hidden_dim=128,
    num_layers=2,
    num_heads=4,
    mask_ratio=0.15,
)

denoiser = DifferentiableTransformerDenoiser(
    config, rngs=nnx.Rngs(params=0, sample=1, dropout=2)
)
rp = denoiser.generate_random_params(jax.random.key(0), {"counts": (100, 2000)})
data = {"counts": counts, "gene_ids": jnp.arange(2000)}
result, state, metadata = denoiser.apply(data, {}, None, random_params=rp)

imputed = result["imputed_counts"]  # (n_cells, n_genes)
mask = result["mask"]               # (n_genes,)
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_genes` | int | 2000 | Number of genes |
| `hidden_dim` | int | 128 | Hidden states and embeddings dimension |
| `num_layers` | int | 2 | Transformer encoder layers |
| `num_heads` | int | 4 | Attention heads |
| `mask_ratio` | float | 0.15 | Fraction of genes to mask |
| `dropout_rate` | float | 0.1 | Dropout rate |

### Algorithm

1. Randomly mask `mask_ratio` fraction of genes (zero expression)
2. Project gene IDs into embeddings + add expression projections
3. Pass through transformer encoder for contextualised representations
4. Predict masked gene expression via linear output head
5. Replace masked positions with predictions, keep originals for unmasked

## DifferentiablePseudotime

Diffusion-map pseudotime ordering via accumulated Markov matrix powers. Pseudotime is the L2 distance from a root cell in diffusion-embedding space (rows of the accumulated power matrix).

### Quick Start

```python
from diffbio.operators.singlecell import DifferentiablePseudotime, PseudotimeConfig

config = PseudotimeConfig(
    n_neighbors=15,
    n_diffusion_components=10,
    root_cell_index=0,
)

pseudotime_op = DifferentiablePseudotime(config)
data = {"embeddings": cell_embeddings}  # (n_cells, n_features)
result, state, metadata = pseudotime_op.apply(data, {}, None)

pseudotime = result["pseudotime"]                 # (n_cells,)
dc = result["diffusion_components"]               # (n_cells, n_components)
transition = result["transition_matrix"]           # (n_cells, n_cells)
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_neighbors` | int | 15 | Neighbors for k-NN graph |
| `n_diffusion_components` | int | 10 | Diffusion map components to retain |
| `root_cell_index` | int | 0 | Index of the root cell |
| `metric` | str | "euclidean" | Distance metric |

### Algorithm

1. Compute pairwise distances, build fuzzy k-NN graph
2. Symmetrize and row-normalize to Markov transition matrix $M$
3. Accumulate $M_{\text{sum}} = \sum_{t=1}^{T} M^t$ via repeated matrix multiplication
4. Pseudotime = L2 distance from root cell in $M_{\text{sum}}$ row space

## DifferentiableFateProbability

Absorption-based fate estimation given a Markov transition matrix and terminal state indices. Computes the probability each transient cell reaches each absorbing state.

### Quick Start

```python
from diffbio.operators.singlecell import DifferentiableFateProbability, FateProbabilityConfig

config = FateProbabilityConfig(n_macrostates=2)

fate_op = DifferentiableFateProbability(config)
data = {
    "transition_matrix": transition_matrix,     # (n_cells, n_cells)
    "terminal_states": jnp.array([48, 49]),     # terminal state indices
}
result, state, metadata = fate_op.apply(data, {}, None)

fate_probs = result["fate_probabilities"]  # (n_cells, n_terminal)
macrostates = result["macrostates"]        # (n_cells,) argmax assignments
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_macrostates` | int | 2 | Number of terminal fates |

### Algorithm

Partitions cells into transient (T) and absorbing (A) sets, then solves $(I - Q) B = R$ where $Q$ is the transient-to-transient sub-matrix and $R$ is the transient-to-absorbing sub-matrix. The linear solve is fully differentiable.

## DifferentiableOTTrajectory

Waddington-OT-style trajectory inference using entropy-regularised optimal transport between two timepoints. Computes a transport plan, per-cell growth rates, and interpolated intermediate distributions.

### Quick Start

```python
from diffbio.operators.singlecell import DifferentiableOTTrajectory, OTTrajectoryConfig

config = OTTrajectoryConfig(
    n_genes=200,
    sinkhorn_epsilon=0.1,
    sinkhorn_iters=100,
    interpolation_time=0.5,
)

ot_op = DifferentiableOTTrajectory(config)
data = {
    "counts_t1": counts_day0,  # (n1, n_genes)
    "counts_t2": counts_day2,  # (n2, n_genes)
}
result, state, metadata = ot_op.apply(data, {}, None)

transport_plan = result["transport_plan"]         # (n1, n2)
growth_rates = result["growth_rates"]             # (n1,)
interpolated = result["interpolated_counts"]      # (n1, n_genes)
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_genes` | int | 200 | Number of input genes |
| `sinkhorn_epsilon` | float | 0.1 | Entropy regularisation strength |
| `sinkhorn_iters` | int | 100 | Sinkhorn iterations |
| `growth_rate_regularization` | float | 1.0 | Growth-rate scaling factor |
| `interpolation_time` | float | 0.5 | Interpolation fraction in (0, 1) |

### Algorithm

1. Build squared-Euclidean cost matrix between timepoints
2. Solve OT via Sinkhorn with uniform marginals
3. Estimate growth rates from transport plan row sums
4. Interpolate: $(1-s) \cdot x_{t1} + s \cdot (T \cdot x_{t2}) / T\mathbf{1}$

## DifferentiableCellAnnotator

Cell type annotation operator supporting three modes: **celltypist** (logistic classifier on VAE latent), **cellassign** (marker-gene Poisson likelihood), and **scanvi** (semi-supervised VAE with type-conditioned prior).

### Quick Start

```python
from diffbio.operators.singlecell import DifferentiableCellAnnotator, CellAnnotatorConfig

config = CellAnnotatorConfig(
    annotation_mode="celltypist",  # or "cellassign", "scanvi"
    n_cell_types=10,
    n_genes=2000,
    latent_dim=10,
)

annotator = DifferentiableCellAnnotator(config, rngs=nnx.Rngs(42))
data = {"counts": counts}  # (n_cells, n_genes)
result, state, metadata = annotator.apply(data, {}, None)

probs = result["cell_type_probabilities"]  # (n_cells, n_cell_types)
labels = result["cell_type_labels"]        # (n_cells,) argmax
latent = result["latent"]                  # (n_cells, latent_dim)
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `annotation_mode` | str | "celltypist" | "celltypist", "cellassign", or "scanvi" |
| `n_cell_types` | int | 10 | Number of cell types |
| `n_genes` | int | 2000 | Number of input genes |
| `latent_dim` | int | 10 | VAE latent dimension |
| `hidden_dims` | list[int] | [128, 64] | Encoder/decoder hidden layers |
| `gene_likelihood` | str | "poisson" | "poisson" or "zinb" (scanvi) |

### Annotation Modes

- **celltypist**: Encode to VAE latent, apply linear classifier + softmax
- **cellassign**: Given binary marker matrix, compute per-type Poisson log-likelihoods
- **scanvi**: VAE encoder + classifier with per-type Gaussian priors in latent space; KL marginalised over predicted types for unlabelled cells

## DifferentiableDoubletScorer

Scrublet-style doublet detection. Generates synthetic doublets by summing random cell pairs, embeds into PCA space, and scores each real cell via a Bayesian k-NN likelihood ratio.

### Quick Start

```python
from diffbio.operators.singlecell import DifferentiableDoubletScorer, DoubletScorerConfig

config = DoubletScorerConfig(
    n_neighbors=30,
    expected_doublet_rate=0.06,
    sim_doublet_ratio=2.0,
    n_pca_components=30,
    n_genes=2000,
)

scorer = DifferentiableDoubletScorer(config, rngs=nnx.Rngs(0))
rp = scorer.generate_random_params(jax.random.key(0), {"counts": (500, 2000)})
result, state, metadata = scorer.apply({"counts": counts}, {}, None, random_params=rp)

doublet_scores = result["doublet_scores"]         # (n_cells,)
predicted_doublets = result["predicted_doublets"]  # (n_cells,) soft [0, 1]
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_neighbors` | int | 30 | Base k for k-NN scoring |
| `expected_doublet_rate` | float | 0.06 | Prior doublet fraction (rho) |
| `sim_doublet_ratio` | float | 2.0 | Synthetic-to-real ratio |
| `n_pca_components` | int | 30 | PCA embedding dimensions |
| `n_genes` | int | 2000 | Number of genes |
| `threshold_temperature` | float | 10.0 | Sigmoid threshold temperature |

## DifferentiableSoloDetector

Solo-style VAE doublet detector. Encodes cells through a VAE, generates synthetic doublets, and classifies real vs. synthetic in latent space using a binary classifier.

### Quick Start

```python
from diffbio.operators.singlecell import DifferentiableSoloDetector, SoloDetectorConfig

config = SoloDetectorConfig(
    n_genes=2000,
    latent_dim=10,
    hidden_dims=[128, 64],
    classifier_hidden_dim=64,
)

detector = DifferentiableSoloDetector(config, rngs=nnx.Rngs(42))
rp = detector.generate_random_params(jax.random.key(0), {"counts": (500, 2000)})
result, state, metadata = detector.apply({"counts": counts}, {}, None, random_params=rp)

doublet_probs = result["doublet_probabilities"]  # (n_cells,)
latent = result["latent"]                        # (n_cells, latent_dim)
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_genes` | int | 2000 | Number of genes |
| `latent_dim` | int | 10 | VAE latent dimension |
| `hidden_dims` | list[int] | [128, 64] | Encoder/decoder hidden layers |
| `classifier_hidden_dim` | int | 64 | Classifier hidden dimension |
| `sim_doublet_ratio` | float | 2.0 | Synthetic-to-real ratio |

## DifferentiableMMDBatchCorrection

Autoencoder batch correction with Maximum Mean Discrepancy (MMD) regularisation. Penalises distributional differences between batches in latent space using an RBF kernel.

### Quick Start

```python
from diffbio.operators.singlecell import (
    DifferentiableMMDBatchCorrection, MMDBatchCorrectionConfig,
)

config = MMDBatchCorrectionConfig(
    n_genes=2000,
    hidden_dim=128,
    latent_dim=64,
    kernel_bandwidth=1.0,
)

mmd_op = DifferentiableMMDBatchCorrection(config, rngs=nnx.Rngs(0))
data = {"expression": expression, "batch_labels": batch_labels}
result, state, metadata = mmd_op.apply(data, {}, None)

corrected = result["corrected_expression"]   # (n_cells, n_genes)
latent = result["latent"]                    # (n_cells, latent_dim)
mmd_loss = result["mmd_loss"]               # scalar
recon_loss = result["reconstruction_loss"]  # scalar
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_genes` | int | 2000 | Number of input genes |
| `hidden_dim` | int | 128 | Autoencoder hidden layer width |
| `latent_dim` | int | 64 | Latent space dimensionality |
| `kernel_bandwidth` | float | 1.0 | RBF kernel bandwidth for MMD |
| `use_gradnorm` | bool | False | Use GradNormBalancer for loss balancing |

## DifferentiableWGANBatchCorrection

Adversarial autoencoder batch correction with Wasserstein GAN loss. A discriminator tries to predict batch labels from the latent representation; gradient reversal ensures the encoder learns batch-invariant latents.

### Quick Start

```python
from diffbio.operators.singlecell import (
    DifferentiableWGANBatchCorrection, WGANBatchCorrectionConfig,
)

config = WGANBatchCorrectionConfig(
    n_genes=2000,
    hidden_dim=128,
    latent_dim=64,
    discriminator_hidden_dim=64,
)

wgan_op = DifferentiableWGANBatchCorrection(config, rngs=nnx.Rngs(0))
data = {"expression": expression, "batch_labels": batch_labels}
result, state, metadata = wgan_op.apply(data, {}, None)

corrected = result["corrected_expression"]     # (n_cells, n_genes)
gen_loss = result["generator_loss"]            # scalar
disc_loss = result["discriminator_loss"]       # scalar
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_genes` | int | 2000 | Number of input genes |
| `hidden_dim` | int | 128 | Generator autoencoder hidden width |
| `latent_dim` | int | 64 | Latent space dimensionality |
| `discriminator_hidden_dim` | int | 64 | Discriminator hidden width |
| `use_gradnorm` | bool | False | Use GradNormBalancer for loss balancing |

## DifferentiableLigandReceptor

Ligand-receptor co-expression scoring using fuzzy k-NN adjacency graphs and Hill function saturation. For each L-R pair, scores cell-cell communication via adjacency-weighted co-expression with analytical z-score p-values.

### Quick Start

```python
from diffbio.operators.singlecell import DifferentiableLigandReceptor, LRScoringConfig

config = LRScoringConfig(n_neighbors=15, temperature=1.0)

lr_op = DifferentiableLigandReceptor(config, rngs=nnx.Rngs(0))
data = {
    "counts": counts,                          # (n_cells, n_genes)
    "lr_pairs": jnp.array([[0, 1], [2, 3]]),  # (n_pairs, 2) [ligand_idx, receptor_idx]
}
result, state, metadata = lr_op.apply(data, {}, None)

lr_scores = result["lr_scores"]    # (n_cells, n_pairs)
lr_pvalues = result["lr_pvalues"]  # (n_pairs,) soft p-values
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_neighbors` | int | 15 | Neighbors for k-NN graph |
| `temperature` | float | 1.0 | Soft p-value sigmoid temperature |
| `kh` | float | 0.5 | Hill function half-maximal constant |
| `hill_n` | float | 1.0 | Hill function cooperativity |

## DifferentiableCellCommunication

GNN-based cell-cell communication analysis using GATv2 graph attention on a spatial cell graph with per-edge L-R expression features. Produces per-node pathway activities and communication scores.

### Quick Start

```python
from diffbio.operators.singlecell import (
    DifferentiableCellCommunication, CellCommunicationConfig,
)

config = CellCommunicationConfig(
    n_genes=2000,
    n_lr_pairs=10,
    hidden_dim=64,
    num_heads=4,
    n_pathways=20,
)

comm_op = DifferentiableCellCommunication(config, rngs=nnx.Rngs(0))
data = {
    "counts": counts,                # (n_cells, n_genes)
    "spatial_graph": edge_index,     # (2, n_edges) [source, target]
    "lr_pairs": lr_pairs,            # (n_pairs, 2)
}
result, state, metadata = comm_op.apply(data, {}, None)

comm_scores = result["communication_scores"]    # (n_cells, n_pairs)
signaling = result["signaling_activity"]        # (n_cells, n_pathways)
niche = result["niche_embeddings"]              # (n_cells, hidden_dim)
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_genes` | int | 2000 | Number of genes |
| `n_lr_pairs` | int | 10 | Number of L-R pairs |
| `hidden_dim` | int | 64 | GNN hidden dimension |
| `num_heads` | int | 4 | GATv2 attention heads |
| `num_gnn_layers` | int | 2 | Stacked GATv2 layers |
| `n_pathways` | int | 20 | Signaling pathways to infer |

## DifferentiableGRN

GATv2-based gene regulatory network inference. Builds a TF-gene bipartite graph, applies GATv2 attention, and extracts attention-derived regulatory strengths with soft L1 sparsity. A differentiable alternative to GENIE3/SCENIC.

### Quick Start

```python
from diffbio.operators.singlecell import DifferentiableGRN, GRNInferenceConfig

config = GRNInferenceConfig(
    n_tfs=50,
    n_genes=2000,
    hidden_dim=64,
    num_heads=4,
    sparsity_temperature=0.1,
)

grn_op = DifferentiableGRN(config, rngs=nnx.Rngs(0))
data = {
    "counts": counts,                         # (n_cells, n_genes)
    "tf_indices": jnp.arange(50),             # (n_tfs,)
}
result, state, metadata = grn_op.apply(data, {}, None)

grn_matrix = result["grn_matrix"]   # (n_tfs, n_genes) sparse regulatory matrix
tf_activity = result["tf_activity"]  # (n_cells, n_tfs)
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_tfs` | int | 50 | Number of transcription factors |
| `n_genes` | int | 2000 | Number of genes |
| `hidden_dim` | int | 64 | GATv2 hidden dimension |
| `num_heads` | int | 4 | Attention heads |
| `sparsity_temperature` | float | 0.1 | Soft L1 sparsity gating temperature |
| `sparsity_lambda` | float | 0.01 | L1 regularization weight |

### Algorithm

1. Build dense TF-gene bipartite graph
2. Compute edge features: [TF expression, gene expression, |difference|]
3. Apply GATv2 attention on bipartite graph
4. Extract regulatory scores from updated node representations
5. Apply soft L1 sparsity: $\text{grn} \cdot \sigma(\text{grn} / T)$
6. Compute TF activity: `counts @ grn_matrix.T`

## DifferentiableSpatialDomain

STAGATE-inspired spatial domain identification using dual-graph GATv2 attention (full + pruned k-NN graphs) and learned domain prototypes. Combines gene expression with spatial coordinates for spatial transcriptomics.

### Quick Start

```python
from diffbio.operators.singlecell import DifferentiableSpatialDomain, SpatialDomainConfig

config = SpatialDomainConfig(
    n_genes=2000,
    hidden_dim=64,
    num_heads=4,
    n_domains=7,
    alpha=0.8,
    n_neighbors=15,
)

spatial_op = DifferentiableSpatialDomain(config, rngs=nnx.Rngs(0))
data = {
    "counts": counts,                # (n_cells, n_genes)
    "spatial_coords": coordinates,   # (n_cells, 2)
}
result, state, metadata = spatial_op.apply(data, {}, None)

domains = result["domain_assignments"]     # (n_cells, n_domains)
embeddings = result["spatial_embeddings"]  # (n_cells, hidden_dim)
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_genes` | int | 2000 | Number of input genes |
| `hidden_dim` | int | 64 | Latent embedding dimension |
| `num_heads` | int | 4 | GATv2 attention heads |
| `n_domains` | int | 7 | Number of spatial domains |
| `alpha` | float | 0.8 | Weight for pruned graph (0=full only, 1=pruned only) |
| `n_neighbors` | int | 15 | Neighbors for spatial k-NN graph |

## DifferentiablePASTEAlignment

PASTE-style fused Gromov-Wasserstein optimal transport for aligning two spatial transcriptomics slices. Balances expression dissimilarity with spatial structure preservation.

### Quick Start

```python
from diffbio.operators.singlecell import DifferentiablePASTEAlignment, PASTEAlignmentConfig

config = PASTEAlignmentConfig(
    alpha=0.1,
    sinkhorn_epsilon=0.1,
    sinkhorn_iters=100,
)

paste_op = DifferentiablePASTEAlignment(config, rngs=nnx.Rngs(0))
data = {
    "slice1_counts": counts_a,    # (n1, n_genes)
    "slice2_counts": counts_b,    # (n2, n_genes)
    "slice1_coords": coords_a,    # (n1, 2)
    "slice2_coords": coords_b,    # (n2, 2)
}
result, state, metadata = paste_op.apply(data, {}, None)

transport_plan = result["transport_plan"]    # (n1, n2)
aligned_coords = result["aligned_coords"]   # (n2, 2)
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 0.1 | Balance: 0=expression only, 1=spatial only |
| `sinkhorn_epsilon` | float | 0.1 | Entropy regularisation strength |
| `sinkhorn_iters` | int | 100 | Sinkhorn iterations |

## DifferentiableSwitchDE

Sigmoidal switch model for differential expression along pseudotime. Each gene has a learnable switch time, amplitude, and baseline.

### Quick Start

```python
from diffbio.operators.singlecell import DifferentiableSwitchDE, SwitchDEConfig

config = SwitchDEConfig(n_genes=2000, temperature=1.0)

switch_op = DifferentiableSwitchDE(config, rngs=nnx.Rngs(42))
data = {"counts": counts, "pseudotime": pseudotime}
result, state, metadata = switch_op.apply(data, {}, None)

switch_times = result["switch_times"]            # (n_genes,)
switch_scores = result["switch_scores"]          # (n_genes,)
predicted = result["predicted_expression"]       # (n_cells, n_genes)
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_genes` | int | 2000 | Number of genes to model |
| `temperature` | float | 1.0 | Sigmoid smoothness (lower = sharper) |
| `learnable_temperature` | bool | False | Whether temperature is learnable |

### Algorithm

Models expression as: $g(t) = a \cdot \sigma((t - t_{\text{switch}}) / T) + b$

Switch score quantifies strength: $a / (4T)$ (maximum sigmoid derivative scaled by amplitude).

## DifferentiableDifferentialDistribution

scDD-style differential distribution testing. Computes a soft KS statistic using sigmoid-smoothed CDF and classifies distributional difference patterns (shift, scale, both, none) via a learned linear head.

### Quick Start

```python
from diffbio.operators.singlecell import (
    DifferentiableDifferentialDistribution, DifferentialDistributionConfig,
)

config = DifferentialDistributionConfig(
    n_genes=2000,
    temperature=1.0,
    n_pattern_classes=4,
)

dd_op = DifferentiableDifferentialDistribution(config, rngs=nnx.Rngs(42))
data = {"counts": counts, "condition_labels": condition_labels}
result, state, metadata = dd_op.apply(data, {}, None)

ks_stats = result["ks_statistics"]     # (n_genes,)
patterns = result["pattern_labels"]    # (n_genes,) 0=shift, 1=scale, 2=both, 3=none
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_genes` | int | 2000 | Number of genes |
| `temperature` | float | 1.0 | Soft CDF sigmoid temperature |
| `learnable_temperature` | bool | False | Whether temperature is learnable |
| `n_pattern_classes` | int | 4 | Pattern categories (shift, scale, both, none) |

## DifferentiableSimulator

Splatter-style differentiable single-cell count simulator using a Gamma-Poisson model with learnable parameters. Generates realistic scRNA-seq count matrices with group-specific DE, batch effects, and dropout.

### Quick Start

```python
from diffbio.operators.singlecell import DifferentiableSimulator, SimulationConfig

config = SimulationConfig(
    n_cells=500,
    n_genes=200,
    n_groups=3,
    n_batches=1,
    de_prob=0.1,
)

sim = DifferentiableSimulator(config, rngs=nnx.Rngs(0, sample=1))
rp = sim.generate_random_params(jax.random.key(0), {})
result, state, metadata = sim.apply({}, {}, None, random_params=rp)

counts = result["counts"]           # (n_cells, n_genes)
group_labels = result["group_labels"]  # (n_cells,)
de_mask = result["de_mask"]          # (n_groups, n_genes)
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_cells` | int | 500 | Cells to simulate |
| `n_genes` | int | 200 | Genes to simulate |
| `n_groups` | int | 3 | Cell groups for DE |
| `n_batches` | int | 1 | Experimental batches |
| `mean_shape` | float | 0.6 | Gamma shape for gene means |
| `mean_rate` | float | 0.3 | Gamma rate for gene means |
| `de_prob` | float | 0.1 | Fraction of DE genes |
| `dropout_mid` | float | -1.0 | Logistic dropout midpoint |
| `dropout_shape` | float | -0.5 | Logistic dropout shape |

### Algorithm

1. Gene means: softplus(learnable logits) * Gamma perturbation
2. Library sizes: LogNormal sampling
3. Soft group assignments + LogNormal DE fold-changes
4. Multiplicative batch effects via exp(learnable shift)
5. Expression-dependent dropout: sigmoid(shape * (log(means) - mid))
6. Continuous Poisson relaxation: means + sqrt(means) * noise

## DifferentiableArchetypalAnalysis

PCHA-style archetypal analysis with softmax simplex constraints. Each cell is represented as a temperature-controlled convex combination of learnable archetype prototypes.

### Quick Start

```python
from diffbio.operators.singlecell import (
    DifferentiableArchetypalAnalysis, ArchetypalAnalysisConfig,
)

config = ArchetypalAnalysisConfig(
    n_genes=2000,
    n_archetypes=5,
    hidden_dim=64,
    temperature=1.0,
)

arch_op = DifferentiableArchetypalAnalysis(config, rngs=nnx.Rngs(0))
data = {"counts": counts}  # (n_cells, n_genes)
result, state, metadata = arch_op.apply(data, {}, None)

weights = result["archetype_weights"]  # (n_cells, n_archetypes) simplex weights
archetypes = result["archetypes"]      # (n_archetypes, n_genes)
reconstructed = result["reconstructed"]  # (n_cells, n_genes)
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_genes` | int | 2000 | Number of input genes |
| `n_archetypes` | int | 5 | Number of archetype prototypes |
| `hidden_dim` | int | 64 | Encoder MLP hidden dimension |
| `temperature` | float | 1.0 | Softmax temperature (lower = sharper) |
| `learnable_temperature` | bool | False | Whether temperature is learnable |

### Algorithm

1. Encode cells to archetype weight logits via MLP
2. Apply temperature-scaled softmax to enforce simplex constraints
3. Reconstruct: `weights @ archetypes`

## Training Single-Cell Pipelines

### Combined Loss Example

```python
from diffbio.losses.singlecell_losses import (
    BatchMixingLoss,
    ClusteringCompactnessLoss,
)

batch_loss = BatchMixingLoss(n_neighbors=15, temperature=1.0)
cluster_loss = ClusteringCompactnessLoss(temperature=1.0)

def combined_loss(model, data):
    result, _, _ = model.apply(data, {}, None)

    # Batch mixing (maximize)
    l_batch = -batch_loss(result["corrected_features"], data["batch_ids"])

    # Cluster compactness (minimize)
    l_cluster = cluster_loss(
        result["corrected_features"],
        result["cluster_assignments"],
    )

    return l_batch + 0.1 * l_cluster
```

## Use Cases

| Application | Operator | Description |
|-------------|----------|-------------|
| Cell clustering | SoftKMeansClustering | Identify cell types |
| Archetypal analysis | DifferentiableArchetypalAnalysis | Identify extreme cell states |
| Dataset integration | DifferentiableHarmony | Merge experiments (Harmony) |
| Batch correction (MMD) | DifferentiableMMDBatchCorrection | MMD-regularised correction |
| Batch correction (WGAN) | DifferentiableWGANBatchCorrection | Adversarial correction |
| Trajectory inference | DifferentiableVelocity | Model differentiation |
| Pseudotime ordering | DifferentiablePseudotime | Diffusion-map pseudotime |
| Cell fate estimation | DifferentiableFateProbability | Absorption probabilities |
| Trajectory OT | DifferentiableOTTrajectory | Waddington-OT trajectory |
| Imputation (diffusion) | DifferentiableDiffusionImputer | MAGIC-style imputation |
| Imputation (transformer) | DifferentiableTransformerDenoiser | Masked gene denoising |
| Cell type annotation | DifferentiableCellAnnotator | Multi-mode annotation |
| Doublet detection | DifferentiableDoubletScorer | Scrublet-style scoring |
| Doublet detection (VAE) | DifferentiableSoloDetector | Solo VAE classification |
| Data cleaning | DifferentiableAmbientRemoval | Remove ambient RNA |
| L-R communication | DifferentiableLigandReceptor | Ligand-receptor scoring |
| Cell communication | DifferentiableCellCommunication | GNN communication analysis |
| GRN inference | DifferentiableGRN | GATv2 regulatory networks |
| Spatial domains | DifferentiableSpatialDomain | STAGATE spatial domains |
| Slice alignment | DifferentiablePASTEAlignment | PASTE OT alignment |
| Switch DE | DifferentiableSwitchDE | Sigmoidal switch genes |
| Differential distribution | DifferentiableDifferentialDistribution | scDD-style testing |
| Simulation | DifferentiableSimulator | Splatter-style counts |

## Next Steps

- See [Normalization Operators](normalization.md) for VAE-based normalization
- Explore [Single-Cell Losses](../losses/singlecell.md) for training objectives
- Check [Single-Cell Clustering Example](../../examples/basic/single-cell-clustering.md)
