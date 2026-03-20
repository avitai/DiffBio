# Core Concepts

This page covers the ideas you need before working with DiffBio. It focuses on
what DiffBio provides — not general JAX or Flax documentation.

---

## Smooth Approximations

Traditional bioinformatics algorithms rely on discrete operations that block
gradient flow. DiffBio replaces each one with a temperature-controlled smooth
approximation:

| Discrete Operation | DiffBio Approximation | Where Used |
|---|---|---|
| `max(a, b)` | `logsumexp(a/τ, b/τ) · τ` | Smith-Waterman recurrence |
| `argmax(x)` | `softmax(x / τ)` | Soft k-means clustering |
| `threshold(x > t)` | `sigmoid((x - t) / τ)` | Quality filtering, doublet scoring |
| Hard counting | Weighted accumulation | Pileup generation |
| Hard assignment | Soft assignment probabilities | Batch correction, cell annotation |

The **temperature** parameter `τ` controls the trade-off:

- **Low τ** — closer to the discrete algorithm, sharper decisions, weaker gradients
- **High τ** — smoother output, stronger gradients, easier to optimize

This is DiffBio's core mechanism: every operator in the library uses some form
of smooth relaxation to stay differentiable.

---

## The Operator Contract

Every DiffBio operator inherits from datarax's `OperatorModule` and exposes
a single entry point:

```python
result, state, metadata = operator.apply(data, state, metadata)
```

| Argument | Type | Typical Value | Purpose |
|---|---|---|---|
| `data` | `dict[str, Array]` | `{"counts": jnp.array(...)}` | Input tensors keyed by name |
| `state` | `dict` | `{}` | Per-element state (empty for stateless ops) |
| `metadata` | `dict \| None` | `None` | Optional metadata |

The result dict contains the original input keys plus new keys added by the
operator. This makes chaining trivial — the output of one operator is the
input to the next.

```python
# Chain: Impute → Cluster → Pseudotime
result, _, _ = imputer.apply({"counts": counts}, {}, None)
result["embeddings"] = result["imputed_counts"]
result, _, _ = clusterer.apply(result, {}, None)
result, _, _ = pseudotime.apply(result, {}, None)

# result now contains counts, imputed_counts, cluster_assignments, pseudotime, ...
```

---

## Operator Configuration

Each operator has a frozen dataclass config that defines its parameters:

```python
from diffbio.operators.singlecell import SoftKMeansClustering, SoftClusteringConfig

config = SoftClusteringConfig(
    n_clusters=10,
    n_features=50,
    temperature=1.0,
)
operator = SoftKMeansClustering(config, rngs=nnx.Rngs(42))
```

Configs are separate from the operator so you can serialize, compare, and
reproduce configurations independently of model weights.

---

## Operator Domains

DiffBio organizes operators by biological domain. Each domain corresponds to
a subpackage under `diffbio.operators`:

| Domain | Subpackage | Examples |
|---|---|---|
| Single-Cell | `singlecell` | Clustering, batch correction, trajectory, imputation, cell annotation |
| Alignment | `alignment` | Smith-Waterman, profile HMM, soft MSA |
| Variant Calling | `variant` | Pileup, classifier, CNV segmentation |
| Normalization | `normalization` | VAE normalizer, UMAP, PHATE |
| Drug Discovery | `drug_discovery` | Molecular fingerprints, ADMET, GNN property prediction |
| Epigenomics | `epigenomics` | Peak calling, chromatin state annotation |
| Multi-omics | `multiomics` | Hi-C contact maps, spatial deconvolution, multi-modal VAE |
| RNA-seq | `rnaseq` | Splicing PSI, motif discovery |
| Preprocessing | `preprocessing` | Adapter removal, error correction, duplicate filtering |
| Statistical | `statistical` | HMM, EM quantification, negative binomial GLM |

Every operator across all domains follows the same `apply()` contract.

---

## Gradient Flow Through Pipelines

The central value proposition of DiffBio: gradients propagate backward through
an entire pipeline of operators.

```
Input Data
      │
      ▼
┌─────────────────┐
│ Operator A      │ ← ∂L/∂params_A
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Operator B      │ ← ∂L/∂params_B
└────────┬────────┘
         │
         ▼
     Loss Function
```

In practice this means you can define a loss at the end of a pipeline and
optimize all upstream operator parameters jointly:

```python
def pipeline_loss(data):
    r1, _, _ = normalizer.apply(data, {}, None)
    r2, _, _ = clusterer.apply(r1, {}, None)
    return r2["cluster_assignments"].sum()

grad = jax.grad(pipeline_loss)(data)
# grad["counts"] contains ∂loss/∂input through both operators
```

Every example in the [Examples](../examples/overview.md) section demonstrates
this with a gradient verification step.

---

## Temperature Scheduling

Because temperature controls the accuracy-trainability trade-off, a common
pattern is **annealing** — start warm (smooth, easy gradients) and cool toward
the discrete solution:

```python
def temperature_schedule(step, initial=10.0, final=0.1, decay_steps=10000):
    """Exponential temperature decay."""
    decay_rate = (final / initial) ** (1.0 / decay_steps)
    return initial * (decay_rate ** step)
```

When temperature is learnable (`nnx.Param`), the optimizer can also discover
the right smoothness level from data.

---

## The Ecosystem

DiffBio is part of a family of libraries that share the datarax operator
contract:

| Library | Role | How DiffBio Uses It |
|---|---|---|
| **datarax** | Pipeline framework | Base `OperatorModule`, config system, data flow |
| **calibrax** | Metrics and evaluation | ARI, NMI, silhouette, AUROC for evaluating results |
| **artifex** | Generative model losses | KL divergence, ELBO components for VAE operators |
| **opifex** | Multi-task training | `GradNormBalancer` for multi-loss optimization |

DiffBio operators produce results; calibrax evaluates them; artifex provides
training losses; opifex balances multiple objectives. The integration is
through standard JAX arrays — no special adapters needed.

```python
# DiffBio operator produces latent representations
result, _, _ = vae_normalizer.apply(data, {}, None)

# calibrax evaluates clustering quality
from calibrax.metrics.functional.clustering import silhouette_score
score = silhouette_score(result["latent_mean"], labels)

# artifex provides training loss components
from artifex.generative_models.core.losses.divergence import gaussian_kl_divergence
kl = gaussian_kl_divergence(result["latent_mean"], result["latent_logvar"])
```

---

## Data Representations

DiffBio operators expect JAX arrays in specific formats depending on the domain:

| Data Type | Shape | Description |
|---|---|---|
| Count matrices | `(n_cells, n_genes)` | Gene expression counts (single-cell) |
| Embeddings | `(n_samples, n_features)` | Latent or reduced representations |
| Sequences (one-hot) | `(length, alphabet_size)` | One-hot encoded DNA/RNA/protein |
| Batch labels | `(n_samples,)` | Integer batch assignments |
| Spatial coordinates | `(n_spots, 2)` | Physical x, y positions |

One-hot encoding is used for sequences because it allows gradients to flow
through sequence-dependent operations:

```python
# "ACGT" → one-hot with DNA alphabet (A=0, C=1, G=2, T=3)
seq_indices = jnp.array([0, 1, 2, 3])
seq_onehot = jnp.eye(4)[seq_indices]
# Shape: (4, 4) — each row is a one-hot vector
```

Count matrices and embeddings are used as-is — no special encoding needed.

---

## Next Steps

- [Quick Start](quickstart.md) — run your first operator
- [Operators Overview](../user-guide/operators/overview.md) — browse available operators by domain
- [Examples](../examples/overview.md) — runnable examples with visual outputs
- [Differentiable Bioinformatics](../user-guide/concepts/differentiable-bioinformatics.md) — deeper theory on smooth relaxations
