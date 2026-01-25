# Statistical Operators

DiffBio provides differentiable statistical operators for probabilistic modeling, including Hidden Markov Models, generalized linear models, and EM algorithms.

<span class="operator-statistical">Statistical</span> <span class="diff-high">Fully Differentiable</span>

## Overview

Statistical operators enable end-to-end optimization of:

- **DifferentiableHMM**: Forward algorithm with logsumexp stability
- **DifferentiableNBGLM**: Negative binomial GLM for differential expression
- **DifferentiableEMQuantifier**: Unrolled EM for transcript quantification

## DifferentiableHMM

Hidden Markov Model with differentiable forward algorithm using logsumexp.

### Quick Start

```python
from flax import nnx
from diffbio.operators.statistical import DifferentiableHMM, HMMConfig

# Configure HMM
config = HMMConfig(
    n_states=5,
    n_observations=4,
    temperature=1.0,
)

# Create operator
rngs = nnx.Rngs(42)
hmm = DifferentiableHMM(config, rngs=rngs)

# Apply to observation sequence
data = {"observations": obs_sequence}  # (seq_length, n_observations)
result, state, metadata = hmm.apply(data, {}, None)

# Get results
log_likelihood = result["log_likelihood"]
forward_probs = result["forward_probs"]      # Alpha values
backward_probs = result["backward_probs"]    # Beta values
posteriors = result["posteriors"]            # State posteriors
viterbi_path = result["viterbi_path"]        # Most likely path
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_states` | int | 5 | Number of hidden states |
| `n_observations` | int | 4 | Observation vocabulary size |
| `temperature` | float | 1.0 | Soft Viterbi temperature |

### HMM Parameters

The HMM learns three parameter sets:

```python
# Transition probabilities: P(state_t | state_{t-1})
hmm.transition_logits  # (n_states, n_states)

# Emission probabilities: P(observation | state)
hmm.emission_logits    # (n_states, n_observations)

# Initial state distribution: P(state_0)
hmm.initial_logits     # (n_states,)
```

### Forward Algorithm

The forward algorithm computes $\alpha_t(i) = P(o_1, \ldots, o_t, s_t = i)$:

$$\alpha_t(j) = b_j(o_t) \sum_i \alpha_{t-1}(i) \cdot a_{ij}$$

DiffBio uses log-space computation with logsumexp for numerical stability.

## DifferentiableNBGLM

Negative binomial GLM for modeling overdispersed count data, as used in differential expression analysis.

### Quick Start

```python
from diffbio.operators.statistical import DifferentiableNBGLM, NBGLMConfig

# Configure NB-GLM
config = NBGLMConfig(
    n_genes=2000,
    n_covariates=10,
    dispersion_model="gene",  # or "shared" or "learned"
)

# Create operator
rngs = nnx.Rngs(42)
nbglm = DifferentiableNBGLM(config, rngs=rngs)

# Fit model
data = {
    "counts": count_matrix,    # (n_samples, n_genes)
    "design": design_matrix,   # (n_samples, n_covariates)
    "size_factors": size_factors,  # (n_samples,)
}
result, state, metadata = nbglm.apply(data, {}, None)

# Get results
betas = result["coefficients"]       # (n_genes, n_covariates)
dispersions = result["dispersions"]  # (n_genes,)
log_likelihood = result["log_likelihood"]
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_genes` | int | 2000 | Number of genes |
| `n_covariates` | int | 10 | Number of design matrix columns |
| `dispersion_model` | str | "gene" | Dispersion estimation approach |

### Negative Binomial Distribution

The NB-GLM models counts as:

$$Y_{ij} \sim NB(\mu_{ij}, \phi_j)$$

Where:

- $\mu_{ij} = s_i \cdot \exp(X_i \cdot \beta_j)$
- $s_i$ = size factor for sample $i$
- $\phi_j$ = dispersion for gene $j$

### Differential Expression Testing

```python
# Compute Wald statistics
wald_stats = result["coefficients"] / result["standard_errors"]

# P-values (two-sided)
from jax.scipy.stats import norm
p_values = 2 * (1 - norm.cdf(jnp.abs(wald_stats)))

# Log2 fold changes
log2fc = result["coefficients"][:, 1] / jnp.log(2)  # Assuming condition is column 1
```

## DifferentiableEMQuantifier

Unrolled EM algorithm for transcript quantification, inspired by tools like Salmon/kallisto.

### Quick Start

```python
from diffbio.operators.statistical import DifferentiableEMQuantifier, EMQuantifierConfig

# Configure EM quantifier
config = EMQuantifierConfig(
    n_transcripts=50000,
    n_iterations=100,
    convergence_tol=1e-6,
)

# Create operator
rngs = nnx.Rngs(42)
em_quant = DifferentiableEMQuantifier(config, rngs=rngs)

# Quantify transcripts
data = {
    "equivalence_classes": eq_classes,  # Read assignments
    "counts": eq_counts,                 # Counts per equivalence class
    "effective_lengths": eff_lengths,    # Transcript effective lengths
}
result, state, metadata = em_quant.apply(data, {}, None)

# Get TPM values
tpm = result["tpm"]                      # (n_transcripts,)
counts = result["estimated_counts"]      # Estimated counts
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_transcripts` | int | 50000 | Number of transcripts |
| `n_iterations` | int | 100 | Max EM iterations |
| `convergence_tol` | float | 1e-6 | Convergence tolerance |

### EM Algorithm

The EM algorithm alternates:

**E-step**: Compute expected read assignments
$$p(t | r) = \frac{\rho_t \cdot w_{rt}}{\sum_t \rho_t \cdot w_{rt}}$$

**M-step**: Update transcript abundances
$$\rho_t = \frac{\sum_r p(t | r)}{L_t}$$

DiffBio unrolls EM iterations for end-to-end gradient flow.

## Training Statistical Models

### HMM Training with Forward Loss

```python
from diffbio.losses.statistical_losses import HMMLikelihoodLoss

hmm_loss = HMMLikelihoodLoss()

def train_hmm(hmm, observations):
    data = {"observations": observations}
    result, _, _ = hmm.apply(data, {}, None)
    return hmm_loss(result["log_likelihood"])

grads = nnx.grad(train_hmm)(hmm, train_observations)
```

### NB-GLM Training

```python
from diffbio.losses.statistical_losses import NegativeBinomialLoss

nb_loss = NegativeBinomialLoss()

def train_nbglm(nbglm, counts, design, size_factors):
    data = {
        "counts": counts,
        "design": design,
        "size_factors": size_factors,
    }
    result, _, _ = nbglm.apply(data, {}, None)

    return nb_loss(
        counts=counts,
        predicted_mean=result["predicted_means"],
        dispersion=result["dispersions"],
    )
```

## Use Cases

| Application | Operator | Description |
|-------------|----------|-------------|
| Sequence segmentation | DifferentiableHMM | Find sequence regions |
| Gene expression | DifferentiableNBGLM | Differential expression |
| Chromatin states | DifferentiableHMM | Annotate chromatin |
| Transcript quantification | DifferentiableEMQuantifier | RNA-seq quantification |

## Next Steps

- See [Epigenomics Operators](epigenomics.md) for chromatin state HMM
- Explore [Differential Expression Pipeline](../pipelines/differential-expression.md)
- Check [Statistical Losses](../losses/statistical.md) for training objectives
