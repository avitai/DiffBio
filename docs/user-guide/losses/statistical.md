# Statistical Losses

DiffBio provides statistical loss functions for probabilistic models and count data.

## NegativeBinomialLoss

Negative binomial log-likelihood loss for overdispersed count data.

### Overview

The negative binomial distribution is commonly used in genomics for modeling RNA-seq counts, which exhibit overdispersion (variance > mean).

### Usage

```python
from diffbio.losses.statistical_losses import NegativeBinomialLoss

# Create loss function
nb_loss = NegativeBinomialLoss()

# Compute loss (signature: counts, mu, theta)
loss = nb_loss(
    observed_counts,    # (n_samples, n_genes)
    model_means,        # (n_samples, n_genes) predicted mean (mu)
    gene_dispersions,   # (n_genes,) dispersion (theta)
)
```

### Parameters

The loss takes no configuration parameters; all inputs are provided at call time.

### Algorithm

Negative binomial log-likelihood:

$$\log P(y | \mu, \alpha) = \log\Gamma(y + \frac{1}{\alpha}) - \log\Gamma(y+1) - \log\Gamma(\frac{1}{\alpha}) + \frac{1}{\alpha}\log(\frac{1}{1+\alpha\mu}) + y\log(\frac{\alpha\mu}{1+\alpha\mu})$$

Where:

- $y$ = observed count
- $\mu$ = predicted mean
- $\alpha$ = dispersion (overdispersion parameter)

### Training Example

```python
from flax import nnx

def de_model_loss(model, counts, design):
    """Train differential expression model."""
    data = {"counts": counts, "design": design}
    result, _, _ = model.apply(data, {}, None)

    dispersion = jnp.exp(model.nb_glm.log_dispersion[...])
    return nb_loss(
        counts,
        result["predicted_mean"],
        dispersion,
    )

grads = nnx.grad(de_model_loss)(model, counts, design)
```

## VAELoss

Evidence Lower Bound (ELBO) loss for variational autoencoders.

### Overview

VAE loss combines reconstruction loss with KL divergence regularization, enabling probabilistic latent space learning.

### Usage

```python
from diffbio.losses.statistical_losses import VAELoss

# Create loss function
vae_loss = VAELoss(
    kl_weight=1.0,
    reconstruction_type="mse",  # or "bce" for binary cross-entropy
)

# Compute loss (signature: x, x_recon, mean, logvar)
loss = vae_loss(
    original_input,         # (n_samples, n_features) x
    decoded_output,         # (n_samples, n_features) x_recon
    latent_mean,            # (n_samples, latent_dim) encoder mean
    latent_log_variance,    # (n_samples, latent_dim) encoder logvar
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kl_weight` | float | 1.0 | KL divergence weight (beta-VAE) |
| `reconstruction_type` | str | "mse" | Reconstruction loss type ("mse" or "bce") |

### Algorithm

ELBO loss:

$$L_{ELBO} = L_{recon} + \beta \cdot D_{KL}(q(z|x) || p(z))$$

Where:

- $L_{recon}$ = reconstruction loss (MSE or NB)
- $D_{KL}$ = KL divergence to prior
- $\beta$ = KL weight (set > 1 for disentanglement)

KL divergence for Gaussian:

$$D_{KL} = -\frac{1}{2}\sum_j (1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

### Training Example

```python
def vae_training_loss(vae, data):
    """Train VAE normalizer."""
    result, _, _ = vae.apply({"counts": data}, {}, None)

    return vae_loss(
        data,
        result["reconstructed"],
        result["mu"],
        result["log_var"],
    )
```

### Beta-VAE Scheduling

For better disentanglement:

```python
# Warmup KL weight
def get_kl_weight(epoch, warmup_epochs=100):
    return min(1.0, epoch / warmup_epochs)

vae_loss = VAELoss(kl_weight=get_kl_weight(current_epoch))
```

## HMMLikelihoodLoss

Negative log-likelihood loss for Hidden Markov Models.

### Overview

HMM likelihood loss uses the forward algorithm to compute sequence log-likelihood, enabling end-to-end HMM training.

### Usage

```python
from flax import nnx
from diffbio.losses.statistical_losses import HMMLikelihoodLoss

# Create loss function (HMM parameters are learnable members)
hmm_loss = HMMLikelihoodLoss(n_states=3, n_emissions=4, rngs=nnx.Rngs(42))

# Compute loss (signature: observations)
loss = hmm_loss(observations)  # (batch, seq_len) integer-encoded sequences
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_states` | int | required | Number of hidden states |
| `n_emissions` | int | required | Number of emission symbols |

The loss owns learnable initial-state, transition, and emission log-probabilities; the forward algorithm runs internally on the supplied observations.

### Algorithm

Returns the mean negative log-likelihood over the batch:

$$L_{HMM} = -\frac{1}{B}\sum_b \log P(O^{(b)} | \theta)$$

Where $P(O | \theta)$ is computed via the forward algorithm with `logsumexp` for stability.

### Training Example

```python
def hmm_training_loss(hmm_loss_fn, observations):
    """Train HMM model with learnable parameters held inside the loss."""
    return hmm_loss_fn(observations)

# Train
grads = nnx.grad(hmm_training_loss)(hmm_loss, train_sequences)
```

## Combining Statistical Losses

### VAE with NB Reconstruction

For scRNA-seq VAE (scVI-style):

```python
def scvi_loss(vae, counts):
    """scVI-style loss with NB reconstruction."""
    result, _, _ = vae.apply({"counts": counts}, {}, None)

    # NB reconstruction loss (signature: counts, mu, theta)
    recon_loss = nb_loss(
        counts,
        result["reconstructed"],
        result["dispersion"],
    )

    # KL divergence
    kl_loss = -0.5 * jnp.mean(
        1 + result["log_var"] - result["mu"]**2 - jnp.exp(result["log_var"])
    )

    return recon_loss + kl_loss
```

### HMM with Emission Learning

```python
def hmm_emission_loss(hmm_loss_fn, observations):
    """HMM with learnable initial/transition/emission log-probabilities."""
    return hmm_loss_fn(observations)
```

## Numerical Stability

All statistical losses use numerically stable implementations:

```python
# Log-space computation
def stable_nb_logprob(y, mu, alpha):
    # Use lgamma for log factorial
    log_prob = (
        jax.scipy.special.gammaln(y + 1/alpha)
        - jax.scipy.special.gammaln(y + 1)
        - jax.scipy.special.gammaln(1/alpha)
        + (1/alpha) * jnp.log(1/(1 + alpha*mu))
        + y * jnp.log(alpha*mu/(1 + alpha*mu) + 1e-10)
    )
    return log_prob
```

## Next Steps

- See [Statistical Operators](../operators/statistical.md) for NB-GLM and HMM
- Explore [Single-Cell Losses](singlecell.md) for batch correction
- Check [Differential Expression Pipeline](../pipelines/differential-expression.md)
