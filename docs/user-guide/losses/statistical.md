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

# Compute loss
loss = nb_loss(
    counts=observed_counts,          # (n_samples, n_genes)
    predicted_mean=model_means,      # (n_samples, n_genes)
    dispersion=gene_dispersions,     # (n_genes,) or scalar
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

    return nb_loss(
        counts=counts,
        predicted_mean=result["predicted_means"],
        dispersion=result["dispersions"],
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
    reconstruction="mse",  # or "nb" for negative binomial
)

# Compute loss
loss = vae_loss(
    reconstructed=decoded_output,    # (n_samples, n_features)
    target=original_input,           # (n_samples, n_features)
    mu=latent_mean,                  # (n_samples, latent_dim)
    log_var=latent_log_variance,     # (n_samples, latent_dim)
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kl_weight` | float | 1.0 | KL divergence weight (beta-VAE) |
| `reconstruction` | str | "mse" | Reconstruction loss type |

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
        reconstructed=result["reconstructed"],
        target=data,
        mu=result["mu"],
        log_var=result["log_var"],
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
from diffbio.losses.statistical_losses import HMMLikelihoodLoss

# Create loss function
hmm_loss = HMMLikelihoodLoss()

# Compute loss
loss = hmm_loss(log_likelihood=forward_log_likelihood)
```

### Parameters

The loss takes no configuration parameters.

### Algorithm

Simply returns negative log-likelihood:

$$L_{HMM} = -\log P(O | \theta)$$

Where $P(O | \theta)$ is computed via the forward algorithm.

### Training Example

```python
def hmm_training_loss(hmm, observations):
    """Train HMM model."""
    data = {"observations": observations}
    result, _, _ = hmm.apply(data, {}, None)

    return hmm_loss(log_likelihood=result["log_likelihood"])

# Train
grads = nnx.grad(hmm_training_loss)(hmm, train_sequences)
```

## Combining Statistical Losses

### VAE with NB Reconstruction

For scRNA-seq VAE (scVI-style):

```python
def scvi_loss(vae, counts):
    """scVI-style loss with NB reconstruction."""
    result, _, _ = vae.apply({"counts": counts}, {}, None)

    # NB reconstruction loss
    recon_loss = nb_loss(
        counts=counts,
        predicted_mean=result["reconstructed"],
        dispersion=result["dispersion"],
    )

    # KL divergence
    kl_loss = -0.5 * jnp.mean(
        1 + result["log_var"] - result["mu"]**2 - jnp.exp(result["log_var"])
    )

    return recon_loss + kl_loss
```

### HMM with Emission Learning

```python
def hmm_emission_loss(hmm, observations, emission_network):
    """HMM with learned emissions."""
    # Get emission probabilities from network
    emission_probs = emission_network(observations)

    # Run HMM forward algorithm
    result, _, _ = hmm.apply(
        {"observations": observations, "emissions": emission_probs},
        {}, None
    )

    return hmm_loss(log_likelihood=result["log_likelihood"])
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
