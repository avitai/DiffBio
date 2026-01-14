# Statistical Losses API

Loss functions for statistical modeling in bioinformatics.

## NegativeBinomialLoss

::: diffbio.losses.statistical_losses.NegativeBinomialLoss
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - __call__

## VAELoss

::: diffbio.losses.statistical_losses.VAELoss
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - __call__

## HMMLikelihoodLoss

::: diffbio.losses.statistical_losses.HMMLikelihoodLoss
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - __call__

## Usage Examples

### Negative Binomial Loss

```python
from diffbio.losses import NegativeBinomialLoss

nb_loss = NegativeBinomialLoss()

# For count data (RNA-seq, scRNA-seq)
loss = nb_loss(
    counts=observed_counts,        # (n_samples, n_genes)
    predicted_mean=model_means,    # (n_samples, n_genes)
    dispersion=dispersions,        # (n_genes,) or scalar
)
```

### VAE Loss (ELBO)

```python
from diffbio.losses import VAELoss

vae_loss = VAELoss(kl_weight=1.0)

# Evidence lower bound
loss = vae_loss(
    x=input_data,
    x_reconstructed=decoded,
    z_mean=latent_mean,
    z_logvar=latent_logvar,
)
```

### HMM Likelihood Loss

```python
from diffbio.losses import HMMLikelihoodLoss

hmm_loss = HMMLikelihoodLoss()

# Negative log-likelihood for HMM training
loss = hmm_loss(
    observations=observed_sequence,
    log_initial=log_initial_probs,
    log_transition=log_transition_matrix,
    log_emission=log_emission_probs,
)
```

## Mathematical Details

### Negative Binomial

$$\log P(k | \mu, \alpha) = \log \Gamma(k + \alpha^{-1}) - \log \Gamma(\alpha^{-1}) - \log k! + \alpha^{-1} \log(\alpha^{-1}) - \alpha^{-1} \log(\mu + \alpha^{-1}) + k \log(\mu) - k \log(\mu + \alpha^{-1})$$

### VAE ELBO

$$\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \beta \cdot D_{KL}(q(z|x) || p(z))$$

### HMM Forward Algorithm

$$\alpha_t(j) = p(o_1, ..., o_t, s_t = j | \lambda)$$

Computed in log-space using logsumexp for numerical stability.
