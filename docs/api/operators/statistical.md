# Statistical Operators API

Differentiable statistical operators for probabilistic modeling including HMMs, GLMs, and EM algorithms.

## DifferentiableHMM

::: diffbio.operators.statistical.hmm.DifferentiableHMM
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## HMMConfig

::: diffbio.operators.statistical.hmm.HMMConfig
    options:
      show_root_heading: true
      members: []

## DifferentiableNBGLM

::: diffbio.operators.statistical.nb_glm.DifferentiableNBGLM
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## NBGLMConfig

::: diffbio.operators.statistical.nb_glm.NBGLMConfig
    options:
      show_root_heading: true
      members: []

## DifferentiableEMQuantifier

::: diffbio.operators.statistical.em_quantification.DifferentiableEMQuantifier
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## EMQuantifierConfig

::: diffbio.operators.statistical.em_quantification.EMQuantifierConfig
    options:
      show_root_heading: true
      members: []

## Usage Examples

### Hidden Markov Model

```python
from flax import nnx
from diffbio.operators.statistical import DifferentiableHMM, HMMConfig

config = HMMConfig(num_states=5, num_emissions=4)
hmm = DifferentiableHMM(config, rngs=nnx.Rngs(42))

data = {"observations": observations}  # (seq_length, n_observations)
result, _, _ = hmm.apply(data, {}, None)
log_likelihood = result["log_likelihood"]
posteriors = result["posteriors"]
```

### Negative Binomial GLM

```python
from diffbio.operators.statistical import DifferentiableNBGLM, NBGLMConfig

config = NBGLMConfig(n_features=2000, n_covariates=10)
nbglm = DifferentiableNBGLM(config, rngs=nnx.Rngs(42))

data = {
    "counts": count_matrix,
    "design": design_matrix,
    "size_factors": size_factors,
}
result, _, _ = nbglm.apply(data, {}, None)
coefficients = result["coefficients"]
```

### EM Quantifier

```python
from diffbio.operators.statistical import DifferentiableEMQuantifier, EMQuantifierConfig

config = EMQuantifierConfig(n_transcripts=50000, n_iterations=100)
em_quant = DifferentiableEMQuantifier(config, rngs=nnx.Rngs(42))

data = {
    "equivalence_classes": eq_classes,
    "counts": eq_counts,
    "effective_lengths": eff_lengths,
}
result, _, _ = em_quant.apply(data, {}, None)
tpm = result["tpm"]
```
