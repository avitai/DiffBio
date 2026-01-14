# Alignment Operators API

Advanced differentiable alignment operators for multiple sequence alignment and profile HMMs.

## SoftProgressiveMSA

::: diffbio.operators.alignment.soft_msa.SoftProgressiveMSA
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## SoftProgressiveMSAConfig

::: diffbio.operators.alignment.soft_msa.SoftProgressiveMSAConfig
    options:
      show_root_heading: true
      members: []

## ProfileHMMSearch

::: diffbio.operators.alignment.profile_hmm.ProfileHMMSearch
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## ProfileHMMConfig

::: diffbio.operators.alignment.profile_hmm.ProfileHMMConfig
    options:
      show_root_heading: true
      members: []

## Usage Examples

### Multiple Sequence Alignment

```python
from flax import nnx
from diffbio.operators.alignment import SoftProgressiveMSA, SoftProgressiveMSAConfig

config = SoftProgressiveMSAConfig(
    max_seq_length=100,
    hidden_dim=64,
    alphabet_size=4,
)
msa = SoftProgressiveMSA(config, rngs=nnx.Rngs(42))

data = {"sequences": sequences}  # (n_seqs, seq_len, alphabet_size)
result, _, _ = msa.apply(data, {}, None)
```

### Profile HMM Scoring

```python
from diffbio.operators.alignment import ProfileHMMSearch, ProfileHMMConfig

config = ProfileHMMConfig(profile_length=50, alphabet_size=4)
hmm = ProfileHMMSearch(config, rngs=nnx.Rngs(42))

data = {"sequence": sequence}
result, _, _ = hmm.apply(data, {}, None)
score = result["log_likelihood"]
```
