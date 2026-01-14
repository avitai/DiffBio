# RNA-seq Operators API

Differentiable operators for RNA-seq analysis including splicing PSI and motif discovery.

## SplicingPSI

::: diffbio.operators.rnaseq.splicing_psi.SplicingPSI
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## SplicingPSIConfig

::: diffbio.operators.rnaseq.splicing_psi.SplicingPSIConfig
    options:
      show_root_heading: true
      members: []

## DifferentiableMotifDiscovery

::: diffbio.operators.rnaseq.motif_discovery.DifferentiableMotifDiscovery
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## MotifDiscoveryConfig

::: diffbio.operators.rnaseq.motif_discovery.MotifDiscoveryConfig
    options:
      show_root_heading: true
      members: []

## Usage Examples

### Splicing PSI Calculation

```python
from flax import nnx
from diffbio.operators.rnaseq import SplicingPSI, SplicingPSIConfig

config = SplicingPSIConfig(temperature=1.0, num_exons=3)
psi_calc = SplicingPSI(config, rngs=nnx.Rngs(42))

data = {
    "inclusion_counts": inclusion,
    "exclusion_counts": exclusion,
}
result, _, _ = psi_calc.apply(data, {}, None)
psi_values = result["psi"]
```

### Motif Discovery

```python
from diffbio.operators.rnaseq import DifferentiableMotifDiscovery, MotifDiscoveryConfig

config = MotifDiscoveryConfig(num_motifs=10, motif_length=8)
motif_finder = DifferentiableMotifDiscovery(config, rngs=nnx.Rngs(42))

data = {"sequences": sequences}  # (n_seqs, seq_len, alphabet_size)
result, _, _ = motif_finder.apply(data, {}, None)
pwms = result["pwms"]
```
