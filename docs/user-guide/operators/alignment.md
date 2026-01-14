# Alignment Operators

DiffBio provides advanced differentiable alignment operators for multiple sequence alignment and profile-based homology detection.

<span class="operator-alignment">Alignment</span> <span class="diff-high">Fully Differentiable</span>

## Overview

Beyond pairwise alignment (Smith-Waterman), DiffBio implements:

- **SoftProgressiveMSA**: Multiple sequence alignment with soft guide tree
- **ProfileHMM**: Profile Hidden Markov Models for sequence family detection

## SoftProgressiveMSA

Differentiable progressive multiple sequence alignment using neural sequence encoders.

### Quick Start

```python
from flax import nnx
from diffbio.operators.alignment import SoftProgressiveMSA, SoftProgressiveMSAConfig

# Configure MSA operator
config = SoftProgressiveMSAConfig(
    max_seq_length=100,
    hidden_dim=64,
    num_layers=2,
    alphabet_size=4,  # DNA
    temperature=1.0,
    gap_open_penalty=-10.0,
    gap_extend_penalty=-1.0,
)

# Create operator
rngs = nnx.Rngs(42)
msa = SoftProgressiveMSA(config, rngs=rngs)

# Prepare sequences (n_seqs, seq_len, alphabet_size)
sequences = jnp.stack([seq1, seq2, seq3])

# Perform MSA
data = {"sequences": sequences}
result, state, metadata = msa.apply(data, {}, None)

# Access results
aligned = result["aligned_sequences"]      # Aligned sequences
distances = result["pairwise_distances"]   # Guide tree distances
consensus = result["consensus_profile"]    # Consensus profile
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_seq_length` | int | 100 | Maximum sequence length |
| `hidden_dim` | int | 64 | Hidden dimension for neural encoder |
| `num_layers` | int | 2 | Number of encoder layers |
| `alphabet_size` | int | 4 | Size of sequence alphabet |
| `temperature` | float | 1.0 | Temperature for softmax operations |
| `gap_open_penalty` | float | -10.0 | Gap opening penalty |
| `gap_extend_penalty` | float | -1.0 | Gap extension penalty |

### Algorithm

1. **Sequence Encoding**: Neural network encodes each sequence to a fixed-size embedding
2. **Distance Computation**: Pairwise distances computed via cosine similarity
3. **Progressive Alignment**: Sequences aligned progressively using soft attention
4. **Consensus Building**: Weighted profile built from aligned sequences

## ProfileHMM

Profile Hidden Markov Model for detecting sequence homology to a family profile.

### Quick Start

```python
from diffbio.operators.alignment import ProfileHMM, ProfileHMMConfig

# Configure Profile HMM
config = ProfileHMMConfig(
    profile_length=50,
    alphabet_size=4,
    num_match_states=50,
    temperature=1.0,
)

# Create operator
rngs = nnx.Rngs(42)
profile_hmm = ProfileHMM(config, rngs=rngs)

# Score sequence against profile
data = {"sequence": sequence}  # (seq_len, alphabet_size)
result, state, metadata = profile_hmm.apply(data, {}, None)

# Get homology score
score = result["log_likelihood"]
viterbi_path = result["viterbi_path"]
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `profile_length` | int | 50 | Length of the profile (number of match states) |
| `alphabet_size` | int | 4 | Size of sequence alphabet |
| `num_match_states` | int | 50 | Number of match states |
| `temperature` | float | 1.0 | Temperature for soft operations |

### HMM States

The Profile HMM has three state types:

- **Match (M)**: Position emits from learned distribution
- **Insert (I)**: Handles insertions relative to profile
- **Delete (D)**: Handles deletions (silent states)

## Differentiability Techniques

### Soft Guide Tree

Instead of discrete UPGMA clustering, SoftProgressiveMSA uses:

- Neural embeddings for sequence similarity
- Soft attention for pairwise alignment
- Differentiable weighted consensus

### Soft Viterbi

ProfileHMM uses temperature-scaled logsumexp for soft Viterbi:

```
soft_max(x) = temperature * logsumexp(x / temperature)
```

This maintains gradient flow through the decoding process.

## Use Cases

| Application | Operator | Description |
|-------------|----------|-------------|
| Protein family detection | ProfileHMM | Score query against family profile |
| Homology search | ProfileHMM | Find similar sequences |
| Multiple alignment | SoftProgressiveMSA | Align related sequences |
| Phylogenetic analysis | SoftProgressiveMSA | Build evolutionary relationships |
| Consensus building | SoftProgressiveMSA | Extract conserved features |

## Next Steps

- See the [Smith-Waterman](smith-waterman.md) operator for pairwise alignment
- Explore [Statistical Operators](statistical.md) for more HMM applications
