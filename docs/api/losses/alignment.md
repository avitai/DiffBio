# Alignment Losses API

Loss functions for sequence alignment optimization.

## AlignmentScoreLoss

::: diffbio.losses.alignment_losses.AlignmentScoreLoss
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - __call__

## SoftEditDistanceLoss

::: diffbio.losses.alignment_losses.SoftEditDistanceLoss
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - __call__

## AlignmentConsistencyLoss

::: diffbio.losses.alignment_losses.AlignmentConsistencyLoss
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - __call__

## Usage Example

```python
from diffbio.losses import AlignmentScoreLoss, SoftEditDistanceLoss

# Alignment score loss
alignment_loss = AlignmentScoreLoss()
loss = alignment_loss(
    seq1=sequence1,
    seq2=sequence2,
    alignment=alignment_matrix,
)

# Soft edit distance
edit_loss = SoftEditDistanceLoss(normalize=True, temperature=0.1)
distance = edit_loss(seq1=sequence1, seq2=sequence2)
```
