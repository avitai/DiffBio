# Variant Operators API

Differentiable operators for variant calling and analysis.

## DeepVariantStylePileup

::: diffbio.operators.variant.deepvariant_pileup.DeepVariantStylePileup
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply
        - compute_pileup_image
        - num_channels

## DeepVariantPileupConfig

::: diffbio.operators.variant.deepvariant_pileup.DeepVariantPileupConfig
    options:
      show_root_heading: true
      members: []

## CNNVariantClassifier

::: diffbio.operators.variant.cnn_classifier.CNNVariantClassifier
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## CNNVariantClassifierConfig

::: diffbio.operators.variant.cnn_classifier.CNNVariantClassifierConfig
    options:
      show_root_heading: true
      members: []

## DifferentiableCNVSegmentation

::: diffbio.operators.variant.cnv_segmentation.DifferentiableCNVSegmentation
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## CNVSegmentationConfig

::: diffbio.operators.variant.cnv_segmentation.CNVSegmentationConfig
    options:
      show_root_heading: true
      members: []

## SoftVariantQualityFilter

::: diffbio.operators.variant.quality_recalibration.SoftVariantQualityFilter
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## VariantQualityFilterConfig

::: diffbio.operators.variant.quality_recalibration.VariantQualityFilterConfig
    options:
      show_root_heading: true
      members: []

## Usage Examples

### CNN Variant Classification

```python
from flax import nnx
from diffbio.operators.variant import CNNVariantClassifier, CNNVariantClassifierConfig

config = CNNVariantClassifierConfig(num_classes=3, window_size=21)
classifier = CNNVariantClassifier(config, rngs=nnx.Rngs(42))

data = {"pileup_tensor": pileup}  # (n_positions, window_size, num_channels)
result, _, _ = classifier.apply(data, {}, None)
predictions = result["predictions"]
```

### CNV Segmentation

```python
from diffbio.operators.variant import DifferentiableCNVSegmentation, CNVSegmentationConfig

config = CNVSegmentationConfig(n_states=5, hidden_dim=64)
cnv_seg = DifferentiableCNVSegmentation(config, rngs=nnx.Rngs(42))

data = {"log_ratios": log_ratios, "positions": positions}
result, _, _ = cnv_seg.apply(data, {}, None)
copy_numbers = result["copy_numbers"]
```

### DeepVariant-style Pileup Images

```python
import jax.numpy as jnp
from diffbio.operators.variant import DeepVariantStylePileup, DeepVariantPileupConfig

config = DeepVariantPileupConfig(
    window_size=101,
    max_reads=50,
    include_base_channels=True,
    include_base_quality=True,
    include_mapping_quality=True,
    include_strand=True,
    include_supports_variant=True,
    include_differs_from_ref=True,
)
pileup = DeepVariantStylePileup(config)

# Prepare data
data = {
    "reads": reads,  # (num_reads, read_length, 4)
    "reference": reference,  # (window_size, 4)
    "base_qualities": qualities,  # (num_reads, read_length)
    "mapping_qualities": mapq,  # (num_reads,)
    "strands": strands,  # (num_reads,)
    "positions": positions,  # (num_reads,)
}

result, _, _ = pileup.apply(data, {}, None)
pileup_image = result["pileup_image"]  # (50, 101, 9)
```

### Variant Quality Filtering

```python
from diffbio.operators.variant import SoftVariantQualityFilter, VariantQualityFilterConfig

config = VariantQualityFilterConfig(n_features=10, hidden_dim=32)
qf = SoftVariantQualityFilter(config, rngs=nnx.Rngs(42))

data = {"quality_scores": quality, "context_features": features}
result, _, _ = qf.apply(data, {}, None)
filtered = result["filtered_quality"]
```
