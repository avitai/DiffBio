# Variant Operators API

Differentiable operators for variant calling and analysis.

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

### Variant Quality Filtering

```python
from diffbio.operators.variant import SoftVariantQualityFilter, VariantQualityFilterConfig

config = VariantQualityFilterConfig(n_features=10, hidden_dim=32)
qf = SoftVariantQualityFilter(config, rngs=nnx.Rngs(42))

data = {"quality_scores": quality, "context_features": features}
result, _, _ = qf.apply(data, {}, None)
filtered = result["filtered_quality"]
```
