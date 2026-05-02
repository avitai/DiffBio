# Variant Operators

DiffBio provides differentiable operators for variant calling, including CNN-based classification, copy number analysis, and quality recalibration.

<span class="operator-variant">Variant</span> <span class="diff-high">Fully Differentiable</span>

## Overview

Variant operators enable end-to-end optimization of:

- **CNNVariantClassifier**: CNN-based variant classification
- **DifferentiableCNVSegmentation**: Copy number variation segmentation
- **EnhancedCNVSegmentation**: Multi-signal fusion CNV with pyramidal smoothing
- **CellTypeAwareVariantClassifier**: Per-cell-type variant classification
- **SoftVariantQualityFilter**: Base quality score recalibration

## CNNVariantClassifier

CNN-based variant classification from pileup images.

### Quick Start

```python
from flax import nnx
from diffbio.operators.variant import CNNVariantClassifier, CNNVariantClassifierConfig

# Configure CNN classifier
config = CNNVariantClassifierConfig(
    num_classes=3,                       # ref, SNP, indel
    input_height=100,                    # coverage depth
    input_width=221,                     # context window
    num_channels=6,                      # A, C, G, T, quality, strand
    hidden_channels=(64, 128, 256),      # conv filter sizes
    fc_dims=(256, 128),                  # fully connected dims
    dropout_rate=0.1,
)

# Create operator
rngs = nnx.Rngs(42)
classifier = CNNVariantClassifier(config, rngs=rngs)

# Classify variants from pileup
data = {"pileup_tensor": pileup}  # (n_positions, window_size, num_channels)
result, state, metadata = classifier.apply(data, {}, None)

# Get predictions
logits = result["logits"]              # (n_positions, num_classes)
probabilities = result["probabilities"] # Softmax probabilities
predictions = result["predictions"]     # Argmax predictions
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_classes` | int | 3 | Number of variant classes |
| `window_size` | int | 21 | Context window size |
| `num_channels` | int | 6 | Input channels |
| `num_filters` | list | [32, 64, 128] | Filters per conv layer |

### Pileup Tensor Channels

Standard pileup tensor channels:

| Channel | Description |
|---------|-------------|
| 0 | Reference base (one-hot) |
| 1-4 | Read base frequencies (A, C, G, T) |
| 5 | Average base quality |
| 6 | Read depth (normalized) |
| 7 | Mapping quality |

### CNN Architecture

```mermaid
graph LR
    A["Pileup Tensor<br/>(window, channels)"] --> B["[Conv2D + BN + ReLU + Pool]<br/>× 3"]
    B --> C["Flatten"]
    C --> D["Dense"]
    D --> E["Softmax"]
    E --> F["Probabilities"]

    style A fill:#d1fae5,stroke:#059669,color:#064e3b
    style B fill:#e0e7ff,stroke:#4338ca,color:#312e81
    style C fill:#e0e7ff,stroke:#4338ca,color:#312e81
    style D fill:#e0e7ff,stroke:#4338ca,color:#312e81
    style E fill:#e0e7ff,stroke:#4338ca,color:#312e81
    style F fill:#d1fae5,stroke:#059669,color:#064e3b
```

## DifferentiableCNVSegmentation

Copy number variation segmentation using attention-based changepoint detection.

### Quick Start

```python
from diffbio.operators.variant import DifferentiableCNVSegmentation, CNVSegmentationConfig

# Configure CNV segmentation
config = CNVSegmentationConfig(
    max_segments=100,        # Maximum number of segments to detect
    hidden_dim=64,           # Attention hidden dimension
    attention_heads=4,       # Number of attention heads
    temperature=1.0,
)

# Create operator
rngs = nnx.Rngs(42)
cnv_seg = DifferentiableCNVSegmentation(config, rngs=rngs)

# Segment copy number
data = {
    "coverage": coverage,        # (n_positions,) coverage signal
}
result, state, metadata = cnv_seg.apply(data, {}, None)

# Get segmentation
boundary_probs = result["boundary_probs"]            # Soft boundary probabilities
segment_assignments = result["segment_assignments"]  # Soft segment memberships
segment_means = result["segment_means"]              # Mean value per segment
smoothed_coverage = result["smoothed_coverage"]      # Segmented/smoothed signal
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_segments` | int | 100 | Maximum number of segments to detect |
| `hidden_dim` | int | 64 | Attention hidden dimension |
| `attention_heads` | int | 4 | Number of attention heads |
| `temperature` | float | 1.0 | Changepoint detection sharpness |

### CNV Model

The CNV model combines:

1. **Emission model**: P(log_ratio | copy_number)
2. **Transition model**: Penalizes state changes (segments)
3. **Soft segmentation**: Temperature-controlled changepoints

```python
# Soft changepoint detection
change_scores = neural_network(log_ratios)
changepoint_probs = jax.nn.sigmoid(change_scores / temperature)

# Segment-aware state prediction
state_logits = emission_network(log_ratios)
state_probs = segment_aware_softmax(state_logits, changepoint_probs)
```

## SoftVariantQualityFilter

VQSR-style variant quality recalibration using a differentiable Gaussian
Mixture Model. Variants are scored by their likelihood under the GMM, and
soft sigmoid thresholds maintain gradient flow through the filtering step.

### Quick Start

```python
from diffbio.operators.variant import SoftVariantQualityFilter, VariantQualityFilterConfig

# Configure GMM-based quality filter
config = VariantQualityFilterConfig(
    n_components=3,          # Number of GMM components
    n_features=4,            # depth, qual, strand_bias, mapq
    threshold=0.5,           # Quality threshold for soft filtering
    temperature=1.0,
)

# Create operator
rngs = nnx.Rngs(42)
quality_filter = SoftVariantQualityFilter(config, rngs=rngs)

# Score and filter variants
data = {
    "variant_features": features,        # (n_variants, n_features)
}
result, state, metadata = quality_filter.apply(data, {}, None)

# Get scored / soft-filtered variants
quality_scores = result["quality_scores"]      # GMM-derived quality in [0, 1]
filter_weights = result["filter_weights"]      # Soft pass/fail in [0, 1]
component_probs = result["component_probs"]    # GMM responsibilities
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_components` | int | 3 | Number of GMM components |
| `n_features` | int | 4 | Number of variant features |
| `threshold` | float | 0.5 | Quality threshold for soft filtering |
| `temperature` | float | 1.0 | Softmax/sigmoid sharpness |

### Variant Features

Typical features supplied to the filter:

| Feature | Description |
|---------|-------------|
| Depth | Total read depth at the variant site |
| Qual | Reported variant quality score |
| Strand bias | Imbalance between forward/reverse supporting reads |
| MAPQ | Mean mapping quality of supporting reads |

### Filtering Model

```python
# GMM-based scoring with soft sigmoid threshold
component_log_probs = quality_filter.compute_component_log_probs(features)
quality_scores = quality_filter.compute_quality_scores(features)
filter_weights = jax.nn.sigmoid((quality_scores - threshold) / temperature)
```

## EnhancedCNVSegmentation

Extended CNV segmentation with multi-signal fusion, pyramidal smoothing (infercnvpy-style), dynamic thresholding, and HMM copy-number state mapping. Extends `DifferentiableCNVSegmentation` with additional capabilities for production-quality CNV analysis.

### Quick Start

```python
from diffbio.operators.variant import EnhancedCNVSegmentation, EnhancedCNVSegmentationConfig

config = EnhancedCNVSegmentationConfig(
    max_segments=50,
    use_baf=True,
    smoothing_window=100,
    threshold_scale=1.5,
    n_copy_states=5,
)

rngs = nnx.Rngs(0)
enhanced_cnv = EnhancedCNVSegmentation(config, rngs=rngs)

data = {
    "coverage": log_ratios,      # (n_positions,)
    "baf_signal": baf,           # (n_positions,) B-allele frequency
    "snp_density": snp_density,  # (n_positions,)
}
result, state, metadata = enhanced_cnv.apply(data, {}, None)

copy_states = result["copy_number_posteriors"]   # (n_positions, n_copy_states)
smoothed = result["smoothed_coverage"]           # (n_positions,)
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_segments` | int | 100 | Maximum segments to detect |
| `hidden_dim` | int | 64 | Attention layer hidden dimension |
| `attention_heads` | int | 4 | Number of attention heads |
| `temperature` | float | 1.0 | Softmax temperature |
| `use_baf` | bool | False | Incorporate B-allele frequency |
| `baf_weight` | float | 0.3 | Initial scaling applied to BAF before learnable fusion |
| `smoothing_window` | int | 100 | Pyramidal smoothing window size |
| `threshold_scale` | float | 1.5 | STDDEV-based dynamic threshold multiplier |
| `n_copy_states` | int | 5 | Discrete copy-number states (0-somy to 4-somy) |

### Enhanced Features

1. **Multi-signal fusion**: Learnable linear combination of log-ratio coverage, BAF, and SNP density
2. **Pyramidal smoothing**: Triangular convolution kernel for spatial noise reduction
3. **Dynamic thresholding**: `threshold_scale * std(smoothed)` filters low-amplitude noise
4. **HMM state mapping**: Soft copy-number posteriors via learned emission model

## CellTypeAwareVariantClassifier

Cell-type-aware variant classifier using separate classification heads per cell type, weighted by soft cell-type assignment probabilities. Enables cell-type-specific variant calling thresholds for heterogeneous populations such as single-cell sequencing data.

### Quick Start

```python
from diffbio.operators.variant import (
    CellTypeAwareVariantClassifier,
    CellTypeAwareVariantClassifierConfig,
)

config = CellTypeAwareVariantClassifierConfig(
    n_classes=3,
    hidden_dim=64,
    n_cell_types=5,
    pileup_channels=6,
    pileup_width=100,
)

rngs = nnx.Rngs(42)
classifier = CellTypeAwareVariantClassifier(config, rngs=rngs)

data = {
    "pileup": pileup_batch,                   # (n, channels, width)
    "cell_type_assignments": assignments,      # (n, n_cell_types)
}
result, state, metadata = classifier.apply(data, {}, None)

probs = result["variant_probabilities"]        # (n, n_classes)
per_type = result["per_type_probabilities"]    # (n, n_cell_types, n_classes)
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_classes` | int | 3 | Variant classes (e.g., ref, SNP, indel) |
| `hidden_dim` | int | 64 | Shared encoder hidden dimension |
| `n_cell_types` | int | 5 | Number of cell types |
| `pileup_channels` | int | 6 | Pileup input channels |
| `pileup_width` | int | 100 | Pileup input width |

### Architecture

1. Shared feature encoder: pileup -> flatten -> Linear -> ReLU
2. Per-type classification heads: `n_cell_types` separate Linear layers
3. Each head produces type-specific variant probabilities via softmax
4. Final aggregation: $\sum_t w_t \cdot \text{head}_t(\text{features})$ weighted by cell-type assignments

## Training Variant Models

### CNN Classifier Training

```python
from flax import nnx
import optax

def variant_loss(classifier, pileups, labels):
    """Cross-entropy loss for variant classification."""
    data = {"pileup_tensor": pileups}
    result, _, _ = classifier.apply(data, {}, None)

    # Softmax cross-entropy
    loss = optax.softmax_cross_entropy_with_integer_labels(
        result["logits"], labels
    ).mean()
    return loss

# Train
grads = nnx.grad(variant_loss)(classifier, train_pileups, train_labels)
```

### CNV Training with Known Segments

```python
def cnv_loss(cnv_model, log_ratios, true_segments):
    """Train CNV segmentation."""
    data = {"log_ratios": log_ratios, "positions": positions}
    result, _, _ = cnv_model.apply(data, {}, None)

    # Segment matching loss
    seg_loss = jnp.mean((result["copy_numbers"] - true_segments) ** 2)

    # Sparsity penalty on changepoints
    sparse_loss = jnp.mean(result["changepoints"])

    return seg_loss + 0.1 * sparse_loss
```

## Use Cases

| Application | Operator | Description |
|-------------|----------|-------------|
| SNP/indel calling | CNNVariantClassifier | Classify variant types |
| Somatic variants | CNNVariantClassifier | Cancer variant detection |
| Copy number | DifferentiableCNVSegmentation | Detect CNV regions |
| Tumor purity | DifferentiableCNVSegmentation | Estimate tumor fraction |
| Quality control | SoftVariantQualityFilter | Improve base qualities |
| Enhanced CNV | EnhancedCNVSegmentation | Multi-signal CNV with pyramidal smoothing |
| Cell-type variants | CellTypeAwareVariantClassifier | Per-cell-type variant calling |

## Integration with Pipelines

```python
from diffbio.pipelines import VariantCallingPipeline

# Full variant calling pipeline
pipeline = VariantCallingPipeline(config, rngs=rngs)

# Pipeline includes:
# 1. Quality filtering
# 2. Pileup generation
# 3. Variant classification (uses CNNVariantClassifier internally)

result, _, _ = pipeline.apply(read_data, {}, None)
```

## Next Steps

- See [Variant Calling Pipeline](../pipelines/variant-calling.md) for end-to-end workflow
- Explore [Pileup Operator](pileup.md) for pileup generation
- Check [Quality Filter](quality-filter.md) for preprocessing
