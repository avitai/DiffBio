# Variant Operators

DiffBio provides differentiable operators for variant calling, including CNN-based classification, copy number analysis, and quality recalibration.

<span class="operator-variant">Variant</span> <span class="diff-high">Fully Differentiable</span>

## Overview

Variant operators enable end-to-end optimization of:

- **CNNVariantClassifier**: CNN-based variant classification
- **CNVSegmentation**: Copy number variation segmentation
- **EnhancedCNVSegmentation**: Multi-signal fusion CNV with pyramidal smoothing
- **CellTypeAwareVariantClassifier**: Per-cell-type variant classification
- **QualityRecalibration**: Base quality score recalibration

## CNNVariantClassifier

CNN-based variant classification from pileup images.

### Quick Start

```python
from flax import nnx
from diffbio.operators.variant import CNNVariantClassifier, CNNVariantConfig

# Configure CNN classifier
config = CNNVariantConfig(
    num_classes=3,           # ref, SNP, indel
    window_size=21,          # Pileup window
    num_channels=6,          # Read depth, quality, etc.
    num_filters=[32, 64, 128],
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

## CNVSegmentation

Copy number variation segmentation using changepoint detection.

### Quick Start

```python
from diffbio.operators.variant import CNVSegmentation, CNVConfig

# Configure CNV segmentation
config = CNVConfig(
    n_states=5,              # Ploidy states (0, 1, 2, 3, 4+)
    hidden_dim=64,
    temperature=1.0,
)

# Create operator
rngs = nnx.Rngs(42)
cnv_seg = CNVSegmentation(config, rngs=rngs)

# Segment copy number
data = {
    "log_ratios": log_ratios,    # (n_bins,) or (n_samples, n_bins)
    "positions": bin_positions,  # (n_bins,) genomic positions
}
result, state, metadata = cnv_seg.apply(data, {}, None)

# Get segmentation
segments = result["segments"]              # Segment assignments
copy_numbers = result["copy_numbers"]      # Inferred CN states
changepoints = result["changepoints"]      # Changepoint probabilities
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_states` | int | 5 | Number of copy number states |
| `hidden_dim` | int | 64 | Network hidden dimension |
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

## QualityRecalibration

Base quality score recalibration using machine learning.

### Quick Start

```python
from diffbio.operators.variant import QualityRecalibration, QualityRecalConfig

# Configure recalibration
config = QualityRecalConfig(
    n_features=10,           # Context features
    hidden_dim=32,
    output_quality_bins=50,
)

# Create operator
rngs = nnx.Rngs(42)
recal = QualityRecalibration(config, rngs=rngs)

# Recalibrate quality scores
data = {
    "quality_scores": original_quality,  # (n_bases,)
    "context_features": features,        # (n_bases, n_features)
}
result, state, metadata = recal.apply(data, {}, None)

# Get recalibrated scores
recalibrated = result["recalibrated_quality"]
adjustment = result["quality_adjustment"]
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_features` | int | 10 | Number of context features |
| `hidden_dim` | int | 32 | Network hidden dimension |
| `output_quality_bins` | int | 50 | Quality score range |

### Context Features

Features used for recalibration:

| Feature | Description |
|---------|-------------|
| Original quality | Reported quality score |
| Read position | Position within read |
| Cycle | Sequencing cycle |
| Context | Dinucleotide context |
| Read group | Library/lane information |

### Recalibration Model

```python
# Neural quality recalibration
predicted_error_rate = recal_network(context_features)
recalibrated_quality = -10 * jnp.log10(predicted_error_rate + 1e-10)
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
smoothed = result["smoothed_signal"]             # (n_positions,)
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_segments` | int | 100 | Maximum segments to detect |
| `hidden_dim` | int | 64 | Attention layer hidden dimension |
| `attention_heads` | int | 4 | Number of attention heads |
| `temperature` | float | 1.0 | Softmax temperature |
| `use_baf` | bool | False | Incorporate B-allele frequency |
| `baf_weight` | float | 0.3 | Initial BAF signal fusion weight |
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
| Copy number | CNVSegmentation | Detect CNV regions |
| Tumor purity | CNVSegmentation | Estimate tumor fraction |
| Quality control | QualityRecalibration | Improve base qualities |
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
