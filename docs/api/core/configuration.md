# Configuration Classes

DiffBio uses dataclasses for operator and pipeline configuration. All configurations inherit from Datarax's `OperatorConfig` base class.

## Operator Configurations

Configuration classes for each operator are documented alongside their respective operators:

| Operator | Config | Documentation |
|----------|--------|---------------|
| Smith-Waterman | `SmithWatermanConfig` | [API Reference](../operators/smith-waterman.md) |
| Quality Filter | `QualityFilterConfig` | [API Reference](../operators/quality-filter.md) |
| Pileup | `PileupConfig` | [API Reference](../operators/pileup.md) |
| CNN Variant Classifier | `CNNVariantClassifierConfig` | [API Reference](../operators/variant.md) |
| CNV Segmentation | `CNVSegmentationConfig` | [API Reference](../operators/variant.md) |
| Soft K-Means | `SoftClusteringConfig` | [API Reference](../operators/singlecell.md) |
| Harmony | `BatchCorrectionConfig` | [API Reference](../operators/singlecell.md) |
| RNA Velocity | `VelocityConfig` | [API Reference](../operators/singlecell.md) |

## Pipeline Configurations

| Pipeline | Config | Documentation |
|----------|--------|---------------|
| Variant Calling | `VariantCallingPipelineConfig` | [API Reference](../pipelines/variant-calling.md) |
| Preprocessing | `PreprocessingPipelineConfig` | [API Reference](../pipelines/preprocessing.md) |
| Differential Expression | `DEPipelineConfig` | [API Reference](../pipelines/differential-expression.md) |

## Training Configuration

The training configuration is documented in the training utilities:

| Config | Documentation |
|--------|---------------|
| `TrainingConfig` | [API Reference](../utils/training.md) |

## Configuration Patterns

### Base Configuration

All operator configs inherit from `OperatorConfig`:

```python
from dataclasses import dataclass
from datarax.core.config import OperatorConfig

@dataclass
class MyOperatorConfig(OperatorConfig):
    # Required base fields
    stochastic: bool = False
    stream_name: str | None = None

    # Custom fields
    my_param: float = 1.0
```

### Nested Configuration

Pipelines can contain nested configs:

```python
@dataclass
class PipelineConfig:
    preprocessing: PreprocessingPipelineConfig
    alignment: AlignmentConfig
    classification: ClassificationConfig
```

### Configuration Validation

Add validation in `__post_init__`:

```python
@dataclass
class ValidatedConfig(OperatorConfig):
    temperature: float = 1.0

    def __post_init__(self):
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
```

### Frozen Configurations

Use frozen dataclasses for immutability:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ImmutableConfig(OperatorConfig):
    """Configuration that cannot be modified after creation."""
    param: float = 1.0
```
