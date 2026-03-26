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

@dataclass(frozen=True)
class MyOperatorConfig(OperatorConfig):
    """Configuration for MyOperator."""
    my_param: float = 1.0
```

### Nested Configuration

Pipelines can contain nested configs:

```python
@dataclass(frozen=True)
class PipelineConfig:
    preprocessing: PreprocessingPipelineConfig
    alignment: AlignmentConfig
    classification: ClassificationConfig
```

### Configuration Validation

Add validation in `__post_init__`:

```python
@dataclass(frozen=True)
class ValidatedConfig(OperatorConfig):
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
```

### Stochastic Operators

If an operator needs stochastic behavior, set `stochastic` and `stream_name` in
`__post_init__` using `object.__setattr__()` (required because the dataclass is frozen):

```python
@dataclass(frozen=True)
class StochasticOpConfig(OperatorConfig):
    """Configuration for a stochastic operator."""
    dropout_rate: float = 0.1

    def __post_init__(self) -> None:
        object.__setattr__(self, "stochastic", True)
        object.__setattr__(self, "stream_name", "dropout")
```
