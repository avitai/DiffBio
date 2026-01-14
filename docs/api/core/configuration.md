# Configuration Classes

DiffBio uses dataclasses for operator and pipeline configuration.

## Operator Configurations

### SmithWatermanConfig

Configuration for the differentiable Smith-Waterman alignment operator.

::: diffbio.operators.alignment.smith_waterman.SmithWatermanConfig
    options:
      show_root_heading: true
      members: []

**Example:**

```python
from diffbio.operators.alignment import SmithWatermanConfig

config = SmithWatermanConfig(
    temperature=1.0,    # Logsumexp smoothness
    gap_open=-10.0,     # Gap opening penalty
    gap_extend=-1.0,    # Gap extension penalty
)
```

---

### QualityFilterConfig

Configuration for the differentiable quality filter.

::: diffbio.operators.quality_filter.QualityFilterConfig
    options:
      show_root_heading: true
      members: []

**Example:**

```python
from diffbio.operators import QualityFilterConfig

config = QualityFilterConfig(
    initial_threshold=20.0,  # Phred quality threshold
)
```

---

### PileupConfig

Configuration for the differentiable pileup generator.

::: diffbio.operators.variant.pileup.PileupConfig
    options:
      show_root_heading: true
      members: []

**Example:**

```python
from diffbio.operators.variant import PileupConfig

config = PileupConfig(
    reference_length=1000,
    window_size=21,
    use_quality_weights=True,
)
```

## Pipeline Configurations

### VariantCallingPipelineConfig

Configuration for the end-to-end variant calling pipeline.

::: diffbio.pipelines.variant_calling.VariantCallingPipelineConfig
    options:
      show_root_heading: true
      members: []

**Example:**

```python
from diffbio.pipelines import VariantCallingPipelineConfig

config = VariantCallingPipelineConfig(
    reference_length=10000,
    num_classes=3,
    quality_threshold=20.0,
    pileup_window_size=11,
    classifier_hidden_dim=128,
)
```

## Training Configuration

### TrainingConfig

Configuration for the training loop.

::: diffbio.utils.training.TrainingConfig
    options:
      show_root_heading: true
      members: []

**Example:**

```python
from diffbio.utils.training import TrainingConfig

config = TrainingConfig(
    learning_rate=1e-3,
    num_epochs=100,
    log_every=10,
    grad_clip_norm=1.0,
)
```

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
    preprocessing: PreprocessingConfig
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
