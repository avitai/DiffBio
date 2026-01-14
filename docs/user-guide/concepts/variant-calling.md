# Variant Calling

Variant calling is the process of identifying differences between a sample genome and a reference genome. DiffBio provides differentiable components for building end-to-end trainable variant calling pipelines.

## Background

### What is a Variant?

A genetic variant is a position where the sample differs from the reference:

| Type | Description | Example |
|------|-------------|---------|
| **SNP** | Single nucleotide polymorphism | A → G |
| **Insertion** | Extra bases in sample | AT → AGT |
| **Deletion** | Missing bases in sample | AGT → AT |
| **MNP** | Multiple nucleotide polymorphism | AT → GC |

### Traditional Pipeline

```
Reads → Alignment → Pileup → Statistical Model → Variants
         (BWA)     (samtools)  (GATK/FreeBayes)
```

Each step is optimized independently, limiting overall performance.

### DiffBio Approach

```
Reads → Soft Alignment → Soft Pileup → Neural Classifier → Variants
         (Learnable)     (Learnable)    (Learnable)
```

All components are differentiable, enabling joint optimization.

## The Variant Calling Problem

### Mathematical Formulation

Given:

- Reference sequence $R = r_1, r_2, \ldots, r_n$
- Aligned reads $\{(s_i, p_i, q_i)\}$ where $s_i$ is sequence, $p_i$ is position, $q_i$ is quality

Goal: For each position $j$, determine:

$$
P(G_j | \text{reads}) = P(\text{genotype at position } j | \text{observed reads})
$$

### Genotype Representation

For diploid organisms, genotypes at each position:

- **Homozygous reference**: Both alleles match reference (0/0)
- **Heterozygous**: One reference, one alternate allele (0/1)
- **Homozygous alternate**: Both alleles differ from reference (1/1)

## Differentiable Variant Calling Pipeline

### Step 1: Quality Filtering

Filter low-quality bases before pileup:

```python
from diffbio.operators import DifferentiableQualityFilter, QualityFilterConfig

filter_config = QualityFilterConfig(initial_threshold=20.0)
quality_filter = DifferentiableQualityFilter(filter_config)

# Apply soft quality filtering
# Low-quality positions are down-weighted rather than removed
filtered_data, _, _ = quality_filter.apply(data, {}, None)
```

The threshold is learnable and can adapt during training.

### Step 2: Soft Alignment

If reads aren't pre-aligned, use differentiable alignment:

```python
from diffbio.operators import SmoothSmithWaterman, SmithWatermanConfig

align_config = SmithWatermanConfig(temperature=1.0)
aligner = SmoothSmithWaterman(align_config, scoring_matrix=scoring)

# Soft alignment produces probability distribution over positions
result = aligner.align(read, reference)
```

### Step 3: Pileup Generation

Aggregate reads into position-wise distributions:

```python
from diffbio.operators import DifferentiablePileup, PileupConfig

pileup_config = PileupConfig(
    reference_length=1000,
    use_quality_weights=True
)
pileup_op = DifferentiablePileup(pileup_config)

data = {
    "reads": reads,        # (num_reads, read_length, 4)
    "positions": positions, # (num_reads,)
    "quality": quality,     # (num_reads, read_length)
}

result, _, _ = pileup_op.apply(data, {}, None)
pileup = result['pileup']  # (reference_length, 4)
```

The pileup is a continuous distribution, not discrete counts.

### Step 4: Variant Classification

Use the pileup to predict variant probabilities:

```python
from diffbio.operators.variant import VariantClassifier, VariantClassifierConfig

# Neural network for variant classification
classifier_config = VariantClassifierConfig(
    hidden_dims=[64, 32],
    num_classes=3  # ref/ref, ref/alt, alt/alt
)
classifier = VariantClassifier(classifier_config)

# Predict genotypes from pileup
predictions = classifier(pileup)  # (reference_length, 3)
```

## Loss Functions

### Cross-Entropy Loss

Standard classification loss for variant calling:

```python
import optax

def variant_loss(predictions, true_genotypes):
    return optax.softmax_cross_entropy(predictions, true_genotypes).mean()
```

### Focal Loss

Handle class imbalance (most positions are reference):

```python
def focal_loss(predictions, targets, gamma=2.0):
    probs = jax.nn.softmax(predictions)
    ce = -targets * jnp.log(probs + 1e-8)
    focal_weight = (1 - probs) ** gamma
    return (focal_weight * ce).sum(axis=-1).mean()
```

### F1-Aware Loss

Optimize directly for precision/recall:

```python
from diffbio.losses import F1Loss

f1_loss = F1Loss(beta=1.0)  # F1 score
loss = f1_loss(predictions, targets)
```

## Training Strategy

### Data Preparation

```python
# Load aligned reads (BAM format conceptually)
reads = load_reads(bam_file)

# Load ground truth variants (VCF format conceptually)
true_variants = load_vcf(vcf_file)

# Convert to tensors
train_data = prepare_training_data(reads, true_variants, reference)
```

### Training Loop

```python
import jax
import optax

# Initialize pipeline
pipeline = VariantCallingPipeline(config)

# Optimizer
optimizer = optax.adamw(learning_rate=1e-4)
opt_state = optimizer.init(pipeline.parameters())

@jax.jit
def train_step(params, opt_state, batch):
    def loss_fn(params):
        predictions = pipeline.apply(params, batch['reads'])
        return variant_loss(predictions, batch['variants'])

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Training
for batch in data_loader:
    params, opt_state, loss = train_step(params, opt_state, batch)
```

### Temperature Annealing

Gradually sharpen soft decisions during training:

```python
def get_temperature(epoch, total_epochs):
    # Start at 5.0, anneal to 0.5
    start_temp = 5.0
    end_temp = 0.5
    progress = epoch / total_epochs
    return start_temp * (end_temp / start_temp) ** progress

for epoch in range(total_epochs):
    temp = get_temperature(epoch, total_epochs)
    pipeline.set_temperature(temp)
    # ... train
```

## Evaluation Metrics

### Standard Metrics

```python
def evaluate_variants(predictions, ground_truth):
    # Discretize predictions for evaluation
    called_variants = jnp.argmax(predictions, axis=-1)

    # True positives, false positives, false negatives
    tp = ((called_variants > 0) & (ground_truth > 0)).sum()
    fp = ((called_variants > 0) & (ground_truth == 0)).sum()
    fn = ((called_variants == 0) & (ground_truth > 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {"precision": precision, "recall": recall, "f1": f1}
```

### Stratified Evaluation

Evaluate separately by variant type:

```python
def evaluate_by_type(predictions, ground_truth, variant_types):
    results = {}
    for vtype in ['SNP', 'Insertion', 'Deletion']:
        mask = variant_types == vtype
        results[vtype] = evaluate_variants(
            predictions[mask],
            ground_truth[mask]
        )
    return results
```

## Comparison with Traditional Methods

| Aspect | Traditional | DiffBio |
|--------|-------------|---------|
| Parameter tuning | Manual, grid search | Gradient-based |
| Pipeline coupling | Loose, cascaded errors | Tight, joint optimization |
| Adaptability | Fixed heuristics | Learnable from data |
| Interpretability | High | Moderate |
| Computational cost | Lower | Higher (but GPU-accelerated) |

## Best Practices

1. **Start with pre-trained components**: Initialize from traditional methods
2. **Use temperature annealing**: Smooth → sharp during training
3. **Balance the dataset**: Variants are rare, use focal loss or oversampling
4. **Validate on held-out chromosomes**: Avoid overfitting to specific regions
5. **Post-process for interpretation**: Convert soft outputs to VCF format

## References

1. Poplin, R. et al. (2018). "A universal SNP and small-indel variant caller using deep neural networks." *Nature Biotechnology*, 36(10), 983-987.

2. Luo, R. et al. (2020). "Exploring the limit of using a deep neural network on pileup data for germline variant calling." *Nature Machine Intelligence*, 2(4), 220-227.
