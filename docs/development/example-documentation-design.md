# DiffBio Example Documentation Design Guide

This guide defines how to design, write, maintain, and review DiffBio examples.
It is specific to DiffBio's current runtime and contributor contract:

- reader-facing tutorial examples are dual-format `.py` and `.ipynb` pairs
- verification and maintenance utilities under `examples/` may remain `.py` only
- repository workflows use `uv`
- examples target JAX and Flax NNX on the datarax pipeline framework
- operators follow the `OperatorModule.apply()` contract from datarax
- backend behavior follows JAX defaults instead of custom CUDA forcing
- contributor-facing example docs live under `docs/examples/`

## Purpose

DiffBio examples must do three jobs at once:

1. Teach a concrete bioinformatics concept through differentiable computation.
2. Demonstrate the current supported DiffBio API and ecosystem integration.
3. Run successfully as real code, not documentation theater.

An example is successful only when all three are true.

## Scope

This guide applies to:

- runnable example sources under `examples/`
- paired example notebooks under `examples/`
- example documentation pages under `docs/examples/`
- example templates under `docs/examples/templates/`

It also informs contributor-facing documentation in:

- `CONTRIBUTING.md`
- `docs/community/contributing.md`
- `examples/README.md`

## Design Principles

### Teach Through Real Execution

Examples should run end to end with the commands shown in the docs. Prefer small
working pipelines over aspirational pseudo-workflows. Every code block should be
copy-pasteable and produce the shown output.

### Prefer Present-Tense API Guidance

Document the current supported DiffBio surface. Avoid transitional language and
avoid teaching historical implementation details.

### Progressive Disclosure

Each example should start with the smallest useful path, then add complexity in
clear stages:

1. environment and prerequisites
2. imports and configuration
3. data creation or loading (synthetic or via `AnnDataSource`)
4. operator construction and `apply()`
5. inspection of outputs (shapes, values, plots)
6. gradient flow verification (differentiability is DiffBio's value proposition)
7. optional extensions or experiments

### Differentiability First

Every operator example must include a gradient flow demonstration. This is
DiffBio's core value proposition — showing that bioinformatics pipelines are
end-to-end differentiable. At minimum, show:

```python
# Verify gradient flow
def loss_fn(data):
    result, _, _ = operator.apply(data, {}, None, None)
    return result["output_key"].sum()

grad = jax.grad(loss_fn)(data)
print(f"Gradient shape: {grad['counts'].shape}")
print(f"Gradient is non-zero: {jnp.any(grad['counts'] != 0)}")
```

### JIT Compilation Demo

Every operator example should show JIT compilation works:

```python
# JIT-compiled forward pass
@jax.jit
def jit_apply(data):
    result, _, _ = operator.apply(data, {}, None, None)
    return result

result = jit_apply(data)
```

### CPU-Safe by Default

Examples should run on CPU unless the example is inherently GPU-bound. GPU use is
an optimization path, not an excuse for unclear setup.

### GPU Requirements Must Be Explicit

If an example requires GPU, say so at the top and provide a direct verification
step with:

```bash
source ./activate.sh
```

If GPU is optional, say that clearly and avoid device forcing logic inside the
example.

### Source of Truth Lives in the Example Pair

For dual-format tutorial examples, the runnable `.py` and `.ipynb` pair is the
technical source of truth. The corresponding `docs/examples/...` page explains
the example, but should not drift from the actual source files.

## Documentation Architecture

Every substantial reader-facing tutorial example should have three aligned artifacts:

1. `examples/.../example_name.py`
   The runnable Jupytext-backed Python source.
2. `examples/.../example_name.ipynb`
   The paired notebook generated from the Python source.
3. `docs/examples/.../example-name.md`
   The reader-facing documentation page.

Use this split intentionally:

- `.py` is the easiest source to review and refactor.
- `.ipynb` supports notebook-first exploration.
- `.md` explains context, expected outcomes, and navigation.

Utility scripts under `examples/` are different. If the file exists to verify,
benchmark, or maintain the example surface rather than teach a workflow, it can
stay as a Python-only script with no paired notebook.

## Location Strategy

Put examples where users will expect them, organized by DiffBio's domain
structure:

- `examples/basics/` for foundational patterns (configs, operators, data flow)
- `examples/singlecell/` for single-cell analysis workflows
- `examples/alignment/` for sequence alignment examples
- `examples/variant/` for variant calling examples
- `examples/drug_discovery/` for molecular property prediction
- `examples/multiomics/` for multi-omics integration
- `examples/epigenomics/` for chromatin and peak analysis
- `examples/pipelines/` for end-to-end pipeline composition
- `examples/ecosystem/` for datarax/artifex/calibrax/opifex integration demos

Put documentation under the corresponding `docs/examples/` section so the docs
match the runnable source domain.

## Dual-Format Workflow

Reader-facing tutorial examples should be maintained as Jupytext pairs. Use the
repo tool, not ad hoc manual notebook edits.

### Create a New Example

Start from:

- `docs/examples/templates/example_template.py`
- `docs/examples/templates/example_template.ipynb`

### Sync the Pair

Use:

```bash
source ./activate.sh
uv run jupytext --sync examples/path/to/example.py
```

### Validate the Pair

Use:

```bash
source ./activate.sh
uv run python examples/path/to/example.py
```

The Python file should remain the main review surface. The notebook should be
regenerated from it rather than hand-edited independently.

Do not create notebook pairs for verification or maintenance scripts such as
`examples/verify_examples.py`.

## Runtime and Backend Contract

### Activation

Always show:

```bash
source ./activate.sh
```

Keep the repository-relative `./` prefix when documenting activation.

### Execution

Run examples through `uv`:

```bash
uv run python examples/path/to/example.py
```

Do not document direct `python ...` commands as the primary path.

### Backend Selection

Let JAX select the best available backend by default. Do not teach or embed:

- hard-coded multi-platform fallback lists
- system CUDA toolkit paths
- custom CUDA library path management

### Device Statements

At the top of each example page, state one of:

- `CPU-compatible`
- `GPU-optional`
- `GPU-required`

If GPU-required, include the activation command. If CPU-compatible, say so
explicitly.

## Code Standards for Examples

### Use the Supported Runtime Surface

Examples should prefer public imports:

```python
# Operators
from diffbio.operators.singlecell.trajectory import (
    DifferentiablePseudotime,
    PseudotimeConfig,
)

# Losses
from diffbio.losses.singlecell_losses import ShannonDiversityLoss

# Core utilities
from diffbio.core.graph_utils import compute_pairwise_distances

# Ecosystem
from calibrax.metrics.functional.clustering import silhouette_score
from artifex.generative_models.core.losses.divergence import gaussian_kl_divergence
```

Avoid reaching through private or transitional paths when a supported top-level
surface exists.

### Use Flax NNX

DiffBio examples should use Flax NNX patterns consistently:

- `nnx.Module`
- `nnx.Rngs`
- explicit RNG flow
- `nnx.Param` for learnable parameters
- `nnx.relu`, `nnx.gelu` for activations (not `jax.nn.*`)

### Use Typed Configs

Prefer the frozen dataclass configuration model from datarax:

```python
config = PseudotimeConfig(
    n_neighbors=15,
    n_diffusion_components=10,
    root_cell_index=0,
)
operator = DifferentiablePseudotime(config, rngs=nnx.Rngs(42))
```

### Follow the apply() Contract

All operator examples must use the standard `apply()` signature:

```python
result, state, metadata = operator.apply(data, {}, None, None)
```

Where:
- `data` is a dict of JAX arrays
- `state` is an empty dict `{}` for stateless operators
- `metadata` is `None` for simple cases
- The result is `{**input_data, "new_keys": new_values}`

### Synthetic Data Over Real Data

Examples should generate synthetic data rather than requiring external downloads.
This keeps examples self-contained and fast. Use JAX random for data generation:

```python
key = jax.random.key(42)
counts = jax.random.poisson(key, jnp.ones((100, 200)) * 5.0)
```

For more realistic synthetic data, use `DifferentiableSimulator`:

```python
from diffbio.operators.singlecell.simulation import (
    DifferentiableSimulator,
    SimulationConfig,
)
```

### Keep Side Effects Explicit

Examples may print or log progress because they are educational artifacts, but
they should not mutate global backend state, modify tracked repo files, or depend
on hidden shell state.

### Avoid Fake Code

Do not include placeholder commands, nonexistent files, or invented outputs.
If a section is conceptual, label it as conceptual. If code is runnable, make it
actually runnable.

## DiffBio-Specific Example Patterns

### The Standard Operator Example

Every operator example follows this skeleton:

```python
# %% [markdown]
# # Operator Name
# Brief description of what this operator does.

# %% [markdown]
# ## Setup

# %%
import jax
import jax.numpy as jnp
from flax import nnx

from diffbio.operators.domain.module import OperatorClass, OperatorConfig

# %% [markdown]
# ## Configuration

# %%
config = OperatorConfig(
    param1=value1,
    param2=value2,
)
operator = OperatorClass(config, rngs=nnx.Rngs(42))

# %% [markdown]
# ## Create Synthetic Data

# %%
key = jax.random.key(0)
data = {"counts": jax.random.poisson(key, jnp.ones((50, 100)) * 3.0)}
print(f"Input shape: {data['counts'].shape}")

# %% [markdown]
# ## Run the Operator

# %%
result, state, metadata = operator.apply(data, {}, None, None)
for key_name, value in result.items():
    if hasattr(value, 'shape'):
        print(f"  {key_name}: {value.shape}")

# %% [markdown]
# ## Verify Differentiability

# %%
def loss_fn(input_data):
    result, _, _ = operator.apply(input_data, {}, None, None)
    return result["output_key"].sum()

grad = jax.grad(loss_fn)(data)
print(f"Gradient flows: {jnp.any(grad['counts'] != 0)}")

# %% [markdown]
# ## JIT Compilation

# %%
jit_apply = jax.jit(lambda d: operator.apply(d, {}, None, None))
result_jit, _, _ = jit_apply(data)
print(f"JIT output matches: {jnp.allclose(result['output_key'], result_jit['output_key'])}")
```

### The Pipeline Composition Example

Shows how multiple operators chain:

```python
# Step 1: Normalize
norm_result, _, _ = normalizer.apply(data, {}, None, None)

# Step 2: Impute (uses normalized output)
impute_data = {"counts": norm_result["normalized"]}
impute_result, _, _ = imputer.apply(impute_data, {}, None, None)

# Step 3: Trajectory (uses imputed output)
traj_data = {"embeddings": impute_result["imputed_counts"]}
traj_result, _, _ = pseudotime.apply(traj_data, {}, None, None)
```

### The Ecosystem Integration Example

Shows how DiffBio connects to datarax/artifex/calibrax/opifex:

```python
# calibrax metrics for evaluation
from calibrax.metrics.functional.clustering import silhouette_score
score = silhouette_score(latent, labels)

# artifex losses for training
from artifex.generative_models.core.losses.divergence import gaussian_kl_divergence
kl = gaussian_kl_divergence(mean, logvar, reduction="sum")

# opifex for multi-task balancing
from opifex.core.physics.gradnorm import GradNormBalancer
```

## Writing the Companion Docs Page

Every `docs/examples/...` page should include:

- what the example demonstrates
- runtime estimate
- device requirements
- prerequisites (which DiffBio modules are needed)
- exact execution command
- a short map of the major sections
- a few key code excerpts
- expected outputs or result shapes
- related examples

Good top-of-page structure:

1. title
2. difficulty or duration
3. runtime and device note
4. overview
5. quick start
6. key concepts
7. important code excerpts
8. next steps

## Output Capture Requirements

Examples should make it easy for readers to recognize success.

Preferred techniques:

- show expected tensor shapes
- show a few key scalar outputs (loss values, metric scores)
- describe saved artifacts or plots
- include short expected-output snippets in docs when stable

Avoid long unstructured logs in docs pages.

## Visual and Narrative Style

Write example docs for comprehension, not marketing.

Use:

- concrete section titles
- short explanatory paragraphs
- code excerpts with commentary
- direct statements about tradeoffs and limitations

Avoid:

- vague hype
- unexplained jargon
- giant code dumps with no framing
- duplicated prose between the docs page and the example source

## Example Difficulty Tiers

### Basic (5-10 minutes)

Single-operator examples. One config, one `apply()`, one gradient check.
Target audience: first-time DiffBio users.

Examples: DNA encoding, simple alignment, molecular fingerprints, single-cell
clustering, quality filtering.

### Intermediate (10-20 minutes)

Multi-operator workflows. Two or three operators chained, with training loop
or evaluation. Target audience: users building custom pipelines.

Examples: batch correction + clustering, trajectory inference + fate analysis,
VAE normalization + cell type annotation.

### Advanced (20-40 minutes)

Full pipeline composition with ecosystem integration (calibrax metrics, artifex
losses, opifex training). Training loops, benchmarking, comparison against
reference tools. Target audience: researchers adapting DiffBio for their data.

Examples: scVI benchmark, multi-omics integration pipeline, spatial
transcriptomics domain analysis, GRN inference with SCENIC comparison.

## Maintenance Rules

When an example changes, review all three surfaces:

1. example `.py`
2. paired `.ipynb`
3. `docs/examples/...` page

Also update contributor-facing surfaces if the workflow contract changed:

- `examples/README.md`
- `CONTRIBUTING.md`
- `docs/community/contributing.md`

## Review Checklist

Before merging an example:

- [ ] the example runs with `source ./activate.sh`
- [ ] the example runs with `uv run python examples/path/to/example.py`
- [ ] device requirements are stated accurately
- [ ] the `.ipynb` pair was synced from the `.py` source
- [ ] the docs page matches the current source
- [ ] imports use the supported DiffBio public API surface
- [ ] commands use `uv`
- [ ] the example teaches a concrete bioinformatics concept
- [ ] differentiability is demonstrated (gradient flow check)
- [ ] JIT compilation is demonstrated
- [ ] ecosystem building blocks are used where appropriate (calibrax, artifex, opifex)
- [ ] synthetic data is generated (no external file dependencies)
- [ ] output shapes and key values are printed for verification

## Recommended Author Workflow

```bash
source ./activate.sh

# create or edit the Python source
$EDITOR examples/path/to/example.py

# run the example to verify
uv run python examples/path/to/example.py

# sync the notebook pair
uv run jupytext --sync examples/path/to/example.py

# update the docs page
$EDITOR docs/examples/path/to/example-name.md

# verify docs build
uv run mkdocs build
```

## Example Catalog (Planned)

### Basic

| Example | Domain | Operators | Status |
|---------|--------|-----------|--------|
| DNA Encoding | Sequences | one-hot encoding | Exists (docs only) |
| Simple Alignment | Alignment | SmoothSmithWaterman | Exists (docs only) |
| Molecular Fingerprints | Drug Discovery | CircularFingerprint | Exists (docs only) |
| Single-Cell Clustering | Single-Cell | SoftKMeansClustering | Exists (docs only) |
| Quality Filtering | Preprocessing | DifferentiableQualityFilter | Exists (docs only) |
| Pileup Generation | Variant | DifferentiablePileup | Exists (docs only) |

### Intermediate (New — Single-Cell Roadmap)

| Example | Domain | Operators | Status |
|---------|--------|-----------|--------|
| MAGIC Imputation | Single-Cell | DiffusionImputer | Planned |
| Trajectory Inference | Single-Cell | Pseudotime + FateProbability | Planned |
| Cell Type Annotation | Single-Cell | CellAnnotator (3 modes) | Planned |
| Doublet Detection | Single-Cell | DoubletScorer + SoloDetector | Planned |
| Batch Correction | Single-Cell | MMD + WGAN correction | Planned |
| L-R Communication | Single-Cell | LigandReceptor + CellCommunication | Planned |

### Advanced (New — Single-Cell Roadmap)

| Example | Domain | Operators | Status |
|---------|--------|-----------|--------|
| scVI Benchmark | Benchmarking | VAENormalizer + calibrax metrics | Planned |
| Multi-Omics Integration | Multi-Omics | MultiOmicsVAE + GradNorm | Planned |
| Spatial Transcriptomics | Spatial | SpatialDomain + PASTE alignment | Planned |
| GRN Inference | Regulatory | DifferentiableGRN + SCENIC comparison | Planned |
| Foundation Model | Language Models | GeneTokenizer + masked prediction | Planned |
| Full Single-Cell Pipeline | Pipelines | Simulate → Normalize → Impute → Annotate → Trajectory | Planned |

## Related Files

- `docs/examples/templates/example_template.py` (to be created)
- `docs/examples/templates/example_template.ipynb` (to be created)
- `examples/README.md` (to be created)
- `CONTRIBUTING.md`
- `docs/community/contributing.md`
