# Installation

DiffBio requires Python 3.10+ and works on Linux, macOS, and Windows (via WSL).

## Quick Install

Install DiffBio from PyPI:

```bash
pip install diffbio
```

## Install from Source

For the latest development version:

```bash
git clone https://github.com/avitai/DiffBio.git
cd DiffBio
pip install -e ".[dev]"
```

## Using uv (Recommended)

If you use [uv](https://docs.astral.sh/uv/) for faster package management:

```bash
uv pip install diffbio
```

Or for development:

```bash
git clone https://github.com/avitai/DiffBio.git
cd DiffBio
uv sync --all-extras
```

## Dependencies

DiffBio depends on the following core packages:

| Package | Purpose |
|---------|---------|
| [JAX](https://jax.readthedocs.io/) | Automatic differentiation and XLA compilation |
| [Flax](https://flax.readthedocs.io/) | Neural network library for JAX |
| [Datarax](https://github.com/avitai/datarax) | Composable data pipeline framework |
| [jaxtyping](https://github.com/google/jaxtyping) | Type annotations for JAX arrays |

## GPU Support

For GPU acceleration, install JAX with CUDA support:

=== "CUDA 12.x"
    ```bash
    pip install -U "jax[cuda12]"
    ```

=== "CUDA 11.x"
    ```bash
    pip install -U "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ```

Verify GPU is available:

```python
import jax
print(jax.devices())  # Should show GPU devices
```

## Verify Installation

```python
import diffbio
from diffbio.operators import SmoothSmithWaterman, SmithWatermanConfig
from diffbio.operators.alignment import create_dna_scoring_matrix

print(f"DiffBio version: {diffbio.__version__}")

# Test operator creation
config = SmithWatermanConfig(temperature=1.0)
scoring = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)
aligner = SmoothSmithWaterman(config, scoring_matrix=scoring)
print("Installation successful!")
```

## Development Installation

For contributing to DiffBio:

```bash
git clone https://github.com/avitai/DiffBio.git
cd DiffBio

# Install with all development dependencies
pip install -e ".[dev,docs]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=src/diffbio --cov-report=term-missing
```

### Building Documentation

```bash
# Serve docs locally
mkdocs serve

# Build static site
mkdocs build
```

## Troubleshooting

### Common Issues

**JAX not using GPU:**

Check that CUDA is properly installed and JAX can see it:

```python
import jax
print(jax.default_backend())  # Should print 'gpu' or 'cuda'
```

**Import errors:**

Ensure all dependencies are installed:

```bash
pip install --upgrade diffbio[all]
```

**Memory issues with large sequences:**

DiffBio operators use JAX's XLA compilation which can be memory-intensive for very long sequences. Consider:

- Using smaller batch sizes
- Chunking long sequences
- Enabling memory-efficient attention patterns

For additional help, please [open an issue](https://github.com/avitai/DiffBio/issues) on GitHub.
