# Contributing Guide

Thank you for your interest in contributing to DiffBio! This guide will help you get started.

## Getting Started

### Development Setup

```bash
# Clone the repository
git clone https://github.com/mahdi-shafiei/DiffBio.git
cd DiffBio

# Install in development mode
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

# Run specific test file
pytest tests/test_operators.py -v
```

### Building Documentation

```bash
# Serve docs locally
mkdocs serve

# Build static site
mkdocs build
```

## Code Style

DiffBio follows these conventions:

### Python Style

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Maximum line length: 88 characters (Black default)
- Use type hints for all public functions

### Docstring Example

```python
def my_function(param1: int, param2: str = "default") -> float:
    """Short description of function.

    Longer description if needed, explaining the purpose
    and behavior of the function.

    Args:
        param1: Description of param1
        param2: Description of param2 with default

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is negative

    Example:
        >>> result = my_function(42, "hello")
        >>> print(result)
        3.14
    """
```

### Formatting

We use these tools (via pre-commit):

- **Black**: Code formatting
- **isort**: Import sorting
- **ruff**: Linting

Run manually:

```bash
black src/
isort src/
ruff check src/
```

## Adding New Operators

### 1. Create Configuration

```python
from dataclasses import dataclass
from datarax.core.config import OperatorConfig

@dataclass
class MyOperatorConfig(OperatorConfig):
    """Configuration for MyOperator.

    Attributes:
        my_param: Description
    """
    my_param: float = 1.0
    stochastic: bool = False
    stream_name: str | None = None
```

### 2. Implement Operator

```python
from datarax.core.operator import OperatorModule
from flax import nnx

class MyOperator(OperatorModule):
    """Description of operator.

    Detailed explanation of what the operator does
    and how it achieves differentiability.

    Args:
        config: Operator configuration
        rngs: Random number generators

    Example:
        >>> config = MyOperatorConfig(my_param=1.0)
        >>> op = MyOperator(config)
        >>> result = op.apply(data, {}, None)
    """

    def __init__(self, config: MyOperatorConfig, *, rngs: nnx.Rngs = None):
        super().__init__(config, rngs=rngs)
        self.param = nnx.Param(jnp.array(config.my_param))

    def apply(self, data, state, metadata, random_params=None, stats=None):
        # Implementation
        result = self._process(data)
        return {"output": result, **data}, state, metadata
```

### 3. Add Tests

```python
# tests/test_my_operator.py
import pytest
import jax
import jax.numpy as jnp
from diffbio.operators import MyOperator, MyOperatorConfig

class TestMyOperator:
    def test_basic_operation(self):
        config = MyOperatorConfig()
        op = MyOperator(config)
        data = {"input": jnp.ones((10,))}
        result, _, _ = op.apply(data, {}, None)
        assert "output" in result

    def test_differentiability(self):
        config = MyOperatorConfig()
        op = MyOperator(config)

        def loss(op, data):
            result, _, _ = op.apply(data, {}, None)
            return result["output"].sum()

        # Should not raise
        grads = jax.grad(loss)(op, {"input": jnp.ones((10,))})
        assert grads is not None

    def test_configuration(self):
        config = MyOperatorConfig(my_param=2.0)
        op = MyOperator(config)
        assert op.param[...] == 2.0
```

### 4. Export in __init__.py

```python
# src/diffbio/operators/__init__.py
from diffbio.operators.my_operator import MyOperator, MyOperatorConfig

__all__ = [
    # ... existing exports
    "MyOperator",
    "MyOperatorConfig",
]
```

### 5. Add Documentation

Create `docs/user-guide/operators/my-operator.md`:

```markdown
# My Operator

Description of operator...

## Configuration

...

## Usage

...

## API Reference

::: diffbio.operators.my_operator.MyOperator
```

## Pull Request Process

1. **Fork and branch**: Create a feature branch from `main`
2. **Implement**: Make your changes with tests
3. **Test**: Ensure all tests pass
4. **Document**: Update documentation if needed
5. **Commit**: Use clear, descriptive commit messages
6. **PR**: Open a pull request with description

### Commit Message Format

```
type: short description

Longer description if needed explaining the change.

- Bullet points for specific changes
- Another change
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `chore`

### PR Checklist

- [ ] Tests pass locally
- [ ] New code has tests
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Docstrings added
- [ ] Changelog updated (if applicable)

## Reporting Issues

### Bug Reports

Include:

- DiffBio version
- Python version
- JAX version
- Minimal reproduction code
- Expected vs actual behavior
- Full error traceback

### Feature Requests

Include:

- Use case description
- Proposed API (if applicable)
- Related existing features
- Priority/importance

## Questions?

- Open a [GitHub issue](https://github.com/mahdi-shafiei/DiffBio/issues)
- Check existing issues first
- Tag appropriately (bug, enhancement, question)
