# Contributing to DiffBio

Thank you for your interest in contributing to DiffBio! This document provides guidelines and information for contributors.

## Getting Started

### Development Setup

```bash
# Clone the repository
git clone https://github.com/mahdi-shafiei/DiffBio.git
cd DiffBio

# Run setup script (creates venv, installs dependencies, detects GPU)
./setup.sh

# Activate the environment
source ./activate.sh
```

### Prerequisites

- Python 3.11+
- uv package manager
- Git

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/amazing-feature
```

### 2. Make Your Changes

- Follow the coding standards below
- Write tests for new functionality
- Update documentation as needed

### 3. Run Quality Checks

```bash
# Run all pre-commit checks
uv run pre-commit run --all-files

# Run tests
uv run pytest tests/ -v
```

### 4. Commit Your Changes

```bash
git commit -m "Add amazing feature"
```

### 5. Push and Create PR

```bash
git push origin feature/amazing-feature
```

Then open a Pull Request on GitHub.

## Coding Standards

### Python Style

- Follow PEP 8 guidelines
- Use type annotations for all functions
- Maximum line length: 100 characters
- Use descriptive variable names

### Framework Requirements

**Always use Flax NNX:**

```python
# CORRECT
from flax import nnx

class MyModule(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        super().__init__()
        self.dense = nnx.Linear(10, 10, rngs=rngs)
```

**Never use Flax Linen or PyTorch.**

### Operator Style

All operators inherit from Datarax OperatorModule:

```python
from datarax.core.operator import OperatorModule
from datarax.core.config import OperatorConfig
from dataclasses import dataclass

@dataclass
class MyOperatorConfig(OperatorConfig):
    my_param: float = 1.0
    stochastic: bool = False
    stream_name: str | None = None

class MyOperator(OperatorModule):
    def __init__(self, config: MyOperatorConfig, *, rngs: nnx.Rngs = None):
        super().__init__(config, rngs=rngs)

    def apply(self, data, state, metadata, random_params=None, stats=None):
        # Implementation
        return {**data, "output": result}, state, metadata
```

### Testing Requirements

- Write tests for all new functionality
- Tests should be in the appropriate `tests/` subdirectory
- Aim for minimum 80% coverage on new code
- Use pytest fixtures for common setup

```python
def test_feature():
    """Test description."""
    # Arrange
    config = create_config()

    # Act
    result = function_under_test(config)

    # Assert
    assert result.property == expected_value
```

## Code Quality Tools

### Pre-commit Hooks

Install and run pre-commit hooks:

```bash
# Install hooks
uv run pre-commit install

# Run on all files
uv run pre-commit run --all-files
```

### Linting and Formatting

```bash
# Linting
uv run ruff check src/

# Formatting
uv run ruff format src/

# Type checking
uv run pyright src/
```

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/operators/test_alignment.py -xvs

# Run with coverage
uv run pytest --cov=src/diffbio --cov-report=html

# Run GPU-specific tests
uv run pytest -m gpu
```

### Test Organization

- `tests/operators/`: Operator unit tests
- `tests/pipelines/`: Pipeline tests
- `tests/integration/`: Integration tests
- `tests/utils/`: Utility tests
- Mark GPU tests with `@pytest.mark.gpu`

## Documentation

### Writing Documentation

- Use Markdown for all documentation
- Include code examples where appropriate
- Keep examples runnable and tested
- Update relevant docs when changing code

### Building Documentation

```bash
# Serve docs locally
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

### Documentation Structure

```
docs/
├── getting-started/    # Installation and first steps
├── user-guide/         # How-to guides
├── api/                # API reference
├── examples/           # Example documentation
└── development/        # Contributor guides
```

## Pull Request Guidelines

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests pass locally (`uv run pytest tests/ -v`)
- [ ] Pre-commit hooks pass (`uv run pre-commit run --all-files`)
- [ ] Documentation updated if needed
- [ ] Commit messages are clear and descriptive

### PR Title Format

Use conventional commit format:

- `feat: Add new feature`
- `fix: Fix bug in component`
- `docs: Update documentation`
- `refactor: Refactor module`
- `test: Add tests for feature`
- `chore: Update dependencies`

### PR Description

Include:

- Brief description of changes
- Motivation and context
- How to test the changes
- Any breaking changes

## Issue Guidelines

### Bug Reports

Include:

- DiffBio version
- Python version
- JAX version
- Operating system
- Minimal reproducible example
- Expected vs actual behavior

### Feature Requests

Include:

- Use case description
- Proposed solution
- Alternative approaches considered

## Community

### Getting Help

- Check existing documentation
- Search existing issues
- Open a new issue if needed

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
