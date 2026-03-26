# Contributing

This guide covers the workflow for contributing operators, tests, and
documentation to DiffBio.

---

## Development Setup

```bash
git clone https://github.com/mahdi-shafiei/DiffBio.git
cd DiffBio
./setup.sh
source ./activate.sh
```

The setup script creates a virtual environment, installs all dependencies
via `uv`, and configures GPU detection. Always activate with `source ./activate.sh`
before running any commands.

---

## Code Quality Tools

DiffBio uses **ruff** for linting and formatting. Do not use Black, isort, or
flake8 — they are not configured for this project.

```bash
uv run ruff check src/ --fix   # Lint + autofix
uv run ruff format src/        # Format
uv run pyright src/            # Type check
uv run pre-commit run --all-files  # Run all hooks
```

### Style Rules

| Rule | Value |
|---|---|
| Line length | 100 characters |
| Formatter | ruff format (not Black) |
| Import sorting | ruff (not isort) |
| Type checker | pyright (not mypy) |
| Docstrings | Google style |
| Type hints | Required on all public functions |
| Type syntax | `list`, `dict`, `tuple` (not `List`, `Dict`, `Tuple`) |

### Pre-commit Hooks

Pre-commit runs automatically on `git commit`. All hooks must pass before
committing. The configured hooks include:

- ruff (lint + format)
- pyright (type checking)
- bandit (security)
- interrogate (docstring coverage)
- nbqa-ruff (notebook linting)
- radon (cyclomatic complexity)
- trailing whitespace, end-of-file, YAML/TOML checks

Install hooks after cloning:

```bash
uv run pre-commit install
```

---

## Adding a New Operator

### 1. Define the Configuration

```python
from dataclasses import dataclass
from datarax.core.config import OperatorConfig

@dataclass(frozen=True)
class MyOperatorConfig(OperatorConfig):
    """Configuration for MyOperator.

    Attributes:
        my_param: Controls the smoothing intensity.
    """

    my_param: float = 1.0
```

### 2. Implement the Operator

All operators inherit from `datarax.core.operator.OperatorModule` and
implement the `apply()` contract:

```python
from datarax.core.operator import OperatorModule
from flax import nnx
import jax.numpy as jnp

class MyOperator(OperatorModule):
    """One-line description of what this operator does.

    Detailed explanation of the algorithm, including what smooth
    approximation technique is used for differentiability.

    Args:
        config: Operator configuration.
        rngs: Flax NNX random number generators.
    """

    def __init__(self, config: MyOperatorConfig, *, rngs: nnx.Rngs) -> None:
        super().__init__(config, rngs=rngs)
        self.param = nnx.Param(jnp.array(config.my_param))

    def apply(
        self, data, state, metadata, random_params=None, stats=None,
    ):
        result = self._process(data)
        return {**data, "output": result}, state, metadata
```

Key patterns:
- Always call `super().__init__(config, rngs=rngs)`
- Use `nnx.Param` for learnable parameters
- Return `{**data, "new_key": value}` to preserve input keys
- Use `jax.numpy` inside the operator, never `numpy`

### 3. Write Tests First

Tests go in `tests/operators/<domain>/test_<module>.py`:

```python
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.<domain> import MyOperator, MyOperatorConfig


class TestMyOperator:
    @pytest.fixture
    def config(self) -> MyOperatorConfig:
        return MyOperatorConfig(my_param=1.0)

    @pytest.fixture
    def operator(self, config: MyOperatorConfig) -> MyOperator:
        return MyOperator(config, rngs=nnx.Rngs(42))

    def test_output_shape(self, operator: MyOperator) -> None:
        data = {"input": jnp.ones((10, 20))}
        result, _, _ = operator.apply(data, {}, None)
        assert result["output"].shape == (10, 20)

    def test_differentiability(self, operator: MyOperator) -> None:
        def loss_fn(data):
            result, _, _ = operator.apply(data, {}, None)
            return result["output"].sum()

        data = {"input": jnp.ones((10, 20))}
        grad = jax.grad(loss_fn)(data)
        assert jnp.any(grad["input"] != 0)
        assert jnp.all(jnp.isfinite(grad["input"]))

    def test_jit_compatible(self, operator: MyOperator) -> None:
        data = {"input": jnp.ones((10, 20))}
        eager_result, _, _ = operator.apply(data, {}, None)
        jit_result, _, _ = jax.jit(lambda d: operator.apply(d, {}, None))(data)
        assert jnp.allclose(eager_result["output"], jit_result["output"])
```

Run tests:

```bash
source ./activate.sh
uv run pytest tests/operators/<domain>/test_my_operator.py -xvs
```

### 4. Export in `__init__.py`

Add the operator and config to the domain's `__init__.py`:

```python
from diffbio.operators.<domain>.my_module import MyOperator, MyOperatorConfig

__all__ = [
    # ... existing exports
    "MyOperator",
    "MyOperatorConfig",
]
```

### 5. Add Documentation

Three artifacts:

1. **API reference** — Add mkdocstrings directive to `docs/api/operators/<domain>.md`:

    ```markdown
    ## MyOperator

    ::: diffbio.operators.<domain>.my_module.MyOperator
        options:
          show_root_heading: true
          show_source: false
          members:
            - __init__
            - apply

    ## MyOperatorConfig

    ::: diffbio.operators.<domain>.my_module.MyOperatorConfig
        options:
          show_root_heading: true
          members: []
    ```

2. **User guide** — Add a section to `docs/user-guide/operators/<domain>.md`
   with overview, quick start, config table, and use cases.

3. **Concept page** — If the operator introduces a new biological concept,
   add it to the relevant `docs/user-guide/concepts/` page.

---

## Running Tests

```bash
source ./activate.sh

# Run all tests
uv run pytest tests/ -v

# Run a specific domain
uv run pytest tests/operators/singlecell/ -xvs

# Run with coverage
uv run pytest tests/ -v --cov=src/diffbio --cov-report=term-missing

# Run a single test
uv run pytest tests/operators/alignment/test_profile_hmm.py::TestProfileHMM::test_basic -xvs
```

See [Testing](testing.md) for details on test patterns and fixtures.

---

## Building Documentation

```bash
source ./activate.sh

# Serve locally with live reload
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

See [Example Documentation Design](example-documentation-design.md) for the
example authoring workflow.

---

## Pull Request Process

1. Create a feature branch from `main`
2. Write tests, then implement
3. Run `uv run pre-commit run --all-files` — all hooks must pass
4. Run `uv run pytest tests/ -v` — all tests must pass
5. Update documentation if adding new operators
6. Open a pull request with a clear description

### Commit Messages

```
type: short description

Longer description if needed.

- Specific change 1
- Specific change 2
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`

---

## Reporting Issues

### Bug Reports

Include: DiffBio version, Python version, JAX version, minimal reproduction
code, expected vs actual behavior, full traceback.

### Feature Requests

Include: use case description, proposed API, related existing operators.

File issues at [github.com/mahdi-shafiei/DiffBio/issues](https://github.com/mahdi-shafiei/DiffBio/issues).
