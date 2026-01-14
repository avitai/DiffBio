# Testing Guide

DiffBio uses pytest for testing. This guide covers how to run tests and write new ones.

## Running Tests

### Basic Test Run

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Very verbose (show individual test names)
pytest -vv
```

### With Coverage

```bash
# Basic coverage report
pytest --cov=src/diffbio

# Detailed report showing missing lines
pytest --cov=src/diffbio --cov-report=term-missing

# Generate HTML report
pytest --cov=src/diffbio --cov-report=html
# Open htmlcov/index.html in browser
```

### Filtering Tests

```bash
# Run tests in a specific file
pytest tests/test_operators.py

# Run tests matching a pattern
pytest -k "smith_waterman"

# Run tests in a specific class
pytest tests/test_operators.py::TestSmithWaterman

# Run a specific test
pytest tests/test_operators.py::TestSmithWaterman::test_basic_alignment
```

### Test Output

```bash
# Show print statements
pytest -s

# Show local variables on failure
pytest -l

# Stop on first failure
pytest -x

# Run last failed tests first
pytest --lf
```

## Test Structure

### Directory Layout

```
tests/
├── __init__.py
├── conftest.py          # Shared fixtures
├── integration/         # Integration tests
│   ├── __init__.py
│   └── test_operator_composition.py
├── operators/           # Operator unit tests
│   ├── __init__.py
│   ├── test_alignment.py
│   ├── test_pileup.py
│   └── test_quality_filter.py
├── pipelines/           # Pipeline tests
│   ├── __init__.py
│   └── test_variant_calling.py
└── utils/               # Utility tests
    ├── __init__.py
    └── test_training.py
```

### Test Naming

- Test files: `test_*.py` or `*_test.py`
- Test classes: `Test*`
- Test functions: `test_*`

## Writing Tests

### Basic Test Structure

```python
import pytest
import jax
import jax.numpy as jnp

class TestMyOperator:
    """Tests for MyOperator."""

    def test_basic_operation(self):
        """Test basic functionality."""
        # Arrange
        config = MyOperatorConfig()
        operator = MyOperator(config)
        data = {"input": jnp.ones((10,))}

        # Act
        result, state, metadata = operator.apply(data, {}, None)

        # Assert
        assert "output" in result
        assert result["output"].shape == (10,)

    def test_with_specific_config(self):
        """Test with non-default configuration."""
        config = MyOperatorConfig(my_param=2.0)
        operator = MyOperator(config)
        # ...
```

### Testing Differentiability

```python
def test_gradient_computation(self):
    """Verify operator is differentiable."""
    config = MyOperatorConfig()
    operator = MyOperator(config)

    def loss_fn(op, data):
        result, _, _ = op.apply(data, {}, None)
        return result["output"].sum()

    data = {"input": jnp.ones((10,))}

    # Should not raise any errors
    grads = jax.grad(loss_fn)(operator, data)

    # Gradients should be non-zero (generally)
    assert grads is not None
    # Check specific gradient properties if needed
```

### Testing JIT Compilation

```python
def test_jit_compatible(self):
    """Verify operator works with JAX JIT."""
    config = MyOperatorConfig()
    operator = MyOperator(config)

    @jax.jit
    def apply_op(data):
        result, _, _ = operator.apply(data, {}, None)
        return result

    data = {"input": jnp.ones((10,))}

    # First call compiles
    result1 = apply_op(data)

    # Second call uses compiled version
    result2 = apply_op(data)

    # Results should match
    assert jnp.allclose(result1["output"], result2["output"])
```

### Testing Batched Operations

```python
def test_vmap_compatible(self):
    """Verify operator works with JAX vmap."""
    config = MyOperatorConfig()
    operator = MyOperator(config)

    def single_apply(single_data):
        result, _, _ = operator.apply(single_data, {}, None)
        return result["output"]

    batch_apply = jax.vmap(single_apply)

    # Create batch data
    batch_data = {"input": jnp.ones((5, 10))}

    # Should work without errors
    batch_result = batch_apply(batch_data)
    assert batch_result.shape == (5, 10)
```

## Fixtures

### Shared Fixtures in conftest.py

```python
# tests/conftest.py
import pytest
import jax
import jax.numpy as jnp
from flax import nnx

@pytest.fixture
def random_key():
    """Provide a JAX random key."""
    return jax.random.PRNGKey(42)

@pytest.fixture
def rngs():
    """Provide Flax NNX RNGs."""
    return nnx.Rngs(42)

@pytest.fixture
def sample_reads(random_key):
    """Generate sample read data."""
    num_reads = 10
    read_length = 30
    k1, k2, k3 = jax.random.split(random_key, 3)

    return {
        "reads": jax.nn.one_hot(
            jax.random.randint(k1, (num_reads, read_length), 0, 4),
            4
        ),
        "positions": jax.random.randint(k2, (num_reads,), 0, 70),
        "quality": jax.random.uniform(k3, (num_reads, read_length), 10, 40),
    }
```

### Using Fixtures

```python
class TestPileup:
    def test_with_sample_data(self, sample_reads, rngs):
        """Test using fixtures."""
        config = PileupConfig(reference_length=100)
        pileup_op = DifferentiablePileup(config, rngs=rngs)

        result, _, _ = pileup_op.apply(sample_reads, {}, None)
        assert result["pileup"].shape == (100, 4)
```

## Parameterized Tests

```python
import pytest

class TestSmithWaterman:
    @pytest.mark.parametrize("temperature", [0.1, 1.0, 5.0, 10.0])
    def test_different_temperatures(self, temperature):
        """Test alignment with various temperatures."""
        config = SmithWatermanConfig(temperature=temperature)
        aligner = SmoothSmithWaterman(config, scoring_matrix=scoring)

        result = aligner.align(seq1, seq2)
        assert result.score > 0

    @pytest.mark.parametrize("gap_open,gap_extend", [
        (-5.0, -0.5),
        (-10.0, -1.0),
        (-20.0, -2.0),
    ])
    def test_gap_penalties(self, gap_open, gap_extend):
        """Test with different gap penalties."""
        config = SmithWatermanConfig(gap_open=gap_open, gap_extend=gap_extend)
        # ...
```

## Markers

### Skip Tests

```python
@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature(self):
    pass

@pytest.mark.skipif(not jax.devices("gpu"), reason="Requires GPU")
def test_gpu_specific(self):
    pass
```

### Slow Tests

```python
@pytest.mark.slow
def test_large_scale(self):
    """This test takes a long time."""
    pass

# Run excluding slow tests
# pytest -m "not slow"
```

### Expected Failures

```python
@pytest.mark.xfail(reason="Known issue #123")
def test_known_issue(self):
    pass
```

## Best Practices

1. **Test one thing per test**: Each test should verify a single behavior
2. **Use descriptive names**: `test_alignment_with_gaps_returns_lower_score`
3. **Isolate tests**: Tests should not depend on each other
4. **Use fixtures**: Share setup code via fixtures
5. **Test edge cases**: Empty inputs, boundary conditions, etc.
6. **Check shapes**: Always verify array shapes in numerical tests
7. **Use approximate comparisons**: `jnp.allclose()` for floating point

## Continuous Integration

Tests run automatically on:

- Every push
- Every pull request

CI configuration in `.github/workflows/test.yml`:

```yaml
- name: Run tests
  run: |
    pytest -v --cov=src/diffbio --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```
