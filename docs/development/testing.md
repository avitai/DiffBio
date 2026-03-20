# Testing

DiffBio uses pytest with a test-first methodology. All operators must have
tests covering output shapes, differentiability, JIT compatibility, and edge
cases before implementation.

---

## Running Tests

Always activate the environment first:

```bash
source ./activate.sh
```

### Common Commands

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ -v --cov=src/diffbio --cov-report=term-missing

# Run a specific domain
uv run pytest tests/operators/singlecell/ -xvs

# Run a specific test class
uv run pytest tests/operators/alignment/test_profile_hmm.py::TestProfileHMM -xvs

# Run tests matching a keyword
uv run pytest tests/ -k "smith_waterman" -v

# Stop on first failure
uv run pytest tests/ -x

# Run last failed tests first
uv run pytest tests/ --lf

# Show print output
uv run pytest tests/ -s
```

### Full Test + Coverage Command

The complete command used for CI-equivalent local testing:

```bash
uv run pytest -vv \
  --json-report --json-report-file=temp/test-results.json \
  --json-report-indent=2 --json-report-verbosity=2 \
  --cov=src/ --cov-report=json:temp/coverage.json \
  --cov-report=term-missing
```

---

## Test Directory Structure

Tests mirror the source tree:

```
tests/
├── conftest.py                    # Shared fixtures (RNGs, sample data)
├── core/                          # Core utilities
├── integration/                   # Cross-operator integration tests
│   ├── test_operator_composition.py
│   └── test_singlecell_pipeline.py
├── losses/                        # Loss function tests
├── operators/                     # Operator unit tests (by domain)
│   ├── alignment/
│   │   ├── test_profile_hmm.py
│   │   └── test_soft_msa.py
│   ├── assembly/
│   ├── drug_discovery/
│   ├── epigenomics/
│   ├── language_models/
│   ├── mapping/
│   ├── multiomics/
│   ├── normalization/
│   ├── preprocessing/
│   ├── rnaseq/
│   ├── singlecell/               # Largest domain (~20 test files)
│   ├── statistical/
│   └── variant/
├── pipelines/                     # End-to-end pipeline tests
├── sequences/                     # DNA/RNA encoding tests
├── sources/                       # Data source tests
├── splitters/                     # Dataset splitting tests
└── utils/                         # Training utility tests
```

---

## Required Test Patterns

Every operator must have tests for these four properties.

### 1. Output Shape and Keys

Verify the operator produces the expected output dictionary:

```python
def test_output_shape(self, operator: MyOperator) -> None:
    data = {"counts": jnp.ones((50, 100))}
    result, state, metadata = operator.apply(data, {}, None)

    # Check expected output keys exist
    assert "output" in result
    assert result["output"].shape == (50, 100)

    # Check input keys are preserved
    assert "counts" in result
```

### 2. Differentiability

Verify gradients flow through the operator:

```python
def test_differentiability(self, operator: MyOperator) -> None:
    def loss_fn(data):
        result, _, _ = operator.apply(data, {}, None)
        return result["output"].sum()

    data = {"counts": jnp.ones((50, 100))}
    grad = jax.grad(loss_fn)(data)

    # Gradients must be non-zero and finite
    assert jnp.any(grad["counts"] != 0)
    assert jnp.all(jnp.isfinite(grad["counts"]))
```

### 3. JIT Compatibility

Verify the operator works under `jax.jit`:

```python
def test_jit_compatible(self, operator: MyOperator) -> None:
    data = {"counts": jnp.ones((50, 100))}
    eager_result, _, _ = operator.apply(data, {}, None)

    jit_fn = jax.jit(lambda d: operator.apply(d, {}, None))
    jit_result, _, _ = jit_fn(data)

    assert jnp.allclose(eager_result["output"], jit_result["output"], atol=1e-5)
```

### 4. Configuration

Verify config parameters affect operator behavior:

```python
def test_config_affects_output(self) -> None:
    data = {"counts": jnp.ones((50, 100))}

    config_a = MyOperatorConfig(temperature=0.1)
    op_a = MyOperator(config_a, rngs=nnx.Rngs(42))
    result_a, _, _ = op_a.apply(data, {}, None)

    config_b = MyOperatorConfig(temperature=10.0)
    op_b = MyOperator(config_b, rngs=nnx.Rngs(42))
    result_b, _, _ = op_b.apply(data, {}, None)

    # Different temperatures should produce different outputs
    assert not jnp.allclose(result_a["output"], result_b["output"])
```

---

## Fixtures

### Shared Fixtures (conftest.py)

The project-level `conftest.py` provides common fixtures:

```python
@pytest.fixture
def rngs():
    """Flax NNX random number generators."""
    return nnx.Rngs(42)

@pytest.fixture
def random_key():
    """JAX random key."""
    return jax.random.key(42)
```

### Domain-Specific Fixtures

Use `@pytest.fixture` within test classes for operator-specific setup:

```python
class TestSoftKMeansClustering:
    @pytest.fixture
    def config(self) -> SoftClusteringConfig:
        return SoftClusteringConfig(n_clusters=5, n_features=20)

    @pytest.fixture
    def operator(self, config: SoftClusteringConfig) -> SoftKMeansClustering:
        return SoftKMeansClustering(config, rngs=nnx.Rngs(42))

    @pytest.fixture
    def sample_data(self) -> dict[str, jax.Array]:
        return {"embeddings": jax.random.normal(jax.random.key(0), (100, 20))}
```

---

## Parameterized Tests

Use `@pytest.mark.parametrize` for testing across configurations:

```python
@pytest.mark.parametrize("temperature", [0.1, 1.0, 5.0, 10.0])
def test_temperature_range(self, temperature: float) -> None:
    config = SmithWatermanConfig(temperature=temperature)
    aligner = SmoothSmithWaterman(config, scoring_matrix=scoring, rngs=nnx.Rngs(0))
    data = {"seq1": seq1, "seq2": seq2}
    result, _, _ = aligner.apply(data, {}, None)
    assert jnp.isfinite(result["score"])
```

---

## Test Markers

```python
@pytest.mark.gpu
def test_gpu_operation(self):
    """Requires CUDA GPU."""
    ...

@pytest.mark.slow
def test_large_scale(self):
    """Takes >30 seconds."""
    ...
```

Run subsets:

```bash
uv run pytest tests/ -m gpu       # GPU tests only
uv run pytest tests/ -m "not slow" # Skip slow tests
```

---

## Assertion Best Practices

Use focused assertions that produce clean failure messages:

```python
# Good: targeted, informative on failure
assert result["output"].shape == (50, 20)
assert float(loss) < 1.0
assert jnp.allclose(result_a, result_b, atol=1e-5)

# Bad: massive diff output, potential timeouts
assert result == expected_full_dict
assert config_a == config_b
```

For floating-point comparisons, always use `jnp.allclose` with an explicit
tolerance rather than exact equality.

---

## Integration Tests

Integration tests in `tests/integration/` verify operator composition:

```python
def test_pipeline_chain(self) -> None:
    """Verify operators chain correctly via data dict."""
    # Step 1: Impute
    result, _, _ = imputer.apply({"counts": counts}, {}, None)

    # Step 2: Cluster (uses imputed output)
    result["embeddings"] = result["imputed_counts"]
    result, _, _ = clusterer.apply(result, {}, None)

    # Verify both outputs present
    assert "imputed_counts" in result
    assert "cluster_assignments" in result
```

---

## Coverage Requirements

New code must achieve 80% coverage minimum. Check coverage after adding tests:

```bash
uv run pytest tests/ --cov=src/diffbio --cov-report=term-missing --cov-fail-under=80
```

Untested lines appear in the `Missing` column. Focus on testing public API
methods (`__init__`, `apply`) rather than private helpers.
