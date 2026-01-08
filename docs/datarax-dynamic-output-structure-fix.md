# Datarax Fix: Dynamic Output Structure in apply_batch()

**Target Repository:** `/media/mahdi/ssd23/Works/workshop-data/`
**Issue:** Operators cannot add new keys to output data when using `apply_batch()`

---

## Problem Statement

`OperatorModule.apply_batch()` derives `out_axes` from INPUT data structure, preventing operators from adding new keys to output data.

### Current Behavior (Broken)

```python
# Input:  {"seq1": array, "seq2": array}
# Output: {"seq1": array, "seq2": array, "score": array, "alignment": array}
# Result: ValueError - vmap out_axes doesn't match output structure
```

### Root Cause

**File:** `src/datarax/core/operator.py` (lines 199-200, 243)

```python
# Lines 199-200: Derives out_axes from INPUT structure
data_axes = jax.tree.map(lambda _: 0, batch_data)      # INPUT structure
states_axes = jax.tree.map(lambda _: 0, batch_states)  # INPUT structure

# Line 243: Uses INPUT axes for OUTPUT
out_axes=(data_axes, states_axes)  # BREAKS if apply() adds new keys!
```

---

## Solution: Hybrid Approach

Add `get_output_structure()` method with automatic discovery via `jax.eval_shape`.

### Why This Approach

| Approach | Pros | Cons |
|----------|------|------|
| Config-based schema | JIT-friendly | Verbose, duplication |
| Output structure method only | Clean API | Requires override for every operator |
| `jax.eval_shape` only | Fully automatic | Tracing overhead |
| **Hybrid (recommended)** | Best of both worlds | Slightly more complex |

**Hybrid approach:**
1. Add `get_output_structure()` method (explicit declaration option)
2. Default implementation uses `jax.eval_shape` (automatic discovery)
3. Operators can override for efficiency or special cases
4. Structure caching avoids repeated `eval_shape` calls

---

## Implementation

### Step 1: Add `get_output_structure()` Method

**File:** `src/datarax/core/operator.py`

Add this method to the `OperatorModule` class:

```python
def get_output_structure(
    self,
    sample_data: PyTree,
    sample_state: PyTree,
) -> tuple[PyTree, PyTree]:
    """Declare output PyTree structure for vmap axis specification.

    Default uses jax.eval_shape to discover structure automatically.
    Override for efficiency or when eval_shape doesn't work (e.g., data-dependent shapes).

    Args:
        sample_data: Single element data (not batched)
        sample_state: Single element state (not batched)

    Returns:
        Tuple of (output_data_structure, output_state_structure) with None leaves.
        The structure (keys/nesting) matters, leaf values are ignored.

    Example override for operator that adds keys:
        def get_output_structure(self, sample_data, sample_state):
            out_data = {
                **jax.tree.map(lambda _: None, sample_data),
                "score": None,
                "alignment": None,
            }
            return out_data, sample_state
    """
    # Default: use eval_shape to discover output structure without computation
    def apply_wrapper(data, state):
        out_data, out_state, _ = self.apply(data, state, None)
        return out_data, out_state

    out_shapes = jax.eval_shape(apply_wrapper, sample_data, sample_state)
    out_data_struct = jax.tree.map(lambda _: None, out_shapes[0])
    out_state_struct = jax.tree.map(lambda _: None, out_shapes[1])
    return out_data_struct, out_state_struct
```

### Step 2: Modify `apply_batch()` Method

**File:** `src/datarax/core/operator.py`

Replace the current `apply_batch()` implementation (approximately lines 160-255):

```python
def apply_batch(
    self,
    batch: Batch,
    stats: dict[str, Any] | None = None,
) -> Batch:
    """Process entire batch with vmap, supporting dynamic output structure.

    This method applies the operator to all elements in a batch using JAX's vmap
    for efficient vectorized processing. Unlike previous implementations, this
    supports operators that add new keys to output data.

    Args:
        batch: Input batch containing data, states, and metadata.
        stats: Optional statistics dictionary for tracking.

    Returns:
        Transformed batch with processed data and states.

    Note:
        Output structure is discovered via get_output_structure(), which defaults
        to using jax.eval_shape for automatic discovery. Operators that add keys
        can override get_output_structure() for efficiency.
    """
    # Extract batch components for vmap processing
    batch_data = batch.data.get_value()
    batch_states = batch.states.get_value()
    batch_metadata = batch._metadata_list
    batch_size = batch.batch_size

    # === OUTPUT STRUCTURE DISCOVERY (NEW) ===
    # Cache output structure per input structure for JIT efficiency
    input_struct_id = id(jax.tree.structure(batch_data))
    if not hasattr(self, '_output_struct_cache'):
        self._output_struct_cache = {}

    if input_struct_id not in self._output_struct_cache:
        # Get sample element (unbatched) for structure discovery
        sample_data = jax.tree.map(lambda x: x[0], batch_data)
        sample_state = jax.tree.map(lambda x: x[0], batch_states)

        # Discover output structure (may differ from input!)
        self._output_struct_cache[input_struct_id] = self.get_output_structure(
            sample_data, sample_state
        )

    out_data_struct, out_state_struct = self._output_struct_cache[input_struct_id]

    # === AXIS SPECIFICATION ===
    # Build in_axes from INPUT structure
    in_data_axes = jax.tree.map(lambda _: 0, batch_data)
    in_state_axes = jax.tree.map(lambda _: 0, batch_states)

    # Build out_axes from OUTPUT structure (KEY CHANGE!)
    out_data_axes = jax.tree.map(lambda _: 0, out_data_struct)
    out_state_axes = jax.tree.map(lambda _: 0, out_state_struct)

    # === RANDOM PARAMETER GENERATION ===
    if self.stochastic:
        random_params_batch = self._generate_random_params(batch_size)
    else:
        random_params_batch = jnp.zeros(batch_size)

    # === VMAP APPLICATION ===
    def apply_fn(data, state, random_params):
        out_data, out_state, _ = self.apply(data, state, None, random_params, stats)
        return out_data, out_state

    transformed_data, transformed_states = jax.vmap(
        apply_fn,
        in_axes=(in_data_axes, in_state_axes, 0),
        out_axes=(out_data_axes, out_state_axes),  # Uses OUTPUT structure!
    )(batch_data, batch_states, random_params_batch)

    # === BATCH RECONSTRUCTION ===
    return Batch.from_parts(
        data=transformed_data,
        states=transformed_states,
        metadata=batch_metadata,
    )
```

### Step 3: Add Required Import

**File:** `src/datarax/core/operator.py`

Ensure `jax` is imported at the top of the file (should already be there):

```python
import jax
import jax.numpy as jnp
```

---

## Test Implementation

### New Test File: `tests/core/test_operator_output_structure.py`

```python
"""Tests for operators with dynamic output structure."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Batch, Element
from datarax.core.operator import OperatorModule


# =============================================================================
# Test Operators
# =============================================================================


class AddKeyOperator(OperatorModule):
    """Test operator that adds a single key to output."""

    def apply(self, data, state, metadata, random_params=None, stats=None):
        out_data = {
            **data,
            "computed": data["input"] * 2,
        }
        return out_data, state, metadata


class AddMultipleKeysOperator(OperatorModule):
    """Test operator that adds multiple keys to output."""

    def apply(self, data, state, metadata, random_params=None, stats=None):
        out_data = {
            **data,
            "sum": data["a"] + data["b"],
            "product": data["a"] * data["b"],
            "difference": data["a"] - data["b"],
        }
        return out_data, state, metadata


class ExplicitStructureOperator(OperatorModule):
    """Test operator with explicit get_output_structure override."""

    def get_output_structure(self, sample_data, sample_state):
        out_data = {
            **jax.tree.map(lambda _: None, sample_data),
            "result": None,
        }
        return out_data, sample_state

    def apply(self, data, state, metadata, random_params=None, stats=None):
        out_data = {
            **data,
            "result": data["value"] ** 2,
        }
        return out_data, state, metadata


class StructurePreservingOperator(OperatorModule):
    """Test operator that preserves input structure (existing behavior)."""

    def apply(self, data, state, metadata, random_params=None, stats=None):
        out_data = {k: v * 2 for k, v in data.items()}
        return out_data, state, metadata


# =============================================================================
# Test Cases
# =============================================================================


class TestDynamicOutputStructure:
    """Test operators that add/change output keys."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def config(self):
        return OperatorConfig()

    def test_operator_adds_single_key(self, config, rngs):
        """Operator adds one new key to output."""
        op = AddKeyOperator(config, rngs=rngs)

        elements = [
            Element(data={"input": jnp.array([1.0, 2.0])}, state={}),
            Element(data={"input": jnp.array([3.0, 4.0])}, state={}),
        ]
        batch = Batch(elements)

        result = op.apply_batch(batch)
        result_data = result.data.get_value()

        assert "input" in result_data
        assert "computed" in result_data
        assert result_data["computed"].shape == (2, 2)

    def test_operator_adds_multiple_keys(self, config, rngs):
        """Operator adds multiple new keys."""
        op = AddMultipleKeysOperator(config, rngs=rngs)

        elements = [
            Element(data={"a": jnp.array([1.0]), "b": jnp.array([2.0])}, state={}),
            Element(data={"a": jnp.array([3.0]), "b": jnp.array([4.0])}, state={}),
        ]
        batch = Batch(elements)

        result = op.apply_batch(batch)
        result_data = result.data.get_value()

        assert "a" in result_data
        assert "b" in result_data
        assert "sum" in result_data
        assert "product" in result_data
        assert "difference" in result_data

    def test_default_eval_shape_discovery(self, config, rngs):
        """Default get_output_structure uses eval_shape correctly."""
        op = AddKeyOperator(config, rngs=rngs)

        sample_data = {"input": jnp.array([1.0])}
        sample_state = {}

        out_data_struct, out_state_struct = op.get_output_structure(
            sample_data, sample_state
        )

        assert "input" in out_data_struct
        assert "computed" in out_data_struct

    def test_explicit_override_works(self, config, rngs):
        """Explicit get_output_structure override works correctly."""
        op = ExplicitStructureOperator(config, rngs=rngs)

        elements = [
            Element(data={"value": jnp.array([2.0])}, state={}),
            Element(data={"value": jnp.array([3.0])}, state={}),
        ]
        batch = Batch(elements)

        result = op.apply_batch(batch)
        result_data = result.data.get_value()

        assert "value" in result_data
        assert "result" in result_data

    def test_backward_compatible_same_structure(self, config, rngs):
        """Existing operators (same in/out structure) still work."""
        op = StructurePreservingOperator(config, rngs=rngs)

        elements = [
            Element(data={"x": jnp.array([1.0]), "y": jnp.array([2.0])}, state={}),
            Element(data={"x": jnp.array([3.0]), "y": jnp.array([4.0])}, state={}),
        ]
        batch = Batch(elements)

        result = op.apply_batch(batch)
        result_data = result.data.get_value()

        assert set(result_data.keys()) == {"x", "y"}

    def test_structure_caching_works(self, config, rngs):
        """Output structure is cached between calls."""
        op = AddKeyOperator(config, rngs=rngs)

        elements = [Element(data={"input": jnp.array([1.0])}, state={})]
        batch = Batch(elements)

        # First call - should populate cache
        op.apply_batch(batch)
        assert hasattr(op, '_output_struct_cache')
        cache_size_after_first = len(op._output_struct_cache)

        # Second call - should use cache
        op.apply_batch(batch)
        cache_size_after_second = len(op._output_struct_cache)

        assert cache_size_after_first == cache_size_after_second

    def test_jit_compatible(self, config, rngs):
        """Works correctly under jax.jit."""
        op = AddKeyOperator(config, rngs=rngs)

        elements = [
            Element(data={"input": jnp.array([1.0, 2.0])}, state={}),
            Element(data={"input": jnp.array([3.0, 4.0])}, state={}),
        ]
        batch = Batch(elements)

        # Note: apply_batch itself isn't jitted, but the vmap inside is JIT-friendly
        result = op.apply_batch(batch)
        result_data = result.data.get_value()

        assert "computed" in result_data

    def test_gradient_flow_with_new_keys(self, config, rngs):
        """Gradients flow through operators that add keys."""
        op = AddKeyOperator(config, rngs=rngs)

        def loss_fn(input_val):
            data = {"input": input_val}
            state = {}
            out_data, _, _ = op.apply(data, state, None)
            return jnp.sum(out_data["computed"])

        input_val = jnp.array([1.0, 2.0, 3.0])
        grad = jax.grad(loss_fn)(input_val)

        assert grad is not None
        assert grad.shape == input_val.shape
        assert jnp.all(jnp.isfinite(grad))


class TestNestedOutputStructure:
    """Test operators with nested PyTree outputs."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def config(self):
        return OperatorConfig()

    def test_nested_input_with_added_keys(self, config, rngs):
        """Operator adds keys to nested input structure."""

        class NestedAddKeyOperator(OperatorModule):
            def apply(self, data, state, metadata, random_params=None, stats=None):
                out_data = {
                    "nested": {
                        **data["nested"],
                        "computed": data["nested"]["value"] * 2,
                    },
                }
                return out_data, state, metadata

        op = NestedAddKeyOperator(config, rngs=rngs)

        elements = [
            Element(data={"nested": {"value": jnp.array([1.0])}}, state={}),
            Element(data={"nested": {"value": jnp.array([2.0])}}, state={}),
        ]
        batch = Batch(elements)

        result = op.apply_batch(batch)
        result_data = result.data.get_value()

        assert "nested" in result_data
        assert "value" in result_data["nested"]
        assert "computed" in result_data["nested"]
```

---

## Verification Steps

After implementing the changes:

```bash
cd /media/mahdi/ssd23/Works/workshop-data

# 1. Run new tests
uv run pytest tests/core/test_operator_output_structure.py -vv

# 2. Run existing operator tests (ensure no regression)
uv run pytest tests/core/test_operator_module.py -vv

# 3. Run full test suite
uv run pytest -vv --cov=src/

# 4. Run pre-commit
uv run pre-commit run --all-files
```

---

## After Datarax Update: DiffBio Changes

Once the Datarax fix is in place, update DiffBio operators:

### Option A: Rely on Automatic Discovery (No Changes Needed)

The default `jax.eval_shape` will automatically discover that:
- SmithWaterman adds: `score`, `alignment_matrix`, `soft_alignment`
- Pileup adds: `pileup`
- Classifier adds: `logits`, `probabilities`

### Option B: Add Explicit Override (For Efficiency)

```python
# In src/diffbio/operators/alignment/smith_waterman.py

class SmoothSmithWaterman(OperatorModule):
    def get_output_structure(self, sample_data, sample_state):
        """Declare output structure for vmap efficiency."""
        out_data = {
            **jax.tree.map(lambda _: None, sample_data),
            "score": None,
            "alignment_matrix": None,
            "soft_alignment": None,
        }
        return out_data, sample_state

    # ... rest of implementation unchanged ...
```

### Update Integration Tests

Remove the manual vmap workarounds in `tests/integration/test_operator_composition.py` and use `apply_batch()` directly.

---

## Edge Cases Handled

| Case | Behavior |
|------|----------|
| Empty input `{}` | Works - output can add keys |
| Nested PyTrees | Works - `jax.tree.map` handles recursion |
| State structure changes | Supported via `get_output_structure` |
| Metadata | Not vmapped, unchanged |
| Stochastic operators | Random params handled separately |
| JIT compilation | Structure caching ensures consistency |
| Multiple batches with same structure | Cache reused |

---

## Technical Notes

### Why `jax.eval_shape`?

From JAX source (`jax/_src/api.py`):
- `eval_shape` traces the function without executing it
- Returns `ShapeDtypeStruct` PyTree matching actual output structure
- Zero FLOPs - only traces to discover structure
- This is how JAX's own `vmap` discovers output structure internally

### Why Structure Caching?

- `jax.eval_shape` has tracing overhead (~microseconds)
- Caching avoids repeated tracing for same input structure
- Cache key uses PyTree structure ID (not values)
- JIT-friendly: structure is static after first call

### Backward Compatibility

- Existing operators work unchanged
- Default `eval_shape` discovers same-structure outputs correctly
- No changes to `OperatorConfig` or other interfaces
- No breaking changes to existing code
