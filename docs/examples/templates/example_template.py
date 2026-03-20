# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # [Example Title]
#
# **Duration:** [X minutes] | **Level:** [Basic/Intermediate/Advanced]
# | **Device:** [CPU-compatible/GPU-optional/GPU-required]
#
# ## Learning Objectives
#
# By the end of this example, you will:
# 1. Understand [bioinformatics concept]
# 2. Be able to [use DiffBio operator]
# 3. Know how to [verify differentiability]
#
# ## Prerequisites
#
# - DiffBio installed (see setup instructions)
# - Basic understanding of [domain concept]
#
# ```bash
# source ./activate.sh
# uv run python examples/path/to/example.py
# ```
#
# ---

# %%
# Environment setup
import jax
import jax.numpy as jnp
from flax import nnx

print(f"JAX version: {jax.__version__}")
print(f"Backend: {jax.default_backend()}")
print(f"Devices: {jax.device_count()}")

# %% [markdown]
# ## 1. Configuration
#
# [Explain what the operator does and what parameters control.]

# %%
# Configure the operator
from diffbio.operators.domain.module import OperatorClass, OperatorConfig

config = OperatorConfig(
    param1=value1,
    param2=value2,
)
operator = OperatorClass(config, rngs=nnx.Rngs(42))

print(f"Operator created: {type(operator).__name__}")

# %% [markdown]
# ## 2. Create Data
#
# [Explain the data format and what it represents biologically.]

# %%
# Generate synthetic data
key = jax.random.key(0)
n_cells, n_genes = 50, 100
counts = jax.random.poisson(key, jnp.ones((n_cells, n_genes)) * 3.0)

data = {"counts": counts}
print(f"Input shape: {data['counts'].shape}")
print(f"Mean expression: {data['counts'].mean():.2f}")

# %% [markdown]
# ## 3. Run the Operator
#
# [Explain what the operator computes and what outputs to expect.]

# %%
# Apply the operator
result, state, metadata = operator.apply(data, {}, None, None)

# Inspect outputs
for key_name, value in result.items():
    if hasattr(value, "shape"):
        print(f"  {key_name}: shape={value.shape}, dtype={value.dtype}")

# %% [markdown]
# ## 4. Verify Differentiability
#
# DiffBio operators are end-to-end differentiable. This is the core
# value proposition: gradients flow through bioinformatics computations.

# %%
# Gradient flow test


def loss_fn(input_data):
    """Compute a scalar loss from the operator output."""
    result, _, _ = operator.apply(input_data, {}, None, None)
    return result["output_key"].sum()


grad = jax.grad(loss_fn)(data)
print(f"Gradient shape: {grad['counts'].shape}")
print(f"Gradient is non-zero: {bool(jnp.any(grad['counts'] != 0))}")
print(f"Gradient is finite: {bool(jnp.all(jnp.isfinite(grad['counts'])))}")

# %% [markdown]
# ## 5. JIT Compilation
#
# All DiffBio operators are compatible with JAX's JIT compiler for
# accelerated execution on GPU/TPU.

# %%
# JIT-compiled forward pass
jit_apply = jax.jit(lambda d: operator.apply(d, {}, None, None))
result_jit, _, _ = jit_apply(data)
print(f"JIT output matches: {bool(jnp.allclose(result['output_key'], result_jit['output_key']))}")

# %% [markdown]
# ## 6. Experiments
#
# [Suggest parameter variations and what to observe.]

# %%
# Experiment: vary a key parameter
# TODO: Show how changing a parameter affects the output

# %% [markdown]
# ## Summary
#
# In this example, you learned:
# - How to configure and run [operator name]
# - That gradients flow through the computation (differentiability)
# - That JIT compilation works for accelerated execution
#
# ## Next Steps
#
# - [Link to related example]
# - [Link to API reference]
# - [Link to more advanced pipeline example]
