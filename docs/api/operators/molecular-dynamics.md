# Molecular Dynamics Operators API

Differentiable operators for molecular dynamics simulations using JAX-MD.

## ForceFieldOperator

::: diffbio.operators.molecular_dynamics.force_field.ForceFieldOperator
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## ForceFieldConfig

::: diffbio.operators.molecular_dynamics.force_field.ForceFieldConfig
    options:
      show_root_heading: true
      members: []

## MDIntegratorOperator

::: diffbio.operators.molecular_dynamics.integrator.MDIntegratorOperator
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## MDIntegratorConfig

::: diffbio.operators.molecular_dynamics.integrator.MDIntegratorConfig
    options:
      show_root_heading: true
      members: []

## PotentialType

::: diffbio.operators.molecular_dynamics.primitives.PotentialType
    options:
      show_root_heading: true
      members: []

## Factory Functions

### create_force_field

::: diffbio.operators.molecular_dynamics.force_field.create_force_field
    options:
      show_root_heading: true

### create_lennard_jones_operator

::: diffbio.operators.molecular_dynamics.force_field.create_lennard_jones_operator
    options:
      show_root_heading: true

### create_integrator

::: diffbio.operators.molecular_dynamics.integrator.create_integrator
    options:
      show_root_heading: true

### create_verlet_integrator

::: diffbio.operators.molecular_dynamics.integrator.create_verlet_integrator
    options:
      show_root_heading: true

## Primitive Functions

### create_displacement_fn

::: diffbio.operators.molecular_dynamics.primitives.create_displacement_fn
    options:
      show_root_heading: true

### create_energy_fn

::: diffbio.operators.molecular_dynamics.primitives.create_energy_fn
    options:
      show_root_heading: true

### create_force_fn

::: diffbio.operators.molecular_dynamics.primitives.create_force_fn
    options:
      show_root_heading: true

## Usage Examples

### Force Field Computation

```python
from diffbio.operators.molecular_dynamics import (
    create_force_field,
    PotentialType,
)
import jax
import jax.numpy as jnp

# Create Lennard-Jones force field
force_field = create_force_field(
    potential_type=PotentialType.LENNARD_JONES,
    sigma=1.0,
    epsilon=1.0,
    box_size=10.0,
)

# Generate positions
positions = jax.random.uniform(
    jax.random.PRNGKey(0), (20, 3), minval=0, maxval=10.0
)

# Compute energy and forces
result, _, _ = force_field.apply({"positions": positions}, {}, None)
energy = result["energy"]  # scalar
forces = result["forces"]  # (20, 3)
```

### MD Simulation

```python
from diffbio.operators.molecular_dynamics import create_verlet_integrator
import jax

# Create integrator
integrator = create_verlet_integrator(
    dt=0.001,
    n_steps=1000,
    box_size=10.0,
)

# Initial conditions
key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(key)
positions = jax.random.uniform(key1, (20, 3), minval=2, maxval=8.0)
velocities = jax.random.normal(key2, (20, 3)) * 0.1

# Run simulation
result, _, _ = integrator.apply(
    {"positions": positions, "velocities": velocities}, {}, None
)
trajectory = result["trajectory"]  # (1001, 20, 3)
```

### Full Configuration

```python
from diffbio.operators.molecular_dynamics import (
    ForceFieldOperator,
    ForceFieldConfig,
    MDIntegratorOperator,
    MDIntegratorConfig,
)
from flax import nnx

# Force field with custom config
ff_config = ForceFieldConfig(
    potential_type="morse",
    sigma=1.0,
    epsilon=2.0,
    alpha=5.0,
    box_size=15.0,
)
force_field = ForceFieldOperator(ff_config, rngs=nnx.Rngs(42))

# Integrator with custom config
int_config = MDIntegratorConfig(
    integrator_type="nvt_langevin",
    dt=0.002,
    n_steps=500,
    box_size=15.0,
    kT=1.0,
    gamma=0.5,
)
integrator = MDIntegratorOperator(int_config, rngs=nnx.Rngs(42))
```

### Batched Processing

```python
import jax

# Batch of configurations
batch_size = 8
n_particles = 20
dim = 3

positions = jax.random.uniform(
    jax.random.PRNGKey(0),
    (batch_size, n_particles, dim),
    minval=0,
    maxval=10.0,
)

# Force field handles batched input
result, _, _ = force_field.apply({"positions": positions}, {}, None)
energies = result["energy"]  # (8,)
forces = result["forces"]    # (8, 20, 3)
```

### Gradient Computation

```python
import jax
from flax import nnx

force_field = create_force_field(box_size=10.0)

def loss_fn(positions):
    result, _, _ = force_field.apply({"positions": positions}, {}, None)
    return result["energy"]

# Compute gradients w.r.t. positions
grads = jax.grad(loss_fn)(positions)
```

## Input Specifications

### ForceFieldOperator

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `positions` | (n, dim) or (batch, n, dim) | float32 | Particle positions |

### MDIntegratorOperator

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `positions` | (n, dim) | float32 | Initial positions |
| `velocities` | (n, dim) | float32 | Initial velocities |

## Output Specifications

### ForceFieldOperator

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `positions` | same as input | float32 | Original positions |
| `energy` | () or (batch,) | float32 | Total potential energy |
| `forces` | same as positions | float32 | Force vectors |

### MDIntegratorOperator

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `positions` | (n, dim) | float32 | Final positions |
| `velocities` | (n, dim) | float32 | Final velocities |
| `trajectory` | (steps+1, n, dim) | float32 | Position trajectory |

## Potential Parameters

| Potential | Parameters | Description |
|-----------|------------|-------------|
| Lennard-Jones | sigma, epsilon, cutoff | Standard 12-6 potential |
| Soft Sphere | sigma, epsilon | Purely repulsive |
| Morse | sigma, epsilon, alpha | Anharmonic bonded potential |
