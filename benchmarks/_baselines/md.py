"""Published molecular dynamics baselines for throughput comparison.

Values are approximate steps/sec for Lennard-Jones systems on A100 GPU:
- jax-md: ~447 steps/sec (Schoenholz & Cubuk, NeurIPS 2020, 64K LJ)
- LAMMPS: ~500 steps/sec (GPU-accelerated, 64K LJ particles, A100)

Source references:
  Schoenholz & Cubuk, "JAX, M.D.: A Framework for Differentiable Physics"
  NeurIPS 2020. https://arxiv.org/abs/1912.04232

  Thompson et al., "LAMMPS - a flexible simulation tool for particle-based
  materials modeling at the atomic, meso, and continuum scales"
  Comp Phys Comm 271, 108171 (2022).
"""

from __future__ import annotations

from calibrax.core.models import Metric, Point

_JAX_MD_SOURCE = "Schoenholz & Cubuk, NeurIPS 2020"
_LAMMPS_SOURCE = "Thompson et al., Comp Phys Comm 2022"

MD_BASELINES: dict[str, Point] = {
    "jax-md": Point(
        name="jax-md",
        scenario="lj_64k_a100",
        tags={"framework": "jax-md", "source": _JAX_MD_SOURCE},
        metrics={
            "steps_per_sec": Metric(value=447.0),
        },
    ),
    "LAMMPS": Point(
        name="LAMMPS",
        scenario="lj_64k_a100",
        tags={"framework": "lammps-gpu", "source": _LAMMPS_SOURCE},
        metrics={
            "steps_per_sec": Metric(value=500.0),
        },
    ),
}
