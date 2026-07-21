"""Computational cost of frozen vs jointly optimized preprocessing.

Measures wall-clock time per training step and peak GPU memory for the pipeline arms, to
quantify the claim that freezing the preprocessing statistics and mini-batching keeps the
joint-optimization overhead modest and bounds peak memory to one batch rather than the whole
atlas. One configuration is measured per process (via command-line arguments) so the reported
peak memory is clean; ``_run_compute_cost.sh`` drives the grid and collects the JSON lines.

Arms:
- ``frozen``: the classifier alone over precomputed $k$-dimensional PCA features (mini-batch).
- ``joint_minibatch``: the PCA-initialized learnable projection plus classifier over the full
  feature matrix, mini-batched with frozen fit-once statistics -- the paper's design.
- ``joint_fullbatch``: the same joint model trained full-batch (one step over all cells), i.e.
  back-propagating through the whole batch at once -- the naive alternative whose activation
  memory scales with the number of cells.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from benchmarks.singlecell._gate2_arms import (
    _ProjectionProbe,
    _embedding_probe,
    _probe_forward,
    _project_probe_forward,
)
from diffbio.operators.normalization.learnable_projection import (
    LearnableProjection,
    LearnableProjectionConfig,
)
from diffbio.utils.training import cross_entropy_loss

_HIDDEN = 128


def _build(arm: str, n_features: int, n_components: int, n_classes: int, seed: int):
    """Return ``(model, forward_fn)`` for the requested arm."""
    if arm == "frozen":
        probe = _embedding_probe(n_components, n_classes, _HIDDEN, seed)
        return probe, _probe_forward
    projection = LearnableProjection(
        LearnableProjectionConfig(n_genes=n_features, n_components=n_components),
        init_loadings=jnp.asarray(
            np.random.default_rng(seed).standard_normal((n_features, n_components)) * 0.01,
            dtype=jnp.float32,
        ),
        rngs=nnx.Rngs(seed),
    )
    model = _ProjectionProbe(projection, _embedding_probe(n_components, n_classes, _HIDDEN, seed))
    return model, _project_probe_forward


def _peak_gpu_mb() -> float:
    """Return peak GPU bytes-in-use since process start, in MiB (0 if unavailable)."""
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:  # noqa: BLE001 - memory stats are backend-specific; absence is non-fatal
        return 0.0
    return float(stats.get("peak_bytes_in_use", 0)) / (1024**2)


def measure(
    *,
    arm: str,
    n_cells: int,
    n_features: int,
    n_components: int,
    n_classes: int,
    batch_size: int | None,
    steps: int,
    seed: int = 0,
) -> dict[str, object]:
    """Time ``steps`` training steps of one arm and report peak GPU memory."""
    rng = np.random.default_rng(seed)
    feature_dim = n_components if arm == "frozen" else n_features
    effective_batch = n_cells if batch_size is None else batch_size
    # Keep the dataset on the host and place only the resident batch on the device, so peak
    # device memory reflects the model plus one batch of activations/gradients -- bounded by
    # the batch for mini-batch training, and by the whole atlas for full-batch training.
    features = jax.device_put(
        rng.standard_normal((effective_batch, feature_dim)).astype(np.float32)
    )
    labels = jax.device_put(rng.integers(0, n_classes, size=effective_batch).astype(np.int32))
    model, forward = _build(arm, n_features, n_components, n_classes, seed)
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)

    # donate_argnames lets XLA reuse the model/optimizer buffers in place (Flax NNX
    # best practice), so peak memory reflects realistic in-place training, not double
    # buffering.
    @nnx.jit(donate_argnames=("model", "opt"))
    def train_step(model: nnx.Module, opt: nnx.Optimizer, x: jnp.ndarray, y: jnp.ndarray):
        def loss_fn(m: nnx.Module) -> jax.Array:
            return cross_entropy_loss(forward(m, x), y, num_classes=n_classes)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    xb, yb = features, labels
    # Warm up (compilation is the slow first call) and block before timing.
    jax.block_until_ready(train_step(model, optimizer, xb, yb))

    start = time.perf_counter()
    for _ in range(steps):
        loss = train_step(model, optimizer, xb, yb)
    # Block on both the loss and the in-place-updated model state (async dispatch).
    jax.block_until_ready((loss, nnx.state(model)))
    ms_per_step = 1000.0 * (time.perf_counter() - start) / steps

    return {
        "arm": arm,
        "n_cells": n_cells,
        "n_features": n_features,
        "n_components": n_components,
        "batch_size": effective_batch,
        "ms_per_step": round(ms_per_step, 3),
        "peak_gpu_mb": round(_peak_gpu_mb(), 1),
    }


def main() -> None:
    """Measure one configuration and print it as a JSON line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=("frozen", "joint_minibatch", "joint_fullbatch"), required=True
    )
    parser.add_argument("--n-cells", type=int, default=100_000)
    parser.add_argument("--n-features", type=int, default=2000)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--n-classes", type=int, default=59)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    arm = "joint_minibatch" if args.arm.startswith("joint") else args.arm
    batch = None if args.arm == "joint_fullbatch" else args.batch_size
    result = measure(
        arm=arm,
        n_cells=args.n_cells,
        n_features=args.n_features,
        n_components=args.n_components,
        n_classes=args.n_classes,
        batch_size=batch,
        steps=args.steps,
    )
    result["config"] = args.arm
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
