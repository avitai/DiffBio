#!/usr/bin/env python3
"""VAE integration benchmark: VAENormalizer on immune_human.

Trains DiffBio's VAENormalizer with ZINB likelihood on the scib
immune human dataset (33,506 cells, 10 batches, 16 cell types),
extracts latent representations, and evaluates with scib-metrics.

Results are compared against published baselines (scVI, scANVI,
Harmony, Scanorama) from Luecken et al. 2022.

Usage:
    python benchmarks/singlecell/bench_vae_integration.py
    python benchmarks/singlecell/bench_vae_integration.py --quick
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from calibrax.core.models import Metric
from calibrax.core.result import BenchmarkResult
from calibrax.profiling.timing import TimingCollector
from flax import nnx

from benchmarks._baselines.scib import INTEGRATION_BASELINES
from benchmarks._gradient import check_gradient_flow
from benchmarks._metrics.scib_bridge import evaluate_integration
from diffbio.operators.normalization.vae_normalizer import (
    VAENormalizer,
    VAENormalizerConfig,
)
from diffbio.sources.immune_human import (
    ImmuneHumanConfig,
    ImmuneHumanSource,
)

logger = logging.getLogger(__name__)


def _create_train_step(
    model: VAENormalizer,
    optimizer: nnx.Optimizer,
) -> Any:
    """Create a JIT-compiled training step for the VAE.

    Args:
        model: VAENormalizer to train.
        optimizer: NNX optimizer wrapping the model.

    Returns:
        JIT-compiled train_step function.
    """

    @nnx.jit
    def train_step(
        m: VAENormalizer,
        opt: nnx.Optimizer,
        counts_batch: jax.Array,
        library_size_batch: jax.Array,
    ) -> jax.Array:
        """Single training step computing mean ELBO loss.

        Args:
            m: VAENormalizer model.
            opt: NNX optimizer.
            counts_batch: Count vectors (n_cells, n_genes).
            library_size_batch: Library sizes (n_cells,).

        Returns:
            Scalar loss value.
        """

        def loss_fn(
            model_inner: VAENormalizer,
        ) -> jax.Array:
            """Compute mean ELBO loss over batch."""

            def per_cell_loss(
                counts_i: jax.Array,
                lib_i: jax.Array,
            ) -> jax.Array:
                return model_inner.compute_elbo_loss(
                    counts_i, lib_i
                )

            losses = jax.vmap(per_cell_loss)(
                counts_batch, library_size_batch
            )
            return jnp.mean(losses)

        loss, grads = nnx.value_and_grad(
            loss_fn, argnums=nnx.DiffState(0, nnx.Param)
        )(m)
        opt.update(m, grads)
        return loss

    return train_step


def _extract_latent_means(
    model: VAENormalizer,
    counts: jax.Array,
) -> jax.Array:
    """Extract latent means for all cells via vmap.

    Uses the encoder to get (mean, logvar), then returns means
    only (no sampling noise for evaluation).

    Args:
        model: Trained VAENormalizer.
        counts: Count matrix (n_cells, n_genes).

    Returns:
        Latent means of shape (n_cells, latent_dim).
    """

    def encode_cell(
        counts_i: jax.Array,
    ) -> jax.Array:
        """Encode a single cell, return mean only."""
        mean, _ = model.encode(counts_i)
        return mean

    return jax.vmap(encode_cell)(counts)


def _compute_reconstruction_mse(
    model: VAENormalizer,
    counts: jax.Array,
    library_size: jax.Array,
) -> float:
    """Compute mean squared reconstruction error.

    Encodes to latent means (no noise), decodes, and compares
    reconstructed rates to original counts.

    Args:
        model: Trained VAENormalizer.
        counts: Count matrix (n_cells, n_genes).
        library_size: Library sizes (n_cells,).

    Returns:
        Scalar MSE value.
    """

    def reconstruct_cell(
        counts_i: jax.Array,
        lib_i: jax.Array,
    ) -> jax.Array:
        """Encode and decode a single cell."""
        mean, _ = model.encode(counts_i)
        decode_out = model.decode(mean, lib_i)
        return jnp.exp(decode_out["log_rate"])

    reconstructed = jax.vmap(reconstruct_cell)(
        counts, library_size
    )
    return float(jnp.mean((counts - reconstructed) ** 2))


class VAEIntegrationBenchmark:
    """Evaluate VAENormalizer on the scib immune human dataset.

    Trains a VAE with ZINB likelihood, extracts latent
    representations, and computes the full scib-metrics integration
    suite. Compares against published baselines.

    Args:
        quick: If True, subsample to 2000 cells and train 10 epochs.
        data_dir: Directory containing the downloaded h5ad file.
    """

    def __init__(
        self,
        *,
        quick: bool = False,
        data_dir: str = "/media/mahdi/ssd23/Data/scib",
    ) -> None:
        self.quick = quick
        self.data_dir = data_dir

    def run(self) -> BenchmarkResult:
        """Execute the benchmark and return a calibrax result."""
        subsample = 2000 if self.quick else None
        n_epochs = 10 if self.quick else 50

        # 1. Load dataset via DataSource
        print("Loading immune_human dataset...")
        source_config = ImmuneHumanConfig(
            data_dir=self.data_dir,
            subsample=subsample,
        )
        source = ImmuneHumanSource(source_config)
        data = source.load()

        n_cells = data["n_cells"]
        n_genes = data["n_genes"]
        n_batches = data["n_batches"]
        n_types = data["n_types"]
        counts = data["counts"]
        batch_labels = data["batch_labels"]
        cell_type_labels = data["cell_type_labels"]

        library_size = jnp.sum(counts, axis=-1)

        print(
            f"  {n_cells} cells, {n_genes} genes, "
            f"{n_batches} batches, {n_types} types"
        )

        # 2. Create VAENormalizer with ZINB likelihood
        vae_config = VAENormalizerConfig(
            n_genes=n_genes,
            latent_dim=10,
            hidden_dims=[128],
            likelihood="zinb",
        )
        rngs = nnx.Rngs(42)
        model = VAENormalizer(vae_config, rngs=rngs)

        # 3. Train with nnx.Optimizer + optax.adam
        print(f"Training VAENormalizer ({n_epochs} epochs)...")
        optimizer = nnx.Optimizer(
            model, optax.adam(1e-3), wrt=nnx.Param
        )
        train_step = _create_train_step(model, optimizer)

        start = time.perf_counter()
        loss = jnp.array(0.0)
        for epoch in range(n_epochs):
            loss = train_step(
                model, optimizer, counts, library_size
            )
            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(f"  epoch {epoch + 1}/{n_epochs} "
                      f"loss={float(loss):.2f}")
        train_time = time.perf_counter() - start
        print(f"  Training completed in {train_time:.2f}s")

        # 4. Extract latent means (no sampling noise)
        print("Extracting latent representations...")
        latent_means = _extract_latent_means(model, counts)

        # 5. Compute ELBO and reconstruction MSE
        elbo_value = float(loss)
        recon_mse = _compute_reconstruction_mse(
            model, counts, library_size
        )
        print(f"  ELBO: {elbo_value:.2f}")
        print(f"  Reconstruction MSE: {recon_mse:.4f}")

        # 6. Compute full scib-metrics on latent space
        print("Computing scib-metrics...")
        quality = evaluate_integration(
            corrected_embeddings=np.asarray(latent_means),
            labels=np.asarray(cell_type_labels),
            batch=np.asarray(batch_labels),
        )

        for key, value in sorted(quality.items()):
            print(f"  {key}: {value:.4f}")

        # 7. Check gradient flow
        print("Checking gradient flow...")

        def loss_fn(
            m: VAENormalizer,
            c: jax.Array,
            ls: jax.Array,
        ) -> jnp.ndarray:
            """Scalar loss for gradient verification."""

            def per_cell(
                ci: jax.Array, li: jax.Array
            ) -> jax.Array:
                return m.compute_elbo_loss(ci, li)

            return jnp.mean(jax.vmap(per_cell)(c, ls))

        grad = check_gradient_flow(
            loss_fn, model, counts, library_size
        )
        print(
            f"  Gradient norm: {grad.gradient_norm:.4f}, "
            f"nonzero: {grad.gradient_nonzero}"
        )

        # 8. Measure throughput
        print("Measuring throughput...")
        n_timing_iters = n_epochs
        collector = TimingCollector(warmup_iterations=3)

        def _forward_pass(_: Any) -> Any:
            """Run a single forward pass for timing."""
            return jax.vmap(
                lambda ci, li: model.compute_elbo_loss(ci, li)
            )(counts, library_size)

        timing = collector.measure_iteration(
            iterator=iter(range(n_timing_iters)),
            num_batches=n_timing_iters,
            process_fn=_forward_pass,
            count_fn=lambda _: n_cells,
        )
        cells_per_sec = timing.num_elements / timing.wall_clock_sec
        print(f"  {cells_per_sec:.0f} cells/sec")

        # 9. Print comparison table vs baselines
        baselines = INTEGRATION_BASELINES.get("immune_human", {})
        print("\nComparison Table:")
        header = (
            f"{'Method':<20} {'Aggregate':>10} {'Sil.Lab':>8}"
        )
        print(header)
        print("-" * len(header))
        agg = quality["aggregate_score"]
        sil = quality["silhouette_label"]
        print(
            f"{'DiffBio VAE':<20} {agg:>10.4f} {sil:>8.4f}"
        )
        for name, point in baselines.items():
            b_agg = point.metrics.get(
                "aggregate_score", Metric(value=0)
            ).value
            b_sil = point.metrics.get(
                "silhouette_label", Metric(value=0)
            ).value
            print(f"{name:<20} {b_agg:>10.4f} {b_sil:>8.4f}")

        # 10. Build calibrax BenchmarkResult
        metrics = {
            k: Metric(value=v) for k, v in quality.items()
        }
        metrics["elbo"] = Metric(value=elbo_value)
        metrics["reconstruction_mse"] = Metric(value=recon_mse)
        metrics["gradient_norm"] = Metric(
            value=grad.gradient_norm
        )
        metrics["gradient_nonzero"] = Metric(
            value=1.0 if grad.gradient_nonzero else 0.0
        )
        metrics["cells_per_sec"] = Metric(value=cells_per_sec)

        return BenchmarkResult(
            name="singlecell/vae_integration",
            domain="diffbio_benchmarks",
            tags={
                "operator": "VAENormalizer",
                "dataset": "immune_human",
                "framework": "diffbio",
            },
            timing=timing,
            metrics=metrics,
            config={
                "n_genes": n_genes,
                "latent_dim": vae_config.latent_dim,
                "hidden_dims": vae_config.hidden_dims,
                "likelihood": vae_config.likelihood,
                "n_epochs": n_epochs,
                "quick": self.quick,
                "subsample": subsample,
            },
            metadata={
                "dataset_info": {
                    "name": "immune_human",
                    "n_cells": n_cells,
                    "n_genes": n_genes,
                    "n_batches": n_batches,
                    "n_types": n_types,
                },
            },
        )


def main() -> None:
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)
    quick = "--quick" in sys.argv

    print("=" * 60)
    print("DiffBio Benchmark: VAE Integration")
    print(
        f"Mode: {'quick (2K cells)' if quick else 'full'}"
    )
    print("=" * 60)

    bench = VAEIntegrationBenchmark(quick=quick)
    result = bench.run()

    # Save result
    from pathlib import Path  # noqa: PLC0415

    output_dir = Path("benchmarks/results/singlecell")
    output_dir.mkdir(parents=True, exist_ok=True)
    result.save(output_dir / "vae_integration.json")
    print(
        f"\nResult saved to: "
        f"{output_dir / 'vae_integration.json'}"
    )

    print("\n" + "=" * 60)
    print(
        f"Aggregate score: "
        f"{result.metrics['aggregate_score'].value:.4f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
