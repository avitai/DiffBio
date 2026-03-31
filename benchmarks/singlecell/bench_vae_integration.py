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
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from benchmarks._base import DiffBioBenchmark, DiffBioBenchmarkConfig
from benchmarks._baselines.scib import INTEGRATION_BASELINES
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

_CONFIG = DiffBioBenchmarkConfig(
    name="singlecell/vae_integration",
    domain="singlecell",
    quick_subsample=2000,
    n_iterations_quick=10,
    n_iterations_full=50,
)


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
                return model_inner.compute_elbo_loss(counts_i, lib_i)

            losses = jax.vmap(per_cell_loss)(counts_batch, library_size_batch)
            return jnp.mean(losses)

        loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(m)
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

    reconstructed = jax.vmap(reconstruct_cell)(counts, library_size)
    return float(jnp.mean((counts - reconstructed) ** 2))


class VAEIntegrationBenchmark(DiffBioBenchmark):
    """Evaluate VAENormalizer on the scib immune human dataset."""

    def __init__(
        self,
        config: DiffBioBenchmarkConfig = _CONFIG,
        *,
        quick: bool = False,
        data_dir: str = "/media/mahdi/ssd23/Data/scib",
    ) -> None:
        super().__init__(config, quick=quick, data_dir=data_dir)

    def _run_core(self) -> dict[str, Any]:
        """Train VAE, extract latents, compute scib-metrics."""
        subsample = self.config.quick_subsample if self.quick else None
        n_epochs = 10 if self.quick else 50

        # 1. Load dataset
        logger.info("Loading immune_human dataset...")
        source = ImmuneHumanSource(ImmuneHumanConfig(data_dir=self.data_dir, subsample=subsample))
        data = source.load()

        n_cells = data["n_cells"]
        n_genes = data["n_genes"]
        n_batches = data["n_batches"]
        n_types = data["n_types"]
        counts = data["counts"]
        batch_labels = data["batch_labels"]
        cell_type_labels = data["cell_type_labels"]

        library_size = jnp.sum(counts, axis=-1)

        logger.info(
            "  %d cells, %d genes, %d batches, %d types",
            n_cells,
            n_genes,
            n_batches,
            n_types,
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
        logger.info("Training VAENormalizer (%d epochs)...", n_epochs)
        optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
        train_step = _create_train_step(model, optimizer)

        loss = jnp.array(0.0)
        for epoch in range(n_epochs):
            loss = train_step(model, optimizer, counts, library_size)
            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                logger.info(
                    "  epoch %d/%d loss=%.2f",
                    epoch + 1,
                    n_epochs,
                    float(loss),
                )

        # 4. Extract latent means (no sampling noise)
        logger.info("Extracting latent representations...")
        latent_means = _extract_latent_means(model, counts)

        # 5. Compute ELBO and reconstruction MSE
        elbo_value = float(loss)
        recon_mse = _compute_reconstruction_mse(model, counts, library_size)
        logger.info("  ELBO: %.2f, Recon MSE: %.4f", elbo_value, recon_mse)

        # 6. Compute scib-metrics on latent space
        logger.info("Computing scib-metrics...")
        quality = evaluate_integration(
            corrected_embeddings=np.asarray(latent_means),
            labels=np.asarray(cell_type_labels),
            batch=np.asarray(batch_labels),
        )
        quality["elbo"] = elbo_value
        quality["reconstruction_mse"] = recon_mse

        for key, value in sorted(quality.items()):
            logger.info("  %s: %.4f", key, value)

        # Loss function for gradient check (dict-based)
        def loss_fn(m: VAENormalizer, d: dict[str, Any]) -> jnp.ndarray:
            """Scalar loss for gradient verification."""
            c = d["counts"]
            ls = d["library_size"]

            def per_cell(ci: jax.Array, li: jax.Array) -> jax.Array:
                return m.compute_elbo_loss(ci, li)

            return jnp.mean(jax.vmap(per_cell)(c, ls))

        grad_input = {
            "counts": counts,
            "library_size": library_size,
        }

        baselines = INTEGRATION_BASELINES.get("immune_human", {})

        return {
            "metrics": quality,
            "operator": model,
            "input_data": grad_input,
            "loss_fn": loss_fn,
            "n_items": n_cells,
            "iterate_fn": lambda: jax.vmap(lambda ci, li: model.compute_elbo_loss(ci, li))(
                counts, library_size
            ),
            "baselines": baselines,
            "dataset_info": {
                "name": "immune_human",
                "n_cells": n_cells,
                "n_genes": n_genes,
                "n_batches": n_batches,
                "n_types": n_types,
            },
            "operator_config": {
                "n_genes": n_genes,
                "latent_dim": vae_config.latent_dim,
                "hidden_dims": vae_config.hidden_dims,
                "likelihood": vae_config.likelihood,
                "n_epochs": n_epochs,
                "subsample": subsample,
            },
            "operator_name": "VAENormalizer",
            "dataset_name": "immune_human",
        }


def main() -> None:
    """CLI entry point."""
    DiffBioBenchmark.cli_main(
        VAEIntegrationBenchmark,
        _CONFIG,
        data_dir="/media/mahdi/ssd23/Data/scib",
    )


if __name__ == "__main__":
    main()
