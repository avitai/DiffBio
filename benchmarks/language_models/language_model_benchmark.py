#!/usr/bin/env python3
"""Language Model Benchmark for DiffBio.

This benchmark evaluates DiffBio's biological language model operators:
- TransformerSequenceEncoder (DNA/RNA transformer encoding)
- DifferentiableFoundationModel (single-cell foundation model)

Metrics:
- Output shape correctness and value finiteness
- Gradient flow for both operators
- Throughput (sequences/second for transformer, cells/second for
  foundation model)

Usage:
    python benchmarks/language_models/language_model_benchmark.py
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime

import jax
import jax.numpy as jnp
from flax import nnx

from benchmarks._common import (
    check_gradient_flow,
    generate_synthetic_expression,
    generate_synthetic_sequences,
    measure_throughput,
    save_benchmark_result,
)
from diffbio.operators.language_models.foundation_model import (
    DifferentiableFoundationModel,
    FoundationModelConfig,
)
from diffbio.operators.language_models.transformer_encoder import (
    TransformerSequenceEncoder,
    TransformerSequenceEncoderConfig,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class LanguageModelBenchmarkResult:
    """Results from the language model benchmark suite.

    Attributes:
        timestamp: ISO-formatted timestamp of the run.
        n_sequences: Number of sequences used for transformer
            benchmark.
        seq_len: Length of each input sequence.
        n_cells: Number of cells used for foundation model
            benchmark.
        n_genes: Number of genes used for foundation model
            benchmark.
        transformer_shape_ok: Whether transformer embedding output
            has the correct shape.
        transformer_values_finite: Whether all transformer output
            values are finite.
        foundation_shape_ok: Whether foundation model embedding
            output has the correct shape.
        foundation_values_finite: Whether all foundation model
            output values are finite.
        transformer_gradient: Gradient flow results for the
            transformer encoder.
        foundation_gradient: Gradient flow results for the
            foundation model.
        transformer_throughput: Throughput metrics for the
            transformer encoder.
        foundation_throughput: Throughput metrics for the
            foundation model.
    """

    timestamp: str
    n_sequences: int
    seq_len: int
    n_cells: int
    n_genes: int
    # Shape correctness
    transformer_shape_ok: bool
    transformer_values_finite: bool
    foundation_shape_ok: bool
    foundation_values_finite: bool
    # Gradient flow
    transformer_gradient: dict[str, float | bool] = field(
        default_factory=dict,
    )
    foundation_gradient: dict[str, float | bool] = field(
        default_factory=dict,
    )
    # Throughput
    transformer_throughput: dict[str, float] = field(
        default_factory=dict,
    )
    foundation_throughput: dict[str, float] = field(
        default_factory=dict,
    )


# ------------------------------------------------------------------
# Operator tests
# ------------------------------------------------------------------


def _test_transformer(
    sequences: jnp.ndarray,
    hidden_dim: int,
) -> tuple[dict[str, bool], TransformerSequenceEncoder]:
    """Test TransformerSequenceEncoder on synthetic one-hot sequences.

    Args:
        sequences: One-hot encoded sequences of shape
            ``(n_seqs, seq_len, alphabet_size)``.
        hidden_dim: Hidden dimension for the encoder.

    Returns:
        Tuple of (metrics dict, operator instance).
    """
    n_seqs, seq_len, alphabet_size = sequences.shape

    config = TransformerSequenceEncoderConfig(
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
        intermediate_dim=4 * hidden_dim,
        max_length=seq_len,
        alphabet_size=alphabet_size,
        dropout_rate=0.0,
        pooling="mean",
    )
    rngs = nnx.Rngs(params=0, dropout=1)
    encoder = TransformerSequenceEncoder(config, rngs=rngs)

    data = {"sequence": sequences}
    result, _, _ = encoder.apply(data, {}, None)

    embedding = result["embedding"]
    position_embeddings = result["position_embeddings"]

    shape_ok = (
        embedding.shape == (n_seqs, hidden_dim)
        and position_embeddings.shape
        == (n_seqs, seq_len, hidden_dim)
    )
    values_finite = bool(
        jnp.all(jnp.isfinite(embedding))
        and jnp.all(jnp.isfinite(position_embeddings))
    )

    return {
        "shape_ok": shape_ok,
        "values_finite": values_finite,
    }, encoder


def _test_foundation_model(
    counts: jnp.ndarray,
    n_genes: int,
) -> tuple[dict[str, bool], DifferentiableFoundationModel]:
    """Test DifferentiableFoundationModel on synthetic expression data.

    Args:
        counts: Gene expression matrix of shape
            ``(n_cells, n_genes)``.
        n_genes: Number of genes.

    Returns:
        Tuple of (metrics dict, operator instance).
    """
    n_cells = counts.shape[0]

    config = FoundationModelConfig(
        n_genes=n_genes,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        mask_ratio=0.0,
        dropout_rate=0.0,
    )
    rngs = nnx.Rngs(params=0, sample=1, dropout=2)
    model = DifferentiableFoundationModel(config, rngs=rngs)

    gene_ids = jnp.arange(n_genes)
    data = {"counts": counts, "gene_ids": gene_ids}
    result, _, _ = model.apply(data, {}, None)

    cell_embeddings = result["cell_embeddings"]
    gene_embeddings = result["gene_embeddings"]
    predicted = result["predicted_expression"]

    shape_ok = (
        cell_embeddings.shape == (n_cells, config.hidden_dim)
        and gene_embeddings.shape == (n_genes, config.hidden_dim)
        and predicted.shape == (n_cells, n_genes)
    )
    values_finite = bool(
        jnp.all(jnp.isfinite(cell_embeddings))
        and jnp.all(jnp.isfinite(gene_embeddings))
        and jnp.all(jnp.isfinite(predicted))
    )

    return {
        "shape_ok": shape_ok,
        "values_finite": values_finite,
    }, model


# ------------------------------------------------------------------
# Main benchmark
# ------------------------------------------------------------------


def run_benchmark(
    *, quick: bool = False,
) -> LanguageModelBenchmarkResult:
    """Run the complete language model benchmark.

    Args:
        quick: If True, use smaller data for faster execution.

    Returns:
        Benchmark results dataclass.
    """
    # Data dimensions
    n_sequences = 10 if quick else 20
    seq_len = 30 if quick else 50
    n_cells = 30 if quick else 100
    n_genes = 20 if quick else 50
    hidden_dim = 64
    n_throughput_iters = 20 if quick else 100

    print("=" * 60)
    print("DiffBio Language Model Benchmark")
    print("=" * 60)
    print(f"  Sequences       : {n_sequences} x {seq_len}")
    print(f"  Cells x genes   : {n_cells} x {n_genes}")
    print(f"  Quick mode      : {quick}")

    # ----- Synthetic data -----
    print("\nGenerating synthetic data...")
    sequences = generate_synthetic_sequences(
        n_seqs=n_sequences,
        seq_len=seq_len,
        alphabet_size=4,
        seed=42,
    )

    expr_data = generate_synthetic_expression(
        n_cells=n_cells,
        n_genes=n_genes,
        seed=42,
    )
    counts = expr_data["counts"]

    # ----- Transformer Encoder -----
    print("\nTesting TransformerSequenceEncoder...")
    tx_metrics, encoder = _test_transformer(sequences, hidden_dim)
    print(f"  Shape OK        : {tx_metrics['shape_ok']}")
    print(f"  Values finite   : {tx_metrics['values_finite']}")

    # ----- Foundation Model -----
    print("\nTesting DifferentiableFoundationModel...")
    fm_metrics, foundation = _test_foundation_model(
        counts, n_genes,
    )
    print(f"  Shape OK        : {fm_metrics['shape_ok']}")
    print(f"  Values finite   : {fm_metrics['values_finite']}")

    # ----- Gradient flow -----
    print("\nChecking gradient flow...")

    def _tx_loss(
        model: TransformerSequenceEncoder,
    ) -> jax.Array:
        """Loss for transformer gradient check."""
        out, _, _ = model.apply(
            {"sequence": sequences}, {}, None,
        )
        return jnp.sum(out["embedding"])

    tx_grad = check_gradient_flow(_tx_loss, encoder)
    print(
        f"  Transformer      : norm={tx_grad.gradient_norm:.6f}"
        f"  nonzero={tx_grad.gradient_nonzero}"
    )

    gene_ids = jnp.arange(n_genes)

    def _fm_loss(
        model: DifferentiableFoundationModel,
    ) -> jax.Array:
        """Loss for foundation model gradient check."""
        out, _, _ = model.apply(
            {"counts": counts, "gene_ids": gene_ids},
            {},
            None,
        )
        return jnp.sum(out["cell_embeddings"])

    fm_grad = check_gradient_flow(_fm_loss, foundation)
    print(
        f"  Foundation       : norm={fm_grad.gradient_norm:.6f}"
        f"  nonzero={fm_grad.gradient_nonzero}"
    )

    # ----- Throughput -----
    print("\nMeasuring throughput...")

    tx_tp = measure_throughput(
        lambda: encoder.apply(
            {"sequence": sequences}, {}, None,
        ),
        args=(),
        n_iterations=n_throughput_iters,
        warmup=3,
    )
    tx_seqs_per_sec = n_sequences * tx_tp["items_per_sec"]
    print(
        f"  Transformer      : {tx_seqs_per_sec:.0f} seqs/s"
        f"  ({tx_tp['per_item_ms']:.2f} ms/call)"
    )

    fm_tp = measure_throughput(
        lambda: foundation.apply(
            {"counts": counts, "gene_ids": gene_ids},
            {},
            None,
        ),
        args=(),
        n_iterations=n_throughput_iters,
        warmup=3,
    )
    fm_cells_per_sec = n_cells * fm_tp["items_per_sec"]
    print(
        f"  Foundation       : {fm_cells_per_sec:.0f} cells/s"
        f"  ({fm_tp['per_item_ms']:.2f} ms/call)"
    )

    # ----- Compile result -----
    result = LanguageModelBenchmarkResult(
        timestamp=datetime.now().isoformat(),
        n_sequences=n_sequences,
        seq_len=seq_len,
        n_cells=n_cells,
        n_genes=n_genes,
        transformer_shape_ok=tx_metrics["shape_ok"],
        transformer_values_finite=tx_metrics["values_finite"],
        foundation_shape_ok=fm_metrics["shape_ok"],
        foundation_values_finite=fm_metrics["values_finite"],
        transformer_gradient=tx_grad,
        foundation_gradient=fm_grad,
        transformer_throughput={
            **tx_tp,
            "sequences_per_sec": tx_seqs_per_sec,
        },
        foundation_throughput={
            **fm_tp,
            "cells_per_sec": fm_cells_per_sec,
        },
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(
        f"  Transformer shapes OK   : {tx_metrics['shape_ok']}"
    )
    print(
        f"  Foundation shapes OK    : {fm_metrics['shape_ok']}"
    )
    all_values_finite = tx_metrics["values_finite"] and fm_metrics["values_finite"]
    all_grads_nonzero = tx_grad.gradient_nonzero and fm_grad.gradient_nonzero
    print(f"  All values finite       : {all_values_finite}")
    print(f"  All gradients nonzero   : {all_grads_nonzero}")
    print("=" * 60)

    return result


def main() -> None:
    """Entry point for the language model benchmark."""
    result = run_benchmark()
    output_path = save_benchmark_result(
        asdict(result),
        domain="language_models",
        benchmark_name="language_model_benchmark",
    )
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
