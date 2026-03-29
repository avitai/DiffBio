"""Task adapters mapping benchmark problems to DiffBio operator invocations.

The ``TaskAdapter`` dispatches benchmark tasks to the appropriate DiffBio
operators, executes them on the provided data, and extracts grader-ready
answers from operator outputs.

Calibrax metrics are used to compute quality scores on operator outputs
(clustering silhouette, batch correction MMD, etc.) alongside the grader
answer.
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from diffbio.evaluation.problem import BenchmarkProblem
from diffbio.sources.anndata_interop import to_grader_answer

logger = logging.getLogger(__name__)


class TaskAdapter:
    """Dispatches benchmark tasks to DiffBio operators.

    Each public method handles a specific task category, instantiates the
    appropriate operator with default or problem-specific config, runs it
    on the data dict, and returns the answer in grader-expected format.

    Args:
        seed: Random seed for operator initialisation.
    """

    def __init__(self, *, seed: int = 42) -> None:
        self._seed = seed
        self._rngs = nnx.Rngs(seed)
        self._dispatch: dict[str, Any] = {
            "qc_filtering": self._run_qc_filtering,
            "clustering": self._run_clustering,
            "differential_expression": self._run_de,
            "batch_correction": self._run_batch_correction,
            "normalization": self._run_normalization,
            "trajectory": self._run_trajectory,
            "spatial_analysis": self._run_spatial_analysis,
            "cell_annotation": self._run_cell_annotation,
        }

    def solve(
        self,
        problem: BenchmarkProblem,
        data_dict: dict[str, Any],
    ) -> Any:
        """Run the appropriate operator for a benchmark problem.

        Args:
            problem: The benchmark problem definition.
            data_dict: Operator-ready input dict (from
                ``from_anndata_to_operator_input``).

        Returns:
            Answer in the format expected by the problem's grader.

        Raises:
            ValueError: If the task_type is not supported.
        """
        handler = self._dispatch.get(problem.task_type)
        if handler is None:
            raise ValueError(
                f"Unsupported task_type {problem.task_type!r}. Supported: {sorted(self._dispatch)}"
            )
        task_config = problem.task_config
        operator_output = handler(data_dict, task_config)
        return to_grader_answer(operator_output, problem.task_type)

    def solve_with_metrics(
        self,
        problem: BenchmarkProblem,
        data_dict: dict[str, Any],
    ) -> tuple[Any, dict[str, float]]:
        """Run operator and compute calibrax quality metrics on the output.

        Like ``solve``, but also returns quality metrics computed via
        calibrax on the raw operator output (e.g., silhouette score for
        clustering, MMD for batch correction).

        Args:
            problem: The benchmark problem definition.
            data_dict: Operator-ready input dict.

        Returns:
            Tuple of (grader_answer, quality_metrics_dict).
        """
        handler = self._dispatch.get(problem.task_type)
        if handler is None:
            raise ValueError(
                f"Unsupported task_type {problem.task_type!r}. Supported: {sorted(self._dispatch)}"
            )
        operator_output = handler(data_dict, problem.task_config)
        answer = to_grader_answer(operator_output, problem.task_type)
        metrics = compute_quality_metrics(operator_output, problem.task_type, data_dict)
        return answer, metrics

    def _run_qc_filtering(
        self, data_dict: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Run quality filtering and return output dict."""
        from diffbio.operators.quality_filter import (  # noqa: PLC0415
            DifferentiableQualityFilter,
            QualityFilterConfig,
        )

        threshold = config.get("initial_threshold", 20.0)
        op_config = QualityFilterConfig(initial_threshold=threshold)
        operator = DifferentiableQualityFilter(op_config, rngs=nnx.Rngs(self._seed))

        # QC filter expects sequence + quality_scores; for count-based tasks,
        # synthesise quality scores from library size
        if "sequence" not in data_dict and "counts" in data_dict:
            counts = data_dict["counts"]
            lib_size = jnp.sum(counts, axis=1)
            # Simulate quality as log library size scaled to Phred range
            quality = jnp.log1p(lib_size) * 5.0
            n_cells = counts.shape[0]
            # Create per-cell sequence (just ones) and quality
            adapted = {
                "sequence": jnp.ones((n_cells, 1)),
                "quality_scores": quality[:, None],
            }
        else:
            adapted = data_dict

        result, _, _ = operator.apply(adapted, {}, None)
        # Propagate retention info
        if "retention_weights" not in result and "quality_scores" in adapted:
            quality = adapted["quality_scores"]
            result["retention_weights"] = jax.nn.sigmoid(quality - threshold)
        return result

    def _run_clustering(self, data_dict: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        """Run soft k-means clustering."""
        from diffbio.operators.singlecell import (  # noqa: PLC0415
            SoftClusteringConfig,
            SoftKMeansClustering,
        )

        embeddings = data_dict["embeddings"]
        n_clusters = config.get("n_clusters", 5)
        n_features = embeddings.shape[-1]
        temperature = config.get("temperature", 1.0)

        op_config = SoftClusteringConfig(
            n_clusters=n_clusters,
            n_features=n_features,
            temperature=temperature,
        )
        operator = SoftKMeansClustering(op_config, rngs=nnx.Rngs(self._seed))
        result, _, _ = operator.apply(data_dict, {}, None)
        return result

    def _run_de(self, data_dict: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        """Run differential expression pipeline."""
        from diffbio.pipelines.differential_expression import (  # noqa: PLC0415
            DEPipelineConfig,
            DifferentialExpressionPipeline,
        )

        n_genes = data_dict["counts"].shape[1]
        n_conditions = data_dict["design"].shape[1] if "design" in data_dict else 1

        op_config = DEPipelineConfig(
            n_genes=n_genes,
            n_conditions=n_conditions,
        )
        pipeline = DifferentialExpressionPipeline(op_config, rngs=nnx.Rngs(self._seed))
        result, _, _ = pipeline.apply(data_dict, {}, None)

        # Add gene names for grader extraction
        if "gene_names" not in result:
            result["gene_names"] = [f"Gene_{i}" for i in range(n_genes)]
        return result

    def _run_batch_correction(
        self, data_dict: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Run Harmony batch correction."""
        from diffbio.operators.singlecell import (  # noqa: PLC0415
            BatchCorrectionConfig,
            DifferentiableHarmony,
        )

        n_features = data_dict["embeddings"].shape[-1]
        n_clusters = config.get("n_clusters", 5)

        op_config = BatchCorrectionConfig(
            n_features=n_features,
            n_clusters=n_clusters,
        )
        operator = DifferentiableHarmony(op_config, rngs=nnx.Rngs(self._seed))
        result, _, _ = operator.apply(data_dict, {}, None)
        return result

    def _run_normalization(
        self, data_dict: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Run VAE normalization."""
        from diffbio.operators.normalization import (  # noqa: PLC0415
            VAENormalizer,
            VAENormalizerConfig,
        )

        n_genes = data_dict["counts"].shape[1]
        latent_dim = config.get("latent_dim", 10)

        op_config = VAENormalizerConfig(
            n_genes=n_genes,
            latent_dim=latent_dim,
        )
        operator = VAENormalizer(op_config, rngs=nnx.Rngs(self._seed))
        result, _, _ = operator.apply(data_dict, {}, None)
        return result

    def _run_trajectory(self, data_dict: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        """Run pseudotime trajectory inference."""
        from diffbio.operators.singlecell import (  # noqa: PLC0415
            DifferentiablePseudotime,
            PseudotimeConfig,
        )

        n_neighbors = config.get("n_neighbors", 15)
        n_diffusion_components = config.get("n_diffusion_components", 10)

        op_config = PseudotimeConfig(
            n_neighbors=n_neighbors,
            n_diffusion_components=n_diffusion_components,
        )
        operator = DifferentiablePseudotime(op_config, rngs=nnx.Rngs(self._seed))
        result, _, _ = operator.apply(data_dict, {}, None)
        return result

    def _run_spatial_analysis(
        self, data_dict: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Run spatial domain identification."""
        from diffbio.operators.singlecell import (  # noqa: PLC0415
            DifferentiableSpatialDomain,
            SpatialDomainConfig,
        )

        n_genes = data_dict["counts"].shape[1]
        n_domains = config.get("n_domains", 5)

        op_config = SpatialDomainConfig(
            n_genes=n_genes,
            n_domains=n_domains,
        )
        operator = DifferentiableSpatialDomain(op_config, rngs=nnx.Rngs(self._seed))
        result, _, _ = operator.apply(data_dict, {}, None)
        return result

    def _run_cell_annotation(
        self, data_dict: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Run cell annotation via clustering and majority vote."""
        # Use clustering as a proxy for annotation
        result = self._run_clustering(data_dict, config)

        # Extract dominant cluster label as "cell_type"
        if "cluster_labels" in result:
            labels = np.asarray(result["cluster_labels"])
            unique, counts = np.unique(labels, return_counts=True)
            result["cell_type"] = str(unique[np.argmax(counts)])

        return result


# ---------------------------------------------------------------------------
# Calibrax quality metrics
# ---------------------------------------------------------------------------


def _extract_labels(output: dict[str, Any]) -> jnp.ndarray | None:
    """Extract integer cluster labels from operator output."""
    if "cluster_labels" in output:
        return jnp.asarray(output["cluster_labels"], dtype=jnp.int32)
    if "cluster_assignments" in output:
        assignments = jnp.asarray(output["cluster_assignments"])
        if assignments.ndim == 2:
            return jnp.argmax(assignments, axis=-1)
        return jnp.asarray(assignments, dtype=jnp.int32)
    return None


def compute_quality_metrics(
    operator_output: dict[str, Any],
    task_type: str,
    data_dict: dict[str, Any],
) -> dict[str, float]:
    """Compute calibrax quality metrics on operator output.

    Returns task-appropriate quality metrics using calibrax's functional
    metrics. These supplement the grader pass/fail with continuous quality
    signals.

    Args:
        operator_output: Raw output dict from operator apply().
        task_type: Task category determining which metrics to compute.
        data_dict: Original input data dict (needed for features/labels).

    Returns:
        Dict mapping metric names to float values. Empty dict if no
        metrics are applicable or computation fails.
    """
    metrics: dict[str, float] = {}

    try:
        if task_type in ("clustering", "cell_annotation"):
            metrics.update(_clustering_metrics(operator_output, data_dict))
        elif task_type == "batch_correction":
            metrics.update(_batch_correction_metrics(operator_output, data_dict))
    except Exception as exc:
        logger.debug("Quality metrics computation failed for %s: %s", task_type, exc)

    return metrics


def _clustering_metrics(
    output: dict[str, Any],
    data_dict: dict[str, Any],
) -> dict[str, float]:
    """Compute clustering quality metrics via calibrax.

    Metrics:
        - silhouette_score: Mean silhouette coefficient (higher is better).
        - calinski_harabasz_score: Variance ratio criterion (higher is better).
    """
    from calibrax.metrics.functional.clustering import (  # noqa: PLC0415
        calinski_harabasz_score,
        silhouette_score,
    )

    labels = _extract_labels(output)
    if labels is None:
        return {}

    features = data_dict.get("embeddings")
    if features is None:
        return {}

    features = jnp.asarray(features, dtype=jnp.float32)
    n_unique = len(jnp.unique(labels))
    if n_unique < 2:
        return {}

    metrics: dict[str, float] = {}
    metrics["silhouette_score"] = float(silhouette_score(features, labels))
    metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(features, labels))
    return metrics


def _batch_correction_metrics(
    output: dict[str, Any],
    data_dict: dict[str, Any],
) -> dict[str, float]:
    """Compute batch correction quality metrics via calibrax.

    Metrics:
        - mmd: Maximum Mean Discrepancy between batches (lower is better).
        - kl_divergence: KL divergence between batch distributions (lower is better).
    """
    from calibrax.metrics.functional.divergence import (  # noqa: PLC0415
        kl_divergence,
        mmd,
    )

    corrected = output.get("corrected_embeddings")
    batch_labels = data_dict.get("batch_labels")
    if corrected is None or batch_labels is None:
        return {}

    corrected = jnp.asarray(corrected, dtype=jnp.float32)
    batch_labels = jnp.asarray(batch_labels, dtype=jnp.int32)
    unique_batches = jnp.unique(batch_labels)

    if len(unique_batches) < 2:
        return {}

    # Compute MMD between first two batches
    mask_0 = batch_labels == unique_batches[0]
    mask_1 = batch_labels == unique_batches[1]
    batch_0 = corrected[mask_0]
    batch_1 = corrected[mask_1]

    metrics: dict[str, float] = {}
    if batch_0.shape[0] > 0 and batch_1.shape[0] > 0:
        metrics["mmd"] = float(mmd(batch_0, batch_1))

        # KL on marginal distributions (mean over features)
        eps = 1e-8
        p = jax.nn.softmax(jnp.mean(batch_0, axis=0))
        q = jax.nn.softmax(jnp.mean(batch_1, axis=0))
        metrics["kl_divergence"] = float(kl_divergence(p + eps, q + eps))

    return metrics
