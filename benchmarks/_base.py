"""Base class for all DiffBio benchmarks.

Provides the shared structure that every benchmark follows:
config-driven setup, operator profiling, gradient flow check,
comparison table rendering, and calibrax BenchmarkResult output.

This eliminates code duplication across benchmark implementations
by extracting the common 9-step pattern into reusable methods.
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from calibrax.core.models import Metric, Point
from calibrax.core.result import BenchmarkResult
from calibrax.profiling.timing import TimingCollector

from benchmarks._gradient import GradientFlowResult, check_gradient_flow
from diffbio.operators.foundation_models.contracts import decode_foundation_text

logger = logging.getLogger(__name__)

_FOUNDATION_TAG_KEYS = (
    "model_family",
    "adapter_mode",
    "artifact_id",
    "preprocessing_version",
)
_FOUNDATION_METADATA_KEYS = _FOUNDATION_TAG_KEYS + ("pooling_strategy",)
_FOUNDATION_COMPARISON_AXES = [
    "dataset",
    "task",
    "model_family",
    "adapter_mode",
    "artifact_id",
    "preprocessing_version",
]


@dataclass(frozen=True, kw_only=True)
class DiffBioBenchmarkConfig:
    """Configuration shared by all DiffBio benchmarks.

    Attributes:
        name: Benchmark identifier (e.g. "singlecell/batch_correction").
        domain: Domain category (e.g. "singlecell", "alignment").
        quick_subsample: Default subsample size for quick mode.
        n_iterations_quick: Throughput iterations in quick mode.
        n_iterations_full: Throughput iterations in full mode.
    """

    name: str
    domain: str
    quick_subsample: int = 2000
    n_iterations_quick: int = 10
    n_iterations_full: int = 50


class DiffBioBenchmark(ABC):
    """Abstract base class for all DiffBio benchmarks.

    Provides shared infrastructure for the common benchmark pattern:

    1. Load real data via a datarax DataSource
    2. Create and run a DiffBio operator
    3. Compute domain-specific quality metrics
    4. Check gradient flow
    5. Measure throughput via calibrax TimingCollector
    6. Compare against published baselines
    7. Return a calibrax BenchmarkResult

    Subclasses implement ``_run_core()`` which contains the
    domain-specific logic (steps 1-3). The base class handles
    gradient checking, throughput measurement, comparison tables,
    and result construction (steps 4-7).
    """

    def __init__(
        self,
        config: DiffBioBenchmarkConfig,
        *,
        quick: bool = False,
        data_dir: str = "",
    ) -> None:
        self.config = config
        self.quick = quick
        self.data_dir = data_dir

    @abstractmethod
    def _run_core(self) -> dict[str, Any]:
        """Execute the domain-specific benchmark logic.

        Must return a dict with at least these keys:
            - ``metrics``: dict[str, float] of quality metrics
            - ``operator``: the nnx.Module operator (for gradient check)
            - ``input_data``: dict fed to operator.apply()
            - ``loss_fn``: callable (model, data) -> scalar
            - ``n_items``: int, number of items processed
            - ``iterate_fn``: callable for throughput measurement

        May also include:
            - ``baselines``: dict[str, Point] for comparison table
            - ``dataset_info``: dict with dataset metadata
            - ``operator_config``: dict with operator config values
            - ``task_name``: override for the benchmark task tag
            - ``result_data``: raw operator output for metadata extraction
            - ``benchmark_tags``: extra string tags merged into BenchmarkResult.tags
            - ``benchmark_metadata``: extra metadata merged into BenchmarkResult.metadata
        """

    def run(self) -> BenchmarkResult:
        """Execute the full benchmark pipeline.

        Returns:
            calibrax BenchmarkResult with quality metrics,
            gradient flow, throughput, and comparison baselines.
        """
        core = self._run_core()

        metrics = core["metrics"]
        operator = core["operator"]
        input_data = core["input_data"]
        loss_fn = core["loss_fn"]
        n_items = core["n_items"]
        iterate_fn = core["iterate_fn"]
        baselines = core.get("baselines", {})
        dataset_info = core.get("dataset_info", {})
        operator_config = core.get("operator_config", {})
        operator_name = core.get("operator_name", "Unknown")
        dataset_name = core.get("dataset_name", "unknown")
        task_name = core.get("task_name", self.config.name.rsplit("/", 1)[-1])
        benchmark_tags = core.get("benchmark_tags", {})
        benchmark_metadata = core.get("benchmark_metadata", {})
        result_data = core.get("result_data", {})
        foundation_metadata = self._extract_foundation_metadata(result_data)

        # Gradient flow check
        logger.info("Checking gradient flow...")
        grad = self._check_gradient(loss_fn, operator, input_data)
        logger.info(
            "  Gradient norm: %.4f, nonzero: %s",
            grad.gradient_norm,
            grad.gradient_nonzero,
        )

        # Throughput measurement
        n_iters = self.config.n_iterations_quick if self.quick else self.config.n_iterations_full
        logger.info("Measuring throughput...")
        timing = self._measure_throughput(iterate_fn, n_items, n_iters)
        items_per_sec = timing.num_elements / timing.wall_clock_sec
        logger.info("  %.0f items/sec", items_per_sec)

        # Build calibrax metrics
        calibrax_metrics = {k: Metric(value=float(v)) for k, v in metrics.items()}
        calibrax_metrics["gradient_norm"] = Metric(value=grad.gradient_norm)
        calibrax_metrics["gradient_nonzero"] = Metric(value=1.0 if grad.gradient_nonzero else 0.0)
        calibrax_metrics["items_per_sec"] = Metric(value=items_per_sec)

        # Print comparison table
        if baselines:
            self._print_comparison(metrics, baselines)

        tags = {
            "operator": operator_name,
            "dataset": dataset_name,
            "framework": "diffbio",
            "task": task_name,
        }
        for key in _FOUNDATION_TAG_KEYS:
            if key in foundation_metadata:
                tags[key] = foundation_metadata[key]
        tags.update(benchmark_tags)

        metadata = {
            "dataset_info": dataset_info,
            "baselines": {k: p.to_dict() for k, p in baselines.items()},
            "comparison_axes": (
                _FOUNDATION_COMPARISON_AXES if foundation_metadata else ["dataset", "task"]
            ),
        }
        if foundation_metadata:
            metadata["foundation_model"] = foundation_metadata
        metadata.update(benchmark_metadata)

        return BenchmarkResult(
            name=self.config.name,
            domain="diffbio_benchmarks",
            tags=tags,
            timing=timing,
            metrics=calibrax_metrics,
            config={
                **operator_config,
                "quick": self.quick,
            },
            metadata=metadata,
        )

    @staticmethod
    def _check_gradient(
        loss_fn: Any,
        operator: Any,
        input_data: dict[str, Any],
    ) -> GradientFlowResult:
        """Check gradient flow through the operator."""
        try:
            return check_gradient_flow(loss_fn, operator, input_data)
        except (ValueError, TypeError, RuntimeError) as exc:
            # JAX/NNX gradient computation can fail for operators
            # without learnable parameters or incompatible loss_fn
            logger.warning("Gradient check failed: %s", exc)
            return GradientFlowResult(gradient_norm=0.0, gradient_nonzero=False)

    @staticmethod
    def _measure_throughput(
        iterate_fn: Any,
        n_items: int,
        n_iterations: int,
    ) -> Any:
        """Measure throughput using calibrax TimingCollector."""
        collector = TimingCollector(warmup_iterations=3)
        return collector.measure_iteration(
            iterator=iter(range(n_iterations)),
            num_batches=n_iterations,
            process_fn=lambda _: iterate_fn(),
            count_fn=lambda _: n_items,
        )

    @staticmethod
    def _print_comparison(
        metrics: dict[str, float],
        baselines: dict[str, Point],
    ) -> None:
        """Print a comparison table of DiffBio vs baselines."""
        # Collect all metric keys present in baselines
        all_keys: list[str] = []
        for point in baselines.values():
            for k in point.metrics:
                if k not in all_keys:
                    all_keys.append(k)

        # Filter to keys we also computed
        keys = [k for k in all_keys if k in metrics]
        if not keys:
            return

        # Log comparison table
        header = f"  {'Method':<22}"
        for k in keys:
            header += f" {k:>12}"
        logger.info("\n%s", header)
        logger.info("  %s", "-" * (len(header) - 2))

        # DiffBio row
        row = f"  {'DiffBio':<22}"
        for k in keys:
            row += f" {metrics.get(k, 0):>12.4f}"
        logger.info("%s", row)

        # Baseline rows
        for name, point in baselines.items():
            row = f"  {name:<22}"
            for k in keys:
                val = point.metrics.get(k, Metric(value=0)).value
                row += f" {val:>12.4f}"
            logger.info("%s", row)

    @staticmethod
    def _extract_foundation_metadata(result_data: Any) -> dict[str, str]:
        """Decode optional foundation-model metadata from operator outputs."""
        if not isinstance(result_data, dict):
            return {}

        raw_metadata = result_data.get("foundation_model")
        if not isinstance(raw_metadata, dict):
            return {}

        decoded: dict[str, str] = {}
        for key in _FOUNDATION_METADATA_KEYS:
            value = raw_metadata.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                decoded[key] = value
                continue
            decoded[key] = decode_foundation_text(value)
        return decoded

    @classmethod
    def cli_main(
        cls,
        benchmark_cls: type[DiffBioBenchmark],
        config: DiffBioBenchmarkConfig,
        **kwargs: Any,
    ) -> None:
        """Shared CLI entry point for all benchmarks.

        Args:
            benchmark_cls: The benchmark class to instantiate.
            config: Benchmark configuration.
            **kwargs: Extra kwargs passed to benchmark constructor.
        """
        logging.basicConfig(level=logging.INFO)
        quick = "--quick" in sys.argv

        print("=" * 60)
        print(f"DiffBio Benchmark: {config.name}")
        print("=" * 60)

        bench = benchmark_cls(config, quick=quick, **kwargs)
        result = bench.run()

        out = Path("benchmarks/results") / config.domain
        out.mkdir(parents=True, exist_ok=True)
        filename = config.name.rsplit("/", 1)[-1]
        result.save(out / f"{filename}.json")
        print(f"\nSaved to: {out / f'{filename}.json'}")
