"""Composition adapters for wiring operators into a pipeline.

Operators exchange data through a shared string-keyed dictionary, but adjacent
operators do not always agree on key names (for example, one writes
``"normalized"`` while the next reads ``"features"``). :class:`RenameField` is a
minimal pass-through operator that moves a value from one key to another, so a
sequential composition can bridge these mismatches without hand-rolled glue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx


@dataclass(frozen=True)
class RenameFieldConfig(OperatorConfig):
    """Configuration for :class:`RenameField`.

    Attributes:
        source: Existing data-dict key to move.
        target: Key the value is moved to.
    """

    source: str = ""
    target: str = ""

    def __post_init__(self) -> None:
        """Validate the configuration at construction, failing fast on empty keys.

        Raises:
            ValueError: If ``source`` or ``target`` is empty.
        """
        super().__post_init__()
        if not self.source:
            raise ValueError("source key must be a non-empty string")
        if not self.target:
            raise ValueError("target key must be a non-empty string")


class RenameField(OperatorModule):
    """Move a value from one data-dict key to another (a pass-through rename).

    Holds no learnable parameters and leaves the value untouched, so gradients
    flow through unchanged; it only bridges key-name mismatches between adjacent
    operators in a sequential composition.
    """

    def __init__(
        self,
        config: RenameFieldConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            config: Rename configuration.
            rngs: Optional RNG state (unused; kept for interface compatibility).
            name: Optional module name.
        """
        super().__init__(config, rngs=rngs, name=name)

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict | None,
        random_params: dict | None = None,
        stats: dict | None = None,
    ) -> tuple[dict, dict, dict | None]:
        """Move ``data[source]`` to ``data[target]`` and drop the source key.

        Args:
            data: Dictionary that must contain the ``source`` key.
            state: Operator state dictionary.
            metadata: Optional metadata dictionary.
            random_params: Optional random parameters (unused).
            stats: Optional statistics dictionary (unused).

        Returns:
            Tuple of ``(output_data, state, metadata)`` with the value moved from
            ``source`` to ``target``.

        Raises:
            KeyError: If the ``source`` key is absent from ``data``.
        """
        del random_params, stats
        config: RenameFieldConfig = self.config
        if config.source not in data:
            raise KeyError(f"RenameField source key {config.source!r} not found in data")
        output_data = {key: value for key, value in data.items() if key != config.source}
        output_data[config.target] = data[config.source]
        return output_data, state, metadata
