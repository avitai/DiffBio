"""Stage HDCytoData CyTOF gating datasets into the ``cytof_gating`` .npz schema.

Provenance (Weber & Robinson 2016 gating truth, distributed via the HDCytoData
Bioconductor package):

- ``Samusik_all``  (Samusik et al., *Nat Methods* 2016): mouse bone marrow, 10 samples
  -> the primary dataset, split by held-out sample/donor.
- ``Levine_32dim`` (Levine et al., *Cell* 2015): healthy human bone marrow, 2 patients
  -> the secondary dataset, split by stratified held-out cells.

HDCytoData ships each dataset as a ``SummarizedExperiment`` whose assay is the
marker-by-cell expression matrix and whose ``rowData`` carries the manually-gated
``population_id`` and the ``sample_id``. Export that to a flat per-cell table (marker
columns + a population-label column + a sample/donor column) and point this converter at
it. Reading the raw FlowRepository FCS directly is intentionally out of scope -- this
converter takes an already-extracted table so the staging tool (R/HDCytoData, or any FCS
reader) stays pluggable and this step is deterministic and testable.

Output schema (one ``.npz`` per dataset), consumed by ``cytof_gating.py``:

    intensities  float32 (n_cells, n_markers)  raw (untransformed) marker intensities
    labels       int32   (n_cells,)            gated population id, 0..n_types-1
    donor        int32   (n_cells,)            sample/donor id, 0..n_donors-1
    marker_names <U..    (n_markers,)
    type_names   <U..    (n_types,)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_DROP_LABELS = ("unassigned", "unknown", "nan")


def encode_categories(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map categorical values to contiguous int ids and return ``(ids, names)``.

    Args:
        values: A 1-D array of hashable category values.

    Returns:
        A tuple ``(ids, names)`` where ``ids`` is an ``int32`` array of positions into
        the sorted unique ``names`` array.
    """
    names, inverse = np.unique(np.asarray(values), return_inverse=True)
    return inverse.astype(np.int32), names


def _resolve_marker_columns(
    frame: pd.DataFrame, label_column: str, donor_column: str | None
) -> list[str]:
    """Return the numeric marker columns, i.e. all numeric columns bar label/donor."""
    reserved = {label_column} | ({donor_column} if donor_column is not None else set())
    markers = [
        column
        for column in frame.columns
        if column not in reserved and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not markers:
        raise ValueError("no numeric marker columns found after excluding label/donor columns")
    return markers


def table_to_arrays(
    frame: pd.DataFrame,
    *,
    label_column: str,
    donor_column: str | None = None,
    marker_columns: list[str] | None = None,
    drop_labels: tuple[str, ...] = _DEFAULT_DROP_LABELS,
) -> dict[str, np.ndarray]:
    """Convert a per-cell table into the ``cytof_gating`` array schema.

    Args:
        frame: One row per cell; marker columns plus a label column and optional donor
            column.
        label_column: Column holding the manually-gated population label.
        donor_column: Column holding the sample/donor id; ``None`` assigns every cell to
            a single donor (forcing a stratified split downstream).
        marker_columns: Explicit marker columns; ``None`` infers all numeric columns
            except the label/donor columns.
        drop_labels: Case-insensitive label values to drop (unassigned/ungated cells).

    Returns:
        A dict with the ``intensities``/``labels``/``donor``/``marker_names``/
        ``type_names`` arrays of the output schema.

    Raises:
        ValueError: If no cells remain after dropping unassigned labels.
    """
    if marker_columns is None:
        marker_columns = _resolve_marker_columns(frame, label_column, donor_column)

    label_text = frame[label_column].astype(str)
    keep = ~label_text.str.strip().str.lower().isin({value.lower() for value in drop_labels})
    keep &= frame[label_column].notna()
    kept = frame.loc[keep]
    if kept.empty:
        raise ValueError("no cells remain after dropping unassigned labels")

    intensities = kept[marker_columns].to_numpy(dtype=np.float32)
    labels, type_names = encode_categories(kept[label_column].astype(str).to_numpy())
    if donor_column is None:
        donor = np.zeros(len(kept), dtype=np.int32)
    else:
        donor, _ = encode_categories(kept[donor_column].astype(str).to_numpy())

    return {
        "intensities": intensities,
        "labels": labels,
        "donor": donor,
        "marker_names": np.asarray(marker_columns, dtype=np.str_),
        "type_names": type_names.astype(np.str_),
    }


def save_npz(arrays: dict[str, np.ndarray], out_path: Path) -> None:
    """Write the schema arrays to ``out_path`` (creating parent directories)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        intensities=arrays["intensities"],
        labels=arrays["labels"],
        donor=arrays["donor"],
        marker_names=arrays["marker_names"],
        type_names=arrays["type_names"],
    )


def _read_table(source: Path) -> pd.DataFrame:
    """Read a per-cell table from CSV/TSV/Parquet by file extension."""
    suffix = source.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(source, sep="\t")
    return pd.read_csv(source)


def _self_check() -> None:
    """Round-trip a synthetic table to validate the conversion logic without real data."""
    rng = np.random.default_rng(0)
    n_cells = 200
    frame = pd.DataFrame(
        {
            "CD3": rng.gamma(2.0, 40.0, n_cells).astype(np.float32),
            "CD19": rng.gamma(2.0, 40.0, n_cells).astype(np.float32),
            "population": rng.choice(["Tcell", "Bcell", "unassigned"], n_cells),
            "sample": rng.choice(["s1", "s2", "s3"], n_cells),
        }
    )
    arrays = table_to_arrays(frame, label_column="population", donor_column="sample")
    assert arrays["intensities"].dtype == np.float32
    assert arrays["intensities"].shape[1] == 2
    assert "unassigned" not in set(arrays["type_names"].tolist())
    assert arrays["labels"].max() < len(arrays["type_names"])
    assert arrays["donor"].max() < 3
    assert arrays["intensities"].shape[0] == arrays["labels"].shape[0]
    print(
        f"self-check OK: {arrays['intensities'].shape[0]} cells, "
        f"{len(arrays['type_names'])} types, markers {arrays['marker_names'].tolist()}"
    )


def main() -> None:
    """Convert a staged per-cell table into the ``cytof_gating`` .npz schema."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="per-cell table (CSV/TSV/Parquet)")
    parser.add_argument("--out", type=Path, help="output .npz path")
    parser.add_argument("--label-column", default="population_id")
    parser.add_argument("--donor-column", default="sample_id")
    parser.add_argument("--markers", nargs="*", default=None, help="explicit marker columns")
    parser.add_argument("--self-check", action="store_true", help="validate on synthetic data")
    args = parser.parse_args()

    if args.self_check:
        _self_check()
        return
    if args.source is None or args.out is None:
        parser.error("--source and --out are required unless --self-check is given")

    frame = _read_table(args.source)
    arrays = table_to_arrays(
        frame,
        label_column=args.label_column,
        donor_column=args.donor_column,
        marker_columns=args.markers,
    )
    save_npz(arrays, args.out)
    print(
        f"wrote {args.out}: {arrays['intensities'].shape[0]} cells x "
        f"{arrays['intensities'].shape[1]} markers, {len(arrays['type_names'])} types, "
        f"{int(arrays['donor'].max()) + 1} donors"
    )


if __name__ == "__main__":
    main()
