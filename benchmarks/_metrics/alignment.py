"""SP (Sum of Pairs) and TC (Total Column) alignment quality metrics.

Implements the two standard metrics for evaluating multiple sequence
alignment quality against a reference alignment:

- **SP score**: Fraction of correctly aligned residue pairs in the
  predicted alignment that match the reference.
- **TC score**: Fraction of columns in the reference alignment that
  are perfectly reproduced in the predicted alignment.

Both metrics follow the BAliBASE convention: only uppercase positions
in the reference are scored. Lowercase positions are structural
inserts that are not part of the core alignment.
"""

from __future__ import annotations


def _build_pair_set(
    alignment: list[tuple[str, str]],
) -> set[tuple[str, int, str, int, int]]:
    """Extract scored residue pairs from an alignment.

    For each column, generates all (name_i, name_j, col_i, col_j)
    pairs of uppercase (scored) residues aligned together. A pair
    is identified by the two sequence names and the original residue
    positions within each sequence (excluding gaps).

    Args:
        alignment: List of (name, aligned_sequence) tuples.

    Returns:
        Set of (name_a, pos_a, name_b, pos_b, column_index) tuples
        representing scored residue pairs.
    """
    if not alignment:
        return set()

    n_cols = len(alignment[0][1])
    pairs: set[tuple[str, int, str, int, int]] = set()

    # Precompute residue indices (position within the ungapped
    # sequence) for each scored position in each sequence
    seq_positions: list[list[tuple[str, int]]] = []
    for name, seq in alignment:
        positions: list[tuple[str, int]] = []
        residue_idx = 0
        for char in seq:
            if char == ".":
                positions.append(("gap", -1))
            elif char.isupper():
                positions.append((name, residue_idx))
                residue_idx += 1
            else:
                # Lowercase = unscored insert, still a residue
                positions.append(("insert", residue_idx))
                residue_idx += 1
        seq_positions.append(positions)

    # For each column, pair all scored (uppercase) residues
    for col in range(n_cols):
        scored_in_col: list[tuple[str, int]] = []
        for seq_idx in range(len(alignment)):
            tag, pos = seq_positions[seq_idx][col]
            if tag not in ("gap", "insert"):
                scored_in_col.append((tag, pos))

        # Generate all ordered pairs
        for i in range(len(scored_in_col)):
            for j in range(i + 1, len(scored_in_col)):
                name_a, pos_a = scored_in_col[i]
                name_b, pos_b = scored_in_col[j]
                # Canonical ordering to avoid duplicates
                if (name_a, pos_a) > (name_b, pos_b):
                    name_a, pos_a, name_b, pos_b = (
                        name_b,
                        pos_b,
                        name_a,
                        pos_a,
                    )
                pairs.add((name_a, pos_a, name_b, pos_b, col))

    return pairs


def _build_residue_pair_set(
    alignment: list[tuple[str, str]],
) -> set[tuple[str, int, str, int]]:
    """Extract scored residue pair identities (column-independent).

    Like ``_build_pair_set`` but without column index, so pairs are
    matched by residue identity only (not column position).

    Args:
        alignment: List of (name, aligned_sequence) tuples.

    Returns:
        Set of (name_a, pos_a, name_b, pos_b) tuples.
    """
    full_pairs = _build_pair_set(alignment)
    return {(na, pa, nb, pb) for na, pa, nb, pb, _ in full_pairs}


def sp_score(
    predicted: list[tuple[str, str]],
    reference: list[tuple[str, str]],
) -> float:
    """Compute the Sum of Pairs (SP) score.

    Measures the fraction of correctly aligned residue pairs in the
    reference that are also correctly aligned in the prediction.
    Only uppercase (core) positions in the reference are scored.

    Args:
        predicted: Predicted alignment as (name, aligned_seq) tuples.
        reference: Reference alignment as (name, aligned_seq) tuples.

    Returns:
        SP score in [0.0, 1.0]. Returns 0.0 if the reference has
        no scored pairs.
    """
    ref_pairs = _build_residue_pair_set(reference)
    if not ref_pairs:
        return 0.0

    pred_pairs = _build_residue_pair_set(predicted)
    correct = len(ref_pairs & pred_pairs)
    return correct / len(ref_pairs)


def tc_score(
    predicted: list[tuple[str, str]],
    reference: list[tuple[str, str]],
) -> float:
    """Compute the Total Column (TC) score.

    Measures the fraction of reference columns that are exactly
    reproduced in the predicted alignment. A column is counted
    only if it contains at least two scored (uppercase) residues.

    Args:
        predicted: Predicted alignment as (name, aligned_seq) tuples.
        reference: Reference alignment as (name, aligned_seq) tuples.

    Returns:
        TC score in [0.0, 1.0]. Returns 0.0 if the reference has
        no scorable columns.
    """
    if not reference:
        return 0.0

    n_cols = len(reference[0][1])

    # Build predicted column lookup: for each pair of scored
    # residues, record whether they share a column
    pred_pair_cols: dict[tuple[str, int, str, int], int] = {}
    if predicted:
        pred_n_cols = len(predicted[0][1])
        for col in range(pred_n_cols):
            scored: list[tuple[str, int]] = []
            for name, seq in predicted:
                residue_idx = 0
                for c_idx, char in enumerate(seq):
                    if char == ".":
                        if c_idx == col:
                            break
                        continue
                    if c_idx == col:
                        if char.isupper():
                            scored.append((name, residue_idx))
                        break
                    residue_idx += 1

            for i in range(len(scored)):
                for j in range(i + 1, len(scored)):
                    a, b = scored[i], scored[j]
                    if a > b:
                        a, b = b, a
                    key = (*a, *b)
                    pred_pair_cols[key] = col

    # Check each reference column
    total_columns = 0
    correct_columns = 0

    for col in range(n_cols):
        scored: list[tuple[str, int]] = []
        for name, seq in reference:
            residue_idx = 0
            for c_idx, char in enumerate(seq):
                if char == ".":
                    if c_idx == col:
                        break
                    continue
                if c_idx == col:
                    if char.isupper():
                        scored.append((name, residue_idx))
                    break
                residue_idx += 1

        if len(scored) < 2:
            continue

        total_columns += 1

        # Column is correct if all scored pairs share a column
        # in the prediction
        all_correct = True
        target_col = None
        for i in range(len(scored)):
            for j in range(i + 1, len(scored)):
                a, b = scored[i], scored[j]
                if a > b:
                    a, b = b, a
                key = (*a, *b)
                pred_col = pred_pair_cols.get(key)
                if pred_col is None:
                    all_correct = False
                    break
                if target_col is None:
                    target_col = pred_col
                elif pred_col != target_col:
                    all_correct = False
                    break
            if not all_correct:
                break

        if all_correct:
            correct_columns += 1

    if total_columns == 0:
        return 0.0
    return correct_columns / total_columns
