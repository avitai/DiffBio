"""Tests for benchmarks._metrics.alignment module.

Verifies SP (Sum of Pairs) and TC (Total Column) alignment scores.
"""

from __future__ import annotations

from benchmarks._metrics.alignment import sp_score, tc_score


# -----------------------------------------------------------------------
# Fixtures: reusable alignment data
# -----------------------------------------------------------------------

_SIMPLE_ALIGNMENT: list[tuple[str, str]] = [
    ("seq1", "ACGT"),
    ("seq2", "ACGT"),
    ("seq3", "ACGT"),
]

_ALIGNMENT_WITH_GAPS: list[tuple[str, str]] = [
    ("seq1", "AC.GT"),
    ("seq2", "A.CGT"),
]


class TestSPScore:
    """Tests for sp_score (Sum of Pairs)."""

    def test_identical_alignments_score_one(self) -> None:
        """SP score is 1.0 when predicted matches reference."""
        ref = [("s1", "ACGT"), ("s2", "ACGT")]
        assert sp_score(ref, ref) == 1.0

    def test_completely_different_alignments_score_zero(self) -> None:
        """SP score is 0.0 when no residue pairs match."""
        # Reference: residues A(s1)-A(s2) aligned in col 0
        ref = [("s1", "A."), ("s2", "A.")]
        # Predicted: residues never share a column
        pred = [("s1", "A."), ("s2", ".A")]
        assert sp_score(pred, ref) == 0.0

    def test_partial_match(self) -> None:
        """SP score is between 0 and 1 for partial matches."""
        ref = [("s1", "AB"), ("s2", "AB"), ("s3", "AB")]
        # Shift one sequence by one position
        pred = [("s1", "AB."), ("s2", "AB."), ("s3", ".AB")]
        score = sp_score(pred, ref)
        assert 0.0 < score < 1.0

    def test_lowercase_positions_ignored(self) -> None:
        """Lowercase (insert) positions in reference are not scored."""
        # Reference: only uppercase A is scored, lowercase 'b' is not
        ref = [("s1", "Ab"), ("s2", "Ab")]
        # Only the uppercase pair (A,A) matters
        pred = [("s1", "Ab"), ("s2", "Ab")]
        score = sp_score(pred, ref)
        assert score == 1.0

    def test_gaps_handled(self) -> None:
        """Gaps (dots) do not form scored pairs."""
        ref = [("s1", "A.B"), ("s2", "A.B")]
        pred = [("s1", "A.B"), ("s2", "A.B")]
        assert sp_score(pred, ref) == 1.0

    def test_empty_reference_returns_zero(self) -> None:
        """Empty reference alignment returns 0.0."""
        pred = [("s1", "ACGT")]
        ref: list[tuple[str, str]] = []
        assert sp_score(pred, ref) == 0.0

    def test_three_sequence_alignment(self) -> None:
        """Score is correct with three sequences."""
        score = sp_score(_SIMPLE_ALIGNMENT, _SIMPLE_ALIGNMENT)
        assert score == 1.0


class TestTCScore:
    """Tests for tc_score (Total Column)."""

    def test_identical_alignments_score_one(self) -> None:
        """TC score is 1.0 when predicted matches reference."""
        ref = [("s1", "ACGT"), ("s2", "ACGT")]
        assert tc_score(ref, ref) == 1.0

    def test_completely_different_alignments_score_zero(self) -> None:
        """TC score is 0.0 when no columns are reproduced."""
        ref = [("s1", "AB"), ("s2", "AB")]
        # Shift sequence 2 so no column matches
        pred = [("s1", "AB."), ("s2", ".AB")]
        assert tc_score(pred, ref) == 0.0

    def test_partial_column_match(self) -> None:
        """TC score reflects fraction of correctly reproduced columns."""
        # Two scorable columns: col 0 (A,A) and col 1 (B,B)
        ref = [("s1", "AB"), ("s2", "AB")]
        # Predicted: only first column matches
        pred = [("s1", "A.B"), ("s2", "A.B")]
        score = tc_score(pred, ref)
        # Both columns should match since both sequences are shifted same
        assert score == 1.0

    def test_lowercase_positions_not_scored_in_columns(self) -> None:
        """Columns with only lowercase positions are not counted."""
        # Only uppercase positions define scorable columns
        ref = [("s1", "Aa"), ("s2", "Aa")]
        pred = [("s1", "Aa"), ("s2", "Aa")]
        # Column 0 has uppercase A from both = scorable
        # Column 1 has lowercase a from both = not scored
        score = tc_score(pred, ref)
        assert score == 1.0

    def test_gaps_in_alignment(self) -> None:
        """Gaps (dots) are handled correctly in TC scoring."""
        ref = [("s1", "A.B"), ("s2", "A.B")]
        pred = [("s1", "A.B"), ("s2", "A.B")]
        assert tc_score(pred, ref) == 1.0

    def test_empty_reference_returns_zero(self) -> None:
        """Empty reference returns 0.0."""
        assert tc_score([("s1", "A")], []) == 0.0

    def test_single_sequence_no_scorable_columns(self) -> None:
        """Single sequence has no pairs, so no scorable columns."""
        ref = [("s1", "ACGT")]
        pred = [("s1", "ACGT")]
        # No pairs possible with one sequence -> 0.0
        assert tc_score(pred, ref) == 0.0

    def test_three_sequence_perfect(self) -> None:
        """Three sequences perfectly aligned yield 1.0."""
        score = tc_score(_SIMPLE_ALIGNMENT, _SIMPLE_ALIGNMENT)
        assert score == 1.0
