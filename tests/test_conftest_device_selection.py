"""Device selection reads a test's markers, never its keywords.

``item.keywords`` holds a test's marker names, but also its parametrize id, its function and
parent node names and its function attributes, so a membership test on it matches tests that carry
no marker at all (pytest ``_pytest/mark/structures.py`` ``NodeKeywords``, ``_pytest/python.py``
``keywords={callspec.id: True}``). A GPU test is one marked ``gpu`` or ``cuda``; the root conftest
reads that with ``item.get_closest_marker``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

TESTS_CONFTEST = Path(__file__).parent / "conftest.py"


def test_an_unmarked_test_whose_parametrize_id_is_gpu_runs(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(TESTS_CONFTEST.read_text())
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.parametrize("backend", ["gpu"])
        def test_backend_name(backend):
            assert backend == "gpu"
        """
    )

    result = pytester.runpytest("-p", "no:randomly", "-p", "no:cacheprovider")

    result.assert_outcomes(passed=1)
