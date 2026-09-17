"""Library code never seeds its own randomness.

A constructor that fell back to ``nnx.Rngs(0)`` when handed no ``rngs``, or a method that
drew from ``jax.random.key(0)`` when its stream was missing, built identical parameters in
every operator constructed without streams and drew the same noise on every call, without
a word. Whoever builds an operator names its streams (``nnx.Rngs``), and whoever applies a
stochastic one passes the record's key; the sources therefore contain no seed literal. This
test reads every module in the package and refuses one. Docstring examples are strings, not
code, so ``nnx.Rngs(42)`` in a usage example is not a finding.
"""

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "diffbio"
SOURCE_FILES = sorted(PACKAGE_ROOT.rglob("*.py"))
KEY_CONSTRUCTORS = ("jax.random.key", "jax.random.PRNGKey", "random.key", "random.PRNGKey")


def _is_int_literal(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) is int


def _seed_literals(tree: ast.AST) -> list[int]:
    """Line numbers of every ``nnx.Rngs(<int>)`` or ``jax.random.key(<int>)`` call."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = ast.unparse(node.func)
        arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
        seeds_rngs = callee.endswith("Rngs") and any(_is_int_literal(a) for a in arguments)
        seeds_key = callee in KEY_CONSTRUCTORS and any(_is_int_literal(a) for a in node.args)
        if seeds_rngs or seeds_key:
            lines.append(node.lineno)
    return sorted(lines)


def test_the_check_recognises_a_seed_literal() -> None:
    """Positive control: every fallback shape the sweep must catch, and one it must not."""
    caught = ast.parse(
        "a = nnx.Rngs(0)\n"
        "b = rngs or nnx.Rngs(0)\n"
        "c = nnx.Rngs(params=0, sample=1)\n"
        "d = rngs.params() if 'params' in rngs else jax.random.key(0)\n"
        "e = jax.random.PRNGKey(42)\n"
    )
    assert _seed_literals(caught) == [1, 2, 3, 4, 5]
    allowed = ast.parse(
        "a = nnx.Rngs(seed)\n"
        "b = jax.random.key(config.seed)\n"
        "c = rngs_from_seed(0, streams)\n"
        '"""nnx.Rngs(42) in a docstring"""\n'
    )
    assert _seed_literals(allowed) == []
    assert len(SOURCE_FILES) > 100


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: str(p.relative_to(PACKAGE_ROOT)))
def test_no_seed_literal_in_library_code(path: Path) -> None:
    """No module constructs ``nnx.Rngs`` or a key from a literal seed."""
    lines = _seed_literals(ast.parse(path.read_text(encoding="utf-8")))
    assert lines == [], f"seed literal in {path.relative_to(PACKAGE_ROOT)} at lines {lines}"
