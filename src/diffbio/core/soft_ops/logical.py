"""Soft logical operators (fuzzy logic).

Provides differentiable fuzzy logic operations on SoftBool values.
No ``softness`` parameter -- these operate purely on probability
values in [0, 1].

Fuzzy logic semantics:
- NOT: ``1 - x``
- AND (product): ``prod(x)`` or geometric mean
- OR: ``1 - AND(NOT(x))``
- XOR: ``AND(x, NOT(y)) OR AND(NOT(x), y)``
"""

import jax.numpy as jnp

from diffbio.core.soft_ops._types import SoftBool


def logical_not(x: SoftBool) -> SoftBool:
    """Soft logical NOT: ``1 - x``.

    Args:
        x: SoftBool input in [0, 1].

    Returns:
        Complement in [0, 1].
    """
    return 1.0 - x


def all(
    x: SoftBool,
    axis: int = -1,
    epsilon: float = 1e-10,
    use_geometric_mean: bool = False,
) -> SoftBool:
    """Soft logical AND reduction along axis.

    Uses product (default) or geometric mean to combine probabilities.

    Args:
        x: SoftBool input in [0, 1].
        axis: Axis along which to reduce.
        epsilon: Minimum value for numerical stability in log.
        use_geometric_mean: If True, use geometric mean instead of product.

    Returns:
        Reduced SoftBool.
    """
    if use_geometric_mean:
        return jnp.exp(
            jnp.mean(jnp.log(jnp.clip(x, min=epsilon)), axis=axis),
        )
    return jnp.prod(x, axis=axis)


def any(
    x: SoftBool,
    axis: int = -1,
    use_geometric_mean: bool = False,
) -> SoftBool:
    """Soft logical OR reduction along axis.

    Implemented as ``1 - all(1 - x)``.

    Args:
        x: SoftBool input in [0, 1].
        axis: Axis along which to reduce.
        use_geometric_mean: If True, use geometric mean in the inner AND.

    Returns:
        Reduced SoftBool.
    """
    return logical_not(
        all(logical_not(x), axis=axis, use_geometric_mean=use_geometric_mean),
    )


def logical_and(
    x: SoftBool,
    y: SoftBool,
    use_geometric_mean: bool = False,
) -> SoftBool:
    """Soft logical AND between two SoftBools.

    Stacks inputs and applies :func:`all` along the stack axis.

    Args:
        x: First SoftBool.
        y: Second SoftBool.
        use_geometric_mean: If True, use geometric mean.

    Returns:
        SoftBool in [0, 1].
    """
    return all(
        jnp.stack([x, y], axis=-1),
        axis=-1,
        use_geometric_mean=use_geometric_mean,
    )


def logical_or(
    x: SoftBool,
    y: SoftBool,
    use_geometric_mean: bool = False,
) -> SoftBool:
    """Soft logical OR between two SoftBools.

    Stacks inputs and applies :func:`any` along the stack axis.

    Args:
        x: First SoftBool.
        y: Second SoftBool.
        use_geometric_mean: If True, use geometric mean in inner AND.

    Returns:
        SoftBool in [0, 1].
    """
    return any(
        jnp.stack([x, y], axis=-1),
        axis=-1,
        use_geometric_mean=use_geometric_mean,
    )


def logical_xor(
    x: SoftBool,
    y: SoftBool,
    use_geometric_mean: bool = False,
) -> SoftBool:
    """Soft logical XOR between two SoftBools.

    Implemented as ``(x AND NOT y) OR (NOT x AND y)``.

    Args:
        x: First SoftBool.
        y: Second SoftBool.
        use_geometric_mean: If True, use geometric mean in AND/OR.

    Returns:
        SoftBool in [0, 1].
    """
    t1 = logical_and(x, logical_not(y), use_geometric_mean=use_geometric_mean)
    t2 = logical_and(logical_not(x), y, use_geometric_mean=use_geometric_mean)
    return logical_or(t1, t2, use_geometric_mean=use_geometric_mean)
