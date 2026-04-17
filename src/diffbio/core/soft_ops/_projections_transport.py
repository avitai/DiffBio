"""Differentiable projection onto the transport polytope.

Projects a cost matrix onto the transport polytope between two marginal
distributions using regularized optimal transport.  Multiple regularizers
control the smoothness of the resulting gradient:

- **smooth** (C-infinity): Entropic/softmax regularizer.  Solved via
  Sinkhorn or L-BFGS on the dual.
- **c0** (continuous): Euclidean/L2 regularizer (p=2 p-norm).  Solved
  via L-BFGS.
- **c1** (once differentiable): p=3/2 p-norm regularizer.  Solved via
  L-BFGS.
- **c2** (twice differentiable): p=4/3 p-norm regularizer.  Solved via
  L-BFGS.

All mathematical implementations are preserved exactly and support JAX
autodiff via implicit differentiation or recursive checkpointing.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from diffbio.core.soft_ops._utils import validate_softness

# -- Optional dependency: optimistix (L-BFGS solver) ----------------------
try:
    import optimistix as optx

    HAS_OPTIMISTIX = True
except ImportError:
    HAS_OPTIMISTIX = False

# -- Optional dependency: lineax (linear solvers) -------------------------
try:
    import lineax as lx

    HAS_LINEAX = True
except ImportError:
    HAS_LINEAX = False

# -- Optional dependency: OTT-JAX (Sinkhorn solver) -----------------------
try:
    from ott.geometry import geometry
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import (
        implicit_differentiation as idiff,
        sinkhorn,
    )

    HAS_OTT = True
except ImportError:
    HAS_OTT = False


def _transport_solver_dtype(dtype: jnp.dtype) -> jnp.dtype:
    """Choose the highest-precision solver dtype supported by the active JAX config."""
    return jnp.float64 if jax.config.jax_enable_x64 else dtype


def _promote_transport_inputs(
    C: jax.Array,
    mu: jax.Array,
    nu: jax.Array,
    scalar: float | Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Promote OT solver inputs without requesting unsupported float64 precision."""
    target_dtype = _transport_solver_dtype(C.dtype)
    C = C.astype(target_dtype)
    mu = mu.astype(target_dtype)
    nu = nu.astype(target_dtype)
    return C, mu, nu, jnp.asarray(scalar, dtype=target_dtype)


# ----------------------------------------------------------------------- #
# Entropic projection via Sinkhorn (requires OTT-JAX)
# ----------------------------------------------------------------------- #


def _proj_transport_polytope_entropic_sinkhorn(
    C: jax.Array,
    mu: jax.Array,
    nu: jax.Array,
    tol: float = 1e-6,
    max_iter: int = 1000,
    epsilon: float | Array = 1.0,
) -> jax.Array:
    """Solve entropic OT via Sinkhorn with implicit differentiation.

    Args:
        C: Cost matrix of shape ``(n, m)``.
        mu: Source marginal of shape ``(n,)``.
        nu: Target marginal of shape ``(m,)``.
        tol: Convergence tolerance.
        max_iter: Maximum Sinkhorn iterations.
        epsilon: Entropic regularization strength.

    Returns:
        Transport plan of shape ``(n, m)``.
    """
    if not HAS_OTT:
        msg = (
            "OTT-JAX is required for Sinkhorn-based entropic "
            "projection. Install it with: uv pip install ott-jax"
        )
        raise ImportError(msg)

    orig_dtype = C.dtype
    C, mu, nu, epsilon = _promote_transport_inputs(C, mu, nu, epsilon)

    # Avoid exact zeros (helps implicit differentiation a lot)
    tiny = 1e-12
    mu = jnp.clip(mu, tiny)
    mu = mu / jnp.sum(mu)
    nu = jnp.clip(nu, tiny)
    nu = nu / jnp.sum(nu)

    geom = geometry.Geometry(cost_matrix=C, epsilon=epsilon)  # pyright: ignore[reportArgumentType]
    prob = linear_problem.LinearProblem(geom, a=mu, b=nu)

    implicit = idiff.ImplicitDiff(
        solver_kwargs={"ridge_identity": 1e-6},
    )

    solver = sinkhorn.Sinkhorn(
        lse_mode=True,
        threshold=tol,
        max_iterations=max_iter,
        implicit_diff=implicit,
    )

    out = solver(prob)
    return out.matrix.astype(orig_dtype)


# ----------------------------------------------------------------------- #
# Entropic projection via L-BFGS (requires optimistix + lineax)
# ----------------------------------------------------------------------- #


def _proj_transport_polytope_entropic_lbfgs(
    C: jnp.ndarray,  # (n, m)
    mu: jnp.ndarray,  # (n,)
    nu: jnp.ndarray,  # (m,)
    epsilon: float | Array,  # scalar
    tol: float,
    max_steps: int,
    gauge_fix: bool = True,
    implicit_diff: bool = True,
) -> jnp.ndarray:
    """Solve entropic OT via L-BFGS on the dual.

    Args:
        C: Cost matrix of shape ``(n, m)``.
        mu: Source marginal of shape ``(n,)``.
        nu: Target marginal of shape ``(m,)``.
        epsilon: Entropic regularization strength (scalar).
        tol: Convergence tolerance.
        max_steps: Maximum L-BFGS steps.
        gauge_fix: If True, fix ``g[0] = 0`` to avoid singular
            Hessians in implicit differentiation.
        implicit_diff: If True, use implicit adjoint; otherwise
            recursive checkpointing.

    Returns:
        Transport plan of shape ``(n, m)``.
    """
    if not HAS_OPTIMISTIX:
        msg = (
            "optimistix is required for L-BFGS transport "
            "projection. Install it with: uv pip install optimistix"
        )
        raise ImportError(msg)
    if not HAS_LINEAX:
        msg = (
            "lineax is required for L-BFGS transport "
            "projection. Install it with: uv pip install lineax"
        )
        raise ImportError(msg)

    orig_dtype = C.dtype
    C, mu, nu, epsilon = _promote_transport_inputs(C, mu, nu, epsilon)

    mu = jnp.clip(mu, 1e-12)
    nu = jnp.clip(nu, 1e-12)
    mu = mu / jnp.sum(mu)
    nu = nu / jnp.sum(nu)
    n, m = C.shape

    if gauge_fix:
        # Gauge fix: set g0 = 0, optimise f and g_rest to avoid
        # singular system on implicit diff
        y0 = (
            jnp.zeros((n,), C.dtype),
            jnp.zeros((m - 1,), C.dtype),
        )  # (f, g_rest)
    else:
        y0 = (
            jnp.zeros((n,), C.dtype),
            jnp.zeros((m,), C.dtype),
        )  # (f, g)

    def neg_dual(
        y: tuple[jnp.ndarray, jnp.ndarray],
        args: tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            bool,
        ],
    ) -> jnp.ndarray:
        """Compute the negative entropic OT dual objective for L-BFGS minimization."""
        C_, mu_, nu_, eps_, gauge_fix_ = args
        if gauge_fix_:
            f, g_rest = y
            g = jnp.concatenate(
                [jnp.zeros((1,), C_.dtype), g_rest],
                axis=0,
            )  # (m,)
        else:
            f, g = y
        Z = (f[:, None] + g[None, :] - C_) / eps_
        return -(jnp.dot(mu_, f) + jnp.dot(nu_, g) - eps_ * jnp.sum(jnp.exp(Z)))

    solver = optx.LBFGS(rtol=tol, atol=tol)
    if implicit_diff:
        adj = optx.ImplicitAdjoint(
            linear_solver=lx.AutoLinearSolver(well_posed=False),
        )
    else:
        adj = optx.RecursiveCheckpointAdjoint()
    sol = optx.minimise(
        neg_dual,
        solver=solver,
        y0=y0,
        args=(C, mu, nu, epsilon, gauge_fix),
        max_steps=max_steps,
        adjoint=adj,
        throw=True,
    )

    if gauge_fix:
        f, g_rest = sol.value
        g = jnp.concatenate(
            [jnp.zeros((1,), C.dtype), g_rest],
            axis=0,
        )
    else:
        f, g = sol.value
    Gamma = jnp.exp((f[:, None] + g[None, :] - C) / epsilon)
    return Gamma.astype(orig_dtype)


# ----------------------------------------------------------------------- #
# P-norm projection via L-BFGS (requires optimistix + lineax)
# ----------------------------------------------------------------------- #


def _proj_transport_polytope_pnorm_lbfgs(
    C: jnp.ndarray,  # (n, m)
    mu: jnp.ndarray,  # (n,)
    nu: jnp.ndarray,  # (m,)
    lam: float | Array,  # scalar
    tol: float,
    max_steps: int,
    gauge_fix: bool = True,
    p: float = 6 / 5,  # 1 < p <= 2
    implicit_diff: bool = True,
) -> jnp.ndarray:
    """Solve p-norm regularized OT via L-BFGS on the dual.

    Args:
        C: Cost matrix of shape ``(n, m)``.
        mu: Source marginal of shape ``(n,)``.
        nu: Target marginal of shape ``(m,)``.
        lam: Regularization strength (scalar).
        tol: Convergence tolerance.
        max_steps: Maximum L-BFGS steps.
        gauge_fix: If True, fix ``g[0] = 0`` to avoid singular
            Hessians in implicit differentiation.
        p: Exponent for p-norm regularizer (1 < p <= 2).
        implicit_diff: If True, use implicit adjoint; otherwise
            recursive checkpointing.

    Returns:
        Transport plan of shape ``(n, m)``.
    """
    if not HAS_OPTIMISTIX:
        msg = (
            "optimistix is required for L-BFGS transport "
            "projection. Install it with: uv pip install optimistix"
        )
        raise ImportError(msg)
    if not HAS_LINEAX:
        msg = (
            "lineax is required for L-BFGS transport "
            "projection. Install it with: uv pip install lineax"
        )
        raise ImportError(msg)

    orig_dtype = C.dtype
    C, mu, nu, lam = _promote_transport_inputs(C, mu, nu, lam)

    mu = jnp.clip(mu, 1e-12)
    nu = jnp.clip(nu, 1e-12)
    mu = mu / jnp.sum(mu)
    nu = nu / jnp.sum(nu)
    n, m = C.shape
    q = p / (p - 1.0)  # conjugate exponent
    lam_pow = lam ** (-(q - 1.0))  # lam^{-(q-1)}

    if gauge_fix:
        # Gauge fix: set g0 = 0, optimise f and g_rest to avoid
        # singular system on implicit diff
        y0 = (
            jnp.zeros((n,), C.dtype),
            jnp.zeros((m - 1,), C.dtype),
        )
    else:
        y0 = (
            jnp.zeros((n,), C.dtype),
            jnp.zeros((m,), C.dtype),
        )

    def neg_dual(
        y: tuple[jnp.ndarray, jnp.ndarray],
        args: tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            bool,
        ],
    ) -> jnp.ndarray:
        """Compute the negative p-norm OT dual objective for L-BFGS minimization."""
        C_, mu_, nu_, lam_pow_, gauge_fix_ = args
        if gauge_fix_:
            f, g_rest = y
            g = jnp.concatenate(
                [jnp.zeros((1,), C_.dtype), g_rest],
                axis=0,
            )
        else:
            f, g = y
        S = f[:, None] + g[None, :] - C_
        P = jnp.maximum(S, 0.0)
        dual = jnp.dot(mu_, f) + jnp.dot(nu_, g) - (lam_pow_ / q) * jnp.sum(P**q)
        return -dual

    solver = optx.LBFGS(rtol=tol, atol=tol)
    if implicit_diff:
        adj = optx.ImplicitAdjoint(
            linear_solver=lx.AutoLinearSolver(well_posed=False),
        )
    else:
        adj = optx.RecursiveCheckpointAdjoint()
    sol = optx.minimise(
        neg_dual,
        solver=solver,
        y0=y0,
        args=(C, mu, nu, lam_pow, gauge_fix),
        max_steps=max_steps,
        adjoint=adj,
        throw=True,
    )

    if gauge_fix:
        f, g_rest = sol.value
        g = jnp.concatenate(
            [jnp.zeros((1,), C.dtype), g_rest],
            axis=0,
        )
    else:
        f, g = sol.value
    S = f[:, None] + g[None, :] - C
    # = lambda^{-(q-1)} [S]_+^{q-1}
    Gamma = lam_pow * jnp.maximum(S, 0.0) ** (q - 1.0)
    return Gamma.astype(orig_dtype)


# ----------------------------------------------------------------------- #
# Public dispatch function
# ----------------------------------------------------------------------- #


def proj_transport_polytope(
    cost: Array,  # (..., n, m)
    mu: Array,  # ([n],)
    nu: Array,  # ([m],)
    softness: float | Array = 0.1,
    mode: Literal["smooth", "c0", "c1", "c2"] = "smooth",
    use_entropic_ot_sinkhorn_on_entropic: bool = True,
    sinkhorn_tol: float = 1e-5,
    sinkhorn_max_iter: int = 10000,
    lbfgs_tol: float = 1e-5,
    lbfgs_max_iter: int = 10000,
    implicit_diff: bool = True,
) -> Array:  # (..., [n], m)
    """Project a cost matrix onto the transport polytope.

    Solves the regularized optimal transport problem::

        min_G  <C, G> + softness * R(G)
        s.t.   G 1_m = mu,  G^T 1_n = nu,  G >= 0

    where ``R(G)`` is the regularizer determined by ``mode``.

    Args:
        cost: Input cost array of shape ``(..., n, m)``.
        mu: Source marginal distribution of shape ``([n],)``.
        nu: Target marginal distribution of shape ``([m],)``.
        softness: Controls the strength of the regularizer.
            Must be positive.
        mode: Controls the type of regularizer:

            - ``"smooth"``: C-infinity smooth (entropic/softmax
              regularizer).  Solved via Sinkhorn or L-BFGS.
            - ``"c0"``: C0 continuous (Euclidean/L2 regularizer).
              Solved via L-BFGS.
            - ``"c1"``: C1 differentiable (p=3/2 p-norm). Solved
              via L-BFGS.
            - ``"c2"``: C2 twice differentiable (p=4/3 p-norm).
              Solved via L-BFGS.
        use_entropic_ot_sinkhorn_on_entropic: If True (default),
            use Sinkhorn for ``"smooth"`` mode. If False, use
            L-BFGS on the dual.
        sinkhorn_tol: Convergence tolerance for Sinkhorn.
        sinkhorn_max_iter: Maximum Sinkhorn iterations.
        lbfgs_tol: Convergence tolerance for L-BFGS.
        lbfgs_max_iter: Maximum L-BFGS iterations.
        implicit_diff: If True (default), use implicit
            differentiation for L-BFGS backward pass.  More
            numerically stable gradients, especially at low
            softness.

    Returns:
        Positive array of shape ``(..., [n], m)`` representing
        the transport plan between ``mu`` and ``nu``.  Sums to 1
        over the second-to-last dimension and approximately sums
        to 1 over the last dimension.

    Note:
        Internal solvers upcast to float64 when possible for
        numerical stability.  This requires
        ``jax.config.update("jax_enable_x64", True)`` (or the
        ``JAX_ENABLE_X64=1`` env var).  Without it the upcast is
        silently ignored and the solver may produce non-finite
        gradients at larger problem sizes (typically n >= 2048).
    """
    validate_softness(softness)
    *batch_sizes, n, m = cost.shape
    C = cost.reshape(-1, n, m)  # (B, n, m)

    if mode == "smooth":
        use_entropic_ot_sinkhorn = use_entropic_ot_sinkhorn_on_entropic

        if use_entropic_ot_sinkhorn:
            proj_fn = lambda c: _proj_transport_polytope_entropic_sinkhorn(
                c,
                mu=mu,
                nu=nu,
                max_iter=sinkhorn_max_iter,
                tol=sinkhorn_tol,
                epsilon=softness,
            )
        else:
            proj_fn = lambda c: _proj_transport_polytope_entropic_lbfgs(
                c,
                mu=mu,
                nu=nu,
                epsilon=softness,
                tol=lbfgs_tol,
                max_steps=lbfgs_max_iter,
                implicit_diff=implicit_diff,
            )

    else:
        if mode == "c0":
            # Curvature of (1/2)||y||^2 at transport polytope
            # center: R''=1
            p = 2
        elif mode == "c1":
            p = 3 / 2
        elif mode == "c2":
            p = 4 / 3
        else:
            msg = f"Invalid mode: {mode}"
            raise ValueError(msg)
        proj_fn = lambda c: _proj_transport_polytope_pnorm_lbfgs(
            c,
            mu=mu,
            nu=nu,
            lam=softness,
            tol=lbfgs_tol,
            max_steps=lbfgs_max_iter,
            p=p,
            implicit_diff=implicit_diff,
        )

    Gamma = jax.vmap(proj_fn, in_axes=(0,))(C)  # (B, n, m)

    y = (Gamma * n).reshape(*batch_sizes, n, m)  # (..., [n], m)
    return y
