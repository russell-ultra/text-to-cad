"""Mechanics per sweep, SI: ``getFT`` wrappers, projection, clearance, energy,
equilibria, the convergence protocol, the verdict, the field slice (lane A3).

- ``F, T = getFT(fixed, moving_with_paths, pivot=P, squeeze=False)`` ALWAYS with
  ``squeeze=False``. magpylib 5.2.3 returns ``(n_sources, p, n_targets, 3)`` (a
  source axis the design's ``(p, t, 3)`` shorthand folds away); the net load on
  the moving GROUP is the sum over sources AND targets, ``F.sum(axis=(0, 2))``
  -> ``(p, 3)``. Shapes 1x1, 1x201, 2x201 are unit-tested.
- Slider / ``travel``: ``Q = F . a``, ``F_perp = F - Q a``. Revolute / ``turn``:
  ``Q = tau . a`` (N m), ``F_perp = F`` (the whole force is a guide load).
- Clearance: for each pose transform the moving solids by ``D(q)`` (metres ->
  millimetres) and test ``cadgen.geometry.overlap_volume`` against every fixed
  solid (magnets and non-magnets alike) behind a bounding-box prefilter; overlap
  above 1e-3 mm^3 is ``inaccessible`` -- excluded from equilibrium search, energy
  integrated per accessible interval, hatched in the report.
- ``U(q) - U(q0) = -integral Q dq`` by trapezoid within each accessible interval;
  ``Q = -sin(q)`` therefore gives ``U = 1 - cos(q)`` up to the interval offset.
- Equilibria: sign changes of ``Q`` present at BOTH pose-grid resolutions,
  refined by bisection on fresh single-pose ``getFT`` calls; stable iff
  ``dQ/dq < 0``.
- Convergence protocol at the verdict pose and each equilibrium: mesh in
  ``{20, 50, 100}`` at eps 1e-5 m gives ``dQ_mesh``; eps in ``{1e-5, 1e-6}`` at
  mesh 100 gives ``dQ_eps``; the finest level at ``N`` and ``2N - 1`` nested
  poses gives ``dQ_grid``, equilibrium agreement and ``dU_max``.
  ``uncertainty = max(dQ_mesh, dQ_eps, dQ_grid)``. ``converged`` is true when
  each relative delta (guarded by ``Q_scale = 0.02 * max|Q|``) is under
  ``converge_tol``, equilibria agree and ``dU_max < 2 %`` of the energy span;
  otherwise false with ``reasons``. ``quick`` runs mesh {20, 50} and grid only;
  ``full`` everything; ``off`` the finest level once, ``converged: None``.
- Verdict at ``--at q`` (default lower limit): ``tendency`` +axis / -axis /
  indeterminate (``|Q| < uncertainty``); ``exceeds_threshold`` against the
  friction threshold, ``None`` without one or when the band straddles it;
  ``admissible`` false when the tendency points outside the limits at a limit
  pose or into an inaccessible interval. Rest under friction is a band around
  each stable equilibrium, never a point; ``|F_perp|`` and torque are guide
  loads, never friction inputs.
- ``--relax``: at every swept pose the held cylindrical sub-DOF is set where its
  own generalized force vanishes (``Q_turn = tau . a`` for a held turn,
  ``Q_travel = F . a`` for a held travel) by bracketed bisection over its
  limits on fresh single-pose calls; no zero in range marks the pose
  ``wedged`` -- reported, excluded from equilibrium search, hatched, never an error.

No magpylib/numpy/scipy import at module scope -- use ``cadgen.magnetics._deps``.
No CAD-kernel import at module scope (``cadgen.geometry``/``build123d`` are
imported inside the functions that touch clearance and outlines).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from cadgen.magnetics import _deps
from cadgen.magnetics.types import (  # noqa: F401 - part of the contract
    Convergence,
    Equilibrium,
    FieldSlice,
    MagnetSpec,
    MagScene,
    MateError,
    MateSpec,
    Sample,
    SceneError,
    SweepConfig,
    SweepResult,
    Verdict,
)

__all__ = [
    "CLEARANCE_THRESHOLD_MM3",
    "DEFAULT_GRID",
    "PLANE_PARALLEL_TOL",
    "force_torque",
    "project",
    "clearance",
    "energy",
    "find_equilibria",
    "converge",
    "make_verdict",
    "relax_held",
    "sweep",
    "field",
]

#: Overlap volume (model units, mm^3) above which a pose is inaccessible.
CLEARANCE_THRESHOLD_MM3 = 1e-3

#: Default ``--grid`` of the field slice.
DEFAULT_GRID = 120

#: ``|axis x up|`` below which the slice's ``up`` falls back to the next bbox axis.
PLANE_PARALLEL_TOL = 1e-6

#: Metres per model millimetre; ``delta`` is SI, the clearance solids are mm.
_M_TO_MM = 1e3

#: Bisection ceiling for :func:`relax_held` and equilibrium refinement.
_BISECT_MAX_ITERS = 60

#: Mesh / eps for the single-pose ``getFT`` calls :func:`relax_held` bisects on.
_RELAX_MESHING = 50
_RELAX_EPS = 1e-5

#: Fraction of the energy span ``dU_max`` must stay under to count as converged.
_ENERGY_TOL = 0.02


# --------------------------------------------------------------------------- forces


def force_torque(
    fixed: Sequence[Any],
    moving_pathed: Sequence[Any],
    pivot: Any,
    meshing: int,
    eps: float,
) -> tuple[Any, Any]:
    """Net ``(F (p, 3), T (p, 3))`` in N and N m on the moving group over its path.

    ``magpylib.getFT(fixed, moving_pathed, pivot=pivot, eps=eps, squeeze=False)``
    with ``meshing`` set on every target first; magpylib returns
    ``(n_sources, p, n_targets, 3)`` which is reduced to the net group load by an
    explicit ``.sum(axis=(0, 2))``. ``pivot`` is ``(n_targets, p, 3)`` from
    ``kinematics.pivot_path``. A test pins ``inspect.signature(getFT)`` so a
    magpylib upgrade fails loudly.
    """
    magpy = _deps.magpylib()
    np = _deps.numpy()
    targets = list(moving_pathed)
    for target in targets:
        target.meshing = meshing
    F, T = magpy.getFT(list(fixed), targets, pivot=pivot, eps=eps, squeeze=False)
    F = np.asarray(F, dtype=float)
    T = np.asarray(T, dtype=float)
    # (n_sources, p, n_targets, 3) -> (p, 3): sum every source's pull on every target.
    return F.sum(axis=(0, 2)), T.sum(axis=(0, 2))


def _swept_is_rotation(mate: MateSpec, dof: str | None) -> bool:
    """Whether the swept sub-DOF is a rotation (torque projection) or translation."""
    if mate.kind == "revolute":
        return True
    if mate.kind == "slider":
        return False
    key = dof.rsplit(".", 1)[-1] if dof else "travel"
    return key == "turn"


def project(F: Any, T: Any, mate: MateSpec, dof: str | None) -> tuple[Any, Any]:
    """``(Q (p,), F_perp (p, 3))``: the generalized force along the swept DOF.

    Translation (slider, ``travel``): ``Q = F . a``, ``F_perp = F - Q a``.
    Rotation (revolute, ``turn``): ``Q = T . a`` (N m), ``F_perp = F`` (the
    whole force is a guide load).
    """
    np = _deps.numpy()
    F = np.atleast_2d(np.asarray(F, dtype=float))
    T = np.atleast_2d(np.asarray(T, dtype=float))
    axis = np.asarray(mate.dir, dtype=float)
    axis = axis / np.linalg.norm(axis)
    if _swept_is_rotation(mate, dof):
        Q = T @ axis
        F_perp = F
    else:
        Q = F @ axis
        F_perp = F - np.outer(Q, axis)
    return Q, F_perp


# --------------------------------------------------------------------------- clearance


def _bbox_of(solid: Any) -> tuple[Any, Any]:
    box = solid.bounding_box()
    return box.min, box.max


def _boxes_disjoint(a_min: Any, a_max: Any, b_min: Any, b_max: Any) -> bool:
    return (
        a_max.X < b_min.X or b_max.X < a_min.X
        or a_max.Y < b_min.Y or b_max.Y < a_min.Y
        or a_max.Z < b_min.Z or b_max.Z < a_min.Z
    )


def _location_from_delta(D: Any) -> Any:
    """A build123d ``Location`` for the SI premultiplier ``D`` in model millimetres."""
    from cadgen import build123d as bd

    R = D[:3, :3]
    t = D[:3, 3] * _M_TO_MM
    plane = bd.Plane(
        origin=(float(t[0]), float(t[1]), float(t[2])),
        x_dir=(float(R[0, 0]), float(R[1, 0]), float(R[2, 0])),
        z_dir=(float(R[0, 2]), float(R[1, 2]), float(R[2, 2])),
    )
    return plane.location


# CONTRACT-CHANGE: added the keyword-only ``swept`` (default "value") so a
# cylindrical sweep can tell clearance which sub-DOF ``q`` moves. Only
# :func:`sweep` (this module) calls clearance, so no other lane is affected;
# B1 can fold this into the frozen signature note if desired.
def clearance(
    scene: MagScene,
    qs: Sequence[float],
    held_values: Sequence[float] | None = None,
    *,
    swept: str = "value",
) -> Any:
    """Boolean ``accessible`` mask ``(p,)``: no moving solid overlaps a fixed one.

    Moves each moving magnet's ``solid`` (model units) by ``delta(q)`` (SI,
    scaled back to millimetres), prefilters by bounding box against
    ``scene.solids_fixed``, and calls ``cadgen.geometry.overlap_volume`` on the
    survivors; overlap above :data:`CLEARANCE_THRESHOLD_MM3` marks the pose
    inaccessible. ``swept`` is the swept limits key (added keyword; a cylindrical
    sweep needs to know which sub-DOF ``q`` moves). :class:`MateError` naming the
    mate when EVERY pose is inaccessible is raised by :func:`sweep`, not here.
    """
    from cadgen import geometry as cad_geometry
    from cadgen.magnetics import kinematics

    np = _deps.numpy()
    qs = np.asarray(qs, dtype=float)
    accessible = np.ones(len(qs), dtype=bool)
    moving_solids = [scene.magnets[i].solid for i in scene.moving]
    if not moving_solids or not scene.solids_fixed:
        return accessible

    fixed_boxes = [(_bbox_of(s), s) for s in scene.solids_fixed]
    for p, q in enumerate(qs):
        held = None if held_values is None else float(held_values[p])
        if held is not None and np.isnan(held):
            # A wedged pose has no held value; clearance is undefined -- leave it
            # accessible here and let :func:`sweep` drop it as wedged.
            continue
        D = np.asarray(kinematics.delta(scene.mate, float(q), held, swept=swept), dtype=float)
        loc = _location_from_delta(D)
        blocked = False
        for solid in moving_solids:
            moved = solid.moved(loc)
            m_min, m_max = _bbox_of(moved)
            for (f_min, f_max), fixed_solid in fixed_boxes:
                if _boxes_disjoint(m_min, m_max, f_min, f_max):
                    continue
                if cad_geometry.overlap_volume(moved, fixed_solid) > CLEARANCE_THRESHOLD_MM3:
                    blocked = True
                    break
            if blocked:
                break
        accessible[p] = not blocked
    return accessible


# --------------------------------------------------------------------------- energy


def _neg_cumtrapz(Q: Any, qs: Any, np: Any) -> Any:
    """``-integral Q dq`` cumulatively, starting at 0 at ``qs[0]`` (trapezoid)."""
    U = np.zeros(len(qs), dtype=float)
    for k in range(1, len(qs)):
        U[k] = U[k - 1] - 0.5 * (Q[k] + Q[k - 1]) * (qs[k] - qs[k - 1])
    return U


def energy(Q: Any, qs: Sequence[float], accessible: Any) -> Any:
    """``U (p,)`` in joules: ``-integral Q dq`` by trapezoid per accessible interval.

    Each accessible interval starts at ``U = 0`` at its own first pose;
    inaccessible poses carry ``nan`` (``Sample.U_J`` is ``None`` there).
    Synthetic ``Q = -sin(q)`` gives ``U = 1 - cos(q)`` up to the interval offset.
    """
    np = _deps.numpy()
    Q = np.asarray(Q, dtype=float)
    qs = np.asarray(qs, dtype=float)
    acc = np.asarray(accessible, dtype=bool)
    U = np.full(len(qs), np.nan, dtype=float)
    i = 0
    n = len(qs)
    while i < n:
        if not acc[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and acc[j + 1]:
            j += 1
        U[i : j + 1] = _neg_cumtrapz(Q[i : j + 1], qs[i : j + 1], np)
        i = j + 1
    return U


# --------------------------------------------------------------------------- equilibria


def _crossing_brackets(Q: Any, qs: Any, np: Any) -> list[tuple[float, float]]:
    """``(lo, hi)`` q-brackets across every finite sign change of ``Q`` on ``qs``.

    A strict sign change ``Q[k]*Q[k+1] < 0`` brackets ``(qs[k], qs[k+1])``. A
    zero that lands exactly on an interior node (``Q[k+1] == 0`` with opposite-
    signed neighbours) brackets the node, ``(qs[k], qs[k+2])`` -- so a root that
    happens to fall on a sampled pose is found, not straddled.
    """
    brackets = []
    n = len(qs)
    k = 0
    while k < n - 1:
        a, b = Q[k], Q[k + 1]
        if np.isfinite(a) and np.isfinite(b):
            if a * b < 0.0:
                brackets.append((float(qs[k]), float(qs[k + 1])))
                k += 1
                continue
            if a != 0.0 and b == 0.0 and k + 2 < n and np.isfinite(Q[k + 2]) and a * Q[k + 2] < 0.0:
                brackets.append((float(qs[k]), float(qs[k + 2])))
                k += 2
                continue
        k += 1
    return brackets


def _fine_has_sign_change(lo: float, hi: float, Q_fine: Any, qs_fine: Any, np: Any) -> bool:
    inside = (qs_fine >= lo - 1e-15) & (qs_fine <= hi + 1e-15)
    seg = Q_fine[inside]
    seg = seg[np.isfinite(seg)]
    if len(seg) == 0:
        return False
    if np.any(seg == 0.0):
        return True
    signs = np.sign(seg)
    return bool(np.any(signs[:-1] * signs[1:] < 0))


def _bisect_root(
    lo: float, hi: float, refine_fn: Callable[[float], float], np: Any
) -> tuple[float, float, float]:
    """``(root, dQdq, half_width)`` from bisecting ``refine_fn`` over ``[lo, hi]``."""
    Q_lo = float(refine_fn(lo))
    Q_hi = float(refine_fn(hi))
    tol = max(abs(hi - lo) * 1e-6, 1e-12)
    root = 0.5 * (lo + hi)
    for _ in range(_BISECT_MAX_ITERS):
        mid = 0.5 * (lo + hi)
        root = mid
        Q_mid = float(refine_fn(mid))
        if Q_mid == 0.0 or (hi - lo) / 2.0 <= tol:
            break
        if (Q_mid > 0) == (Q_lo > 0):
            lo, Q_lo = mid, Q_mid
        else:
            hi, Q_hi = mid, Q_mid
    # The bracket keeps its opposite-signed endpoints, so the slope is well
    # defined even when the midpoint landed exactly on the root.
    width = hi - lo
    dQdq = (Q_hi - Q_lo) / width if width > 0 else 0.0
    return root, dQdq, 0.5 * width


def find_equilibria(
    Q_coarse: Any,
    Q_fine: Any,
    qs_c: Sequence[float],
    qs_f: Sequence[float],
    refine_fn: Callable[[float], float],
) -> list[Equilibrium]:
    """Roots of ``Q`` present at BOTH grids, refined and classified.

    A sign change on the coarse grid whose bracket contains no sign change on
    the fine (nested, ``2N - 1``) grid is rejected as a grid artefact.
    Surviving brackets are bisected with ``refine_fn(q) -> Q`` (fresh
    single-pose ``getFT``), ``stable`` iff ``dQ/dq < 0`` across the final
    bracket, ``uncertainty_q`` the final bracket half-width. No sign change
    anywhere -> ``[]``. Inaccessible and wedged poses are masked out (``nan``)
    by the caller before this runs.
    """
    np = _deps.numpy()
    Q_coarse = np.asarray(Q_coarse, dtype=float)
    Q_fine = np.asarray(Q_fine, dtype=float)
    qs_c = np.asarray(qs_c, dtype=float)
    qs_f = np.asarray(qs_f, dtype=float)
    equilibria: list[Equilibrium] = []
    for lo, hi in _crossing_brackets(Q_coarse, qs_c, np):
        if not _fine_has_sign_change(lo, hi, Q_fine, qs_f, np):
            continue
        root, dQdq, half_width = _bisect_root(lo, hi, refine_fn, np)
        equilibria.append(
            Equilibrium(q=root, stable=bool(dQdq < 0), dQdq=dQdq, uncertainty_q=half_width)
        )
    return equilibria


# --------------------------------------------------------------------------- convergence


def _grid_forces(
    scene: MagScene,
    cfg: SweepConfig,
    qs: Sequence[float],
    held_values: Sequence[float] | None,
    meshing: int,
    eps: float,
) -> tuple[Any, Any, Any, Any]:
    """``(Q, F, T, F_perp)`` over ``qs`` at ``(meshing, eps)``.

    The single kinematics touch of the whole convergence/sweep path: builds the
    pathed magpylib sources and the pivot path for ``qs`` (holding the other
    cylindrical sub-DOF per ``held_values``), calls :func:`force_torque` and
    :func:`project`. Tests patch this seam to feed synthetic ``Q`` without the
    lane A2 kinematics.
    """
    from cadgen.magnetics import kinematics

    swept_key, _ = kinematics.sweep_dof(scene.mate, cfg.dof)
    moving = kinematics.pathed_sources(scene, qs, held_values, swept=swept_key)
    fixed = [scene.magnets[i].source for i in scene.fixed]
    pivot = kinematics.pivot_path(scene, qs, held_values, swept=swept_key)
    F, T = force_torque(fixed, moving, pivot, meshing, eps)
    Q, F_perp = project(F, T, scene.mate, cfg.dof)
    return Q, F, T, F_perp


def _held_at(qs: Any, held_values: Sequence[float] | None, q: float, np: Any) -> float | None:
    if held_values is None:
        return None
    return float(np.interp(q, qs, np.nan_to_num(np.asarray(held_values, dtype=float))))


def _rel(dq_abs: float, q_ref: float, q_scale: float) -> float:
    return dq_abs / max(abs(q_ref), q_scale)


def converge(
    scene: MagScene,
    cfg: SweepConfig,
    qs: Sequence[float],
    at_q: float,
    equilibria: Sequence[Equilibrium],
    held_values: Sequence[float] | None = None,
) -> Convergence:
    """Run the convergence protocol from the module docstring and record it.

    Levels come from ``cfg.meshing_levels``/``cfg.eps_levels`` cut by
    ``cfg.convergence`` (``quick``: first two mesh levels + grid; ``full``: all;
    ``off``: nothing run, ``converged=None``). The relative-delta guard
    ``Q_scale = 0.02 * max|Q|`` keeps a ``Q ~ 0`` pose from dividing by zero.
    ``reasons`` lists every failing axis: ``"mesh"``, ``"eps"``,
    ``"pose_grid"``, ``"equilibria"``, ``"energy"``.
    """
    np = _deps.numpy()
    if cfg.convergence == "off":
        return Convergence(mesh=None, eps=None, pose_grid=None, converged=None, reasons=())

    qs = np.asarray(qs, dtype=float)
    mesh_levels = list(cfg.meshing_levels)
    if cfg.convergence == "quick":
        mesh_levels = mesh_levels[:2]
    eps_levels = list(cfg.eps_levels)
    finest_mesh = mesh_levels[-1]
    coarse_eps = eps_levels[0]
    finest_eps = eps_levels[-1]
    run_eps = cfg.convergence == "full" and len(eps_levels) >= 2

    # Grid axis first: it also fixes Q_scale and the energy span.
    Q_N = np.asarray(_grid_forces(scene, cfg, qs, held_values, finest_mesh, finest_eps)[0], dtype=float)
    max_absQ = float(np.nanmax(np.abs(Q_N))) if len(Q_N) else 0.0
    q_scale = 0.02 * max_absQ if max_absQ > 0 else 1e-30

    qs_fine = np.linspace(float(qs[0]), float(qs[-1]), 2 * len(qs) - 1)
    held_fine = None
    if held_values is not None:
        held_fine = np.interp(qs_fine, qs, np.nan_to_num(np.asarray(held_values, dtype=float)))
    Q_2N = np.asarray(_grid_forces(scene, cfg, qs_fine, held_fine, finest_mesh, finest_eps)[0], dtype=float)
    dQ_grid = float(np.nanmax(np.abs(Q_2N[::2] - Q_N))) if len(Q_N) else 0.0

    acc_all = np.ones(len(qs), dtype=bool)
    U_N = energy(Q_N, qs, acc_all)
    U_2N = energy(Q_2N, qs_fine, np.ones(len(qs_fine), dtype=bool))
    span = float(np.nanmax(U_N) - np.nanmin(U_N)) if len(U_N) else 0.0
    dU_max = float(np.nanmax(np.abs(U_2N[::2] - U_N))) if len(U_N) else 0.0
    n_coarse = len(_crossing_brackets(Q_N, qs, np))
    n_fine = len(_crossing_brackets(Q_2N, qs_fine, np))
    equilibria_agree = n_coarse == n_fine

    reasons: list[str] = []

    # Probe poses for the mesh and eps axes: the verdict pose and each equilibrium.
    probes = [float(at_q)] + [float(e.q) for e in equilibria]

    mesh_block: dict[str, Any] | None = None
    if len(mesh_levels) >= 2:
        dQ_mesh = 0.0
        rel_mesh = 0.0
        for pose in probes:
            held = _held_at(qs, held_values, pose, np)
            hv = None if held is None else [held]
            q_prev = float(_grid_forces(scene, cfg, [pose], hv, mesh_levels[-2], coarse_eps)[0][0])
            q_fin = float(_grid_forces(scene, cfg, [pose], hv, mesh_levels[-1], coarse_eps)[0][0])
            dq = abs(q_fin - q_prev)
            dQ_mesh = max(dQ_mesh, dq)
            rel_mesh = max(rel_mesh, _rel(dq, q_fin, q_scale))
        mesh_block = {"levels": mesh_levels, "eps": coarse_eps, "dQ": dQ_mesh}
        if rel_mesh >= cfg.converge_tol:
            reasons.append("mesh")

    eps_block: dict[str, Any] | None = None
    if run_eps:
        dQ_eps = 0.0
        rel_eps = 0.0
        for pose in probes:
            held = _held_at(qs, held_values, pose, np)
            hv = None if held is None else [held]
            q_coarse = float(_grid_forces(scene, cfg, [pose], hv, finest_mesh, eps_levels[0])[0][0])
            q_fine = float(_grid_forces(scene, cfg, [pose], hv, finest_mesh, eps_levels[-1])[0][0])
            dq = abs(q_fine - q_coarse)
            dQ_eps = max(dQ_eps, dq)
            rel_eps = max(rel_eps, _rel(dq, q_fine, q_scale))
        eps_block = {"levels": eps_levels, "meshing": finest_mesh, "dQ": dQ_eps}
        if rel_eps >= cfg.converge_tol:
            reasons.append("eps")

    pose_grid_block = {
        "samples": [int(len(qs)), int(len(qs_fine))],
        "dQ": dQ_grid,
        "equilibria_agree": bool(equilibria_agree),
        "dU_max": dU_max,
    }
    if _rel(dQ_grid, max_absQ, q_scale) >= cfg.converge_tol:
        reasons.append("pose_grid")
    if not equilibria_agree:
        reasons.append("equilibria")
    if span > 0 and dU_max / span >= _ENERGY_TOL:
        reasons.append("energy")

    return Convergence(
        mesh=mesh_block,
        eps=eps_block,
        pose_grid=pose_grid_block,
        converged=len(reasons) == 0,
        reasons=tuple(reasons),
    )


def _uncertainty_from(conv: Convergence) -> float:
    """``max(dQ_mesh, dQ_eps, dQ_grid)`` -- the verdict's band half-width."""
    values = [
        block["dQ"]
        for block in (conv.mesh, conv.eps, conv.pose_grid)
        if block is not None and "dQ" in block
    ]
    return max(values) if values else 0.0


# --------------------------------------------------------------------------- verdict


def _admissible(
    at_q: float,
    tendency: str,
    lo: float,
    hi: float,
    qs: Any,
    acc: Any,
    np: Any,
) -> bool:
    if tendency == "indeterminate":
        return True
    direction = 1 if tendency == "+axis" else -1
    span = abs(hi - lo)
    edge = 1e-12 + span * 1e-6
    if direction < 0 and abs(at_q - lo) <= edge:
        return False
    if direction > 0 and abs(at_q - hi) <= edge:
        return False
    idx = int(np.argmin(np.abs(qs - at_q)))
    nxt = idx + direction
    if 0 <= nxt < len(qs) and not acc[nxt]:
        return False
    return True


def make_verdict(
    qs: Sequence[float],
    Q: Any,
    accessible: Any,
    at_q: float,
    uncertainty: float,
    threshold: float | None,
    limits: tuple[float, float],
) -> Verdict:
    """The split verdict at ``at_q`` (interpolated onto the sample grid).

    ``tendency`` is ``"+axis"``/``"-axis"`` by the sign of ``Q(at_q)`` and
    ``"indeterminate"`` when ``|Q| < uncertainty``. ``exceeds_threshold`` is
    ``None`` when ``threshold`` is ``None`` or the band ``[|Q| - u, |Q| + u]``
    straddles it, else the comparison. ``admissible`` is ``False`` when the
    tendency points outside ``limits`` at a limit pose or into an inaccessible
    interval adjacent to ``at_q``.
    """
    np = _deps.numpy()
    qs = np.asarray(qs, dtype=float)
    Q = np.asarray(Q, dtype=float)
    acc = np.asarray(accessible, dtype=bool)
    lo, hi = float(limits[0]), float(limits[1])
    Q_at = float(np.interp(at_q, qs, Q))
    abs_Q = abs(Q_at)
    if abs_Q < uncertainty:
        tendency = "indeterminate"
    elif Q_at > 0:
        tendency = "+axis"
    else:
        tendency = "-axis"
    if threshold is None:
        exceeds: bool | None = None
    elif abs_Q - uncertainty <= threshold <= abs_Q + uncertainty:
        exceeds = None
    else:
        exceeds = abs_Q > threshold
    admissible = _admissible(float(at_q), tendency, lo, hi, qs, acc, np)
    return Verdict(
        at_q=float(at_q),
        tendency=tendency,  # type: ignore[arg-type]
        Q=Q_at,
        uncertainty=float(uncertainty),
        threshold=threshold,
        exceeds_threshold=exceeds,
        admissible=admissible,
    )


# --------------------------------------------------------------------------- relax


def _held_generalized_force(
    scene: MagScene,
    q: float,
    held_val: float,
    held_key: str,
    swept_key: str,
) -> float:
    """The held sub-DOF's own generalized force at swept ``q``, held at ``held_val``.

    ``Q_turn = tau . a`` (held turn) or ``Q_travel = F . a`` (held travel), from
    a single-pose ``getFT``. A seam :func:`relax_held`'s tests patch so the
    bisection is exercised without the lane A2 kinematics.
    """
    from cadgen.magnetics import kinematics

    np = _deps.numpy()
    posed = kinematics.posed_sources(scene, q, held_val, swept=swept_key)
    fixed = [scene.magnets[i].source for i in scene.fixed]
    origin = np.asarray(scene.mate.origin_m, dtype=float)
    pivot = np.tile(origin.reshape(1, 1, 3), (len(posed), 1, 1))
    F, T = force_torque(fixed, posed, pivot, _RELAX_MESHING, _RELAX_EPS)
    axis = np.asarray(scene.mate.dir, dtype=float)
    axis = axis / np.linalg.norm(axis)
    load = T[0] if held_key == "turn" else F[0]
    return float(np.dot(load, axis))


def relax_held(
    scene: MagScene,
    q: float,
    held_key: str,
    limits: tuple[float, float],
) -> float | None:
    """The held sub-DOF's value at swept pose ``q`` where its own generalized force vanishes.

    Bracketed bisection over ``limits`` on fresh single-pose ``getFT`` calls
    (``Q_turn = tau . a`` for ``held_key == "turn"``, ``Q_travel = F . a`` for
    ``"travel"``). ``None`` when the force has no zero in range: the pose is
    ``wedged``. Tested on ``Q_turn = -sin(turn - 0.3)`` -> 0.3 rad and on a
    monotone force -> ``None``.
    """
    np = _deps.numpy()
    swept_key = "turn" if held_key == "travel" else "travel"

    def force_at(held_val: float) -> float:
        return _held_generalized_force(scene, float(q), float(held_val), held_key, swept_key)

    lo, hi = float(limits[0]), float(limits[1])
    Q_lo = force_at(lo)
    Q_hi = force_at(hi)
    if Q_lo == 0.0:
        return lo
    if Q_hi == 0.0:
        return hi
    if (Q_lo > 0) == (Q_hi > 0):
        return None  # no sign change in range: wedged
    root, _dQ, _u = _bisect_root(lo, hi, force_at, np)
    return root


# --------------------------------------------------------------------------- sweep


def _mask_to_intervals(qs: Any, mask: Any, np: Any) -> tuple[tuple[float, float], ...]:
    """Closed ``(lo, hi)`` q-intervals over every run of ``True`` in ``mask``."""
    intervals = []
    i = 0
    n = len(qs)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        intervals.append((float(qs[i]), float(qs[j])))
        i = j + 1
    return tuple(intervals)


def sweep(scene: MagScene, cfg: SweepConfig) -> SweepResult:
    """The whole analysis: samples, clearance, energy, equilibria, verdict, convergence.

    ``scene.mate`` must be set (:class:`MateError` otherwise). Samples ``q``
    across the swept DOF's limits (``cfg.samples`` poses, ``kinematics.sweep_dof``
    picks the key), holds or relaxes the other cylindrical sub-DOF, computes the
    path-vectorised ``force_torque`` at the finest configured level, projects,
    masks inaccessible (clearance) and wedged poses, integrates energy, finds
    equilibria against the nested ``2N - 1`` grid, runs :func:`converge`, and
    builds the :class:`Verdict` at ``cfg.at`` (default lower limit) with
    ``uncertainty`` from the convergence record. :class:`MateError` naming the
    mate when every sampled pose is inaccessible.
    """
    from cadgen.magnetics import kinematics

    np = _deps.numpy()
    if scene.mate is None:
        raise MateError("sweep needs a mate, but the scene selected none")
    mate = scene.mate
    swept_key, held_key = kinematics.sweep_dof(mate, cfg.dof)
    lo, hi = mate.limits[swept_key]
    qs = np.linspace(float(lo), float(hi), int(cfg.samples))

    # Held cylindrical sub-DOF: fixed, relaxed per pose, or absent.
    held_values: Any | None
    wedged_mask = np.zeros(len(qs), dtype=bool)
    if held_key is None:
        held_values = None
    elif cfg.relax:
        held_limits = mate.limits[held_key]
        relaxed = np.array(
            [relax_held(scene, float(q), held_key, held_limits) for q in qs], dtype=object
        )
        wedged_mask = np.array([v is None for v in relaxed], dtype=bool)
        held_values = np.array([np.nan if v is None else float(v) for v in relaxed], dtype=float)
    else:
        held_values = np.full(len(qs), float(cfg.hold), dtype=float)

    accessible = np.asarray(
        clearance(scene, qs, held_values, swept=swept_key), dtype=bool
    )
    accessible = accessible & ~wedged_mask
    if not accessible.any():
        raise MateError(f"mate {mate.name!r}: every sampled pose is inaccessible (clearance)")

    finest_mesh = cfg.meshing_levels[-1]
    finest_eps = cfg.eps_levels[-1]
    Q, F, T, F_perp = _grid_forces(scene, cfg, qs, held_values, finest_mesh, finest_eps)
    Q = np.asarray(Q, dtype=float)
    F = np.asarray(F, dtype=float)
    T = np.asarray(T, dtype=float)
    F_perp = np.asarray(F_perp, dtype=float)

    U = energy(Q, qs, accessible)

    # Equilibria: mask inaccessible/wedged out (nan) so no bracket forms there,
    # cross-check against the nested 2N-1 grid, refine with single-pose calls.
    searchable = accessible.copy()
    Q_masked = np.where(searchable, Q, np.nan)
    qs_fine = np.linspace(float(lo), float(hi), 2 * len(qs) - 1)
    held_fine = None
    if held_values is not None:
        held_fine = np.interp(qs_fine, qs, np.nan_to_num(held_values))
    Q_fine = np.asarray(_grid_forces(scene, cfg, qs_fine, held_fine, finest_mesh, finest_eps)[0], dtype=float)
    # Mask the fine grid where its nearest coarse pose is not searchable.
    fine_ok = np.interp(qs_fine, qs, searchable.astype(float)) > 0.5
    Q_fine_masked = np.where(fine_ok, Q_fine, np.nan)

    def refine_fn(q: float) -> float:
        held = _held_at(qs, held_values, q, np)
        hv = None if held is None else [held]
        return float(_grid_forces(scene, cfg, [q], hv, finest_mesh, finest_eps)[0][0])

    equilibria = find_equilibria(Q_masked, Q_fine_masked, qs, qs_fine, refine_fn)

    convergence = converge(scene, cfg, qs, cfg.at if cfg.at is not None else float(lo), equilibria, held_values)
    uncertainty = _uncertainty_from(convergence)
    at_q = float(cfg.at) if cfg.at is not None else float(lo)
    verdict = make_verdict(qs, Q, accessible, at_q, uncertainty, cfg.friction, (float(lo), float(hi)))

    samples = tuple(
        Sample(
            q=float(qs[p]),
            F_N=(float(F[p, 0]), float(F[p, 1]), float(F[p, 2])),
            Q=float(Q[p]),
            F_transverse_N=(float(F_perp[p, 0]), float(F_perp[p, 1]), float(F_perp[p, 2])),
            torque_Nm=(float(T[p, 0]), float(T[p, 1]), float(T[p, 2])),
            U_J=None if not np.isfinite(U[p]) else float(U[p]),
            accessible=bool(accessible[p]),
            held_value=(
                None
                if held_values is None or wedged_mask[p]
                else float(held_values[p])
            ),
        )
        for p in range(len(qs))
    )

    held_record: dict[str, Any] | None = None
    if held_key is not None:
        held_record = {
            "name": f"{mate.name}.{held_key}",
            "mode": "relax" if cfg.relax else "hold",
            "value": None if cfg.relax else float(cfg.hold),
        }

    inaccessible = _mask_to_intervals(qs, ~accessible & ~wedged_mask, np)
    wedged = _mask_to_intervals(qs, wedged_mask, np)

    return SweepResult(
        magnets=scene.magnets,
        mate=mate,
        dof=cfg.dof,
        held=held_record,
        samples=samples,
        inaccessible=inaccessible,
        wedged=wedged,
        equilibria=tuple(equilibria),
        verdict=verdict,
        convergence=convergence,
    )


# --------------------------------------------------------------------------- field


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_AXIS_UNIT = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}


def _bbox_center_extent(scene: MagScene, np: Any) -> tuple[Any, Any]:
    lo = np.asarray(scene.bbox_m[0], dtype=float)
    hi = np.asarray(scene.bbox_m[1], dtype=float)
    return 0.5 * (lo + hi), (hi - lo)


def _shortest_bbox_axis(extent: Any, np: Any) -> str:
    order = ["x", "y", "z"]
    return order[int(np.argmin(extent))]


def _plane_from_spec(scene: MagScene, plane_spec: str, np: Any) -> dict[str, Any]:
    """The plane dict (``origin_m``, ``normal``, ``u_axis``, ``v_axis``) for ``plane_spec``.

    :class:`SceneError` when an axis-normal plane's coordinate misses the bbox;
    :class:`MateError` when ``mate`` is asked for and ``scene.mate`` is ``None``.
    """
    center, extent = _bbox_center_extent(scene, np)
    lo = np.asarray(scene.bbox_m[0], dtype=float)
    hi = np.asarray(scene.bbox_m[1], dtype=float)

    if "=" in plane_spec:
        axis_letter, _, value = plane_spec.partition("=")
        axis_letter = axis_letter.strip()
        if axis_letter not in _AXIS_INDEX:
            raise SceneError(f"slice {plane_spec!r}: unknown axis {axis_letter!r}")
        coord_m = float(value) * 1e-3  # model millimetres -> metres
        ai = _AXIS_INDEX[axis_letter]
        if not (lo[ai] - 1e-12 <= coord_m <= hi[ai] + 1e-12):
            raise SceneError(f"slice {plane_spec!r} misses the model bounding box")
        normal = np.asarray(_AXIS_UNIT[axis_letter], dtype=float)
        others = [a for a in ("x", "y", "z") if a != axis_letter]
        u_axis = np.asarray(_AXIS_UNIT[others[0]], dtype=float)
        v_axis = np.asarray(_AXIS_UNIT[others[1]], dtype=float)
        origin = center.copy()
        origin[ai] = coord_m
        return {
            "spec": plane_spec,
            "origin_m": tuple(float(c) for c in origin),
            "normal": tuple(float(c) for c in normal),
            "u_axis": tuple(float(c) for c in u_axis),
            "v_axis": tuple(float(c) for c in v_axis),
        }

    head, _, up_letter = plane_spec.partition(":")
    if head != "mate":
        raise SceneError(f"slice {plane_spec!r}: expected x=|y=|z=|mate[:up]")
    if scene.mate is None:
        raise MateError("slice 'mate' needs a mate, but the scene selected none")
    axis = np.asarray(scene.mate.dir, dtype=float)
    axis = axis / np.linalg.norm(axis)
    if up_letter:
        if up_letter not in _AXIS_INDEX:
            raise SceneError(f"slice {plane_spec!r}: unknown up axis {up_letter!r}")
        up_order = [up_letter] + [a for a in ("x", "y", "z") if a != up_letter]
    else:
        up_order = [_shortest_bbox_axis(extent, np)] + ["x", "y", "z"]
    normal = None
    for cand in up_order:
        up = np.asarray(_AXIS_UNIT[cand], dtype=float)
        cross = np.cross(axis, up)
        if np.linalg.norm(cross) >= PLANE_PARALLEL_TOL:
            normal = cross / np.linalg.norm(cross)
            break
    if normal is None:
        raise SceneError(f"slice {plane_spec!r}: cannot build a plane from the mate axis")
    u_axis = axis
    v_axis = np.cross(normal, u_axis)
    v_axis = v_axis / np.linalg.norm(v_axis)
    origin = np.asarray(scene.mate.origin_m, dtype=float)
    return {
        "spec": plane_spec,
        "origin_m": tuple(float(c) for c in origin),
        "normal": tuple(float(c) for c in normal),
        "u_axis": tuple(float(c) for c in u_axis),
        "v_axis": tuple(float(c) for c in v_axis),
    }


def _field_sources(scene: MagScene, q: float, held: float | None, np: Any) -> list[Any]:
    sources = [m.source for m in scene.magnets]
    if q != 0.0 and scene.moving and scene.mate is not None:
        from cadgen.magnetics import kinematics

        posed = kinematics.posed_sources(scene, q, held)
        for src, idx in zip(posed, scene.moving):
            sources[idx] = src
    return sources


def _outline_on_plane(solid: Any, plane: dict[str, Any], np: Any) -> Any | None:
    """The magnet's cross-section on the plane as ``(k, 2)`` ``(u, v)`` metres, or ``None``."""
    try:
        from cadgen import build123d as bd

        origin = np.asarray(plane["origin_m"], dtype=float) * _M_TO_MM
        normal = np.asarray(plane["normal"], dtype=float)
        u_axis = np.asarray(plane["u_axis"], dtype=float)
        v_axis = np.asarray(plane["v_axis"], dtype=float)
        cut = bd.Plane(
            origin=tuple(float(c) for c in origin),
            x_dir=tuple(float(c) for c in u_axis),
            z_dir=tuple(float(c) for c in normal),
        )
        section = solid & cut
        pts = []
        for vertex in section.vertices():
            rel = np.array([vertex.X, vertex.Y, vertex.Z], dtype=float) * 1e-3 - np.asarray(
                plane["origin_m"], dtype=float
            )
            pts.append((float(np.dot(rel, u_axis)), float(np.dot(rel, v_axis))))
        if len(pts) < 3:
            return None
        return np.asarray(pts, dtype=float)
    except Exception:
        return None


def field(
    scene: MagScene,
    plane_spec: str,
    grid: int = DEFAULT_GRID,
    q: float = 0.0,
    held: float | None = None,
) -> FieldSlice:
    """``B`` on one plane at one pose, plus the magnet outlines on it.

    ``plane_spec`` is ``x=<mm>``, ``y=<mm>``, ``z=<mm>`` (an axis-normal plane
    at that model coordinate) or ``mate[:<up>]``: the plane containing the
    mate axis with normal ``axis x up``, ``up`` defaulting to the model bounding
    box's shortest axis and falling back to the next axis when ``|axis x up| <
    PLANE_PARALLEL_TOL``. Sources are ``kinematics.posed_sources(scene, q,
    held)`` (single-pose copies). Samples ``getB`` on a ``grid x grid`` lattice
    in metres spanning the bbox on that plane; outlines are each world-placed
    magnet solid intersected with the plane. :class:`SceneError` naming the
    plane when it misses the model bounding box; :class:`MateError` when
    ``mate`` is asked for and ``scene.mate`` is ``None``.
    """
    magpy = _deps.magpylib()
    np = _deps.numpy()

    plane = _plane_from_spec(scene, plane_spec, np)
    origin = np.asarray(plane["origin_m"], dtype=float)
    u_axis = np.asarray(plane["u_axis"], dtype=float)
    v_axis = np.asarray(plane["v_axis"], dtype=float)
    _, extent = _bbox_center_extent(scene, np)

    # Half-extent of the bbox along each in-plane axis (with a small margin).
    half_u = 0.5 * float(np.dot(np.abs(u_axis), extent)) * 1.1 or 1e-3
    half_v = 0.5 * float(np.dot(np.abs(v_axis), extent)) * 1.1 or 1e-3
    u = np.linspace(-half_u, half_u, grid)
    v = np.linspace(-half_v, half_v, grid)
    UU, VV = np.meshgrid(u, v, indexing="ij")
    pts = origin[None, None, :] + UU[..., None] * u_axis + VV[..., None] * v_axis

    sources = _field_sources(scene, q, held, np)
    B = np.asarray(magpy.getB(sources, pts.reshape(-1, 3)), dtype=float)
    if B.ndim == 3:  # (n_sources, n_pts, 3): sum the sources
        B = B.sum(axis=0)
    B = B.reshape(grid, grid, 3)

    outlines = []
    for magnet in scene.magnets:
        outline = _outline_on_plane(magnet.solid, plane, np)
        if outline is not None:
            outlines.append(outline)

    return FieldSlice(plane=plane, u=u, v=v, B=B, outlines=tuple(outlines))
