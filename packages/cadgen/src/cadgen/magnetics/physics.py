"""Mechanics per sweep, SI: ``getFT`` wrappers, projection, clearance, energy,
equilibria, the convergence protocol, the verdict, the field slice (lane A3).

- ``F, T = getFT(fixed, moving_with_paths, pivot=P, squeeze=False)`` ALWAYS with
  ``squeeze=False`` so the result is ``(p, t, 3)`` whatever ``p`` and ``t`` are;
  ``P`` is ``(t, p, 3)``. Reduce explicitly: ``F.sum(axis=1)``, ``T.sum(axis=1)``
  -> ``(p, 3)``. Shapes 1x1, 1x201, 2x201 are unit-tested.
- Slider: ``Q = F . a``, ``F_perp = F - Q a``. Revolute: ``Q = tau . a``.
- Clearance: for each pose transform the moving solids by ``D(q)`` and test
  ``cadgen.geometry.overlap_volume`` against every fixed solid (magnets and
  non-magnets alike) behind a bounding-box prefilter; overlap above 1e-3 mm^3
  is ``inaccessible`` -- excluded from equilibrium search, energy integrated
  per accessible interval, hatched in the report.
- ``U(q) - U(q0) = -integral Q dq`` by trapezoid within each accessible interval.
- Equilibria: sign changes of ``Q`` present at BOTH pose-grid resolutions,
  refined by bisection on fresh single-pose ``getFT`` calls; stable iff
  ``dQ/dq < 0``.
- Convergence protocol at the verdict pose and each equilibrium: mesh in
  ``{20, 50, 100}`` at eps 1e-5 m gives ``dQ_mesh``; eps in ``{1e-5, 1e-6}`` at
  mesh 100 gives ``dQ_eps``; the finest level at ``N`` and ``2N - 1`` nested
  poses gives ``dQ_grid``, equilibrium agreement and ``dU_max``.
  ``uncertainty = max(dQ_mesh, dQ_eps, dQ_grid)``. ``converged`` is true when
  each relative delta (guarded by ``Q_scale = 0.02 * max|Q|``) is under
  ``converge_tol``, equilibria agree within 1 % of range and ``dU_max < 2 %``
  of the energy span; otherwise false with ``reasons``. ``quick`` runs mesh
  {20, 50} and grid only; ``full`` everything; ``off`` the finest level once,
  ``converged: None``.
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

Every function below is a stub for lane A3: ``raise NotImplementedError("lane A3")``.
No magpylib/numpy/scipy import at module scope -- use ``cadgen.magnetics._deps``.
No CAD-kernel import at module scope (``cadgen.geometry`` is imported inside
``clearance``).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from cadgen.magnetics.types import (  # noqa: F401 - part of the contract
    Convergence,
    Equilibrium,
    FieldSlice,
    MagScene,
    MateError,
    MateSpec,
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


def force_torque(
    fixed: Sequence[Any],
    moving_pathed: Sequence[Any],
    pivot: Any,
    meshing: int,
    eps: float,
) -> tuple[Any, Any]:
    """Net ``(F (p, 3), T (p, 3))`` in N and N m on the moving group over its path.

    ``magpylib.getFT(fixed, moving_pathed, pivot=pivot, eps=eps, squeeze=False)``
    with ``meshing`` set on every target first; result ``(p, t, 3)`` reduced by
    an explicit ``.sum(axis=1)``. ``pivot`` is ``(t, p, 3)`` from
    ``kinematics.pivot_path``. A test pins ``inspect.signature(getFT)`` so a
    magpylib upgrade fails loudly.
    """
    raise NotImplementedError("lane A3")


def project(F: Any, T: Any, mate: MateSpec, dof: str | None) -> tuple[Any, Any]:
    """``(Q (p,), F_perp (p, 3))``: the generalized force along the swept DOF.

    Translation (slider, ``travel``): ``Q = F . a``, ``F_perp = F - Q a``.
    Rotation (revolute, ``turn``): ``Q = T . a`` (N m), ``F_perp = F`` (the
    whole force is a guide load).
    """
    raise NotImplementedError("lane A3")


def clearance(
    scene: MagScene,
    qs: Sequence[float],
    held_values: Sequence[float] | None = None,
) -> Any:
    """Boolean ``accessible`` mask ``(p,)``: no moving solid overlaps a fixed one.

    Moves each moving magnet's ``solid`` (model units) by ``delta(q)`` scaled
    back to millimetres, prefilters by bounding box against
    ``scene.solids_fixed``, and calls ``cadgen.geometry.overlap_volume`` on the
    survivors; overlap above :data:`CLEARANCE_THRESHOLD_MM3` marks the pose
    inaccessible. Two tests: overlapping boxes yield an interval, separated
    boxes none. :class:`MateError` naming the mate when EVERY pose is
    inaccessible is raised by :func:`sweep`, not here.
    """
    raise NotImplementedError("lane A3")


def energy(Q: Any, qs: Sequence[float], accessible: Any) -> Any:
    """``U (p,)`` in joules: ``-integral Q dq`` by trapezoid per accessible interval.

    Each accessible interval starts at ``U = 0`` at its own first pose;
    inaccessible poses carry ``nan`` (``Sample.U_J`` is ``None`` there).
    Synthetic ``Q = -sin(q)`` gives ``U = cos(q) - 1`` up to the interval offset.
    """
    raise NotImplementedError("lane A3")


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
    anywhere -> ``[]``. Inaccessible and wedged poses are masked out by the
    caller before this runs.
    """
    raise NotImplementedError("lane A3")


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
    raise NotImplementedError("lane A3")


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
    raise NotImplementedError("lane A3")


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
    raise NotImplementedError("lane A3")


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
    raise NotImplementedError("lane A3")


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
    raise NotImplementedError("lane A3")
