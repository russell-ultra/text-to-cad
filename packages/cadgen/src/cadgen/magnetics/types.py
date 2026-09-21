"""The frozen data contract of :mod:`cadgen.magnetics`.

Every value here is SI: metres, radians, tesla, newtons, newton-metres, joules.
Model units (millimetres, degrees) enter the package in exactly one place,
:func:`cadgen.magnetics.scene.load`, and nothing downstream of it sees them.

These are data classes and exceptions only -- no behaviour lives here. The one
function, :func:`sweep_result_to_json`, is the serializer for the
``magnetics-report/1`` schema that ``sweep.json`` carries; it is pure over the
dataclasses above it.

stdlib only: this module is imported by ``cadgen.cli.magnetics`` on every
invocation, including ``--help``, and cadgen must import with the ``magnetics``
extra absent. magpylib objects and numpy arrays are carried as ``Any``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "SCHEMA",
    "UNITS",
    "DOF_UNITS",
    "MagnetShape",
    "MateKind",
    "Vec3",
    "Quat",
    "MagnetSpec",
    "MateSpec",
    "MagScene",
    "SweepConfig",
    "Sample",
    "Equilibrium",
    "Verdict",
    "Convergence",
    "SweepResult",
    "FieldSlice",
    "MagneticsError",
    "MagneticsDependencyError",
    "SceneError",
    "MateError",
    "ConvergenceWarning",
    "sweep_result_to_json",
]

#: The ``sweep.json`` schema tag.
SCHEMA = "magnetics-report/1"

#: The ``units`` block every ``sweep.json`` carries.
UNITS = {"length": "m", "force": "N", "torque": "N m", "energy": "J", "B": "T"}

#: Unit of each limits key a mate can carry. ``value`` is the single DOF of a
#: slider (m) or revolute (rad) and is resolved per mate kind by ``scene.load``;
#: a cylindrical mate's two sub-DOFs have fixed units.
DOF_UNITS = {"travel": "m", "turn": "rad"}

MagnetShape = Literal["cuboid", "cylinder", "mesh"]
MateKind = Literal["slider", "revolute", "cylindrical"]
Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


# --------------------------------------------------------------------------- errors


class MagneticsError(Exception):
    """Base of every analysis error; the CLI maps a subclass to exit 1."""


class MagneticsDependencyError(MagneticsError):
    """The ``magnetics`` extra is not installed; the message names the fix."""


class SceneError(MagneticsError):
    """A STEP leaf, its name, its solid or a slice plane cannot become physics.

    Raised naming the leaf ref (``#o1.2.3``) and the token or check at fault:
    unknown grade, malformed ``Br=``, zero-length direction, open or
    disconnected mesh, no ``mag:`` leaf in the file, a plane missing the model.
    """


class MateError(MagneticsError):
    """The mate cannot drive a sweep.

    Raised naming the mate: none or several mates and no ``--mate``, an unknown
    name, an unsupported kind, a missing ``childId``, a moving subtree without
    magnets, reversed or degenerate limits, every sampled pose inaccessible.
    """


class ConvergenceWarning(UserWarning):
    """Emitted when ``converged`` is ``False``: the run completed (exit 0) but
    a refinement axis (``mesh``, ``eps``, ``pose_grid``) moved the answer more
    than ``converge_tol``."""


# --------------------------------------------------------------------------- scene


@dataclass(frozen=True)
class MagnetSpec:
    """One magpylib source, with the record ``inspect`` prints about it.

    ``source`` is the magpylib object at REST (``q = 0``), in metres, with
    body-frame ``dimension``/``polarization`` rotated by ``orientation`` --
    never rotated twice. ``solid`` is the world-placed build123d solid in
    MODEL units, kept for clearance; nothing but ``physics.clearance`` reads it.
    """

    ref: str
    label: str
    shape: MagnetShape
    source: Any
    polarization_world_T: Vec3
    position_m: Vec3
    orientation_quat: Quat
    moving: bool | None
    solid: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class MateSpec:
    """The one mate a sweep drives, in SI.

    ``limits`` is keyed ``"value"`` for a slider (m) or revolute (rad) and
    ``"travel"`` (m) / ``"turn"`` (rad) for a cylindrical mate. ``q_unit`` is
    the unit of ``limits["value"]`` for the single-DOF kinds and ``"m"`` (the
    ``travel`` unit) for a cylindrical mate; the swept sub-DOF chosen by
    ``--dof`` decides the unit ``sweep.json`` reports (``DOF_UNITS``).
    ``origin_m``/``dir`` are the world-at-rest axis; ``dir`` is unit length.
    """

    name: str
    kind: MateKind
    child_id: str
    origin_m: Vec3
    dir: Vec3
    limits: dict[str, tuple[float, float]]
    q_unit: str


@dataclass(frozen=True)
class MagScene:
    """A STEP document as physics: the output of ``scene.load``, frozen and SI.

    ``fixed``/``moving`` index ``magnets``. ``mate`` is ``None`` when no mate
    was selected (``inspect`` without ``--mate``), in which case ``moving`` is
    empty and every ``MagnetSpec.moving`` is ``None``. ``bbox_m`` is
    ``(min_xyz, max_xyz)`` over every solid in the model, magnets or not.
    ``solids_fixed`` are the world build123d solids (MODEL units) of every
    non-moving occurrence, magnets and non-magnets alike, for clearance.
    """

    step_path: str
    magnets: tuple[MagnetSpec, ...]
    fixed: tuple[int, ...]
    moving: tuple[int, ...]
    mate: MateSpec | None
    bbox_m: tuple[Vec3, Vec3]
    solids_fixed: tuple[Any, ...] = field(repr=False, compare=False)


# --------------------------------------------------------------------------- sweep


@dataclass(frozen=True)
class SweepConfig:
    """What ``cadgen magnetics sweep`` asked for, already validated by argparse.

    ``friction`` is the threshold in the swept DOF's generalized unit (N for a
    translation, N m for a rotation); ``None`` means no threshold. ``at`` is
    the verdict pose (``None`` = lower limit). ``dof`` is ``"<mate>.travel"``
    or ``"<mate>.turn"`` for a cylindrical mate, ``None`` otherwise. ``hold``
    is the held sub-DOF's value unless ``relax`` is set. ``meshing_levels``
    and ``eps_levels`` are the refinement ladders; ``convergence`` picks how
    much of them runs (``"off"`` = finest level once, ``converged: None``).
    """

    samples: int = 201
    convergence: Literal["off", "quick", "full"] = "quick"
    converge_tol: float = 0.02
    friction: float | None = None
    at: float | None = None
    dof: str | None = None
    hold: float = 0.0
    relax: bool = False
    meshing_levels: tuple[int, ...] = (20, 50, 100)
    eps_levels: tuple[float, ...] = (1e-5, 1e-6)


@dataclass(frozen=True)
class Sample:
    """One pose of the sweep. ``U_J`` is ``None`` at an inaccessible pose
    (energy is integrated per accessible interval). ``held_value`` is the
    held sub-DOF's value at this pose (``--hold`` or the relaxed root),
    ``None`` for single-DOF mates and for a wedged pose."""

    q: float
    F_N: Vec3
    Q: float
    F_transverse_N: Vec3
    torque_Nm: Vec3
    U_J: float | None
    accessible: bool
    held_value: float | None = None


@dataclass(frozen=True)
class Equilibrium:
    """A root of ``Q(q)`` present at both pose-grid resolutions and refined by
    bisection; ``stable`` iff ``dQdq < 0``."""

    q: float
    stable: bool
    dQdq: float
    uncertainty_q: float


@dataclass(frozen=True)
class Verdict:
    """The split verdict at ``at_q``.

    ``tendency`` is ``"+axis"``, ``"-axis"`` or ``"indeterminate"`` (when
    ``|Q| < uncertainty``). ``exceeds_threshold`` is ``None`` when no threshold
    was given or the band ``[|Q| - u, |Q| + u]`` straddles it. ``admissible``
    is ``False`` when the tendency points outside the limits at a limit pose
    or into an inaccessible interval.
    """

    at_q: float
    tendency: Literal["+axis", "-axis", "indeterminate"]
    Q: float
    uncertainty: float
    threshold: float | None
    exceeds_threshold: bool | None
    admissible: bool


@dataclass(frozen=True)
class Convergence:
    """The protocol's record. Each axis block is the ``sweep.json`` dict for it
    (``mesh``: levels, eps, dQ; ``eps``: levels, meshing, dQ; ``pose_grid``:
    samples, dQ, equilibria_agree, dU_max) or ``None`` when that axis did not
    run. ``converged`` is ``None`` for ``--convergence off``."""

    mesh: dict[str, Any] | None
    eps: dict[str, Any] | None
    pose_grid: dict[str, Any] | None
    converged: bool | None
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class SweepResult:
    """Everything ``sweep.json`` and ``report.html`` are written from.

    ``dof`` is the swept sub-DOF id for a cylindrical mate (``None`` otherwise);
    ``held`` is ``{"name", "mode": "hold" | "relax", "value"}`` for it, with
    ``value`` ``None`` under ``relax`` (per-sample values ride ``Sample``).
    ``inaccessible`` and ``wedged`` are closed ``(lo, hi)`` intervals in ``q``.
    """

    magnets: tuple[MagnetSpec, ...]
    mate: MateSpec
    dof: str | None
    held: dict[str, Any] | None
    samples: tuple[Sample, ...]
    inaccessible: tuple[tuple[float, float], ...]
    wedged: tuple[tuple[float, float], ...]
    equilibria: tuple[Equilibrium, ...]
    verdict: Verdict
    convergence: Convergence


# --------------------------------------------------------------------------- field


@dataclass(frozen=True)
class FieldSlice:
    """A sampled ``B`` on one plane, at one pose.

    ``plane`` is ``{"spec", "origin_m", "normal", "u_axis", "v_axis"}`` (world,
    metres, unit axes); ``u``/``v`` are the grid coordinates ``(N,)`` along
    those axes; ``B`` is ``(N, N, 3)`` tesla in WORLD components; ``outlines``
    are the magnet solids intersected with the plane, each a ``(k, 2)`` array
    of ``(u, v)`` points in metres.
    """

    plane: dict[str, Any]
    u: Any
    v: Any
    B: Any
    outlines: tuple[Any, ...]


# --------------------------------------------------------------------------- json


def _swept_unit(result: SweepResult) -> str:
    if result.dof is not None:
        key = result.dof.rsplit(".", 1)[-1]
        return DOF_UNITS.get(key, result.mate.q_unit)
    return result.mate.q_unit


def _limits_for_json(result: SweepResult) -> list[float]:
    key = "value"
    if result.dof is not None:
        key = result.dof.rsplit(".", 1)[-1]
    lo, hi = result.mate.limits[key]
    return [lo, hi]


def sweep_result_to_json(result: SweepResult) -> dict[str, Any]:
    """The ``magnetics-report/1`` document for ``result``.

    Pure over the dataclasses: lists for tuples, ``None`` for JSON ``null``,
    nothing rounded (the report and the sweep.json reader do that). The shape
    is the design's example: ``schema``, ``units``, ``q_unit``, ``magnets``,
    ``mate`` (with ``dof`` and ``held`` only for a cylindrical sweep),
    ``samples``, ``inaccessible``, ``wedged``, ``equilibria``, ``verdict``,
    ``convergence``.
    """
    mate: dict[str, Any] = {
        "name": result.mate.name,
        "kind": result.mate.kind,
        "axis": list(result.mate.dir),
        "origin_m": list(result.mate.origin_m),
        "limits": _limits_for_json(result),
    }
    if result.dof is not None:
        mate["dof"] = result.dof
        mate["held"] = dict(result.held) if result.held is not None else None
    conv = result.convergence
    return {
        "schema": SCHEMA,
        "units": dict(UNITS),
        "q_unit": _swept_unit(result),
        "magnets": [
            {
                "ref": m.ref,
                "label": m.label,
                "shape": m.shape,
                "polarization_world_T": list(m.polarization_world_T),
                "position_m": list(m.position_m),
                "orientation_quat": list(m.orientation_quat),
                "moving": m.moving,
            }
            for m in result.magnets
        ],
        "mate": mate,
        "samples": [
            {
                "q": s.q,
                "F_N": list(s.F_N),
                "Q": s.Q,
                "F_transverse_N": list(s.F_transverse_N),
                "torque_Nm": list(s.torque_Nm),
                "U_J": s.U_J,
                "accessible": s.accessible,
                **({"held_value": s.held_value} if result.dof is not None else {}),
            }
            for s in result.samples
        ],
        "inaccessible": [list(interval) for interval in result.inaccessible],
        "wedged": [list(interval) for interval in result.wedged],
        "equilibria": [
            {"q": e.q, "stable": e.stable, "dQdq": e.dQdq, "uncertainty_q": e.uncertainty_q}
            for e in result.equilibria
        ],
        "verdict": {
            "at_q": result.verdict.at_q,
            "tendency": result.verdict.tendency,
            "Q": result.verdict.Q,
            "uncertainty": result.verdict.uncertainty,
            "threshold": result.verdict.threshold,
            "exceeds_threshold": result.verdict.exceeds_threshold,
            "admissible": result.verdict.admissible,
        },
        "convergence": {
            "mesh": dict(conv.mesh) if conv.mesh is not None else None,
            "eps": dict(conv.eps) if conv.eps is not None else None,
            "pose_grid": dict(conv.pose_grid) if conv.pose_grid is not None else None,
            "converged": conv.converged,
            "reasons": list(conv.reasons),
        },
    }
