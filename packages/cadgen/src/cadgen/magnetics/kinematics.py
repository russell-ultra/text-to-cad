"""Mates as motion: sidecar -> :class:`MateSpec`, the moving set by ``childId``,
``delta(mate, q)``, posed and pathed magpylib copies (lane A2).

The sidecar is loaded through ``cadgen._internal.source_sidecar.read_source_sidecar``,
so schema version and ``documentHash`` are enforced against the STEP bytes; a
stale pair propagates cadgen's binding error (the CLI maps it to exit 1). Each
mate is ``{"name", "kind", "parent", "child", "childId", "axis": {"origin",
"dir"}, "limits": {"value": [lo, hi]}}`` (cylindrical: ``limits`` keyed
``"turn"``/``"travel"``) with ``axis`` in world-at-rest MODEL units and
revolute/turn limits in DEGREES. One mate of kind ``slider``, ``revolute`` or
``cylindrical`` drives a sweep: ``--mate`` selects it, a sole mate is the
default, otherwise :class:`MateError`. The moving set is
``scene.resolve("#" + childId)`` plus all descendants; the authored ``#label``
is never used for selection.

Pose convention matches the viewer's forward kinematics
(``packages/cadgen-js/src/common/kinematicsRuntime.js``, the convention source):
the artifact as written is ``q = 0`` and ``world(q) = D(q) @ world(0)``,
premultiplied, with

    slider       D = T(dir * q)
    revolute     D = T(origin) @ R(dir, q) @ T(-origin)
    cylindrical  D = T(dir * travel) @ T(origin) @ R(dir, turn) @ T(-origin)

(the two cylindrical factors commute: both are about the same axis). Slider
``q`` in metres, revolute ``q`` in radians. Fixed slider orientation is an
explicit simplification of the physical pin-slot.

No magpylib/numpy/scipy import at module scope -- use ``cadgen.magnetics._deps``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from cadgen.magnetics import _deps
from cadgen.magnetics.types import MagnetSpec, MagScene, MateError, MateSpec  # noqa: F401 - part of the contract

__all__ = [
    "SWEPT_SINGLE",
    "select_mate",
    "moving_indices",
    "sweep_dof",
    "delta",
    "posed_sources",
    "pathed_sources",
    "pivot_path",
]

#: The limits key of a single-DOF mate (slider, revolute).
SWEPT_SINGLE = "value"

#: cadgen models are millimetres; every length crossing into SI is scaled here.
_MM_TO_M = 1e-3

#: The mate kinds a sweep can drive.
_SUPPORTED_KINDS = ("slider", "revolute", "cylindrical")

#: The two sub-DOFs of a cylindrical mate and the unit each limit is stored in.
_CYLINDRICAL_KEYS = ("travel", "turn")


# --------------------------------------------------------------------------- sidecar


def _mates_block(source: Path | str | Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The ``kinematics.mates`` list, from a STEP path or an already-read mapping.

    A STEP path goes through ``read_source_sidecar`` so ``documentHash`` and the
    schema are enforced (a stale/wrong-schema pair raises cadgen's own error,
    which propagates). A mapping is either a whole sidecar payload (with a
    ``"kinematics"`` block) or the kinematics block itself. Missing sidecar or
    no mates raises :class:`MateError`.
    """
    if isinstance(source, Mapping):
        payload: Mapping[str, Any] = source
        where = "the model"
    else:
        from cadgen._internal.source_sidecar import read_source_sidecar

        step = Path(source)
        payload = read_source_sidecar(step)  # SidecarSchemaError/SidecarBindingError propagate
        where = step.name
        if payload is None:
            raise MateError(f"{where} declares no mates: it has no kinematics sidecar")
    kinematics = payload.get("kinematics", payload)
    mates = kinematics.get("mates") if isinstance(kinematics, Mapping) else None
    if not mates:
        raise MateError(f"{where} declares no mates to drive")
    return list(mates)


def _mate_names(mates: Sequence[Mapping[str, Any]]) -> str:
    return ", ".join(repr(str(m.get("name", ""))) for m in mates) or "(none)"


def _unit_dir(raw: Any, *, mate_name: str) -> tuple[float, float, float]:
    axis = raw if isinstance(raw, (list, tuple)) and len(raw) == 3 else None
    if axis is None:
        raise MateError(f"mate {mate_name!r}: axis dir is not a 3-vector")
    x, y, z = (float(axis[0]), float(axis[1]), float(axis[2]))
    length = math.sqrt(x * x + y * y + z * z)
    if length == 0.0:
        raise MateError(f"mate {mate_name!r}: axis direction has zero length")
    return (x / length, y / length, z / length)


def _limit_pair(limits: Mapping[str, Any], key: str, scale: float, *, mate_name: str, unit: str) -> tuple[float, float]:
    raw = limits.get(key) if isinstance(limits, Mapping) else None
    if not (isinstance(raw, (list, tuple)) and len(raw) == 2):
        raise MateError(f"mate {mate_name!r}: limits[{key!r}] is not a [lo, hi] pair")
    lo, hi = float(raw[0]) * scale, float(raw[1]) * scale
    if lo >= hi:
        raise MateError(
            f"mate {mate_name!r}: limits[{key!r}] = [{lo}, {hi}] {unit} is reversed or degenerate (lo must be < hi)"
        )
    return (lo, hi)


def select_mate(sidecar: Path | str | Mapping[str, Any], mate_name: str | None = None) -> MateSpec:
    """The mate a sweep drives, in SI.

    ``sidecar`` is a STEP path -- loaded through ``read_source_sidecar`` so
    ``documentHash`` and the schema are enforced (a stale or missing sidecar
    raises cadgen's own ``SidecarBindingError``/``SidecarSchemaError``, or
    :class:`MateError` when the sidecar is absent or carries no ``kinematics``
    block) -- or an already-read sidecar mapping. ``mate_name`` selects among
    the block's mates; ``None`` takes the sole mate. :class:`MateError` names
    the available mates when the name matches none or the model has zero or
    two-plus mates and none was given; names the mate and its kind when the
    kind is not slider/revolute/cylindrical; names the mate when ``childId`` is
    missing or ``limits`` is reversed or degenerate (``lo == hi``).
    Conversions: origin mm -> m, revolute and ``turn`` limits degrees -> radians,
    ``travel`` and slider limits mm -> m; ``dir`` normalised; ``q_unit`` per
    :class:`MateSpec`.
    """
    mates = _mates_block(sidecar)
    if mate_name is None:
        if len(mates) != 1:
            raise MateError(
                f"the model has {len(mates)} mates ({_mate_names(mates)}); pass --mate to choose one"
            )
        mate = mates[0]
    else:
        matched = [m for m in mates if str(m.get("name", "")) == mate_name]
        if not matched:
            raise MateError(f"no mate named {mate_name!r}; available mates: {_mate_names(mates)}")
        mate = matched[0]

    name = str(mate.get("name", ""))
    kind = str(mate.get("kind", ""))
    if kind not in _SUPPORTED_KINDS:
        raise MateError(f"mate {name!r} is a {kind!r} mate; only {', '.join(_SUPPORTED_KINDS)} drive a sweep")

    child_id = str(mate.get("childId", "")).strip()
    if not child_id:
        raise MateError(f"mate {name!r} has no childId; rebuild the model so the mate resolves to an occurrence")

    axis = mate.get("axis") if isinstance(mate.get("axis"), Mapping) else {}
    origin_raw = axis.get("origin", (0.0, 0.0, 0.0))
    if not (isinstance(origin_raw, (list, tuple)) and len(origin_raw) == 3):
        raise MateError(f"mate {name!r}: axis origin is not a 3-vector")
    origin_m = tuple(float(v) * _MM_TO_M for v in origin_raw)
    direction = _unit_dir(axis.get("dir"), mate_name=name)

    raw_limits = mate.get("limits") if isinstance(mate.get("limits"), Mapping) else {}
    if kind == "cylindrical":
        limits = {
            "travel": _limit_pair(raw_limits, "travel", _MM_TO_M, mate_name=name, unit="m"),
            "turn": _limit_pair(_radian_limits(raw_limits, "turn"), "turn", 1.0, mate_name=name, unit="rad"),
        }
        q_unit = "m"  # the travel unit; the swept sub-DOF decides the reported unit
    elif kind == "revolute":
        limits = {"value": _limit_pair(_radian_limits(raw_limits, "value"), "value", 1.0, mate_name=name, unit="rad")}
        q_unit = "rad"
    else:  # slider
        limits = {"value": _limit_pair(raw_limits, "value", _MM_TO_M, mate_name=name, unit="m")}
        q_unit = "m"

    return MateSpec(
        name=name,
        kind=kind,  # type: ignore[arg-type]
        child_id=child_id,
        origin_m=origin_m,  # type: ignore[arg-type]
        dir=direction,
        limits=limits,
        q_unit=q_unit,
    )


def _radian_limits(limits: Mapping[str, Any], key: str) -> dict[str, Any]:
    """A copy of ``limits`` with ``key`` converted degrees -> radians in place.

    Degrees are the model unit cadgen stores for a revolute or a ``turn``
    sub-DOF; ``_limit_pair`` then applies a unit scale of 1.0 to the result.
    """
    raw = limits.get(key) if isinstance(limits, Mapping) else None
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return {**limits, key: (math.radians(float(raw[0])), math.radians(float(raw[1])))}
    return dict(limits) if isinstance(limits, Mapping) else {}


# --------------------------------------------------------------------------- moving set


def _occurrence_ref(ref: str) -> str:
    """The occurrence portion of a magnet ref: ``#o1.2`` or ``#o1.2.s1`` -> ``#o1.2``.

    A magnet's ref is a leaf occurrence, possibly carrying an entity suffix
    (``.sN``/``.fN``/``.eN``/``.vN``). Occurrence path segments are pure
    integers, so the first non-integer segment begins the entity suffix.
    """
    if not ref.startswith("#o"):
        return ref
    segments = ref[2:].split(".")
    kept: list[str] = []
    for segment in segments:
        if segment.isdigit():
            kept.append(segment)
        else:
            break
    return "#o" + ".".join(kept)


def moving_indices(scene_obj: Any, child_id: str, magnets: Sequence[MagnetSpec]) -> tuple[int, ...]:
    """Indices into ``magnets`` of every magnet under the mate's child occurrence.

    ``scene_obj`` is the ``cadgen.read_scene`` StepScene; the subtree is
    ``scene_obj.resolve("#" + child_id)`` and its ``.children`` recursively,
    matched to ``MagnetSpec.ref`` (a leaf may hold several magnets, all move).
    Selection is by numeric occurrence id, so duplicate labels still select the
    right subtree. :class:`MateError` naming the mate's child ref when the
    subtree contains no magnets.
    """
    ref = "#" + str(child_id).lstrip("#")
    root = scene_obj.resolve(ref)

    subtree: set[str] = set()
    stack = [root]
    while stack:
        occ = stack.pop()
        subtree.add(occ.ref)
        stack.extend(occ.children)

    indices = tuple(i for i, magnet in enumerate(magnets) if _occurrence_ref(magnet.ref) in subtree)
    if not indices:
        raise MateError(f"the moving subtree {ref} carries no magnets")
    return indices


# --------------------------------------------------------------------------- sub-DOFs


def sweep_dof(mate: MateSpec, dof: str | None) -> tuple[str, str | None]:
    """``(swept_key, held_key)`` -- the limits keys a sweep moves and holds.

    Single-DOF mates: ``("value", None)``; ``dof`` must be ``None`` there. A
    cylindrical mate needs ``dof`` = ``"<mate>.travel"`` or ``"<mate>.turn"``
    and returns ``("travel", "turn")`` or ``("turn", "travel")``.
    :class:`MateError` naming the mate and its sub-DOFs when ``dof`` is missing
    for a cylindrical mate or names a sub-DOF the mate lacks (the CLI checks
    the ``--dof`` spelling first and maps this row to exit 2).
    """
    if mate.kind != "cylindrical":
        if dof is not None:
            raise MateError(
                f"mate {mate.name!r} is a single-DOF {mate.kind} mate; --dof applies only to cylindrical mates"
            )
        return (SWEPT_SINGLE, None)

    subs = f"{mate.name}.travel or {mate.name}.turn"
    if dof is None:
        raise MateError(f"cylindrical mate {mate.name!r} needs --dof to choose the swept sub-DOF ({subs})")
    swept = dof.rsplit(".", 1)[-1]
    if swept not in _CYLINDRICAL_KEYS:
        raise MateError(f"--dof {dof!r} names no sub-DOF of cylindrical mate {mate.name!r}; use {subs}")
    held = "turn" if swept == "travel" else "travel"
    return (swept, held)


# --------------------------------------------------------------------------- poses


def _rigid_transform(mate: MateSpec, q: float, held: float | None, swept: str):
    """The ``(4, 4)`` premultiplier plus its ``(R, t)`` blocks, all in metres."""
    np = _deps.numpy()
    Rotation = _deps.scipy_rotation()

    axis = np.asarray(mate.dir, dtype=float)
    length = float(np.linalg.norm(axis))
    if length == 0.0:
        raise MateError(f"mate {mate.name!r}: axis direction has zero length")
    axis = axis / length
    origin = np.asarray(mate.origin_m, dtype=float)

    def translation(vec):
        matrix = np.eye(4)
        matrix[:3, 3] = vec
        return matrix

    def rotation_about(angle: float):
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_rotvec(axis * float(angle)).as_matrix()
        # T(origin) @ R @ T(-origin): rotate about the world-at-rest axis point.
        to_origin = translation(-origin)
        back = translation(origin)
        return back @ matrix @ to_origin

    if mate.kind == "slider":
        matrix = translation(axis * float(q))
    elif mate.kind == "revolute":
        matrix = rotation_about(float(q))
    elif mate.kind == "cylindrical":
        other = 0.0 if held is None else float(held)
        if swept == "travel":
            travel, turn = float(q), other
        elif swept == "turn":
            turn, travel = float(q), other
        else:
            raise MateError(
                f"mate {mate.name!r}: swept sub-DOF {swept!r} is neither 'travel' nor 'turn'"
            )
        # D = T(dir * travel) @ T(origin) @ R(dir, turn) @ T(-origin); the two commute.
        matrix = translation(axis * travel) @ rotation_about(turn)
    else:
        raise MateError(f"mate {mate.name!r}: unsupported kind {mate.kind!r}")

    return matrix, matrix[:3, :3], matrix[:3, 3]


def delta(mate: MateSpec, q: float, held: float | None = None, *, swept: str = SWEPT_SINGLE) -> Any:
    """``D(q)``: the ``(4, 4)`` world premultiplier that poses the moving set, metres.

    ``swept`` is the limits key ``q`` moves (``"value"``, ``"travel"`` or
    ``"turn"``) and ``held`` the other cylindrical sub-DOF's value (``None`` for
    single-DOF mates, treated as 0 for cylindrical). The three forms are in the
    module docstring; each is tested against a hand-derived matrix (slider
    10 mm along x; revolute 90 degrees about z through (1, 0, 0); cylindrical
    10 mm + 90 degrees about the same axis) with the JS runtime cited as the
    convention source.
    """
    matrix, _, _ = _rigid_transform(mate, q, held, swept)
    return matrix


def _apply_to_source(source: Any, rotation_matrix: Any, translation: Any) -> Any:
    """A fresh copy of ``source`` posed by ``D``: ``p' = R @ p + t``, ``orient' = R @ orient``."""
    Rotation = _deps.scipy_rotation()
    posed = source.copy()
    posed.rotate(Rotation.from_matrix(rotation_matrix), anchor=(0, 0, 0))
    posed.move(translation)
    return posed


def posed_sources(
    scene: MagScene, q: float, held_value: float | None = None, *, swept: str = SWEPT_SINGLE
) -> list[Any]:
    """Fresh SINGLE-POSE copies of the moving magnets at ``q`` (``MagScene.at(q)``).

    Each copy's ``position``/``orientation`` is ``delta(q) @ rest``; the scene's
    own sources are never mutated. This is what ``field`` and every bisection
    step use: a path source would broadcast a slice grid to ``(201, 14400, 3)``.
    """
    if scene.mate is None:
        raise MateError("scene has no mate to pose the moving set with")
    _, rotation_matrix, translation = _rigid_transform(scene.mate, q, held_value, swept)
    return [_apply_to_source(scene.magnets[i].source, rotation_matrix, translation) for i in scene.moving]


def pathed_sources(
    scene: MagScene,
    qs: Sequence[float],
    held_values: Sequence[float] | None = None,
    *,
    swept: str = SWEPT_SINGLE,
) -> list[Any]:
    """Copies of the moving magnets carrying a POSITION + ORIENTATION PATH over ``qs``.

    Path length equals ``len(qs)``; ``held_values`` (same length) is the held
    sub-DOF per pose for a cylindrical mate. This is the vectorised sweep's
    target list for ``physics.force_torque``.
    """
    if scene.mate is None:
        raise MateError("scene has no mate to pose the moving set with")
    np = _deps.numpy()
    Rotation = _deps.scipy_rotation()

    rotations = []  # (p, 3, 3)
    translations = []  # (p, 3)
    for index, q in enumerate(qs):
        held = None if held_values is None else held_values[index]
        _, rotation_matrix, translation = _rigid_transform(scene.mate, q, held, swept)
        rotations.append(np.asarray(rotation_matrix))
        translations.append(np.asarray(translation))
    rotations = np.stack(rotations) if rotations else np.empty((0, 3, 3))
    translations = np.stack(translations) if translations else np.empty((0, 3))

    pathed: list[Any] = []
    for i in scene.moving:
        source = scene.magnets[i].source.copy()
        rest_position = np.asarray(source.position, dtype=float)
        rest_matrix = source.orientation.as_matrix()
        # p_k = R_k @ p0 + t_k; orient_k = R_k @ orient0 -- one pose per q.
        positions = np.einsum("kij,j->ki", rotations, rest_position) + translations
        orientations = np.einsum("kij,jl->kil", rotations, rest_matrix)
        source.position = positions
        source.orientation = Rotation.from_matrix(orientations)
        pathed.append(source)
    return pathed


def pivot_path(
    scene: MagScene,
    qs: Sequence[float],
    held_values: Sequence[float] | None = None,
    *,
    swept: str = SWEPT_SINGLE,
) -> Any:
    """The torque pivot per pose, ``(t, p, 3)`` metres, for ``getFT(pivot=...)``.

    ``t`` is the number of moving targets and ``p == len(qs)``. A translation
    (slider, ``travel``) pivots at the moving group's centroid at every pose;
    a rotation (revolute, ``turn``) at a point on the axis, repeated.
    """
    if scene.mate is None:
        raise MateError("scene has no mate to pose the moving set with")
    np = _deps.numpy()
    targets = len(scene.moving)
    poses = len(qs)

    if _is_rotation(scene.mate, swept):
        origin = np.asarray(scene.mate.origin_m, dtype=float)
        return np.broadcast_to(origin, (targets, poses, 3)).copy()

    # Translation: the moving group's centroid, carried by D(q) at each pose.
    rest_positions = np.stack(
        [np.asarray(scene.magnets[i].source.position, dtype=float) for i in scene.moving]
    )
    rest_centroid = rest_positions.mean(axis=0)
    centroids = np.empty((poses, 3))
    for index, q in enumerate(qs):
        held = None if held_values is None else held_values[index]
        _, rotation_matrix, translation = _rigid_transform(scene.mate, q, held, swept)
        centroids[index] = np.asarray(rotation_matrix) @ rest_centroid + np.asarray(translation)
    # Same centroid for every target at a given pose: broadcast across t.
    return np.broadcast_to(centroids, (targets, poses, 3)).copy()


def _is_rotation(mate: MateSpec, swept: str) -> bool:
    """Whether the swept DOF rotates the moving group (torque pivots on the axis)."""
    if mate.kind == "revolute":
        return True
    if mate.kind == "cylindrical":
        return swept == "turn"
    return False
