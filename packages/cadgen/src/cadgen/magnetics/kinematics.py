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

Every function below is a stub for lane A2: ``raise NotImplementedError("lane A2")``.
No magpylib/numpy/scipy import at module scope -- use ``cadgen.magnetics._deps``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

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
    raise NotImplementedError("lane A2")


def moving_indices(scene_obj: Any, child_id: str, magnets: Sequence[MagnetSpec]) -> tuple[int, ...]:
    """Indices into ``magnets`` of every magnet under the mate's child occurrence.

    ``scene_obj`` is the ``cadgen.read_scene`` StepScene; the subtree is
    ``scene_obj.resolve("#" + child_id)`` and its ``.children`` recursively,
    matched to ``MagnetSpec.ref`` (a leaf may hold several magnets, all move).
    Selection is by numeric occurrence id, so duplicate labels still select the
    right subtree. :class:`MateError` naming the mate's child ref when the
    subtree contains no magnets.
    """
    raise NotImplementedError("lane A2")


def sweep_dof(mate: MateSpec, dof: str | None) -> tuple[str, str | None]:
    """``(swept_key, held_key)`` -- the limits keys a sweep moves and holds.

    Single-DOF mates: ``("value", None)``; ``dof`` must be ``None`` there. A
    cylindrical mate needs ``dof`` = ``"<mate>.travel"`` or ``"<mate>.turn"``
    and returns ``("travel", "turn")`` or ``("turn", "travel")``.
    :class:`MateError` naming the mate and its sub-DOFs when ``dof`` is missing
    for a cylindrical mate or names a sub-DOF the mate lacks (the CLI checks
    the ``--dof`` spelling first and maps this row to exit 2).
    """
    raise NotImplementedError("lane A2")


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
    raise NotImplementedError("lane A2")


def posed_sources(
    scene: MagScene, q: float, held_value: float | None = None, *, swept: str = SWEPT_SINGLE
) -> list[Any]:
    """Fresh SINGLE-POSE copies of the moving magnets at ``q`` (``MagScene.at(q)``).

    Each copy's ``position``/``orientation`` is ``delta(q) @ rest``; the scene's
    own sources are never mutated. This is what ``field`` and every bisection
    step use: a path source would broadcast a slice grid to ``(201, 14400, 3)``.
    """
    raise NotImplementedError("lane A2")


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
    raise NotImplementedError("lane A2")


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
    raise NotImplementedError("lane A2")
