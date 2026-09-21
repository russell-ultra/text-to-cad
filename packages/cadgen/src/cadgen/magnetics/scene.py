"""STEP leaves -> magpylib sources: the name grammar, solid classification, the
mesh weld, the occurrence-transform shim and the one SI adapter (lane A1).

Magnet declaration (naming convention, until the ``magnetics=`` slot lands): a
leaf occurrence is a magnet when its name matches ``mag:<material>:<direction>``.
Inside ``compound_from_instances`` each leaf is named by the instance-name
argument, so the convention rides that argument; for a standalone part
``label_shape(cube, "mag", "N42", "+z")`` yields the same string. Duplicates are
legal: the tool iterates ``scene.leaves()`` and never resolves by name.

``<material>``: a grade from :data:`GRADE_REMANENCE_T` or explicit ``Br=1.32T``
(remanence Br used as |J| in tesla). ``<direction>``: the magnetization axis in
the part's LOCAL frame, ``+x -x +y -y +z -z`` or a comma-separated vector,
normalised by the tool.

Frames. magpylib's ``polarization`` and ``dimension`` are BODY-frame and rotated
by ``orientation``; never rotate twice. ``R_occ`` is the occurrence's
local->world rotation. ``J_world = |J| * R_occ @ dir_local``. Cuboid/Cylinder:
body frame ``R_box`` from the classified world face normals, ``dimension``
measured in that frame (not the world AABB), ``position`` = centroid,
``orientation = R_box``, ``polarization = R_box.T @ J_world``. TriangularMesh:
world vertices, identity orientation, ``polarization = J_world``.

Every function below is a stub for lane A1: ``raise NotImplementedError("lane A1")``.
No magpylib/numpy/scipy import at module scope -- use ``cadgen.magnetics._deps``
inside functions. No CAD-kernel import at module scope either: ``cadgen.magnetics``
must import light so ``--help`` never pays for OCP.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from cadgen.magnetics import _deps
from cadgen.magnetics.types import MagnetSpec, MagScene, SceneError  # noqa: F401 - part of the contract

__all__ = [
    "GRADE_REMANENCE_T",
    "MAGNET_NAME_PREFIX",
    "WELD_TOL_M",
    "parse_magnet_name",
    "classify_solid",
    "weld_mesh",
    "occurrence_transform",
    "magnet_from_solid",
    "load",
]

#: Remanence Br in tesla, used as |J|, per grade token (case-sensitive as written
#: in the design: ``N42`` not ``n42``). ``Br=<value>T`` is the explicit form.
GRADE_REMANENCE_T: dict[str, float] = {
    "N35": 1.19,
    "N38": 1.24,
    "N42": 1.30,
    "N45": 1.35,
    "N48": 1.40,
    "N52": 1.45,
    "SmCo26": 1.05,
    "Y30": 0.40,
}

#: A leaf whose name starts with this (followed by ``<material>:<direction>``) is a magnet.
MAGNET_NAME_PREFIX = "mag:"

#: Vertex-merge tolerance of :func:`weld_mesh`, metres.
WELD_TOL_M = 1e-6

#: Model length unit -> metres. cadgen models are millimetres.
MM_TO_M = 1e-3

#: Named local magnetization directions (``<direction>`` shorthand).
_AXIS_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}

#: A face normal is treated as parallel to a body axis when ``|dot| >= this``,
#: and two body axes as orthogonal when ``|dot| <= 1 - this``. Face normals from
#: OCCT are exact for a box; the slack only absorbs float noise.
_ORTHO_TOL = 1e-6

#: Relative tolerance for the "four corners equidistant from the centroid"
#: rectangularity check of a planar face.
_RECT_TOL = 1e-4

#: build123d tessellation tolerance (model units) for a solid that falls to a
#: mesh. Those solids are planar-faced (an oblique prism), so the triangulation
#: is exact and the value only bounds curved fallbacks.
TESSELLATION_TOL_MM = 1e-3


def parse_magnet_name(name: str | None) -> tuple[float, tuple[float, float, float]] | None:
    """``(|J| in tesla, unit dir_local)`` for a ``mag:<material>:<direction>`` name.

    ``None`` when ``name`` does not start with ``mag:`` (the leaf is not a
    magnet). A name that IS a magnet declaration but is malformed raises
    :class:`SceneError` naming the offending token: an unknown grade, ``Br=``
    without a trailing ``T`` or with a value <= 0, a direction that is neither
    one of ``+x -x +y -y +z -z`` nor a three-number comma vector, or a
    direction vector of zero length. The caller prefixes the leaf ref.
    """
    if name is None or not name.startswith(MAGNET_NAME_PREFIX):
        return None
    body = name[len(MAGNET_NAME_PREFIX) :]
    tokens = body.split(":")
    if len(tokens) != 2 or not tokens[0] or not tokens[1]:
        raise SceneError(
            f"magnet name {name!r} is not 'mag:<material>:<direction>'"
        )
    material, direction = tokens
    return _remanence(material, name), _direction_vector(direction, name)


def _remanence(material: str, name: str) -> float:
    """|J| in tesla for a ``<material>`` token: a grade or ``Br=<value>T``."""
    if material in GRADE_REMANENCE_T:
        return GRADE_REMANENCE_T[material]
    if material.startswith("Br=") and material.endswith("T"):
        try:
            value = float(material[3:-1])
        except ValueError:
            raise SceneError(
                f"magnet name {name!r} has a malformed remanence token {material!r}"
            ) from None
        if value <= 0:
            raise SceneError(
                f"magnet name {name!r} remanence {material!r} must be positive"
            )
        return value
    raise SceneError(f"magnet name {name!r} has an unknown grade token {material!r}")


def _direction_vector(direction: str, name: str) -> tuple[float, float, float]:
    """Unit local magnetization direction for a ``<direction>`` token."""
    if direction in _AXIS_DIRECTIONS:
        return _AXIS_DIRECTIONS[direction]
    parts = direction.split(",")
    if len(parts) != 3:
        raise SceneError(
            f"magnet name {name!r} has an unknown direction token {direction!r}"
        )
    try:
        vector = [float(part) for part in parts]
    except ValueError:
        raise SceneError(
            f"magnet name {name!r} has a non-numeric direction vector {direction!r}"
        ) from None
    length = (vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2) ** 0.5
    if length == 0.0:
        raise SceneError(
            f"magnet name {name!r} direction vector {direction!r} has zero length"
        )
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def classify_solid(solid: Any) -> tuple[Any, ...]:
    """Classify one WORLD-placed build123d solid for magpylib.

    Returns one of::

        ("cuboid",   R_box (3,3), dims_m (3,),      center_m (3,))
        ("cylinder", R_box (3,3), (d_m, h_m),       center_m (3,))
        ("mesh",     vertices_m (n,3), faces (m,3))

    Cuboid: six planar faces whose normals form three mutually orthogonal
    pairs and whose faces are rectangles; ``R_box`` has the pair normals as
    right-handed columns and ``dims_m`` is measured in that frame (a 45 degree
    tilted cube is still a cube). Cylinder: two parallel planar caps plus one
    cylindrical face whose axis is normal to the caps; the cap normal is body
    z. Anything else -- an oblique prism must land here -- tessellates
    (``solid.tessellate(tolerance)``) and is welded by :func:`weld_mesh`.
    All outputs are in metres (the model's millimetres times ``MM_TO_M``).
    """
    np = _deps.numpy()
    faces = list(solid.faces())
    planar = [f for f in faces if _geom_name(f) == "PLANE"]
    cylindrical = [f for f in faces if _geom_name(f) == "CYLINDER"]
    center = _vec(np, solid.center()) * MM_TO_M

    if len(faces) == 6 and len(planar) == 6:
        box = _classify_cuboid(np, solid, planar, center)
        if box is not None:
            return box

    if len(faces) == 3 and len(cylindrical) == 1 and len(planar) == 2:
        cyl = _classify_cylinder(np, cylindrical[0], planar, center)
        if cyl is not None:
            return cyl

    vertices, triangles = solid.tessellate(TESSELLATION_TOL_MM)
    world = np.array([[v.X, v.Y, v.Z] for v in vertices], dtype=float) * MM_TO_M
    mesh_faces = np.array(triangles, dtype=np.int64)
    welded_vertices, welded_faces = weld_mesh(world, mesh_faces)
    return ("mesh", welded_vertices, welded_faces)


def _geom_name(face: Any) -> str:
    """The build123d surface kind of ``face`` (``"PLANE"``, ``"CYLINDER"``, ...)."""
    geom_type = face.geom_type
    return getattr(geom_type, "name", str(geom_type).rsplit(".", 1)[-1])


def _vec(np: Any, vector: Any) -> Any:
    """A build123d ``Vector`` as a ``(3,)`` ndarray."""
    return np.array([vector.X, vector.Y, vector.Z], dtype=float)


def _classify_cuboid(np: Any, solid: Any, planar: list[Any], center: Any) -> tuple[Any, ...] | None:
    """A cuboid classification, or ``None`` when the six faces are not a box."""
    normals = [_unit(np, _vec(np, f.normal_at())) for f in planar]
    axes: list[Any] = []
    for normal in normals:
        if not any(abs(float(normal @ axis)) >= 1.0 - _ORTHO_TOL for axis in axes):
            axes.append(normal)
    if len(axes) != 3:
        return None
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(float(axes[i] @ axes[j])) > _ORTHO_TOL:
                return None
    if not all(_is_rectangle(np, f) for f in planar):
        return None

    r_box = _canonical_frame(np, axes)
    corners = np.array([[v.X, v.Y, v.Z] for v in solid.vertices()], dtype=float) * MM_TO_M
    dims = np.array([float(np.ptp(corners @ r_box[:, i])) for i in range(3)], dtype=float)
    return ("cuboid", r_box, dims, center)


def _canonical_frame(np: Any, raw_axes: list[Any]) -> Any:
    """A right-handed orthonormal body->world frame from three orthogonal axes.

    Each column is one of the (unit) input axes, sign-fixed so its dominant
    component is positive and placed at the world-axis slot it most aligns with
    (collisions fall to the next free slot). Axis-aligned boxes get identity;
    a cube's symmetry makes any 45 degree choice equally valid, so the frame is
    only required to be deterministic and physically consistent.
    """
    axes = [_unit(np, axis) for axis in raw_axes]
    columns: list[Any] = [None, None, None]
    for i in sorted(range(3), key=lambda k: -float(np.max(np.abs(axes[k])))):
        axis = axes[i]
        dominant = int(np.argmax(np.abs(axis)))
        if axis[dominant] < 0:
            axis = -axis
        slot = dominant if columns[dominant] is None else next(
            j for j in range(3) if columns[j] is None
        )
        columns[slot] = axis
    frame = np.column_stack(columns)
    if np.linalg.det(frame) < 0:
        frame[:, 2] = -frame[:, 2]
    return frame


def _classify_cylinder(np: Any, cyl_face: Any, caps: list[Any], center: Any) -> tuple[Any, ...] | None:
    """A cylinder classification, or ``None`` when the caps are not parallel."""
    n0 = _unit(np, _vec(np, caps[0].normal_at()))
    n1 = _unit(np, _vec(np, caps[1].normal_at()))
    if abs(float(n0 @ n1)) < 1.0 - _ORTHO_TOL:
        return None
    axis = n0
    radius = float(cyl_face.radius) * MM_TO_M
    c0 = _vec(np, caps[0].center()) * MM_TO_M
    c1 = _vec(np, caps[1].center()) * MM_TO_M
    height = abs(float((c1 - c0) @ axis))
    r_box = _frame_from_axis(np, axis)
    return ("cylinder", r_box, (2.0 * radius, height), center)


def _frame_from_axis(np: Any, axis: Any) -> Any:
    """A right-handed orthonormal matrix whose third column is ``axis`` (body z)."""
    axis = _unit(np, axis)
    seed = np.array([1.0, 0.0, 0.0]) if abs(float(axis[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = _unit(np, seed - float(seed @ axis) * axis)
    y = np.cross(axis, x)
    return np.column_stack([x, y, axis])


def _unit(np: Any, vector: Any) -> Any:
    length = float(np.linalg.norm(vector))
    return vector / length if length else vector


def _is_rectangle(np: Any, face: Any) -> bool:
    """True when ``face`` has four corners equidistant from their centroid."""
    corners = np.array([[v.X, v.Y, v.Z] for v in face.vertices()], dtype=float)
    if corners.shape[0] != 4:
        return False
    centroid = corners.mean(axis=0)
    distances = np.linalg.norm(corners - centroid, axis=1)
    longest = float(distances.max())
    if longest == 0.0:
        return False
    return float(distances.max() - distances.min()) <= _RECT_TOL * longest


def weld_mesh(vertices: Any, faces: Any, tol: float = WELD_TOL_M) -> tuple[Any, Any]:
    """Merge the per-face vertex copies build123d's tessellation emits.

    build123d's ``Shape.tessellate`` is unwelded (a unit ``Box`` gives 24
    vertices, 12 triangles); magpylib accepts that but flags it
    ``status_open``/``status_disconnected`` with a documented-unreliable field.
    Round vertices to ``tol`` (metres), ``np.unique(..., return_inverse=True)``,
    remap ``faces``, drop degenerate triangles. Returns ``(vertices (k,3),
    faces (m,3))`` -- 8 and 12 for a box. The caller constructs the
    ``TriangularMesh`` with ``check_open``, ``check_disconnected`` and
    ``reorient_faces`` on and raises :class:`SceneError` naming the leaf and
    the failing check if the welded mesh is still open or disconnected.
    """
    np = _deps.numpy()
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.shape[0] == 0:
        return vertices, faces

    quantized = np.round(vertices / tol).astype(np.int64)
    _, first_index, inverse = np.unique(
        quantized, axis=0, return_index=True, return_inverse=True
    )
    inverse = inverse.reshape(-1)
    # np.unique sorts the unique rows; index the ORIGINAL float coordinates by
    # the first occurrence of each so the welded vertices keep exact geometry
    # rather than the rounded grid point.
    welded_vertices = vertices[first_index]
    welded_faces = inverse[faces]
    non_degenerate = (
        (welded_faces[:, 0] != welded_faces[:, 1])
        & (welded_faces[:, 1] != welded_faces[:, 2])
        & (welded_faces[:, 0] != welded_faces[:, 2])
    )
    return welded_vertices, welded_faces[non_degenerate]


def occurrence_transform(occ: Any) -> Any:
    """The occurrence's WORLD transform as a ``(4, 4)`` row-major ndarray, model units.

    Reads the public ``Occurrence.world_transform`` accessor (added to
    ``cadgen.step_scene`` by lane A1) and falls back to the private
    ``occ._node.transform`` -- 16 floats, row-major 4x4: rotation in the first
    three entries of rows 0-2, translation at indices 3, 7, 11 -- with a
    comment saying so. This is the ONLY reader of that private attribute; a
    contract test asserts it has 16 floats and that its translation equals the
    placed shape's centroid on ``TILTED_CUBE_MODEL``, so a rename fails CI
    loudly rather than silently mis-posing every magnet.
    """
    np = _deps.numpy()
    transform = getattr(occ, "world_transform", None)
    if transform is None:
        # Documented fallback for the prototype fork: the private world
        # transform, 16 row-major floats (translation at indices 3, 7, 11).
        transform = occ._node.transform
    return np.asarray(transform, dtype=float).reshape(4, 4)


def magnet_from_solid(
    solid: Any,
    J_T: float,
    dir_local: tuple[float, float, float],
    R_occ: Any,
    *,
    ref: str,
    label: str,
    moving: bool | None = None,
) -> MagnetSpec:
    """One :class:`MagnetSpec` from a world-placed solid and its declaration.

    ``J_world = J_T * R_occ @ dir_local``. Then by :func:`classify_solid`:
    cuboid/cylinder get ``position`` = centroid, ``orientation = R_box``,
    ``polarization = R_box.T @ J_world`` and body-frame ``dimension``; a mesh
    gets world vertices, identity orientation and ``polarization = J_world``.
    ``polarization_world_T`` is ``J_world``; ``orientation_quat`` is body->world
    as ``(x, y, z, w)`` (scipy order); ``position_m`` the centroid. ``ref``
    and ``label`` are the leaf's; ``moving`` is whatever the caller knows
    (``None`` until a mate selects the moving set). Raises :class:`SceneError`
    naming ``ref`` when the mesh fails its checks.
    """
    np = _deps.numpy()
    magpy = _deps.magpylib()
    Rotation = _deps.scipy_rotation()

    r_occ = np.asarray(R_occ, dtype=float)
    j_world = float(J_T) * (r_occ @ np.asarray(dir_local, dtype=float))
    classified = classify_solid(solid)
    shape = classified[0]

    if shape in ("cuboid", "cylinder"):
        _, r_box, dims, center = classified
        orientation = Rotation.from_matrix(r_box)
        polarization = r_box.T @ j_world
        factory = magpy.magnet.Cuboid if shape == "cuboid" else magpy.magnet.Cylinder
        source = factory(
            polarization=tuple(float(x) for x in polarization),
            dimension=tuple(float(x) for x in dims),
            position=tuple(float(x) for x in center),
            orientation=orientation,
        )
        position = tuple(float(x) for x in center)
        quat = tuple(float(x) for x in orientation.as_quat())
    else:  # mesh
        _, vertices, faces = classified
        try:
            source = magpy.magnet.TriangularMesh(
                polarization=tuple(float(x) for x in j_world),
                vertices=vertices,
                faces=faces,
                position=(0.0, 0.0, 0.0),
                orientation=Rotation.identity(),
                check_open="raise",
                check_disconnected="raise",
                reorient_faces="warn",
            )
        except Exception as exc:  # magpylib raises on open/disconnected meshes
            raise SceneError(
                f"{ref}: magnet mesh is not a closed solid ({exc})"
            ) from exc
        position = tuple(float(x) for x in np.asarray(vertices, dtype=float).mean(axis=0))
        quat = (0.0, 0.0, 0.0, 1.0)

    return MagnetSpec(
        ref=ref,
        label=label,
        shape=shape,
        source=source,
        polarization_world_T=tuple(float(x) for x in j_world),
        position_m=position,
        orientation_quat=quat,
        moving=moving,
        solid=solid,
    )


def load(step_path: Path | str, mate_name: str | None = None) -> MagScene:
    """The SI boundary: a STEP document (plus its sidecar) as a frozen :class:`MagScene`.

    The one place millimetres and degrees are converted. Walks
    ``cadgen.read_scene(step_path).leaves()``; for each leaf whose label
    :func:`parse_magnet_name` accepts, iterates ``occ.shape().solids()`` (a leaf
    may contain several solids; each becomes its own magnet) through
    :func:`magnet_from_solid` with ``R_occ`` from :func:`occurrence_transform`.
    No ``mag:`` leaf in the file raises :class:`SceneError` naming the file.

    ``mate_name`` ``None`` selects NO mate: ``mate`` is ``None``, ``moving`` is
    empty and every ``MagnetSpec.moving`` is ``None`` (the ``inspect`` shape).
    A name delegates to ``kinematics.select_mate(step_path, mate_name)`` (which
    reads the sidecar through ``read_source_sidecar``, so schema version and
    ``documentHash`` are enforced) and the moving set to
    ``kinematics.moving_indices(scene, mate.child_id, magnets)``. The
    sole-mate default is the CLI's: ``sweep``/``field`` call
    ``kinematics.select_mate(step, args.mate)`` first and pass ``mate.name``
    here. ``solids_fixed`` holds every non-moving world solid, magnets and
    non-magnets alike, in model units for clearance; ``bbox_m`` spans every solid.
    """
    import cadgen
    from cadgen.magnetics import kinematics

    np = _deps.numpy()
    scene = cadgen.read_scene(step_path)

    magnets: list[MagnetSpec] = []
    magnet_occ_refs: list[str] = []
    all_solids: list[tuple[str, Any]] = []  # (owning occurrence ref, world solid)
    for occ in scene.leaves():
        world_solids = list(occ.shape().solids())
        for solid in world_solids:
            all_solids.append((occ.ref, solid))
        parsed = parse_magnet_name(occ.label)
        if parsed is None:
            continue
        j_T, dir_local = parsed
        r_occ = occurrence_transform(occ)[:3, :3]
        for index, solid in enumerate(world_solids):
            ref = occ.ref if len(world_solids) == 1 else f"{occ.ref}.s{index + 1}"
            magnets.append(
                magnet_from_solid(
                    solid, j_T, dir_local, r_occ, ref=ref, label=occ.label, moving=None
                )
            )
            magnet_occ_refs.append(occ.ref)

    if not magnets:
        raise SceneError(f"{step_path}: no 'mag:<grade>:<direction>' leaf in the model")

    if mate_name is None:
        mate = None
        moving = ()
        fixed = tuple(range(len(magnets)))
        moving_occ_refs: set[str] = set()
    else:
        mate = kinematics.select_mate(step_path, mate_name)
        moving_indices = set(kinematics.moving_indices(scene, mate.child_id, tuple(magnets)))
        magnets = [
            replace(m, moving=(i in moving_indices)) for i, m in enumerate(magnets)
        ]
        moving = tuple(sorted(moving_indices))
        fixed = tuple(i for i in range(len(magnets)) if i not in moving_indices)
        moving_occ_refs = {magnet_occ_refs[i] for i in moving_indices}

    solids_fixed = tuple(
        solid for occ_ref, solid in all_solids if occ_ref not in moving_occ_refs
    )
    bbox_m = _bbox_metres(np, [solid for _, solid in all_solids])

    return MagScene(
        step_path=str(step_path),
        magnets=tuple(magnets),
        fixed=fixed,
        moving=moving,
        mate=mate,
        bbox_m=bbox_m,
        solids_fixed=solids_fixed,
    )


def _bbox_metres(np: Any, solids: list[Any]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The ``(min_xyz, max_xyz)`` over every solid's bounding box, in metres."""
    mins = np.array([np.inf, np.inf, np.inf])
    maxs = np.array([-np.inf, -np.inf, -np.inf])
    for solid in solids:
        box = solid.bounding_box()
        mins = np.minimum(mins, [box.min.X, box.min.Y, box.min.Z])
        maxs = np.maximum(maxs, [box.max.X, box.max.Y, box.max.Z])
    mins = mins * MM_TO_M
    maxs = maxs * MM_TO_M
    return (tuple(float(x) for x in mins), tuple(float(x) for x in maxs))
