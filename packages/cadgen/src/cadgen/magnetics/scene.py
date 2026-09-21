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

from pathlib import Path
from typing import Any

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


def parse_magnet_name(name: str | None) -> tuple[float, tuple[float, float, float]] | None:
    """``(|J| in tesla, unit dir_local)`` for a ``mag:<material>:<direction>`` name.

    ``None`` when ``name`` does not start with ``mag:`` (the leaf is not a
    magnet). A name that IS a magnet declaration but is malformed raises
    :class:`SceneError` naming the offending token: an unknown grade, ``Br=``
    without a trailing ``T`` or with a value <= 0, a direction that is neither
    one of ``+x -x +y -y +z -z`` nor a three-number comma vector, or a
    direction vector of zero length. The caller prefixes the leaf ref.
    """
    raise NotImplementedError("lane A1")


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
    raise NotImplementedError("lane A1")


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
    raise NotImplementedError("lane A1")


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
    raise NotImplementedError("lane A1")


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
    raise NotImplementedError("lane A1")


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
    raise NotImplementedError("lane A1")
