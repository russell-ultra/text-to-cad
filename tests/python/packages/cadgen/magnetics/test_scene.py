"""Lane A1: the STEP-leaves -> magpylib-sources adapter (``cadgen.magnetics.scene``).

Covers the name grammar and every parse error, solid classification (cuboid,
45-degree-tilted cuboid measured in its BODY frame, cylinder, oblique prism ->
mesh), the tessellation weld, the open-mesh hard-fail, the occurrence-transform
contract (the public ``world_transform`` accessor plus the private fallback),
the SI boundary (millimetres -> metres) and the body-frame magnetization rule
(no double rotation). Classification runs on build123d solids built in-process;
only the transform-contract, SI-adapter and ``inspect``-shape tests need a
generated STEP, built with :func:`support.build_model`.
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from tests.python.packages.cadgen.magnetics import support
from tests.python.support.paths import add_repo_path

add_repo_path("packages/cadgen/src")

import cadgen  # noqa: E402
from cadgen import build123d as bd  # noqa: E402
from cadgen.magnetics import scene  # noqa: E402
from cadgen.magnetics.types import MagnetSpec, MagScene, SceneError  # noqa: E402


def _cube_solid(side: float = support.MAGNET_MM, location: bd.Location | None = None):
    """One world-placed build123d cube solid, optionally relocated."""
    part = bd.Box(side, side, side)
    if location is not None:
        part = location * part
    return part.solids()[0]


# A model whose only leaf is not a magnet: ``scene.load`` must reject it.
NON_MAGNET_MODEL = f"""\
from cadgen import label_shape, step
from cadgen import build123d as bd


@step
def plate_only():
    plate = label_shape(bd.Box({support.MAGNET_MM}, {support.MAGNET_MM}, 2.0), "plate")
    return bd.Compound(children=[plate])


if __name__ == "__main__":
    plate_only()
"""


class ParseMagnetNameTests(unittest.TestCase):
    def test_non_magnet_names_return_none(self):
        self.assertIsNone(scene.parse_magnet_name(None))
        self.assertIsNone(scene.parse_magnet_name("plate"))
        self.assertIsNone(scene.parse_magnet_name("channel"))

    def test_every_grade_in_the_table(self):
        for grade, remanence in scene.GRADE_REMANENCE_T.items():
            j, direction = scene.parse_magnet_name(f"mag:{grade}:+z")
            self.assertAlmostEqual(j, remanence)
            self.assertEqual(direction, (0.0, 0.0, 1.0))

    def test_every_named_axis(self):
        expected = {
            "+x": (1.0, 0.0, 0.0),
            "-x": (-1.0, 0.0, 0.0),
            "+y": (0.0, 1.0, 0.0),
            "-y": (0.0, -1.0, 0.0),
            "+z": (0.0, 0.0, 1.0),
            "-z": (0.0, 0.0, -1.0),
        }
        for token, vector in expected.items():
            j, direction = scene.parse_magnet_name(f"mag:N42:{token}")
            self.assertEqual(direction, vector)
            self.assertAlmostEqual(j, 1.30)

    def test_explicit_remanence(self):
        j, direction = scene.parse_magnet_name("mag:Br=1.32T:+z")
        self.assertAlmostEqual(j, 1.32)
        self.assertEqual(direction, (0.0, 0.0, 1.0))

    def test_comma_vector_is_normalised(self):
        _, direction = scene.parse_magnet_name("mag:N42:0,3,4")
        self.assertAlmostEqual(direction[0], 0.0)
        self.assertAlmostEqual(direction[1], 0.6)
        self.assertAlmostEqual(direction[2], 0.8)
        self.assertAlmostEqual(sum(c * c for c in direction), 1.0)

    def test_unknown_grade_names_the_token(self):
        with self.assertRaises(SceneError) as ctx:
            scene.parse_magnet_name("mag:N99:+z")
        self.assertIn("N99", str(ctx.exception))

    def test_remanence_without_trailing_t(self):
        with self.assertRaises(SceneError):
            scene.parse_magnet_name("mag:Br=1.32:+z")

    def test_non_positive_remanence(self):
        for token in ("Br=0T", "Br=-1.3T"):
            with self.assertRaises(SceneError):
                scene.parse_magnet_name(f"mag:{token}:+z")

    def test_bad_direction_token(self):
        with self.assertRaises(SceneError) as ctx:
            scene.parse_magnet_name("mag:N42:sideways")
        self.assertIn("sideways", str(ctx.exception))

    def test_two_component_vector(self):
        with self.assertRaises(SceneError):
            scene.parse_magnet_name("mag:N42:1,0")

    def test_non_numeric_vector(self):
        with self.assertRaises(SceneError):
            scene.parse_magnet_name("mag:N42:1,x,0")

    def test_zero_length_vector(self):
        with self.assertRaises(SceneError) as ctx:
            scene.parse_magnet_name("mag:N42:0,0,0")
        self.assertIn("zero length", str(ctx.exception))

    def test_wrong_number_of_colons(self):
        for name in ("mag:N42", "mag:N42:+z:extra", "mag::+z", "mag:N42:"):
            with self.assertRaises(SceneError):
                scene.parse_magnet_name(name)


class ClassifySolidTests(unittest.TestCase):
    def test_axis_aligned_cube(self):
        kind, r_box, dims, center = scene.classify_solid(_cube_solid())
        self.assertEqual(kind, "cuboid")
        np.testing.assert_allclose(r_box, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(dims, [support.MAGNET_MM * 1e-3] * 3, atol=1e-9)
        np.testing.assert_allclose(center, [0.0, 0.0, 0.0], atol=1e-9)

    def test_tilted_cube_dims_are_body_frame_not_aabb(self):
        # 45 degrees about x: the AABB grows to side*sqrt(2) in y and z, but the
        # body-frame dimension stays the cube's side.
        solid = _cube_solid(location=bd.Pos(5, 0, 0) * bd.Rot(45, 0, 0))
        kind, r_box, dims, center = scene.classify_solid(solid)
        self.assertEqual(kind, "cuboid")
        side_m = support.MAGNET_MM * 1e-3
        np.testing.assert_allclose(dims, [side_m] * 3, atol=1e-9)
        # R_box is orthonormal and right-handed.
        np.testing.assert_allclose(r_box.T @ r_box, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(r_box)), 1.0, places=9)
        # Each body axis is parallel to one of the cube's actual edge lines
        # (Rot_x(45) applied to the world axes). A cube's 90-degree symmetry
        # leaves the exact representative free, so parallelism -- not equality --
        # is the invariant.
        cos, sin = np.cos(np.pi / 4), np.sin(np.pi / 4)
        edge_lines = [np.array([1.0, 0, 0]), np.array([0, cos, sin]), np.array([0, -sin, cos])]
        for column in r_box.T:
            self.assertTrue(
                any(abs(abs(float(column @ line)) - 1.0) < 1e-6 for line in edge_lines)
            )
        # The AABB would be side*sqrt(2) ~ 8.98 mm in y and z -- distinctly larger.
        bbox = solid.bounding_box()
        self.assertGreater(bbox.max.Y - bbox.min.Y, support.MAGNET_MM * 1.3)

    def test_cylinder(self):
        kind, r_box, (diameter, height), center = scene.classify_solid(
            bd.Cylinder(radius=3, height=10).solids()[0]
        )
        self.assertEqual(kind, "cylinder")
        self.assertAlmostEqual(diameter, 6.0 * 1e-3)
        self.assertAlmostEqual(height, 10.0 * 1e-3)
        # Body z is the cylinder axis; the frame is orthonormal, right-handed.
        np.testing.assert_allclose(r_box[:, 2], [0.0, 0.0, 1.0], atol=1e-9)
        np.testing.assert_allclose(r_box.T @ r_box, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(r_box)), 1.0, places=9)

    def test_oblique_prism_falls_to_mesh(self):
        prism = bd.extrude(bd.Rectangle(6.35, 6.35), amount=6.35, dir=(0.4, 0, 1)).solids()[0]
        kind, vertices, faces = scene.classify_solid(prism)
        self.assertEqual(kind, "mesh")
        # A parallelepiped is welded to 8 corners and 12 triangles.
        self.assertEqual(vertices.shape, (8, 3))
        self.assertEqual(faces.shape, (12, 3))


class WeldMeshTests(unittest.TestCase):
    def test_box_tessellation_welds_to_eight_vertices(self):
        raw_vertices, raw_faces = _cube_solid().tessellate(0.1)
        vertices = np.array([[v.X, v.Y, v.Z] for v in raw_vertices]) * 1e-3
        faces = np.array(raw_faces)
        self.assertEqual(vertices.shape[0], 24)  # build123d emits per-face copies
        welded_vertices, welded_faces = scene.weld_mesh(vertices, faces)
        self.assertEqual(welded_vertices.shape, (8, 3))
        self.assertEqual(welded_faces.shape, (12, 3))

    def test_degenerate_triangles_are_dropped(self):
        # Two coincident vertices within tolerance collapse; a triangle using
        # both becomes degenerate and is removed.
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1e-9, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        welded_vertices, welded_faces = scene.weld_mesh(vertices, faces)
        self.assertEqual(welded_vertices.shape[0], 3)  # first two merge
        self.assertEqual(welded_faces.shape[0], 1)  # the collapsed triangle is gone

    def test_welded_coordinates_are_exact_not_rounded(self):
        vertices = np.array([[0.123456789, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]])
        welded_vertices, _ = scene.weld_mesh(vertices, faces)
        self.assertIn(0.123456789, welded_vertices[:, 0].tolist())


class MagnetFromSolidTests(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_axis_aligned_cube_is_si_and_identity(self):
        spec = scene.magnet_from_solid(
            _cube_solid(), 1.30, (0, 0, 1), np.eye(3), ref="#o1", label="mag:N42:+z"
        )
        self.assertIsInstance(spec, MagnetSpec)
        self.assertEqual(spec.shape, "cuboid")
        np.testing.assert_allclose(spec.polarization_world_T, [0, 0, 1.30], atol=1e-9)
        np.testing.assert_allclose(spec.orientation_quat, [0, 0, 0, 1], atol=1e-9)
        np.testing.assert_allclose(
            spec.source.dimension, [support.MAGNET_MM * 1e-3] * 3, atol=1e-9
        )
        self.assertIsNone(spec.moving)

    def test_tilted_cube_polarization_is_body_frame(self):
        rot_x45 = np.array(
            [[1, 0, 0], [0, np.cos(np.pi / 4), -np.sin(np.pi / 4)], [0, np.sin(np.pi / 4), np.cos(np.pi / 4)]]
        )
        solid = _cube_solid(location=bd.Rot(45, 0, 0))
        _, r_box, _, _ = scene.classify_solid(solid)
        spec = scene.magnet_from_solid(
            solid, 1.30, (0, 0, 1), rot_x45, ref="#o1", label="mag:N42:+z"
        )
        j_world = 1.30 * (rot_x45 @ np.array([0.0, 0.0, 1.0]))
        # polarization stored on the source is body-frame = R_box.T @ J_world;
        # never rotated twice (orientation carries R_box).
        np.testing.assert_allclose(spec.source.polarization, r_box.T @ j_world, atol=1e-9)
        np.testing.assert_allclose(spec.polarization_world_T, j_world, atol=1e-9)

    def test_tilted_field_matches_a_rotated_untilted_control(self):
        rot_x45 = np.array(
            [[1, 0, 0], [0, np.cos(np.pi / 4), -np.sin(np.pi / 4)], [0, np.sin(np.pi / 4), np.cos(np.pi / 4)]]
        )
        solid = _cube_solid(location=bd.Rot(45, 0, 0))
        _, r_box, dims, _ = scene.classify_solid(solid)
        spec = scene.magnet_from_solid(
            solid, 1.30, (0, 0, 1), rot_x45, ref="#o1", label="mag:N42:+z"
        )
        import magpylib as magpy

        control = magpy.magnet.Cuboid(
            polarization=tuple(spec.source.polarization), dimension=tuple(dims), position=(0, 0, 0)
        )
        points = np.array([[0.01, 0.005, 0.008], [0.0, 0.0, 0.02], [-0.007, 0.003, 0.0]])
        b_tilted = spec.source.getB(points)
        # The rotated source's field equals R_box applied to the control field
        # sampled at the inverse-rotated points -- one rotation, not two.
        b_control = (r_box @ control.getB((r_box.T @ points.T).T).T).T
        np.testing.assert_allclose(b_tilted, b_control, rtol=1e-9, atol=1e-12)

    def test_open_mesh_raises_naming_the_leaf(self):
        raw_vertices, raw_faces = _cube_solid().tessellate(0.1)
        vertices = np.array([[v.X, v.Y, v.Z] for v in raw_vertices]) * 1e-3
        welded_vertices, welded_faces = scene.weld_mesh(vertices, np.array(raw_faces))
        open_faces = welded_faces[:-2]  # a hole in the closed box

        original = scene.classify_solid
        scene.classify_solid = lambda solid: ("mesh", welded_vertices, open_faces)
        try:
            with self.assertRaises(SceneError) as ctx:
                scene.magnet_from_solid(
                    _cube_solid(), 1.30, (0, 0, 1), np.eye(3), ref="#o7.7.7", label="mag:N42:+z"
                )
        finally:
            scene.classify_solid = original
        self.assertIn("#o7.7.7", str(ctx.exception))

    def test_closed_mesh_builds_with_world_polarization(self):
        raw_vertices, raw_faces = _cube_solid().tessellate(0.1)
        vertices = np.array([[v.X, v.Y, v.Z] for v in raw_vertices]) * 1e-3
        welded_vertices, welded_faces = scene.weld_mesh(vertices, np.array(raw_faces))

        original = scene.classify_solid
        scene.classify_solid = lambda solid: ("mesh", welded_vertices, welded_faces)
        try:
            spec = scene.magnet_from_solid(
                _cube_solid(), 1.30, (0, 0, 1), np.eye(3), ref="#o1", label="mag:N42:+z"
            )
        finally:
            scene.classify_solid = original
        self.assertEqual(spec.shape, "mesh")
        # A mesh gets identity orientation and world-frame polarization.
        np.testing.assert_allclose(spec.orientation_quat, [0, 0, 0, 1], atol=1e-9)
        np.testing.assert_allclose(spec.polarization_world_T, [0, 0, 1.30], atol=1e-9)


class OccurrenceTransformTests(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_public_accessor_has_sixteen_floats(self):
        step = support.build_model(self, "tilted_cube", support.TILTED_CUBE_MODEL)
        occ = next(cadgen.read_scene(step).leaves())
        world = occ.world_transform
        self.assertEqual(len(world), 16)
        for value in world:
            self.assertIsInstance(value, float)

    def test_translation_equals_the_placed_centroid(self):
        step = support.build_model(self, "tilted_cube", support.TILTED_CUBE_MODEL)
        occ = next(cadgen.read_scene(step).leaves())
        transform = scene.occurrence_transform(occ)
        self.assertEqual(transform.shape, (4, 4))
        # TILTED_CUBE_MODEL places the instance at Pos(5, 0, 0); the cube is
        # centred at its local origin, so the placed centroid is (5, 0, 0) mm.
        np.testing.assert_allclose(transform[:3, 3], [5.0, 0.0, 0.0], atol=1e-6)
        centroid = occ.shape().solids()[0].center()
        np.testing.assert_allclose(
            transform[:3, 3], [centroid.X, centroid.Y, centroid.Z], atol=1e-6
        )

    def test_falls_back_to_the_private_node_transform(self):
        # An occurrence-shaped object without the public accessor exercises the
        # documented private fallback (occ._node.transform, row-major 16 floats).
        class _Node:
            transform = (
                1.0, 0.0, 0.0, 2.0,
                0.0, 1.0, 0.0, 3.0,
                0.0, 0.0, 1.0, 4.0,
                0.0, 0.0, 0.0, 1.0,
            )

        class _Occurrence:
            _node = _Node()

        transform = scene.occurrence_transform(_Occurrence())
        np.testing.assert_allclose(transform[:3, 3], [2.0, 3.0, 4.0])


class LoadTests(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_two_cube_is_si_and_inspect_shaped_without_a_mate(self):
        step = support.build_model(self, "two_cube", support.TWO_CUBE_MODEL)
        magnetic_scene = scene.load(step)
        self.assertIsInstance(magnetic_scene, MagScene)
        self.assertIsNone(magnetic_scene.mate)
        self.assertEqual(magnetic_scene.moving, ())
        self.assertEqual(len(magnetic_scene.magnets), 2)

        by_label = {m.label: m for m in magnetic_scene.magnets}
        self.assertEqual(set(by_label), {"mag:N42:+z", "mag:N42:-z"})

        plus = by_label["mag:N42:+z"]
        minus = by_label["mag:N42:-z"]
        # No mate selected -> every magnet's moving flag is None (inspect shape).
        self.assertIsNone(plus.moving)
        self.assertIsNone(minus.moving)
        # SI: 6.35 mm cube -> 0.00635 m dimension; 30 mm spacing -> 0.030 m.
        np.testing.assert_allclose(plus.source.dimension, [0.00635] * 3, atol=1e-9)
        np.testing.assert_allclose(plus.position_m, [0, 0, 0], atol=1e-9)
        np.testing.assert_allclose(minus.position_m, [0, 0, 0.030], atol=1e-6)
        np.testing.assert_allclose(plus.polarization_world_T, [0, 0, 1.30], atol=1e-9)
        np.testing.assert_allclose(minus.polarization_world_T, [0, 0, -1.30], atol=1e-9)

    def test_bbox_is_in_metres(self):
        step = support.build_model(self, "two_cube", support.TWO_CUBE_MODEL)
        lo, hi = scene.load(step).bbox_m
        # Two 6.35 mm cubes spanning z in [-3.175, 33.175] mm.
        np.testing.assert_allclose(lo, [-0.003175, -0.003175, -0.003175], atol=1e-6)
        np.testing.assert_allclose(hi, [0.003175, 0.003175, 0.033175], atol=1e-6)

    def test_solids_fixed_holds_every_solid_without_a_mate(self):
        step = support.build_model(self, "two_cube", support.TWO_CUBE_MODEL)
        magnetic_scene = scene.load(step)
        self.assertEqual(len(magnetic_scene.solids_fixed), 2)
        self.assertEqual(magnetic_scene.fixed, (0, 1))

    def test_model_without_a_magnet_is_rejected(self):
        step = support.build_model(self, "plate_only", NON_MAGNET_MODEL)
        with self.assertRaises(SceneError) as ctx:
            scene.load(step)
        self.assertIn(str(step), str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
