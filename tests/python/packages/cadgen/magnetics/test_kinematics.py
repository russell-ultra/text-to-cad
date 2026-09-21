"""Lane A2: mates become motion.

``delta`` is checked against three hand-derived 4x4 matrices whose convention
source is the viewer's forward kinematics
(``packages/cadgen-js/src/common/kinematicsRuntime.js``): the artifact as
written is ``q = 0`` and ``world(q) = D(q) @ world(0)``, premultiplied, with
``D = T(dir*q)`` (slider), ``T(o) R(dir,q) T(-o)`` (revolute) and
``T(dir*travel) T(o) R(dir,turn) T(-o)`` (cylindrical). ``select_mate`` is the
SI boundary for a mate (mm->m, deg->rad) and the source of every ``MateError``
row of the validation matrix that names a mate; ``moving_indices`` selects the
moving set by numeric ``childId`` against a real built scene; ``sweep_dof``,
``posed_sources``, ``pathed_sources`` and ``pivot_path`` are exercised on
hand-built magpylib scenes so nothing here depends on lane A1's ``scene.load``.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from tests.python.packages.cadgen.magnetics import support
from tests.python.support.paths import add_repo_path

add_repo_path("packages/cadgen/src")

import magpylib as magpy  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from cadgen.magnetics import kinematics as kin  # noqa: E402
from cadgen.magnetics.types import MagnetSpec, MagScene, MateError, MateSpec  # noqa: E402


# --------------------------------------------------------------------------- helpers


def _slider(limits_m=(0.0, 0.02), direction=(1.0, 0.0, 0.0), origin_m=(0.0, 0.0, 0.0)) -> MateSpec:
    return MateSpec("travel", "slider", "o1.2", origin_m, direction, {"value": limits_m}, "m")


def _revolute(limits_rad=(0.0, math.pi), direction=(0.0, 0.0, 1.0), origin_m=(1.0, 0.0, 0.0)) -> MateSpec:
    return MateSpec("hinge", "revolute", "o1.2", origin_m, direction, {"value": limits_rad}, "rad")


def _cylindrical(direction=(0.0, 0.0, 1.0), origin_m=(1.0, 0.0, 0.0)) -> MateSpec:
    return MateSpec(
        "cyl", "cylindrical", "o1.2", origin_m, direction,
        {"travel": (0.0, 0.02), "turn": (0.0, math.pi)}, "m",
    )


def _cuboid(position, polarization=(0.0, 0.0, 1.3)) -> magpy.magnet.Cuboid:
    return magpy.magnet.Cuboid(polarization=polarization, dimension=(6.35e-3,) * 3, position=position)


def _magnet(ref, source, moving) -> MagnetSpec:
    position = tuple(float(v) for v in np.asarray(source.position))
    return MagnetSpec(ref, "mag:N42:+z", "cuboid", source, (0.0, 0.0, 1.3), position, (0, 0, 0, 1), moving, None)


def _scene(mate, sources) -> MagScene:
    magnets = tuple(_magnet(f"#o1.{i + 1}", s, i != 0) for i, s in enumerate(sources))
    moving = tuple(i for i in range(1, len(sources)))
    return MagScene("model.step", magnets, (0,), moving, mate, ((0, 0, 0), (0, 0, 0)), ())


def _kin_mapping(*mates) -> dict:
    """A sidecar-shaped mapping (model units, degrees) for ``select_mate``."""
    return {"kinematics": {"mates": list(mates)}}


def _mate_dict(name, kind, child_id, limits, direction=(0, 0, 1), origin=(0, 0, 0)) -> dict:
    entry = {"name": name, "kind": kind, "axis": {"origin": list(origin), "dir": list(direction)}, "limits": limits}
    if child_id is not None:
        entry["childId"] = child_id
    return entry


# --------------------------------------------------------------------------- delta


class DeltaMatrixTests(unittest.TestCase):
    """delta against three hand-derived matrices; convention: kinematicsRuntime.js."""

    def test_slider_ten_mm_along_x(self):
        # world(q) = T(dir*q) @ world(0); a 10 mm slide along +x is a pure translation.
        got = kin.delta(_slider(), 0.01)
        expected = np.eye(4)
        expected[:3, 3] = (0.01, 0.0, 0.0)
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_revolute_ninety_degrees_about_z_through_one_zero_zero(self):
        # T(o) R(z, 90) T(-o) with o=(1,0,0): the point (2,0,0) swings to (1,1,0).
        got = kin.delta(_revolute(), math.pi / 2)
        expected = np.array(
            [
                [0.0, -1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(got, expected, atol=1e-12)
        np.testing.assert_allclose(got @ np.array([2, 0, 0, 1.0]), (1, 1, 0, 1), atol=1e-12)

    def test_cylindrical_ten_mm_plus_ninety_degrees_same_axis(self):
        # D = T(dir*travel) @ T(o) R(dir,turn) T(-o); swept travel or turn agree.
        via_travel = kin.delta(_cylindrical(), 0.01, math.pi / 2, swept="travel")
        via_turn = kin.delta(_cylindrical(), math.pi / 2, 0.01, swept="turn")
        np.testing.assert_allclose(via_travel, via_turn, atol=1e-12)
        # (2,0,0) rotates to (1,1,0) then translates +z by 10 mm.
        np.testing.assert_allclose(via_travel @ np.array([2, 0, 0, 1.0]), (1, 1, 0.01, 1), atol=1e-12)

    def test_held_none_is_zero_for_a_cylindrical_mate(self):
        no_hold = kin.delta(_cylindrical(), 0.01, None, swept="travel")
        zero_hold = kin.delta(_cylindrical(), 0.01, 0.0, swept="travel")
        np.testing.assert_allclose(no_hold, zero_hold, atol=1e-12)

    def test_slider_direction_is_normalised(self):
        got = kin.delta(_slider(direction=(0.0, 0.0, 3.0)), 0.01)
        np.testing.assert_allclose(got[:3, 3], (0.0, 0.0, 0.01), atol=1e-12)

    def test_unknown_swept_key_for_cylindrical_raises(self):
        with self.assertRaises(MateError):
            kin.delta(_cylindrical(), 0.01, 0.0, swept="value")


# --------------------------------------------------------------------------- sweep_dof


class SweepDofTests(unittest.TestCase):
    def test_single_dof_returns_value_and_no_held(self):
        self.assertEqual(kin.sweep_dof(_slider(), None), ("value", None))
        self.assertEqual(kin.sweep_dof(_revolute(), None), ("value", None))

    def test_dof_on_a_single_dof_mate_raises(self):
        with self.assertRaises(MateError):
            kin.sweep_dof(_slider(), "travel.travel")

    def test_cylindrical_needs_a_dof(self):
        with self.assertRaises(MateError) as caught:
            kin.sweep_dof(_cylindrical(), None)
        self.assertIn("cyl.travel", str(caught.exception))

    def test_cylindrical_selects_travel_and_turn(self):
        self.assertEqual(kin.sweep_dof(_cylindrical(), "cyl.travel"), ("travel", "turn"))
        self.assertEqual(kin.sweep_dof(_cylindrical(), "cyl.turn"), ("turn", "travel"))

    def test_cylindrical_dof_naming_a_missing_sub_dof_raises(self):
        with self.assertRaises(MateError) as caught:
            kin.sweep_dof(_cylindrical(), "cyl.spin")
        self.assertIn("cyl", str(caught.exception))


# --------------------------------------------------------------------------- select_mate (units + errors)


class SelectMateConversionTests(unittest.TestCase):
    def test_slider_converts_mm_to_m_and_normalises_dir(self):
        mate = kin.select_mate(
            _kin_mapping(_mate_dict("travel", "slider", "o1.2", {"value": [0, 20]}, direction=(0, 0, 2)))
        )
        self.assertEqual(mate.kind, "slider")
        self.assertEqual(mate.child_id, "o1.2")
        self.assertEqual(mate.q_unit, "m")
        self.assertEqual(mate.limits, {"value": (0.0, 0.02)})
        np.testing.assert_allclose(mate.dir, (0.0, 0.0, 1.0))

    def test_revolute_converts_degrees_to_radians(self):
        mate = kin.select_mate(
            _kin_mapping(_mate_dict("hinge", "revolute", "o1.2", {"value": [0, 90]}))
        )
        self.assertEqual(mate.q_unit, "rad")
        lo, hi = mate.limits["value"]
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, math.pi / 2)

    def test_cylindrical_converts_travel_mm_and_turn_degrees(self):
        mate = kin.select_mate(
            _kin_mapping(_mate_dict("cyl", "cylindrical", "o1.2", {"travel": [0, 20], "turn": [0, 180]}))
        )
        self.assertEqual(mate.q_unit, "m")  # the travel unit; the swept sub-DOF decides the reported unit
        self.assertEqual(mate.limits["travel"], (0.0, 0.02))
        lo, hi = mate.limits["turn"]
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, math.pi)

    def test_origin_converts_mm_to_m(self):
        mate = kin.select_mate(
            _kin_mapping(_mate_dict("travel", "slider", "o1.2", {"value": [0, 20]}, origin=(5, 0, 0)))
        )
        np.testing.assert_allclose(mate.origin_m, (0.005, 0.0, 0.0))

    def test_a_bare_kinematics_block_is_accepted(self):
        block = {"mates": [_mate_dict("travel", "slider", "o1.2", {"value": [0, 20]})]}
        self.assertEqual(kin.select_mate(block).name, "travel")


class SelectMateErrorTests(unittest.TestCase):
    def test_no_mates_raises_naming_the_absence(self):
        with self.assertRaises(MateError):
            kin.select_mate(_kin_mapping())

    def test_two_mates_and_none_given_names_both(self):
        mapping = _kin_mapping(
            _mate_dict("a", "slider", "o1.2", {"value": [0, 20]}),
            _mate_dict("b", "slider", "o1.3", {"value": [0, 20]}),
        )
        with self.assertRaises(MateError) as caught:
            kin.select_mate(mapping)
        self.assertIn("'a'", str(caught.exception))
        self.assertIn("'b'", str(caught.exception))

    def test_two_mates_selects_by_name(self):
        mapping = _kin_mapping(
            _mate_dict("a", "slider", "o1.2", {"value": [0, 20]}),
            _mate_dict("b", "revolute", "o1.3", {"value": [0, 90]}),
        )
        self.assertEqual(kin.select_mate(mapping, "b").kind, "revolute")

    def test_unknown_name_names_the_available_mates(self):
        mapping = _kin_mapping(_mate_dict("travel", "slider", "o1.2", {"value": [0, 20]}))
        with self.assertRaises(MateError) as caught:
            kin.select_mate(mapping, "nope")
        self.assertIn("travel", str(caught.exception))

    def test_unsupported_kind_names_the_mate_and_kind(self):
        mapping = _kin_mapping(_mate_dict("weld", "fastened", "o1.2", {"value": [0, 20]}))
        with self.assertRaises(MateError) as caught:
            kin.select_mate(mapping)
        self.assertIn("weld", str(caught.exception))
        self.assertIn("fastened", str(caught.exception))

    def test_missing_child_id_names_the_mate(self):
        mapping = _kin_mapping(_mate_dict("travel", "slider", None, {"value": [0, 20]}))
        with self.assertRaises(MateError) as caught:
            kin.select_mate(mapping)
        self.assertIn("travel", str(caught.exception))

    def test_reversed_limits_raise(self):
        mapping = _kin_mapping(_mate_dict("travel", "slider", "o1.2", {"value": [20, 0]}))
        with self.assertRaises(MateError):
            kin.select_mate(mapping)

    def test_degenerate_limits_raise(self):
        mapping = _kin_mapping(_mate_dict("travel", "slider", "o1.2", {"value": [5, 5]}))
        with self.assertRaises(MateError):
            kin.select_mate(mapping)

    def test_zero_length_axis_raises(self):
        mapping = _kin_mapping(_mate_dict("travel", "slider", "o1.2", {"value": [0, 20]}, direction=(0, 0, 0)))
        with self.assertRaises(MateError):
            kin.select_mate(mapping)


# --------------------------------------------------------------------------- select_mate + moving_indices on a real STEP


class SelectMateOnDocumentTests(unittest.TestCase):
    def test_select_mate_reads_the_sidecar_in_si(self):
        step = support.build_model(self, "two_cube", support.TWO_CUBE_MODEL)
        mate = kin.select_mate(step)
        self.assertEqual(mate.name, "travel")
        self.assertEqual(mate.kind, "slider")
        self.assertEqual(mate.child_id, "o1.2")
        self.assertEqual(mate.q_unit, "m")
        self.assertEqual(mate.limits, {"value": (0.0, 0.02)})  # 0..20 mm -> 0..0.02 m
        np.testing.assert_allclose(mate.dir, (0.0, 0.0, 1.0))

    def test_a_stale_sidecar_propagates_cadgens_binding_error(self):
        from cadgen._internal.source_sidecar import SidecarBindingError

        step = support.build_model(self, "two_cube", support.TWO_CUBE_MODEL)
        step.write_bytes(step.read_bytes() + b"\n# tampered\n")  # STEP bytes no longer match documentHash
        with self.assertRaises(SidecarBindingError):
            kin.select_mate(step)


class MovingIndicesTests(unittest.TestCase):
    def _scene_and_magnets(self):
        import cadgen

        step = support.build_model(self, "two_cube", support.TWO_CUBE_MODEL)
        scene = cadgen.read_scene(str(step))
        magnets = [
            MagnetSpec(leaf.ref, leaf.label or "", "cuboid", None, (0, 0, 0), (0, 0, 0), (0, 0, 0, 1), None, None)
            for leaf in scene.leaves()
        ]
        return scene, magnets

    def test_selects_the_magnet_under_the_child_subtree(self):
        scene, magnets = self._scene_and_magnets()
        indices = kin.moving_indices(scene, "o1.2", magnets)
        self.assertEqual([magnets[i].ref for i in indices], ["#o1.2.1"])

    def test_a_leading_hash_on_child_id_is_tolerated(self):
        scene, magnets = self._scene_and_magnets()
        self.assertEqual(kin.moving_indices(scene, "#o1.2", magnets), kin.moving_indices(scene, "o1.2", magnets))

    def test_an_empty_subtree_raises_naming_the_child_ref(self):
        scene, magnets = self._scene_and_magnets()
        without_slider = [m for m in magnets if not m.ref.startswith("#o1.2")]
        with self.assertRaises(MateError) as caught:
            kin.moving_indices(scene, "o1.2", without_slider)
        self.assertIn("o1.2", str(caught.exception))

    def test_duplicate_labels_still_select_by_numeric_id(self):
        # Both leaves are named through the mag: convention, not #labels; selection
        # is by numeric childId, so the fixed magnet under #o1.1 is never chosen.
        scene, magnets = self._scene_and_magnets()
        indices = kin.moving_indices(scene, "o1.1", magnets)
        self.assertEqual([magnets[i].ref for i in indices], ["#o1.1.1"])


# --------------------------------------------------------------------------- posed / pathed sources


class PosedSourcesTests(unittest.TestCase):
    def test_a_single_pose_copy_equals_delta_applied_to_rest(self):
        rest = _cuboid((0.0, 0.0, 0.03))
        scene = _scene(_slider(direction=(1, 0, 0)), [_cuboid((0, 0, 0)), rest])
        posed = kin.posed_sources(scene, 0.01)
        self.assertEqual(len(posed), 1)
        np.testing.assert_allclose(posed[0].position, (0.01, 0.0, 0.03), atol=1e-12)

    def test_the_scenes_own_sources_are_never_mutated(self):
        rest = _cuboid((0.0, 0.0, 0.03))
        scene = _scene(_slider(), [_cuboid((0, 0, 0)), rest])
        kin.posed_sources(scene, 0.02)
        np.testing.assert_allclose(rest.position, (0.0, 0.0, 0.03))

    def test_a_revolute_pose_rotates_position_and_orientation(self):
        rest = _cuboid((2.0, 0.0, 0.0))
        scene = _scene(_revolute(), [_cuboid((0, 0, 0)), rest])
        posed = kin.posed_sources(scene, math.pi / 2)
        np.testing.assert_allclose(posed[0].position, (1.0, 1.0, 0.0), atol=1e-12)
        expected = Rotation.from_rotvec([0, 0, math.pi / 2]).as_matrix()
        np.testing.assert_allclose(posed[0].orientation.as_matrix(), expected, atol=1e-12)

    def test_a_scene_without_a_mate_raises(self):
        scene = _scene(None, [_cuboid((0, 0, 0)), _cuboid((0, 0, 0.03))])
        with self.assertRaises(MateError):
            kin.posed_sources(scene, 0.01)


class PathedSourcesTests(unittest.TestCase):
    def test_path_length_equals_len_qs(self):
        scene = _scene(_slider(), [_cuboid((0, 0, 0)), _cuboid((0, 0, 0.03))])
        qs = [0.0, 0.005, 0.01, 0.02]
        pathed = kin.pathed_sources(scene, qs)
        self.assertEqual(np.asarray(pathed[0].position).shape, (len(qs), 3))
        self.assertEqual(len(pathed[0].orientation), len(qs))

    def test_slider_path_translates_each_pose(self):
        scene = _scene(_slider(direction=(1, 0, 0)), [_cuboid((0, 0, 0)), _cuboid((0, 0, 0.03))])
        pathed = kin.pathed_sources(scene, [0.0, 0.01, 0.02])
        np.testing.assert_allclose(
            pathed[0].position, [(0, 0, 0.03), (0.01, 0, 0.03), (0.02, 0, 0.03)], atol=1e-12
        )

    def test_revolute_path_carries_orientation(self):
        scene = _scene(_revolute(), [_cuboid((0, 0, 0)), _cuboid((2, 0, 0))])
        pathed = kin.pathed_sources(scene, [0.0, math.pi / 2])
        np.testing.assert_allclose(pathed[0].position[1], (1.0, 1.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(
            pathed[0].orientation[1].as_matrix(),
            Rotation.from_rotvec([0, 0, math.pi / 2]).as_matrix(),
            atol=1e-12,
        )

    def test_held_values_ride_a_cylindrical_path(self):
        scene = _scene(_cylindrical(), [_cuboid((0, 0, 0)), _cuboid((2, 0, 0))])
        qs = [0.0, 0.01]
        held = [0.0, math.pi / 2]  # the turn sub-DOF held per pose
        pathed = kin.pathed_sources(scene, qs, held, swept="travel")
        # Pose 1: turn 90 deg about z through (1,0,0) then travel +z 10 mm.
        np.testing.assert_allclose(pathed[0].position[1], (1.0, 1.0, 0.01), atol=1e-12)


class PivotPathTests(unittest.TestCase):
    def test_slider_pivot_is_the_group_centroid_at_each_pose(self):
        scene = _scene(_slider(direction=(1, 0, 0)), [_cuboid((0, 0, 0)), _cuboid((0, 0, 0.03))])
        qs = [0.0, 0.01, 0.02]
        pivot = kin.pivot_path(scene, qs)
        self.assertEqual(pivot.shape, (1, 3, 3))  # (t, p, 3), one moving target
        np.testing.assert_allclose(pivot[0], [(0, 0, 0.03), (0.01, 0, 0.03), (0.02, 0, 0.03)], atol=1e-12)

    def test_two_moving_targets_share_the_centroid(self):
        scene = MagScene(
            "m.step",
            (
                _magnet("#o1.1", _cuboid((0, 0, 0)), False),
                _magnet("#o1.2", _cuboid((0, 0, 0.02)), True),
                _magnet("#o1.3", _cuboid((0, 0, 0.04)), True),
            ),
            (0,),
            (1, 2),
            _slider(direction=(1, 0, 0)),
            ((0, 0, 0), (0, 0, 0)),
            (),
        )
        pivot = kin.pivot_path(scene, [0.0, 0.01])
        self.assertEqual(pivot.shape, (2, 2, 3))
        # Rest centroid of the two moving magnets is z = 0.03; both targets share it.
        np.testing.assert_allclose(pivot[0], [(0, 0, 0.03), (0.01, 0, 0.03)], atol=1e-12)
        np.testing.assert_allclose(pivot[0], pivot[1], atol=1e-12)

    def test_revolute_pivot_is_a_point_on_the_axis_repeated(self):
        scene = _scene(_revolute(origin_m=(1, 0, 0)), [_cuboid((0, 0, 0)), _cuboid((2, 0, 0))])
        pivot = kin.pivot_path(scene, [0.0, math.pi / 2, math.pi])
        self.assertEqual(pivot.shape, (1, 3, 3))
        np.testing.assert_allclose(pivot[0], [(1, 0, 0)] * 3, atol=1e-12)

    def test_cylindrical_turn_pivots_on_the_axis_travel_on_the_centroid(self):
        scene = _scene(_cylindrical(origin_m=(1, 0, 0)), [_cuboid((0, 0, 0)), _cuboid((2, 0, 0))])
        turn = kin.pivot_path(scene, [0.0, math.pi / 2], swept="turn")
        np.testing.assert_allclose(turn[0], [(1, 0, 0), (1, 0, 0)], atol=1e-12)
        travel = kin.pivot_path(scene, [0.0, 0.01], swept="travel")
        np.testing.assert_allclose(travel[0], [(2, 0, 0), (2, 0, 0.01)], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
