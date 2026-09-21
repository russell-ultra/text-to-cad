"""Lane A3: the mechanics of a sweep, tested on hand-built magpylib objects and
hand-built :class:`MagScene` instances -- never on the lane A1/A2 code.

What is proven here: ``force_torque`` reduces every ``(sources, path, targets)``
shape to ``(p, 3)`` and matches a direct ``getFT`` and the coaxial dipole
far-field; ``project`` picks force vs torque per DOF; ``clearance`` finds the
overlap interval behind a bbox prefilter; ``energy`` integrates ``-∫Q dq`` per
accessible interval; ``find_equilibria`` keeps only double-bracketed roots and
classifies stability; ``converge`` runs the right levels and never divides by
zero at ``Q ≈ 0``; ``make_verdict`` reads tendency, the threshold band and
admissibility; ``relax_held`` bisects the held DOF's force to zero or marks the
pose wedged; ``field`` samples a finite ``|B|`` and falls back when the mate
axis is parallel to ``up``; and the ``getFT`` signature is pinned so a magpylib
upgrade fails loudly.

The kinematics (lane A2) and force-per-level (``_grid_forces``,
``_held_generalized_force``) are patched where a function reaches for them, so
this suite stands alone.
"""

from __future__ import annotations

import inspect
import math
import unittest
from typing import Any
from unittest import mock

from tests.python.support.paths import add_repo_path

add_repo_path("packages/cadgen/src")

try:
    import magpylib  # noqa: F401
    import numpy as np

    _HAVE_EXTRA = True
except ImportError:  # pragma: no cover - the extra is installed in CI
    _HAVE_EXTRA = False

if _HAVE_EXTRA:
    from cadgen import build123d as bd
    from cadgen.magnetics import physics
    from cadgen.magnetics.types import Equilibrium, MagnetSpec, MagScene, MateSpec, SweepConfig

MAGNET_MM = 6.35
N42 = 1.30
SPACING = 0.030


requires_extra = unittest.skipUnless(
    _HAVE_EXTRA, "the magnetics extra (magpylib, numpy, scipy, plotly) is not installed"
)


def _cuboid(pol, dim=MAGNET_MM * 1e-3, position=(0.0, 0.0, 0.0)):
    import magpylib as magpy

    return magpy.magnet.Cuboid(polarization=pol, dimension=(dim, dim, dim), position=position)


def _magnet(source, *, solid=None, position=(0.0, 0.0, 0.0), moving=False, ref="#o1.1"):
    return MagnetSpec(
        ref=ref,
        label="mag:N42:+z",
        shape="cuboid",
        source=source,
        polarization_world_T=(0.0, 0.0, N42),
        position_m=position,
        orientation_quat=(0.0, 0.0, 0.0, 1.0),
        moving=moving,
        solid=solid,
    )


def _slider(direction=(0.0, 0.0, 1.0), limits=(0.0, 0.02)):
    return MateSpec(
        name="travel", kind="slider", child_id="o1.2", origin_m=(0.0, 0.0, 0.0),
        dir=direction, limits={"value": limits}, q_unit="m",
    )


@requires_extra
class ForceTorqueTests(unittest.TestCase):
    def test_every_shape_reduces_to_p_by_three(self) -> None:
        import magpylib as magpy

        for n_sources, p in ((1, 1), (1, 201), (2, 201)):
            with self.subTest(sources=n_sources, poses=p):
                sources = [_cuboid((0, 0, N42), position=(0.01 * i, 0, 0)) for i in range(n_sources)]
                target = _cuboid((0, 0, -N42))
                zs = np.linspace(SPACING, SPACING + 0.02, p)
                target.position = np.c_[np.zeros(p), np.zeros(p), zs]
                pivot = np.zeros((1, p, 3))
                F, T = physics.force_torque(sources, [target], pivot, meshing=20, eps=1e-5)
                self.assertEqual((p, 3), F.shape)
                self.assertEqual((p, 3), T.shape)

    def test_matches_a_direct_getFT_within_a_tenth_percent(self) -> None:
        import magpylib as magpy

        from tests.python.packages.cadgen.magnetics.support import hand_built_coaxial_pair

        fixed, moving = hand_built_coaxial_pair()
        moving.meshing = 100
        ref_F, _ref_T = magpy.getFT([fixed], [moving], pivot="centroid", eps=1e-5, squeeze=False)
        ref = ref_F.sum(axis=(0, 2))[0]
        fixed2, moving2 = hand_built_coaxial_pair()
        pivot = np.zeros((1, 1, 3))
        F, _T = physics.force_torque([fixed2], [moving2], pivot, meshing=100, eps=1e-5)
        self.assertLess(abs(F[0, 2] - ref[2]) / abs(ref[2]), 1e-3)

    def test_the_coaxial_pair_matches_the_dipole_far_field_within_two_percent(self) -> None:
        from tests.python.packages.cadgen.magnetics.support import hand_built_coaxial_pair

        fixed, moving = hand_built_coaxial_pair()
        pivot = np.zeros((1, 1, 3))
        F, _T = physics.force_torque([fixed], [moving], pivot, meshing=100, eps=1e-5)
        mu0 = 4 * math.pi * 1e-7
        volume = (MAGNET_MM * 1e-3) ** 3
        m = N42 * volume / mu0
        dipole = 3 * mu0 * m**2 / (2 * math.pi * SPACING**4)
        self.assertLess(abs(abs(F[0, 2]) - dipole) / dipole, 0.02)

    def test_the_getFT_signature_is_pinned(self) -> None:
        import magpylib as magpy

        params = inspect.signature(magpy.getFT).parameters
        self.assertIn("squeeze", params)
        self.assertIn("pivot", params)
        self.assertIn("eps", params)
        self.assertEqual(("sources", "targets"), tuple(params)[:2])


@requires_extra
class ProjectTests(unittest.TestCase):
    def test_slider_projects_force_and_leaves_a_transverse_load(self) -> None:
        F = np.array([[0.4, 0.1, 0.9]])
        T = np.array([[0.0, 0.02, 0.0]])
        Q, F_perp = physics.project(F, T, _slider(direction=(0, 0, 1)), dof=None)
        self.assertAlmostEqual(0.9, float(Q[0]))
        np.testing.assert_allclose(F_perp[0], [0.4, 0.1, 0.0])

    def test_revolute_projects_torque_and_the_whole_force_is_a_guide_load(self) -> None:
        F = np.array([[0.4, 0.1, 0.9]])
        T = np.array([[0.0, 0.0, 0.021]])
        mate = MateSpec(name="turn", kind="revolute", child_id="o1.2", origin_m=(0, 0, 0),
                        dir=(0, 0, 1), limits={"value": (0.0, 1.0)}, q_unit="rad")
        Q, F_perp = physics.project(F, T, mate, dof=None)
        self.assertAlmostEqual(0.021, float(Q[0]))
        np.testing.assert_allclose(F_perp[0], F[0])


@requires_extra
class ClearanceTests(unittest.TestCase):
    def _scene(self) -> MagScene:
        moving_solid = bd.Box(10, 10, 10).solids()[0]
        fixed_solid = bd.Box(10, 10, 10).solids()[0]
        magnet = _magnet(_cuboid((0, 0, N42)), solid=moving_solid, moving=True)
        return MagScene(
            step_path="mem", magnets=(magnet,), fixed=(), moving=(0,),
            mate=_slider(direction=(1, 0, 0), limits=(0.0, 0.02)),
            bbox_m=((-0.005, -0.005, -0.005), (0.005, 0.005, 0.005)),
            solids_fixed=(fixed_solid,),
        )

    def _translate(self, q):
        D = np.eye(4)
        D[0, 3] = q  # metres along x
        return D

    def test_overlapping_then_separating_gives_an_inaccessible_interval(self) -> None:
        scene = self._scene()
        qs = np.linspace(0.0, 0.02, 11)  # 0..20 mm along x
        with mock.patch(
            "cadgen.magnetics.kinematics.delta",
            side_effect=lambda mate, q, held, swept: self._translate(q),
        ):
            accessible = physics.clearance(scene, qs, swept="value")
        self.assertFalse(bool(accessible[0]), "coincident boxes overlap")
        self.assertTrue(bool(accessible[-1]), "20 mm apart is clear")
        intervals = physics._mask_to_intervals(qs, ~accessible, np)
        self.assertEqual(1, len(intervals))
        self.assertAlmostEqual(0.0, intervals[0][0])

    def test_separated_boxes_are_all_accessible(self) -> None:
        scene = self._scene()
        qs = np.linspace(0.05, 0.07, 5)  # always >= 50 mm apart
        with mock.patch(
            "cadgen.magnetics.kinematics.delta",
            side_effect=lambda mate, q, held, swept: self._translate(q),
        ):
            accessible = physics.clearance(scene, qs, swept="value")
        self.assertTrue(bool(accessible.all()))


@requires_extra
class EnergyTests(unittest.TestCase):
    def test_negative_sine_integrates_to_one_minus_cosine(self) -> None:
        qs = np.linspace(0.0, math.pi, 400)
        Q = -np.sin(qs)
        U = physics.energy(Q, qs, np.ones(len(qs), dtype=bool))
        np.testing.assert_allclose(U, 1 - np.cos(qs), atol=2e-4)

    def test_inaccessible_poses_are_nan_and_each_interval_restarts_at_zero(self) -> None:
        qs = np.linspace(0.0, 1.0, 11)
        Q = np.ones(len(qs))
        acc = np.ones(len(qs), dtype=bool)
        acc[3:6] = False
        U = physics.energy(Q, qs, acc)
        self.assertTrue(np.all(np.isnan(U[3:6])))
        self.assertAlmostEqual(0.0, U[0])
        self.assertAlmostEqual(0.0, U[6], msg="the second interval restarts at zero")


@requires_extra
class EquilibriaTests(unittest.TestCase):
    def test_a_root_present_at_both_grids_is_classified_stable(self) -> None:
        # 10 samples: the root at q = 0 lands strictly inside a bracket, so the
        # bisection runs to a tight uncertainty rather than landing on a node.
        qs_c = np.linspace(-1.5, 1.5, 10)
        qs_f = np.linspace(-1.5, 1.5, 19)
        Q_c = -np.sin(qs_c)
        Q_f = -np.sin(qs_f)
        eqs = physics.find_equilibria(Q_c, Q_f, qs_c, qs_f, refine_fn=lambda q: -math.sin(q))
        self.assertEqual(1, len(eqs))
        self.assertAlmostEqual(0.0, eqs[0].q, places=4)
        self.assertTrue(eqs[0].stable, "dQ/dq < 0 at q = 0 for Q = -sin(q)")
        self.assertLess(eqs[0].uncertainty_q, 1e-3)

    def test_a_root_landing_exactly_on_a_node_is_still_found(self) -> None:
        qs = np.linspace(-1.5, 1.5, 11)  # includes q = 0 exactly
        Q = -np.sin(qs)
        eqs = physics.find_equilibria(Q, Q, qs, qs, refine_fn=lambda q: -math.sin(q))
        self.assertEqual(1, len(eqs))
        self.assertLess(abs(eqs[0].q), 1e-4)
        self.assertTrue(eqs[0].stable)

    def test_no_sign_change_yields_no_equilibria(self) -> None:
        qs = np.linspace(0.1, 1.0, 10)
        Q = np.cos(qs) + 2.0  # strictly positive
        eqs = physics.find_equilibria(Q, Q, qs, qs, refine_fn=lambda q: math.cos(q) + 2.0)
        self.assertEqual([], eqs)

    def test_a_coarse_only_root_is_rejected_as_a_grid_artefact(self) -> None:
        # A coarse bracket straddles a spike whose fine grid never changes sign.
        qs_c = np.array([0.0, 1.0, 2.0])
        Q_c = np.array([1.0, -1.0, 1.0])  # two coarse sign changes
        qs_f = np.linspace(0.0, 2.0, 5)
        Q_f = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # the fine grid never crosses zero
        eqs = physics.find_equilibria(Q_c, Q_f, qs_c, qs_f, refine_fn=lambda q: 1.0)
        self.assertEqual([], eqs)


@requires_extra
class ConvergeTests(unittest.TestCase):
    def _scene(self) -> MagScene:
        return MagScene(
            step_path="mem", magnets=(_magnet(_cuboid((0, 0, N42)), moving=True),),
            fixed=(), moving=(0,), mate=_slider(), bbox_m=((0, 0, 0), (0, 0, 0.03)),
            solids_fixed=(),
        )

    def test_off_runs_nothing_and_reports_none(self) -> None:
        cfg = SweepConfig(convergence="off")
        conv = physics.converge(self._scene(), cfg, np.linspace(0, 0.02, 5), 0.0, ())
        self.assertIsNone(conv.converged)
        self.assertIsNone(conv.mesh)
        self.assertIsNone(conv.eps)
        self.assertIsNone(conv.pose_grid)

    def test_quick_runs_two_mesh_levels_and_the_grid_but_not_eps(self) -> None:
        cfg = SweepConfig(convergence="quick")
        with mock.patch.object(physics, "_grid_forces", side_effect=self._const_forces(0.5)):
            conv = physics.converge(self._scene(), cfg, np.linspace(0, 0.02, 5), 0.0, ())
        self.assertEqual([20, 50], conv.mesh["levels"])
        self.assertIsNone(conv.eps, "quick does not run the eps axis")
        self.assertIsNotNone(conv.pose_grid)
        self.assertTrue(conv.converged, "a constant field converges on every axis")

    def test_full_runs_three_mesh_levels_and_two_eps_levels(self) -> None:
        cfg = SweepConfig(convergence="full")
        with mock.patch.object(physics, "_grid_forces", side_effect=self._const_forces(0.5)):
            conv = physics.converge(self._scene(), cfg, np.linspace(0, 0.02, 5), 0.0, ())
        self.assertEqual([20, 50, 100], conv.mesh["levels"])
        self.assertEqual([1e-5, 1e-6], conv.eps["levels"])
        self.assertEqual(100, conv.eps["meshing"])

    def test_a_moving_answer_fails_to_converge_and_names_the_axis(self) -> None:
        cfg = SweepConfig(convergence="full")
        # Q depends on meshing, so the mesh axis never settles.
        def forces(scene, c, qs, held, meshing, eps):
            base = 0.5 + 0.4 * (100 - meshing) / 100.0
            return (np.full(len(qs), base), None, None, None)

        with mock.patch.object(physics, "_grid_forces", side_effect=forces):
            conv = physics.converge(self._scene(), cfg, np.linspace(0, 0.02, 5), 0.0, ())
        self.assertFalse(conv.converged)
        self.assertIn("mesh", conv.reasons)

    def test_a_field_that_is_zero_everywhere_does_not_divide_by_zero(self) -> None:
        cfg = SweepConfig(convergence="full")
        with mock.patch.object(physics, "_grid_forces", side_effect=self._const_forces(0.0)):
            conv = physics.converge(self._scene(), cfg, np.linspace(0, 0.02, 5), 0.0, ())
        self.assertTrue(conv.converged)
        self.assertEqual((), conv.reasons)

    @staticmethod
    def _const_forces(value):
        def forces(scene, cfg, qs, held, meshing, eps):
            return (np.full(len(qs), value), None, None, None)

        return forces


@requires_extra
class VerdictTests(unittest.TestCase):
    def test_a_clear_positive_tendency_over_the_threshold(self) -> None:
        qs = np.linspace(0.0, 0.02, 5)
        Q = np.full(5, 0.4)
        v = physics.make_verdict(qs, Q, np.ones(5, bool), at_q=0.01, uncertainty=0.01,
                                 threshold=0.15, limits=(0.0, 0.02))
        self.assertEqual("+axis", v.tendency)
        self.assertTrue(v.exceeds_threshold)
        self.assertTrue(v.admissible)

    def test_indeterminate_when_the_band_covers_zero(self) -> None:
        qs = np.linspace(0.0, 0.02, 5)
        Q = np.full(5, 0.005)
        v = physics.make_verdict(qs, Q, np.ones(5, bool), at_q=0.01, uncertainty=0.01,
                                 threshold=None, limits=(0.0, 0.02))
        self.assertEqual("indeterminate", v.tendency)

    def test_exceeds_threshold_is_null_when_the_band_straddles_it(self) -> None:
        qs = np.linspace(0.0, 0.02, 5)
        Q = np.full(5, 0.15)
        v = physics.make_verdict(qs, Q, np.ones(5, bool), at_q=0.01, uncertainty=0.02,
                                 threshold=0.15, limits=(0.0, 0.02))
        self.assertIsNone(v.exceeds_threshold)

    def test_exceeds_threshold_is_null_without_a_threshold(self) -> None:
        qs = np.linspace(0.0, 0.02, 5)
        v = physics.make_verdict(qs, np.full(5, 0.4), np.ones(5, bool), at_q=0.01,
                                 uncertainty=0.01, threshold=None, limits=(0.0, 0.02))
        self.assertIsNone(v.exceeds_threshold)

    def test_inadmissible_at_the_lower_limit_pointing_out(self) -> None:
        qs = np.linspace(0.0, 0.02, 5)
        Q = np.full(5, -0.4)  # -axis: points below the lower limit
        v = physics.make_verdict(qs, Q, np.ones(5, bool), at_q=0.0, uncertainty=0.01,
                                 threshold=None, limits=(0.0, 0.02))
        self.assertEqual("-axis", v.tendency)
        self.assertFalse(v.admissible)

    def test_inadmissible_when_the_next_pose_is_inaccessible(self) -> None:
        qs = np.linspace(0.0, 0.02, 5)
        Q = np.full(5, 0.4)  # +axis
        acc = np.ones(5, bool)
        acc[3] = False  # the pose just ahead of at_q is blocked
        v = physics.make_verdict(qs, Q, acc, at_q=0.01, uncertainty=0.01,
                                 threshold=None, limits=(0.0, 0.02))
        self.assertFalse(v.admissible)


@requires_extra
class RelaxHeldTests(unittest.TestCase):
    def _scene(self) -> MagScene:
        mate = MateSpec(name="travel", kind="cylindrical", child_id="o1.2", origin_m=(0, 0, 0),
                        dir=(0, 0, 1), limits={"travel": (0.0, 0.02), "turn": (-1.0, 1.0)}, q_unit="m")
        return MagScene(
            step_path="mem", magnets=(_magnet(_cuboid((0, 0, N42)), moving=True),),
            fixed=(), moving=(0,), mate=mate, bbox_m=((0, 0, 0), (0, 0, 0.03)), solids_fixed=(),
        )

    def test_a_sign_changing_force_relaxes_to_its_zero(self) -> None:
        with mock.patch.object(
            physics, "_held_generalized_force",
            side_effect=lambda scene, q, held, held_key, swept: -math.sin(held - 0.3),
        ):
            root = physics.relax_held(self._scene(), q=0.01, held_key="turn", limits=(-1.0, 1.0))
        self.assertIsNotNone(root)
        self.assertAlmostEqual(0.3, root, places=4)

    def test_a_monotone_force_has_no_zero_and_marks_the_pose_wedged(self) -> None:
        with mock.patch.object(
            physics, "_held_generalized_force",
            side_effect=lambda scene, q, held, held_key, swept: held + 2.0,  # never zero on [-1,1]
        ):
            root = physics.relax_held(self._scene(), q=0.01, held_key="turn", limits=(-1.0, 1.0))
        self.assertIsNone(root)


@requires_extra
class FieldTests(unittest.TestCase):
    def _scene(self, mate_dir=(0.0, 0.0, 1.0), bbox=None) -> MagScene:
        bbox = bbox or ((-0.01, -0.01, -0.02), (0.01, 0.01, 0.02))
        solid = bd.Box(MAGNET_MM, MAGNET_MM, MAGNET_MM).solids()[0]
        m0 = _magnet(_cuboid((0, 0, N42), position=(0, 0, 0)), solid=solid, ref="#o1.1")
        m1 = _magnet(_cuboid((0, 0, -N42), position=(0, 0, 0.015)), solid=solid, ref="#o1.2", moving=True)
        return MagScene(
            step_path="mem", magnets=(m0, m1), fixed=(0,), moving=(1,),
            mate=MateSpec(name="travel", kind="slider", child_id="o1.2", origin_m=(0, 0, 0),
                          dir=mate_dir, limits={"value": (0.0, 0.02)}, q_unit="m"),
            bbox_m=bbox, solids_fixed=(solid,),
        )

    def test_a_mate_slice_samples_a_finite_field(self) -> None:
        slice_ = physics.field(self._scene(), "mate", grid=16, q=0.0)
        self.assertEqual((16,), slice_.u.shape)
        self.assertEqual((16, 16, 3), slice_.B.shape)
        self.assertTrue(np.all(np.isfinite(slice_.B)))

    def test_an_axis_normal_slice_samples_a_finite_field(self) -> None:
        slice_ = physics.field(self._scene(), "z=0", grid=16, q=0.0)
        self.assertEqual((16, 16, 3), slice_.B.shape)
        self.assertTrue(np.all(np.isfinite(slice_.B)))

    def test_the_normal_falls_back_when_the_axis_is_parallel_to_up(self) -> None:
        # mate axis is +z and z is the shortest bbox axis, so up defaults to z:
        # axis x up == 0 and the plane must fall back to another up.
        scene = self._scene(mate_dir=(0.0, 0.0, 1.0), bbox=((-0.02, -0.02, -0.001), (0.02, 0.02, 0.001)))
        slice_ = physics.field(scene, "mate", grid=8, q=0.0)
        normal = np.asarray(slice_.plane["normal"], dtype=float)
        self.assertAlmostEqual(1.0, float(np.linalg.norm(normal)), places=6)
        self.assertGreater(float(np.linalg.norm(normal)), physics.PLANE_PARALLEL_TOL)

    def test_a_plane_that_misses_the_bounding_box_is_an_error(self) -> None:
        from cadgen.magnetics.types import SceneError

        with self.assertRaises(SceneError):
            physics.field(self._scene(), "z=500", grid=8, q=0.0)


@requires_extra
class SweepAssemblyTests(unittest.TestCase):
    """The full sweep wiring, with the kinematics and force levels patched."""

    def _scene(self) -> MagScene:
        moving_solid = bd.Box(MAGNET_MM, MAGNET_MM, MAGNET_MM).solids()[0]
        magnet = _magnet(_cuboid((0, 0, N42)), solid=moving_solid, moving=True)
        return MagScene(
            step_path="mem", magnets=(magnet,), fixed=(), moving=(0,),
            mate=_slider(direction=(0, 0, 1), limits=(0.0, 0.02)),
            bbox_m=((0, 0, 0), (0, 0, 0.03)), solids_fixed=(),
        )

    def test_a_slider_sweep_assembles_a_result_with_a_stable_equilibrium(self) -> None:
        scene = self._scene()
        cfg = SweepConfig(samples=21, convergence="off", friction=0.05)
        lo, hi = scene.mate.limits["value"]

        def forces(sc, c, qs, held, meshing, eps):
            qs = np.asarray(qs, dtype=float)
            mid = 0.5 * (lo + hi)
            Q = -(qs - mid) * 50.0  # linear, one stable root at the midpoint
            return (Q, np.zeros((len(qs), 3)), np.zeros((len(qs), 3)), np.zeros((len(qs), 3)))

        with mock.patch("cadgen.magnetics.kinematics.sweep_dof", return_value=("value", None)), \
             mock.patch.object(physics, "clearance", return_value=np.ones(21, dtype=bool)), \
             mock.patch.object(physics, "_grid_forces", side_effect=forces):
            result = physics.sweep(scene, cfg)

        self.assertEqual(21, len(result.samples))
        self.assertEqual(1, len(result.equilibria))
        self.assertTrue(result.equilibria[0].stable)
        self.assertAlmostEqual(0.01, result.equilibria[0].q, places=3)
        self.assertEqual((), result.inaccessible)
        self.assertEqual((), result.wedged)
        self.assertIsNone(result.convergence.converged)

    def test_every_pose_inaccessible_is_a_mate_error(self) -> None:
        from cadgen.magnetics.types import MateError

        scene = self._scene()
        cfg = SweepConfig(samples=5, convergence="off")
        with mock.patch("cadgen.magnetics.kinematics.sweep_dof", return_value=("value", None)), \
             mock.patch.object(physics, "clearance", return_value=np.zeros(5, dtype=bool)):
            with self.assertRaises(MateError):
                physics.sweep(scene, cfg)


if __name__ == "__main__":
    unittest.main()
