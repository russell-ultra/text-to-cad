"""The ``cadgen magnetics`` CLI: the validation matrix, the stream contract and
the model-unit boundary (lane A5, task T5).

Every physics module is still a lane stub, so nothing here runs geometry: the
verb bodies import ``cadgen.magnetics.{scene,kinematics,physics,report}`` inside
the function and call them attribute-style, so each row of the matrix is a
``mock.patch`` on the module function that raises the domain error or returns a
fixture. What is asserted per row is the exit code AND a message substring; the
happy paths assert the stdout/stderr split and the O(1) ``--json`` payload; the
unit tests assert ``--at``/``--hold`` cross the model->SI boundary in the CLI.

End-to-end on a real STEP is B1's ``test_e2e.py``; there is no build here.
"""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from unittest import mock

import numpy as np

from tests.python.support.paths import add_repo_path

add_repo_path("packages/cadgen/src")

from cadgen._internal.source_sidecar import SidecarBindingError  # noqa: E402
from cadgen.cli import magnetics as cli  # noqa: E402
from cadgen.magnetics import types  # noqa: E402

HINT = 'pip install "cadgen[magnetics]"'
SCENE = "cadgen.magnetics.scene.load"
SELECT = "cadgen.magnetics.kinematics.select_mate"
SWEEP_DOF = "cadgen.magnetics.kinematics.sweep_dof"
SWEEP = "cadgen.magnetics.physics.sweep"
FIELD = "cadgen.magnetics.physics.field"
WRITE_JSON = "cadgen.magnetics.report.write_json"
WRITE_REPORT = "cadgen.magnetics.report.write_report"
WRITE_FIELD = "cadgen.magnetics.report.write_field"


# --------------------------------------------------------------------------- fixtures


def _slider_mate() -> types.MateSpec:
    # limits 0..20 mm == 0..0.02 m, SI as scene.load/select_mate produce.
    return types.MateSpec(
        name="travel", kind="slider", child_id="o1.2", origin_m=(0.0, 0.0, 0.0),
        dir=(0.0, 0.0, 1.0), limits={"value": (0.0, 0.02)}, q_unit="m",
    )


def _revolute_mate() -> types.MateSpec:
    # limits 0..90 degrees == 0..pi/2 rad.
    import math

    return types.MateSpec(
        name="hinge", kind="revolute", child_id="o1.2", origin_m=(0.0, 0.0, 0.0),
        dir=(0.0, 0.0, 1.0), limits={"value": (0.0, math.pi / 2)}, q_unit="rad",
    )


def _cylindrical_mate() -> types.MateSpec:
    import math

    return types.MateSpec(
        name="travel", kind="cylindrical", child_id="o1.2", origin_m=(0.0, 0.0, 0.0),
        dir=(1.0, 0.0, 0.0), limits={"travel": (0.0, 0.024), "turn": (0.0, math.pi)}, q_unit="m",
    )


def _magnet(moving: bool | None) -> types.MagnetSpec:
    return types.MagnetSpec(
        ref="#o1.1.1", label="mag:N42:+z", shape="cuboid", source=None,
        polarization_world_T=(0.0, 0.0, 1.3), position_m=(0.0, 0.0, 0.0),
        orientation_quat=(0.0, 0.0, 0.0, 1.0), moving=moving, solid=None,
    )


def _scene(mate: types.MateSpec | None) -> types.MagScene:
    magnets = (_magnet(False), _magnet(True))
    return types.MagScene(
        step_path="m.step", magnets=magnets, fixed=(0,), moving=(1,) if mate else (),
        mate=mate, bbox_m=((-0.01, -0.01, -0.01), (0.01, 0.01, 0.04)), solids_fixed=(),
    )


def _sweep_result(mate: types.MateSpec | None = None, *, converged: bool | None = True) -> types.SweepResult:
    mate = mate or _slider_mate()
    dof = held = None
    if mate.kind == "cylindrical":
        # A cylindrical result must name a swept sub-DOF, or sweep_result_to_json
        # looks for the absent "value" limits key.
        dof = f"{mate.name}.travel"
        held = {"name": f"{mate.name}.turn", "mode": "hold", "value": 0.0}
    samples = tuple(
        types.Sample(q=q, F_N=(0.0, 0.0, f), Q=f, F_transverse_N=(0.0, 0.0, 0.0),
                     torque_Nm=(0.0, 0.0, 0.0), U_J=0.0, accessible=True,
                     held_value=0.0 if dof is not None else None)
        for q, f in ((0.0, 0.5), (0.01, 0.2), (0.02, -0.1))
    )
    return types.SweepResult(
        magnets=(_magnet(False), _magnet(True)), mate=mate, dof=dof, held=held, samples=samples,
        inaccessible=(), wedged=(),
        equilibria=(types.Equilibrium(q=0.015, stable=True, dQdq=-30.0, uncertainty_q=1e-4),),
        verdict=types.Verdict(at_q=0.0, tendency="+axis", Q=0.5, uncertainty=0.01, threshold=0.15,
                              exceeds_threshold=True, admissible=True),
        convergence=types.Convergence(
            mesh={"levels": [20, 50], "eps": 1e-5, "dQ": 0.005}, eps=None,
            pose_grid={"samples": [3, 5], "dQ": 0.001, "equilibria_agree": True, "dU_max": 0.0},
            converged=converged, reasons=() if converged is not False else ("mesh moved Q 8%",),
        ),
    )


def _field_slice() -> types.FieldSlice:
    b = np.zeros((2, 2, 3), dtype=float)
    b[..., 2] = 0.4
    return types.FieldSlice(
        plane={"spec": "mate", "origin_m": [0.0, 0.0, 0.0], "normal": [0.0, 1.0, 0.0],
               "u_axis": [1.0, 0.0, 0.0], "v_axis": [0.0, 0.0, 1.0]},
        u=[0.0, 1.0], v=[0.0, 1.0], B=b, outlines=(),
    )


class _Base(unittest.TestCase):
    def _invoke(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = cli.main(list(argv))
            except SystemExit as exc:  # argparse / parser.error rows
                code = exc.code if isinstance(exc.code, int) else 2
        return code, out.getvalue(), err.getvalue()

    def _write_paths(self) -> dict[str, mock.Mock]:
        """Patches for the three report writers, each returning a fake path."""
        from pathlib import Path

        return {
            WRITE_JSON: mock.patch(WRITE_JSON, return_value=Path("out/sweep.json")),
            WRITE_REPORT: mock.patch(WRITE_REPORT, return_value=Path("out/report.html")),
            WRITE_FIELD: mock.patch(WRITE_FIELD, return_value=Path("out/field.html")),
        }


# --------------------------------------------------------------------------- exit 1


class ExitOneAnalysisErrorTests(_Base):
    """Every analysis-error row: exit 1, the domain message reaches stderr, stdout empty."""

    def _fails_one(self, code: int, out: str) -> None:
        self.assertEqual(1, code, f"stdout={out!r}")
        self.assertEqual("", out, "an analysis error puts nothing on stdout")

    def test_dependency_missing(self) -> None:
        exc = types.MagneticsDependencyError(f'magpylib is required: {HINT}')
        with mock.patch(SCENE, side_effect=exc):
            code, out, err = self._invoke("inspect", "m.step")
        self._fails_one(code, out)
        self.assertIn(HINT, err)

    def test_no_magnet_leaf(self) -> None:
        with mock.patch(SCENE, side_effect=types.SceneError("no mag: leaf in m.step")):
            code, out, err = self._invoke("inspect", "m.step")
        self._fails_one(code, out)
        self.assertIn("m.step", err)

    def test_unknown_grade_names_leaf_and_token(self) -> None:
        with mock.patch(SCENE, side_effect=types.SceneError("#o1.1.1: unknown grade token 'N99'")):
            code, out, err = self._invoke("inspect", "m.step")
        self._fails_one(code, out)
        self.assertIn("#o1.1.1", err)
        self.assertIn("N99", err)

    def test_zero_direction_names_leaf(self) -> None:
        with mock.patch(SCENE, side_effect=types.SceneError("#o1.1.1: direction vector has zero length")):
            code, out, err = self._invoke("inspect", "m.step")
        self._fails_one(code, out)
        self.assertIn("#o1.1.1", err)

    def test_open_mesh_names_leaf_and_check(self) -> None:
        with mock.patch(SCENE, side_effect=types.SceneError("#o1.1.1: solid failed weld check (open edges)")):
            code, out, err = self._invoke("inspect", "m.step")
        self._fails_one(code, out)
        self.assertIn("#o1.1.1", err)
        self.assertIn("weld", err)

    def test_stale_sidecar_propagates_cadgens_error(self) -> None:
        with mock.patch(SELECT, side_effect=SidecarBindingError("m.step.json documentHash mismatch; regenerate")):
            code, out, err = self._invoke("sweep", "m.step")
        self._fails_one(code, out)
        self.assertIn("documentHash", err)

    def test_ambiguous_mate_names_available(self) -> None:
        with mock.patch(SELECT, side_effect=types.MateError("2 mates (travel, hinge); pass --mate")):
            code, out, err = self._invoke("sweep", "m.step")
        self._fails_one(code, out)
        self.assertIn("travel", err)
        self.assertIn("hinge", err)

    def test_unknown_mate_kind_names_mate_and_kind(self) -> None:
        with mock.patch(SELECT, side_effect=types.MateError("mate 'ball' kind ball is not slider/revolute/cylindrical")):
            code, out, err = self._invoke("sweep", "m.step", "--mate", "ball")
        self._fails_one(code, out)
        self.assertIn("ball", err)

    def test_missing_child_id_names_mate(self) -> None:
        with mock.patch(SELECT, side_effect=types.MateError("mate 'travel' has no childId")):
            code, out, err = self._invoke("sweep", "m.step")
        self._fails_one(code, out)
        self.assertIn("travel", err)

    def test_moving_subtree_without_magnets(self) -> None:
        with mock.patch(SELECT, return_value=_slider_mate()), \
                mock.patch(SCENE, side_effect=types.MateError("mate 'travel' child #o1.2 has no magnets")):
            code, out, err = self._invoke("sweep", "m.step")
        self._fails_one(code, out)
        self.assertIn("travel", err)
        self.assertIn("#o1.2", err)

    def test_reversed_limits_names_mate_and_limits(self) -> None:
        with mock.patch(SELECT, side_effect=types.MateError("mate 'travel' limits reversed (20, 0)")):
            code, out, err = self._invoke("sweep", "m.step")
        self._fails_one(code, out)
        self.assertIn("travel", err)

    def test_slice_misses_bounding_box(self) -> None:
        with mock.patch(SCENE, return_value=_scene(None)), \
                mock.patch(FIELD, side_effect=types.SceneError("plane x=99 misses the model bounding box")):
            code, out, err = self._invoke("field", "m.step", "--slice", "x=99")
        self._fails_one(code, out)
        self.assertIn("x=99", err)

    def test_every_pose_inaccessible(self) -> None:
        with mock.patch(SELECT, return_value=_slider_mate()), \
                mock.patch(SCENE, return_value=_scene(_slider_mate())), \
                mock.patch(SWEEP, side_effect=types.MateError("mate 'travel': every sampled pose is inaccessible")):
            code, out, err = self._invoke("sweep", "m.step")
        self._fails_one(code, out)
        self.assertIn("travel", err)


# --------------------------------------------------------------------------- exit 2 (mate-dependent)


class ExitTwoMateDependentTests(_Base):
    """Rows that need the resolved mate to judge, but are usage errors (exit 2)."""

    def _usage(self, code: int, out: str) -> None:
        self.assertEqual(2, code, f"stdout={out!r}")
        self.assertEqual("", out, "a usage error puts nothing on stdout")

    def test_cylindrical_without_dof(self) -> None:
        with mock.patch(SELECT, return_value=_cylindrical_mate()), \
                mock.patch(SWEEP_DOF, side_effect=types.MateError(
                    "cylindrical mate 'travel' needs --dof travel.travel or travel.turn")):
            code, out, err = self._invoke("sweep", "m.step")
        self._usage(code, out)
        self.assertIn("travel", err)
        self.assertIn("--dof", err)

    def test_dof_names_a_subdof_the_mate_lacks(self) -> None:
        with mock.patch(SELECT, return_value=_cylindrical_mate()), \
                mock.patch(SWEEP_DOF, side_effect=types.MateError(
                    "--dof names mate 'other', not the selected 'travel'")):
            code, out, err = self._invoke("sweep", "m.step", "--dof", "other.turn")
        self._usage(code, out)
        self.assertIn("travel", err)

    def test_dof_on_a_non_cylindrical_mate(self) -> None:
        with mock.patch(SELECT, return_value=_slider_mate()):
            code, out, err = self._invoke("sweep", "m.step", "--dof", "travel.turn")
        self._usage(code, out)
        self.assertIn("--dof", err)
        self.assertIn("slider", err)

    def test_at_outside_limits(self) -> None:
        # slider value limits 0..20 mm; --at 30 mm is past the end.
        with mock.patch(SELECT, return_value=_slider_mate()):
            code, out, err = self._invoke("sweep", "m.step", "--at", "30")
        self._usage(code, out)
        self.assertIn("--at", err)
        self.assertIn("30", err)

    def test_hold_outside_held_subdof_limits(self) -> None:
        # cylindrical turn limits 0..180 degrees; --hold 400 degrees is past the end.
        with mock.patch(SELECT, return_value=_cylindrical_mate()), \
                mock.patch(SWEEP_DOF, return_value=("travel", "turn")):
            code, out, err = self._invoke("sweep", "m.step", "--dof", "travel.travel", "--hold", "400")
        self._usage(code, out)
        self.assertIn("--hold", err)
        self.assertIn("turn", err)

    def test_friction_N_on_a_rotation_dof(self) -> None:
        with mock.patch(SELECT, return_value=_revolute_mate()):
            code, out, err = self._invoke("sweep", "m.step", "--friction-N", "0.1")
        self._usage(code, out)
        self.assertIn("--friction-N", err)
        self.assertIn("--friction-Nm", err)

    def test_friction_Nm_on_a_translation_dof(self) -> None:
        with mock.patch(SELECT, return_value=_slider_mate()):
            code, out, err = self._invoke("sweep", "m.step", "--friction-Nm", "0.1")
        self._usage(code, out)
        self.assertIn("--friction-Nm", err)
        self.assertIn("--friction-N", err)


# --------------------------------------------------------------------------- exit 2 (argparse)


class ExitTwoArgparseTests(_Base):
    """The argparse rows, asserted in-process for the code and the flag name."""

    def _usage(self, *argv: str) -> str:
        code, out, err = self._invoke(*argv)
        self.assertEqual(2, code, f"stdout={out!r} stderr={err!r}")
        self.assertEqual("", out)
        return err

    def test_samples_below_three(self) -> None:
        self.assertIn("--samples", self._usage("sweep", "m.step", "--samples", "2"))

    def test_grid_below_eight(self) -> None:
        self.assertIn("--grid", self._usage("field", "m.step", "--grid", "4"))

    def test_bad_slice_syntax(self) -> None:
        self.assertIn("--slice", self._usage("field", "m.step", "--slice", "w=3"))

    def test_bad_dof_syntax(self) -> None:
        self.assertIn("--dof", self._usage("sweep", "m.step", "--dof", "travel"))

    def test_negative_friction(self) -> None:
        self.assertIn("--friction-N", self._usage("sweep", "m.step", "--friction-N", "-1"))

    def test_both_friction_flags(self) -> None:
        err = self._usage("sweep", "m.step", "--friction-N", "0.1", "--friction-Nm", "0.2")
        self.assertIn("--friction-N", err)

    def test_converge_tol_not_positive(self) -> None:
        self.assertIn("--converge-tol", self._usage("sweep", "m.step", "--converge-tol", "0"))

    def test_hold_and_relax_exclusive(self) -> None:
        err = self._usage("sweep", "m.step", "--dof", "t.turn", "--hold", "1", "--relax")
        self.assertIn("--relax", err)

    def test_relax_needs_dof(self) -> None:
        err = self._usage("sweep", "m.step", "--relax")
        self.assertIn("--relax", err)
        self.assertIn("--dof", err)

    def test_at_not_a_number(self) -> None:
        self.assertIn("--at", self._usage("sweep", "m.step", "--at", "abc"))


# --------------------------------------------------------------------------- happy paths


class InspectTests(_Base):
    def test_text_lists_the_magnets_and_mate(self) -> None:
        with mock.patch(SCENE, return_value=_scene(_slider_mate())):
            code, out, err = self._invoke("inspect", "m.step", "--mate", "travel")
        self.assertEqual(0, code, err)
        self.assertIn("magnets 2", out)
        self.assertIn("travel (slider)", out)
        self.assertIn("mag:N42:+z", out)

    def test_json_is_one_compact_line(self) -> None:
        with mock.patch(SCENE, return_value=_scene(_slider_mate())):
            code, out, err = self._invoke("inspect", "m.step", "--mate", "travel", "--json")
        self.assertEqual(0, code, err)
        self.assertEqual(1, len(out.strip().splitlines()), "exactly one line on stdout")
        payload = json.loads(out)  # parses after ignoring stderr
        self.assertEqual("m.step", payload["step"])
        self.assertEqual("travel", payload["mate"]["name"])
        self.assertEqual(2, len(payload["magnets"]))
        self.assertNotIn(" ", out.rstrip("\n"), "compact JSON has no separator spaces")

    def test_json_mate_is_null_without_a_mate(self) -> None:
        with mock.patch(SCENE, return_value=_scene(None)):
            code, out, err = self._invoke("inspect", "m.step", "--json")
        self.assertEqual(0, code, err)
        self.assertIsNone(json.loads(out)["mate"])


class SweepTests(_Base):
    def _run_sweep(self, result: types.SweepResult, *argv: str, field_exc: Exception | None = None):
        patches = self._write_paths()
        with mock.patch(SELECT, return_value=result.mate), \
                mock.patch(SCENE, return_value=_scene(result.mate)), \
                mock.patch(SWEEP, return_value=result), \
                mock.patch(FIELD, side_effect=field_exc, return_value=None if field_exc else _field_slice()) as field, \
                patches[WRITE_JSON] as wj, patches[WRITE_REPORT] as wr, patches[WRITE_FIELD]:
            code, out, err = self._invoke("sweep", "m.step", *argv)
        return code, out, err, field, wj, wr

    def test_text_output_and_stream_split(self) -> None:
        code, out, err, _field, _wj, _wr = self._run_sweep(_sweep_result())
        self.assertEqual(0, code, err)
        self.assertIn("verdict", out)
        self.assertIn("+axis", out)
        self.assertIn("report", out)
        self.assertNotIn("[cadgen]", out, "narration never touches stdout")
        self.assertIn("[cadgen]", err, "narration is on stderr")

    def test_json_is_o1_and_carries_the_paths(self) -> None:
        code, out, err, _field, _wj, _wr = self._run_sweep(_sweep_result(), "--json")
        self.assertEqual(0, code, err)
        self.assertEqual(1, len(out.strip().splitlines()))
        payload = json.loads(out)
        self.assertTrue(payload["ok"])
        self.assertNotIn("samples", payload, "the per-pose table stays in sweep.json, off stdout")
        self.assertEqual("magnetics-report/1", payload["schema"])
        self.assertEqual({"json", "report"}, set(payload["out"]))

    def test_report_gets_the_slice_at_the_verdict_pose(self) -> None:
        code, out, err, field, _wj, wr = self._run_sweep(_sweep_result())
        self.assertEqual(0, code, err)
        # field is computed at verdict.at_q (SI, from physics) for the report.
        self.assertEqual(0.0, field.call_args.kwargs["q"])
        self.assertIs(_field_slice().__class__, wr.call_args.args[1].__class__)

    def test_a_degenerate_slice_downgrades_to_none(self) -> None:
        code, out, err, _field, _wj, wr = self._run_sweep(
            _sweep_result(), field_exc=types.SceneError("mate plane is degenerate")
        )
        self.assertEqual(0, code, err)
        self.assertIsNone(wr.call_args.args[1], "the report is written with no slice")
        self.assertIn("no field slice", err)

    def test_not_converged_warns_but_exits_zero(self) -> None:
        code, out, err, _field, _wj, _wr = self._run_sweep(_sweep_result(converged=False))
        self.assertEqual(0, code, err)
        self.assertIn("not converged", err)
        self.assertIn("mesh moved Q", err)


class FieldTests(_Base):
    def test_json_reports_b_max_and_the_path(self) -> None:
        patches = self._write_paths()
        with mock.patch(SELECT, return_value=_slider_mate()), \
                mock.patch(SCENE, return_value=_scene(_slider_mate())), \
                mock.patch(FIELD, return_value=_field_slice()), \
                patches[WRITE_FIELD]:
            code, out, err = self._invoke("field", "m.step", "--slice", "mate", "--json")
        self.assertEqual(0, code, err)
        payload = json.loads(out)
        self.assertTrue(payload["ok"])
        self.assertAlmostEqual(0.4, payload["B_max_T"])
        self.assertEqual({"field"}, set(payload["out"]))

    def test_axis_slice_needs_no_mate(self) -> None:
        patches = self._write_paths()
        with mock.patch(SELECT, side_effect=AssertionError("select_mate must not be called")) as select, \
                mock.patch(SCENE, return_value=_scene(None)) as scene, \
                mock.patch(FIELD, return_value=_field_slice()), \
                patches[WRITE_FIELD]:
            code, out, err = self._invoke("field", "m.step", "--slice", "z=0")
        self.assertEqual(0, code, err)
        select.assert_not_called()
        self.assertIsNone(scene.call_args.args[1], "scene.load gets mate_name=None")


# --------------------------------------------------------------------------- model -> SI boundary


class UnitConversionTests(_Base):
    """--at and --hold are model units at the CLI and SI in SweepConfig / physics.field."""

    def _config_for(self, mate: types.MateSpec, *argv: str) -> types.SweepConfig:
        captured: dict[str, types.SweepConfig] = {}

        def _capture(scene, cfg):  # noqa: ANN001
            captured["cfg"] = cfg
            return _sweep_result(mate)

        patches = self._write_paths()
        with mock.patch(SELECT, return_value=mate), \
                mock.patch(SCENE, return_value=_scene(mate)), \
                mock.patch(SWEEP_DOF, return_value=("travel", "turn")), \
                mock.patch(SWEEP, side_effect=_capture), \
                mock.patch(FIELD, return_value=_field_slice()), \
                patches[WRITE_JSON], patches[WRITE_REPORT], patches[WRITE_FIELD]:
            code, out, err = self._invoke("sweep", "m.step", *argv)
        self.assertEqual(0, code, err)
        return captured["cfg"]

    def test_slider_at_is_mm_to_m(self) -> None:
        cfg = self._config_for(_slider_mate(), "--at", "10")
        self.assertAlmostEqual(0.010, cfg.at)

    def test_revolute_at_is_degrees_to_rad(self) -> None:
        import math

        cfg = self._config_for(_revolute_mate(), "--at", "45")
        self.assertAlmostEqual(math.radians(45), cfg.at)

    def test_cylindrical_hold_uses_the_held_subdof_unit(self) -> None:
        import math

        # swept travel (mm), held turn (degrees): --hold is degrees -> rad.
        cfg = self._config_for(_cylindrical_mate(), "--dof", "travel.travel", "--hold", "30")
        self.assertAlmostEqual(math.radians(30), cfg.hold)
        self.assertEqual("travel.travel", cfg.dof)

    def test_single_dof_config_has_no_dof_and_zero_hold(self) -> None:
        cfg = self._config_for(_slider_mate())
        self.assertIsNone(cfg.dof)
        self.assertEqual(0.0, cfg.hold)
        self.assertIsNone(cfg.at)

    def test_field_at_is_converted_to_si(self) -> None:
        captured: dict[str, float] = {}

        def _capture(scene, spec, grid, q):  # noqa: ANN001
            captured["q"] = q
            return _field_slice()

        patches = self._write_paths()
        with mock.patch(SELECT, return_value=_slider_mate()), \
                mock.patch(SCENE, return_value=_scene(_slider_mate())), \
                mock.patch(FIELD, side_effect=_capture), \
                patches[WRITE_FIELD]:
            code, out, err = self._invoke("field", "m.step", "--slice", "mate", "--at", "10")
        self.assertEqual(0, code, err)
        self.assertAlmostEqual(0.010, captured["q"])


if __name__ == "__main__":
    unittest.main()
