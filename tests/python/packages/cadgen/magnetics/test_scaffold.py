"""The Phase 0 contract holds: the CLI surface parses, the package imports light,
the dependency gate names the fix, the types are frozen, the fixtures build.

Nothing here runs physics (every physics body is a lane stub). What it proves
is what the eight lanes build on: ``cadgen magnetics`` and each verb answer
``--help``; the argparse rows of the validation matrix exit 2 naming the flag;
``_deps`` raises the ``pip install "cadgen[magnetics]"`` hint; every dataclass
in ``types`` is frozen and ``sweep_result_to_json`` emits the schema;
``support.build_model`` builds the fixtures with the magnet names and the mate
the lanes expect; and ``import cadgen.magnetics`` succeeds with magpylib, numpy,
scipy and plotly all unimportable.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from tests.python.packages.cadgen.magnetics import support
from tests.python.support.paths import add_repo_path, repo_path

add_repo_path("packages/cadgen/src")

from cadgen.magnetics import _deps, types  # noqa: E402

REPO = Path(repo_path("."))
VERBS = ("inspect", "sweep", "field")


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(REPO / "packages" / "cadgen" / "src")])
    env["CADGEN_DAEMON"] = "0"
    return env


def _cli(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "cadgen.cli", "magnetics", *argv],
        cwd=REPO, env=_env(), capture_output=True, text=True, check=False, timeout=300,
    )


class HelpTests(unittest.TestCase):
    def test_the_command_group_answers_help(self) -> None:
        result = _cli("--help")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("cadgen magnetics", result.stdout)
        for verb in VERBS:
            self.assertIn(verb, result.stdout)

    def test_every_verb_answers_help(self) -> None:
        for verb in VERBS:
            with self.subTest(verb=verb):
                result = _cli(verb, "--help")
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertTrue(result.stdout.strip())
                self.assertIn(f"cadgen magnetics {verb}", result.stdout)
                self.assertIn("--mate", result.stdout)
                self.assertIn("--json", result.stdout)

    def test_the_module_form_has_a_guard(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "cadgen.cli.magnetics", "sweep", "--help"],
            cwd=REPO, env=_env(), capture_output=True, text=True, check=False, timeout=300,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("--samples", result.stdout)

    def test_the_registry_names_the_module(self) -> None:
        from cadgen.cli import _COMMANDS

        module, help_text = _COMMANDS["magnetics"]
        self.assertEqual("cadgen.cli.magnetics", module)
        for verb in VERBS:
            self.assertIn(verb, help_text)


class UsageErrorTests(unittest.TestCase):
    """The argparse-only rows of the validation matrix: exit 2, the flag and value named."""

    def _usage_error(self, *argv: str) -> str:
        result = _cli(*argv)
        self.assertEqual(2, result.returncode, f"stdout={result.stdout!r} stderr={result.stderr!r}")
        self.assertEqual("", result.stdout, "a usage error puts nothing on stdout")
        return result.stderr

    def test_samples_below_three(self) -> None:
        err = self._usage_error("sweep", "m.step", "--samples", "2")
        self.assertIn("--samples", err)
        self.assertIn("2", err)

    def test_grid_below_eight(self) -> None:
        err = self._usage_error("field", "m.step", "--grid", "4")
        self.assertIn("--grid", err)
        self.assertIn("4", err)

    def test_bad_slice_syntax(self) -> None:
        for spec in ("w=3", "x", "x=abc", "mate:q", "plane"):
            with self.subTest(spec=spec):
                err = self._usage_error("field", "m.step", "--slice", spec)
                self.assertIn("--slice", err)
                self.assertIn(spec, err)

    def test_bad_dof_syntax(self) -> None:
        for spec in ("travel", "travel.spin", ".turn"):
            with self.subTest(spec=spec):
                err = self._usage_error("sweep", "m.step", "--dof", spec)
                self.assertIn("--dof", err)
                self.assertIn(spec, err)

    def test_hold_and_relax_are_exclusive(self) -> None:
        err = self._usage_error("sweep", "m.step", "--dof", "travel.travel", "--hold", "1", "--relax")
        self.assertIn("--relax", err)
        self.assertIn("--hold", err)

    def test_relax_needs_a_dof(self) -> None:
        err = self._usage_error("sweep", "m.step", "--relax")
        self.assertIn("--relax", err)
        self.assertIn("--dof", err)

    def test_both_friction_flags(self) -> None:
        err = self._usage_error("sweep", "m.step", "--friction-N", "0.1", "--friction-Nm", "0.1")
        self.assertIn("--friction-N", err)
        self.assertIn("--friction-Nm", err)

    def test_converge_tol_must_be_positive(self) -> None:
        err = self._usage_error("sweep", "m.step", "--converge-tol", "0")
        self.assertIn("--converge-tol", err)

    def test_unknown_verb(self) -> None:
        err = self._usage_error("plot", "m.step")
        self.assertIn("plot", err)


class DependencyGateTests(unittest.TestCase):
    """Each accessor turns ImportError into the hint; nothing else about the message matters."""

    HINT = 'pip install "cadgen[magnetics]"'

    def _blocked(self, *names: str) -> dict[str, None]:
        return {name: None for name in names}

    def test_magpylib(self) -> None:
        with mock.patch.dict(sys.modules, self._blocked("magpylib")):
            with self.assertRaises(types.MagneticsDependencyError) as caught:
                _deps.magpylib()
        self.assertIn(self.HINT, str(caught.exception))
        self.assertIn("magpylib", str(caught.exception))

    def test_numpy(self) -> None:
        with mock.patch.dict(sys.modules, self._blocked("numpy")):
            with self.assertRaises(types.MagneticsDependencyError) as caught:
                _deps.numpy()
        self.assertIn(self.HINT, str(caught.exception))

    def test_scipy_rotation(self) -> None:
        with mock.patch.dict(sys.modules, self._blocked("scipy", "scipy.spatial", "scipy.spatial.transform")):
            with self.assertRaises(types.MagneticsDependencyError) as caught:
                _deps.scipy_rotation()
        self.assertIn(self.HINT, str(caught.exception))

    def test_plotly(self) -> None:
        with mock.patch.dict(sys.modules, self._blocked("plotly", "plotly.graph_objects")):
            with self.assertRaises(types.MagneticsDependencyError) as caught:
                _deps.plotly()
        self.assertIn(self.HINT, str(caught.exception))

    def test_the_error_is_a_magnetics_error(self) -> None:
        self.assertTrue(issubclass(types.MagneticsDependencyError, types.MagneticsError))
        self.assertIn(self.HINT, _deps.HINT)

    def test_the_accessors_return_the_real_modules_when_installed(self) -> None:
        self.assertEqual("magpylib", _deps.magpylib().__name__)
        self.assertEqual("numpy", _deps.numpy().__name__)
        self.assertEqual("Rotation", _deps.scipy_rotation().__name__)
        self.assertTrue(hasattr(_deps.plotly(), "graph_objects"))


def _sample_result() -> types.SweepResult:
    magnet = types.MagnetSpec(
        ref="#o1.1.1", label="mag:N42:+z", shape="cuboid", source=None,
        polarization_world_T=(0.0, 0.0, 1.3), position_m=(0.0, 0.0, 0.0),
        orientation_quat=(0.0, 0.0, 0.0, 1.0), moving=False, solid=None,
    )
    mate = types.MateSpec(
        name="travel", kind="slider", child_id="o1.2", origin_m=(0.0, 0.0, 0.0),
        dir=(0.0, 0.0, 1.0), limits={"value": (0.0, 0.02)}, q_unit="m",
    )
    samples = tuple(
        types.Sample(q=q, F_N=(0.0, 0.0, f), Q=f, F_transverse_N=(0.0, 0.0, 0.0),
                     torque_Nm=(0.0, 0.0, 0.0), U_J=0.0, accessible=True)
        for q, f in ((0.0, 0.5), (0.01, 0.2), (0.02, -0.1))
    )
    return types.SweepResult(
        magnets=(magnet,), mate=mate, dof=None, held=None, samples=samples,
        inaccessible=(), wedged=(),
        equilibria=(types.Equilibrium(q=0.015, stable=True, dQdq=-30.0, uncertainty_q=1e-4),),
        verdict=types.Verdict(at_q=0.0, tendency="+axis", Q=0.5, uncertainty=0.01, threshold=0.15,
                              exceeds_threshold=True, admissible=True),
        convergence=types.Convergence(
            mesh={"levels": [20, 50], "eps": 1e-5, "dQ": 0.005}, eps=None,
            pose_grid={"samples": [3, 5], "dQ": 0.001, "equilibria_agree": True, "dU_max": 0.0},
            converged=True, reasons=(),
        ),
    )


class TypesTests(unittest.TestCase):
    DATACLASSES = (
        types.MagnetSpec, types.MateSpec, types.MagScene, types.SweepConfig, types.Sample,
        types.Equilibrium, types.Verdict, types.Convergence, types.SweepResult, types.FieldSlice,
    )

    def test_every_dataclass_is_frozen(self) -> None:
        for cls in self.DATACLASSES:
            with self.subTest(cls=cls.__name__):
                self.assertTrue(dataclasses.is_dataclass(cls))
                self.assertTrue(cls.__dataclass_params__.frozen, f"{cls.__name__} is not frozen")

    def test_a_frozen_instance_rejects_assignment(self) -> None:
        result = _sample_result()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.verdict.Q = 1.0  # type: ignore[misc]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.mate.name = "other"  # type: ignore[misc]
        cfg = types.SweepConfig()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg.samples = 5  # type: ignore[misc]

    def test_sweep_config_defaults_match_the_design(self) -> None:
        cfg = types.SweepConfig()
        self.assertEqual((201, "quick", 0.02, None, None, None, 0.0, False),
                         (cfg.samples, cfg.convergence, cfg.converge_tol, cfg.friction, cfg.at, cfg.dof, cfg.hold, cfg.relax))
        self.assertEqual((20, 50, 100), cfg.meshing_levels)
        self.assertEqual((1e-5, 1e-6), cfg.eps_levels)

    def test_the_exception_tree(self) -> None:
        for cls in (types.MagneticsDependencyError, types.SceneError, types.MateError):
            self.assertTrue(issubclass(cls, types.MagneticsError))
        self.assertTrue(issubclass(types.ConvergenceWarning, Warning))

    def test_sweep_result_to_json_emits_the_schema(self) -> None:
        document = types.sweep_result_to_json(_sample_result())
        self.assertEqual("magnetics-report/1", document["schema"])
        self.assertEqual("m", document["q_unit"])
        self.assertEqual({"length": "m", "force": "N", "torque": "N m", "energy": "J", "B": "T"}, document["units"])
        self.assertEqual(
            {"schema", "units", "q_unit", "magnets", "mate", "samples", "inaccessible", "wedged",
             "equilibria", "verdict", "convergence"},
            set(document),
        )
        self.assertEqual([0.0, 0.02], document["mate"]["limits"])
        self.assertNotIn("dof", document["mate"], "a slider sweep carries no cylindrical keys")
        self.assertNotIn("held_value", document["samples"][0])
        # Compact and round-trippable: the report and stdout both read this.
        text = json.dumps(document, separators=(",", ":"))
        self.assertEqual(document, json.loads(text))
        self.assertNotIn("\n", text)

    def test_a_cylindrical_result_reports_the_swept_unit(self) -> None:
        base = _sample_result()
        mate = dataclasses.replace(base.mate, kind="cylindrical",
                                   limits={"travel": (0.0, 0.02), "turn": (0.0, 3.14)})
        result = dataclasses.replace(
            base, mate=mate, dof="travel.turn", held={"name": "travel.travel", "mode": "hold", "value": 0.0},
            samples=tuple(dataclasses.replace(s, held_value=0.0) for s in base.samples),
        )
        document = types.sweep_result_to_json(result)
        self.assertEqual("rad", document["q_unit"])
        self.assertEqual([0.0, 3.14], document["mate"]["limits"])
        self.assertEqual("travel.turn", document["mate"]["dof"])
        self.assertEqual("hold", document["mate"]["held"]["mode"])
        self.assertEqual(0.0, document["samples"][0]["held_value"])


class ImportLightTests(unittest.TestCase):
    """cadgen must import, and `--help` must answer, with the extra absent and no kernel."""

    def test_the_package_imports_with_every_heavy_dependency_unimportable(self) -> None:
        script = r"""
import sys
for name in ("magpylib", "numpy", "scipy", "scipy.spatial", "scipy.spatial.transform", "plotly", "plotly.graph_objects"):
    sys.modules[name] = None
import cadgen.magnetics
import cadgen.magnetics.types, cadgen.magnetics._deps, cadgen.magnetics.scene
import cadgen.magnetics.kinematics, cadgen.magnetics.physics, cadgen.magnetics.report
import cadgen.cli.magnetics
cadgen.cli.magnetics.build_parser("cadgen magnetics").parse_args(["sweep", "m.step", "--samples", "9"])
assert cadgen.magnetics.SceneError is cadgen.magnetics.types.SceneError
assert cadgen.magnetics.load_scene is cadgen.magnetics.scene.load
assert cadgen.magnetics.sweep is cadgen.magnetics.physics.sweep
assert cadgen.magnetics.field is cadgen.magnetics.physics.field
assert cadgen.magnetics.write_report is cadgen.magnetics.report.write_report
heavy = {"build123d", "OCP", "cadquery"}
loaded = {m.split(".")[0] for m in sys.modules if sys.modules[m] is not None}
assert not (heavy & loaded), f"module-scope kernel import: {sorted(heavy & loaded)}"
try:
    cadgen.magnetics._deps.magpylib()
except cadgen.magnetics.MagneticsDependencyError as exc:
    assert 'pip install "cadgen[magnetics]"' in str(exc)
else:
    raise AssertionError("the gate let a blocked magpylib through")
print("IMPORT_LIGHT_OK")
"""
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=REPO, env=_env(), capture_output=True, text=True, check=False, timeout=300,
        )
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn("IMPORT_LIGHT_OK", result.stdout)

    def test_import_cadgen_cli_never_imports_the_magnetics_module(self) -> None:
        script = "import sys, cadgen.cli; assert 'cadgen.cli.magnetics' not in sys.modules; print('ok')"
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=REPO, env=_env(), capture_output=True, text=True, check=False, timeout=300,
        )
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)


class FixtureTests(unittest.TestCase):
    """The fixtures build, name their magnets as the lanes expect, and bind a mate."""

    def _labels(self, step: Path) -> list[str]:
        from cadgen import read_scene

        return sorted(str(leaf.label) for leaf in read_scene(step).leaves())

    def _sidecar(self, step: Path) -> dict:
        from cadgen._internal.source_sidecar import read_source_sidecar

        return read_source_sidecar(step) or {}

    def test_two_cube_model(self) -> None:
        step = support.build_model(self, "two_cube", support.TWO_CUBE_MODEL)
        self.assertTrue(step.is_file())
        self.assertEqual(["mag:N42:+z", "mag:N42:-z"], self._labels(step))

        (mate,) = self._sidecar(step)["kinematics"]["mates"]
        self.assertEqual("travel", mate["name"])
        self.assertEqual("slider", mate["kind"])
        self.assertEqual("#slider", mate["child"])
        self.assertTrue(mate["childId"], "the sidecar carries the resolved child occurrence id")
        self.assertEqual({"origin": [0.0, 0.0, 0.0], "dir": [0.0, 0.0, 1.0]}, mate["axis"])
        self.assertEqual({"value": [0.0, 20.0]}, mate["limits"])

        # The moving set is selected by childId, never by label: the id must
        # resolve to the group whose one child is the -z magnet.
        from cadgen import read_scene

        scene = read_scene(step)
        child = scene.resolve("#" + mate["childId"])
        self.assertEqual(["mag:N42:-z"], [c.label for c in child.children])

    def test_channel_models(self) -> None:
        for polarity in support.CHANNEL_POLARITIES:
            with self.subTest(polarity=polarity):
                step = support.build_model(self, f"channel_{polarity}", support.channel_model(polarity))
                labels = self._labels(step)
                magnets = [label for label in labels if label.startswith("mag:")]
                self.assertEqual(7, len(magnets), labels)
                self.assertIn("plate", labels)
                flipped = sum(1 for label in magnets if label.endswith("-z"))
                self.assertEqual({"aligned": 0, "alternating": 2, "one_flipped": 1}[polarity], flipped, magnets)
                (mate,) = self._sidecar(step)["kinematics"]["mates"]
                self.assertEqual(("travel", "slider", "#channel", "#slider"),
                                 (mate["name"], mate["kind"], mate["parent"], mate["child"]))
                self.assertTrue(mate["childId"])

    def test_single_magnet_models(self) -> None:
        for name, text in (("tilted_cube", support.TILTED_CUBE_MODEL), ("oblique_prism", support.OBLIQUE_PRISM_MODEL)):
            with self.subTest(model=name):
                step = support.build_model(self, name, text)
                self.assertEqual(["mag:N42:+z"], self._labels(step))
                self.assertNotIn("kinematics", self._sidecar(step))

    def test_hand_built_pair_matches_the_two_cube_fixture(self) -> None:
        fixed, moving = support.hand_built_coaxial_pair()
        self.assertEqual([0.0, 0.0, 0.0], list(fixed.position))
        self.assertAlmostEqual(0.030, float(moving.position[2]))
        self.assertEqual([0.00635] * 3, [round(float(v), 8) for v in fixed.dimension])
        self.assertAlmostEqual(1.30, float(fixed.polarization[2]))
        self.assertAlmostEqual(-1.30, float(moving.polarization[2]))


if __name__ == "__main__":
    unittest.main()
