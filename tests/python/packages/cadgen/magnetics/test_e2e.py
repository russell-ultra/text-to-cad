"""End-to-end integration for the magnetics feature (Phase 2, lane B1).

Unlike the lane suites -- which mock ``scene``/``kinematics``/``physics`` or hand
build magpylib objects -- this file drives the REAL CLI as a subprocess over STEP
fixtures it builds from :mod:`support`, so every seam the eight lanes coded to in
isolation is exercised together: STEP -> magpylib scene, mate selection, the
``getFT`` reduction, projection, clearance, energy, equilibria, convergence, the
verdict, the report render, and the CLI's stream/exit contract.

The design's success criteria, asserted here:

1. two-cube ``inspect`` names both magnets and their polarities;
2. ``sweep`` force matches the hand-built coaxial pair (regression, 0.1 %) and
   the coaxial dipole far field (2 %) THROUGH the CLI;
3. the three channel arrangements produce distinct tendencies or equilibria;
4. ``--convergence full`` reports ``converged`` true on the two-cube pair;
5. ``--dof travel.travel --relax`` on a cylindrical mate fills the held-value
   column, and the report/field HTML render.

Every fixture is this test's own (nothing under ``models/`` is read); the build
uses a private ``CADGEN_CACHE_DIR`` torn down with the class.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.python.support.paths import REPO_ROOT, add_repo_path

add_repo_path("packages/cadgen/src")

from tests.python.packages.cadgen.magnetics import support

# A test-owned cylindrical (travel + turn) fixture: the small aligned channel on
# a cylindrical mate. ``turn`` is capped at +-30 degrees -- the same yaw window
# as the design's channel_yaw model -- so the relaxed roll never swings the
# slider cube into a tilted row and clearance holds across the travel sweep. It
# is the only shape ``support`` does not already provide; the design's real yaw
# model lives under models/, which the suite may not read.
CYLINDRICAL_MODEL = f"""\
import cadgen
from cadgen import compound_from_instances, label_shape, step
from cadgen import build123d as bd

MAGNET_MM = {support.MAGNET_MM}
PITCH_MM = 12.0
GAP_MM = 10.0
TILT_DEG = 45.0
N_PER_ROW = 3
CHANNEL_LEN_MM = PITCH_MM * (N_PER_ROW - 1)

KINEMATICS = {{
    "mates": [
        cadgen.cylindrical("travel", parent="#channel", child="#slider",
                           origin=(0, 0, 0), direction=(1, 0, 0),
                           limits={{"travel": (0, CHANNEL_LEN_MM), "turn": (-30, 30)}}),
    ],
}}


def _row(name, y):
    cube = bd.Box(MAGNET_MM, MAGNET_MM, MAGNET_MM)
    return compound_from_instances(name, [
        (cube, bd.Pos(i * PITCH_MM, y, 0) * bd.Rot(0, TILT_DEG, 0), "mag:N42:+z")
        for i in range(N_PER_ROW)
    ])


@step(kinematics=KINEMATICS)
def channel_cyl():
    y = GAP_MM / 2 + MAGNET_MM / 2
    plate = label_shape(
        bd.Pos(CHANNEL_LEN_MM / 2, 0, -(MAGNET_MM / 2 + 2.0)) * bd.Box(CHANNEL_LEN_MM + 2 * MAGNET_MM, 2 * y + MAGNET_MM, 2.0),
        "plate",
    )
    rows = [plate, _row("row_a", y), _row("row_b", -y)]
    body = bd.Compound(obj=list(rows), children=list(rows), label="channel")
    slider = compound_from_instances("slider", [(bd.Box(MAGNET_MM, MAGNET_MM, MAGNET_MM), bd.Location(), "mag:N42:+z")])
    return bd.Compound(children=[body, slider])


if __name__ == "__main__":
    channel_cyl()
"""


def _cli(*argv: str, cwd: Path, cache: Path) -> subprocess.CompletedProcess:
    """Run ``cadgen magnetics <argv>`` in a subprocess with the checkout on the path.

    Mirrors ``test_store_two_sides._cli``: the worktree's ``packages/cadgen/src``
    leads ``PYTHONPATH`` so the CLI resolves to this checkout, and ``cache`` is a
    private store so a bare ``python -m unittest`` never reads the developer's.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [str(REPO_ROOT / "packages" / "cadgen" / "src"), env.get("PYTHONPATH", "")] if p
    )
    env["CADGEN_CACHE_DIR"] = str(cache)
    return subprocess.run(
        [sys.executable, "-m", "cadgen.cli", "magnetics", *argv],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=600,
    )


class MagneticsEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="magnetics-e2e-")
        cls.work = Path(cls._tmp.name)
        cls.cache = cls.work / "cache"
        cls.cache.mkdir()

        cls._previous_cache = os.environ.get("CADGEN_CACHE_DIR")
        os.environ["CADGEN_CACHE_DIR"] = str(cls.cache)

        from cadgen.catalog import StepImportOptions
        from cadgen.generation import generate_step_targets

        def build(name: str, text: str) -> Path:
            script = cls.work / f"{name}.py"
            script.write_text(text, encoding="utf-8")
            code = generate_step_targets(
                [str(script)], step_options=StepImportOptions(), force=True, verbose=False
            )
            if code != 0:
                raise AssertionError(f"building {name} exited {code}")
            step = script.with_suffix(".step")
            if not step.is_file():
                raise AssertionError(f"building {name} wrote no {step.name}")
            return step

        cls.two_cube = build("two_cube", support.TWO_CUBE_MODEL)
        cls.channels = {
            polarity: build(f"channel_{polarity}", support.channel_model(polarity))
            for polarity in support.CHANNEL_POLARITIES
        }
        cls.cylindrical = build("channel_cyl", CYLINDRICAL_MODEL)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._previous_cache is None:
            os.environ.pop("CADGEN_CACHE_DIR", None)
        else:
            os.environ["CADGEN_CACHE_DIR"] = cls._previous_cache
        cls._tmp.cleanup()

    # -- helpers -----------------------------------------------------------

    def _run(self, *argv: str) -> subprocess.CompletedProcess:
        return _cli(*argv, cwd=self.work, cache=self.cache)

    def _json(self, proc: subprocess.CompletedProcess) -> dict:
        self.assertEqual(proc.returncode, 0, f"exit {proc.returncode}\n{proc.stderr}")
        # The stream contract: the result is exactly one compact line on stdout.
        self.assertEqual(proc.stdout.count("\n"), 1, f"stdout is not one line:\n{proc.stdout!r}")
        return json.loads(proc.stdout)

    def _sweep_json(self, payload: dict) -> dict:
        """The full ``sweep.json`` sidecar the ``--json`` summary points at."""
        return json.loads((self.work / payload["out"]["json"]).read_text(encoding="utf-8"))

    # -- criterion 1: inspect ---------------------------------------------

    def test_inspect_two_cube_names_both_magnets(self) -> None:
        payload = self._json(self._run("inspect", str(self.two_cube), "--json"))
        self.assertIsNone(payload["mate"])
        labels = [m["label"] for m in payload["magnets"]]
        self.assertEqual(sorted(labels), ["mag:N42:+z", "mag:N42:-z"])
        for magnet in payload["magnets"]:
            self.assertEqual(magnet["shape"], "cuboid")
            # inspect selects no mate, so nothing is classified moving.
            self.assertIsNone(magnet["moving"])

    # -- criterion 2: regression + dipole THROUGH the CLI -----------------

    def test_sweep_force_matches_the_hand_built_pair_and_the_dipole(self) -> None:
        payload = self._json(
            self._run("sweep", str(self.two_cube), "--samples", "41", "--convergence", "off", "--json")
        )
        samples = self._sweep_json(payload)["samples"]
        rest = min(samples, key=lambda s: abs(s["q"]))

        # Regression: the CLI's rest-pose Q against the hand-built magpylib pair.
        import numpy as np

        from cadgen.magnetics import physics

        fixed, moving = support.hand_built_coaxial_pair()
        pivot = np.array([[[0.0, 0.0, support.COAXIAL_SPACING_MM * 1e-3]]])
        force, _torque = physics.force_torque([fixed], [moving], pivot, meshing=100, eps=1e-5)
        reference = float(force[0, 2])
        self.assertLess(abs(rest["Q"] - reference) / abs(reference), 1e-3)

        # Dipole far field: 3 mu0 m^2 / (2 pi r^4) within 2 %.
        mu0 = 4e-7 * math.pi
        volume = (support.MAGNET_MM * 1e-3) ** 3
        moment = support.N42_REMANENCE_T * volume / mu0
        separation = support.COAXIAL_SPACING_MM * 1e-3
        dipole = 3 * mu0 * moment**2 / (2 * math.pi * separation**4)
        self.assertLess(abs(abs(rest["Q"]) - dipole) / dipole, 0.02)

        # A repelling pair separating along +z: |Q| falls monotonically.
        ordered = [s["Q"] for s in sorted(samples, key=lambda s: s["q"])]
        magnitudes = [abs(q) for q in ordered]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))

    # -- criterion 3: the three arrangements differ ----------------------

    def test_channel_arrangements_are_distinguishable(self) -> None:
        signatures = set()
        for polarity, step in self.channels.items():
            payload = self._json(
                self._run("sweep", str(step), "--samples", "31", "--convergence", "quick", "--json")
            )
            with self.subTest(polarity=polarity):
                self.assertEqual(len(payload["magnets"]), 7)  # 2x3 rows + slider
            signatures.add((payload["verdict"]["tendency"], len(payload["equilibria"])))
        # (tendency, equilibrium count) tells all three apart.
        self.assertEqual(len(signatures), len(self.channels))

    # -- criterion 4: full convergence ------------------------------------

    def test_full_convergence_converges_on_the_two_cube_pair(self) -> None:
        payload = self._json(
            self._run("sweep", str(self.two_cube), "--samples", "41", "--convergence", "full", "--json")
        )
        convergence = payload["convergence"]
        self.assertIs(convergence["converged"], True)
        self.assertEqual(convergence["reasons"], [])
        for stage in ("mesh", "eps", "pose_grid"):
            self.assertIsNotNone(convergence[stage])

    # -- criterion 5: --relax fills held_value, and the HTML renders ------

    def test_relax_fills_the_held_value_column(self) -> None:
        payload = self._json(
            self._run(
                "sweep", str(self.cylindrical),
                "--dof", "travel.travel", "--relax",
                "--samples", "11", "--convergence", "off", "--json",
            )
        )
        report = self._sweep_json(payload)
        self.assertEqual(report["mate"]["dof"], "travel.travel")
        self.assertEqual(report["mate"]["held"]["mode"], "relax")
        held = [s["held_value"] for s in report["samples"]]
        # Every accessible (non-wedged) pose gets a relaxed roll; a symmetric
        # aligned channel relaxes near zero, but the point is the column is filled.
        filled = [v for v in held if v is not None]
        self.assertTrue(filled, "no pose produced a relaxed held value")
        for value in filled:
            self.assertIsInstance(value, float)
            self.assertLessEqual(abs(value), math.radians(30) + 1e-6)

    def test_sweep_writes_an_embedded_report_and_json(self) -> None:
        payload = self._json(
            self._run("sweep", str(self.two_cube), "--samples", "9", "--convergence", "off", "--json")
        )
        report_path = self.work / payload["out"]["report"]
        html = report_path.read_text(encoding="utf-8")
        # embed is the default: plotly.js is inlined, so no CDN <script src> tag.
        self.assertNotIn('src="https://cdn.plot.ly', html)
        self.assertGreater(report_path.stat().st_size, 1_000_000)

        sweep = self._sweep_json(payload)
        self.assertEqual(sweep["schema"], "magnetics-report/1")
        self.assertEqual(sweep["q_unit"], "m")

    def test_cdn_report_is_small_and_links_the_cdn(self) -> None:
        payload = self._json(
            self._run(
                "sweep", str(self.two_cube), "--samples", "9", "--convergence", "off",
                "--plotly", "cdn", "--out", "cdn-report", "--json",
            )
        )
        report_path = self.work / payload["out"]["report"]
        html = report_path.read_text(encoding="utf-8")
        self.assertIn('src="https://cdn.plot.ly', html)
        # Not inlining plotly.js is the whole point: an order of magnitude smaller.
        self.assertLess(report_path.stat().st_size, 1_000_000)

    def test_field_slice_renders(self) -> None:
        payload = self._json(self._run("field", str(self.two_cube), "--slice", "x=0", "--json"))
        self.assertTrue(payload["ok"])
        self.assertGreater(payload["B_max_T"], 0.0)
        field_path = self.work / payload["out"]["field"]
        self.assertTrue(field_path.is_file())
        self.assertGreater(field_path.stat().st_size, 1_000)

    # -- the stream contract on a non-JSON run ----------------------------

    def test_narration_goes_to_stderr(self) -> None:
        proc = self._run("sweep", str(self.two_cube), "--samples", "5", "--convergence", "off", "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        json.loads(proc.stdout)  # stdout is pure result
        self.assertIn("[cadgen]", proc.stderr)  # narration is on stderr only


if __name__ == "__main__":
    unittest.main()
