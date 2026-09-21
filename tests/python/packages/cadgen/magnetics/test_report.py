"""Lane A4: ``cadgen.magnetics.report`` -- ``sweep.json`` and the plotly HTML.

Every ``SweepResult``/``FieldSlice`` here is built by hand, so these tests
depend on the frozen ``types`` contract and plotly/numpy only -- never on the
scene, kinematics or physics lanes. Run:

    env -u PYTHONPATH PYTHONPATH="$PWD:$PWD/packages/cadgen/src" \
        ./.venv/bin/python -m unittest \
        tests/python/packages/cadgen/magnetics/test_report.py
"""

from __future__ import annotations

import json
import math
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cadgen.magnetics import report
from cadgen.magnetics.types import (
    Convergence,
    Equilibrium,
    FieldSlice,
    MagnetSpec,
    MateSpec,
    Sample,
    SweepResult,
    Verdict,
)

# A <script src="...cdn.plot.ly..."> tag -- the marker that the HTML loads
# plotly.js from the CDN rather than carrying it inline. NB the embedded bundle
# itself contains the literal "https://cdn.plot.ly/un/" (plotly.js's topojson
# default), so a bare substring test cannot tell the two modes apart; the script
# *src* can.
_CDN_SCRIPT = re.compile(r'<script[^>]*src="[^"]*cdn\.plot\.ly', re.IGNORECASE)


def _has_cdn_script(text: str) -> bool:
    return bool(_CDN_SCRIPT.search(text))


def _magnets() -> tuple[MagnetSpec, ...]:
    return (
        MagnetSpec(
            ref="#o1.1",
            label="mag:N42:+z",
            shape="cuboid",
            source=None,
            polarization_world_T=(0.0, 0.0, 0.92),
            position_m=(0.0, 0.0, 0.0),
            orientation_quat=(0.0, 0.0, 0.0, 1.0),
            moving=False,
            solid=None,
        ),
        MagnetSpec(
            ref="#o1.2",
            label="mag:N42:-z",
            shape="cylinder",
            source=None,
            polarization_world_T=(0.0, 0.0, -0.92),
            position_m=(0.0, 0.0, 0.03),
            orientation_quat=(0.0, 0.0, 0.0, 1.0),
            moving=True,
            solid=None,
        ),
    )


def _convergence(converged: bool | None = True) -> Convergence:
    if converged is None:
        return Convergence(mesh=None, eps=None, pose_grid=None, converged=None, reasons=())
    reasons = () if converged else ("mesh dQ 0.05 > 0.02 tol",)
    return Convergence(
        mesh={"levels": [20, 50, 100], "eps": 1e-5, "dQ": 0.005},
        eps={"levels": [1e-5, 1e-6], "meshing": 100, "dQ": 0.002},
        pose_grid={"samples": [201, 401], "dQ": 0.001, "equilibria_agree": True, "dU_max": 0.004},
        converged=converged,
        reasons=reasons,
    )


def make_result(
    *,
    n: int = 41,
    equilibria: tuple[Equilibrium, ...] = (),
    tendency: str = "+axis",
    exceeds: bool | None = True,
    threshold: float | None = 0.15,
    inaccessible: tuple[tuple[float, float], ...] = (),
    wedged: tuple[tuple[float, float], ...] = (),
    converged: bool | None = True,
    slider: bool = True,
) -> SweepResult:
    """A hand-built slider (default) or cylindrical ``SweepResult``."""
    if slider:
        mate = MateSpec("travel", "slider", "o1", (0, 0, 0), (1, 0, 0), {"value": (0.0, 0.08)}, "m")
        dof = None
        held = None
        lo, hi = 0.0, 0.08
    else:
        mate = MateSpec(
            "twist", "cylindrical", "o1", (0, 0, 0), (0, 0, 1),
            {"travel": (0.0, 0.02), "turn": (0.0, math.pi / 2)}, "m",
        )
        dof = "twist.turn"
        held = {"name": "twist.travel", "mode": "hold", "value": 0.0}
        lo, hi = 0.0, math.pi / 2

    qs = np.linspace(lo, hi, n)
    samples = []
    for q in qs:
        inside_gap = any(a <= q <= b for a, b in inaccessible)
        Q = math.sin(q * 40)
        U = None if inside_gap else math.cos(q * 40)
        held_value = None if slider else 0.0
        samples.append(
            Sample(
                q=float(q),
                F_N=(Q, 0.1, 0.0),
                Q=float(Q),
                F_transverse_N=(0.0, 0.1, 0.0),
                torque_Nm=(0.0, 0.0, 0.002),
                U_J=U,
                accessible=not inside_gap,
                held_value=held_value,
            )
        )

    verdict = Verdict(
        at_q=lo,
        tendency=tendency,  # type: ignore[arg-type]
        Q=0.42,
        uncertainty=0.006,
        threshold=threshold,
        exceeds_threshold=exceeds,
        admissible=True,
    )
    return SweepResult(
        magnets=_magnets(),
        mate=mate,
        dof=dof,
        held=held,
        samples=tuple(samples),
        inaccessible=inaccessible,
        wedged=wedged,
        equilibria=equilibria,
        verdict=verdict,
        convergence=_convergence(converged),
    )


def make_slice(n: int = 16) -> FieldSlice:
    u = np.linspace(-0.02, 0.02, n)
    v = np.linspace(-0.02, 0.02, n)
    uu, vv = np.meshgrid(u, v)
    B = np.zeros((n, n, 3))
    B[..., 0] = uu
    B[..., 1] = vv
    B[..., 2] = 0.1
    plane = {
        "spec": "z=0",
        "origin_m": (0.0, 0.0, 0.0),
        "normal": (0.0, 0.0, 1.0),
        "u_axis": (1.0, 0.0, 0.0),
        "v_axis": (0.0, 1.0, 0.0),
    }
    outline = np.array([[-0.005, -0.005], [0.005, -0.005], [0.005, 0.005], [-0.005, 0.005]])
    return FieldSlice(plane=plane, u=u, v=v, B=B, outlines=(outline,))


class WriteJsonTests(unittest.TestCase):
    def test_round_trips_with_schema_and_q_unit(self) -> None:
        result = make_result(equilibria=(Equilibrium(0.0613, True, -12.1, 0.0002),))
        with tempfile.TemporaryDirectory() as d:
            path = report.write_json(result, d)
            self.assertEqual(path.name, report.JSON_NAME)
            text = path.read_text(encoding="utf-8")
            doc = json.loads(text)
            self.assertEqual(doc["schema"], "magnetics-report/1")
            self.assertIn("q_unit", doc)
            self.assertEqual(doc["q_unit"], "m")

    def test_output_is_compact_no_indentation(self) -> None:
        result = make_result()
        with tempfile.TemporaryDirectory() as d:
            text = report.write_json(result, d).read_text(encoding="utf-8")
            # Compact: no pretty-printing whitespace anywhere in the file.
            self.assertNotIn("\n", text)
            self.assertNotIn(": ", text)
            self.assertNotIn(", ", text)

    def test_cylindrical_reports_the_swept_unit(self) -> None:
        result = make_result(slider=False)
        with tempfile.TemporaryDirectory() as d:
            doc = json.loads(report.write_json(result, d).read_text(encoding="utf-8"))
            self.assertEqual(doc["q_unit"], "rad")
            self.assertEqual(doc["mate"]["dof"], "twist.turn")

    def test_creates_missing_output_directory(self) -> None:
        result = make_result()
        with tempfile.TemporaryDirectory() as d:
            nested = Path(d) / "does" / "not" / "exist"
            path = report.write_json(result, nested)
            self.assertTrue(path.is_file())


class WriteReportTests(unittest.TestCase):
    def _write(self, result: SweepResult, slice_: FieldSlice | None, plotly: str) -> str:
        with tempfile.TemporaryDirectory() as d:
            path = report.write_report(result, slice_, d, plotly=plotly)  # type: ignore[arg-type]
            self.assertEqual(path.name, report.REPORT_NAME)
            return path.read_text(encoding="utf-8")

    def test_renders_with_zero_equilibria(self) -> None:
        text = self._write(make_result(equilibria=()), make_slice(), "cdn")
        self.assertIn("no equilibria in range", text)

    def test_renders_with_two_equilibria(self) -> None:
        eqs = (
            Equilibrium(0.03, True, -8.0, 0.0002),
            Equilibrium(0.06, False, 5.0, 0.0002),
        )
        text = self._write(make_result(equilibria=eqs), make_slice(), "cdn")
        self.assertIn("2 equilibria", text)
        self.assertIn("stable", text)
        self.assertIn("unstable", text)

    def test_renders_with_exceeds_threshold_none(self) -> None:
        text = self._write(make_result(exceeds=None, threshold=0.15), None, "cdn")
        # The null case is worded, not dropped.
        self.assertIn("indeterminate", text)
        self.assertIn("straddles", text)

    def test_renders_with_no_threshold(self) -> None:
        text = self._write(make_result(exceeds=None, threshold=None), None, "cdn")
        self.assertIn("no friction threshold", text)

    def test_renders_with_indeterminate_tendency(self) -> None:
        text = self._write(make_result(tendency="indeterminate"), None, "cdn")
        self.assertIn("indeterminate", text)

    def test_renders_with_wedged_interval(self) -> None:
        text = self._write(make_result(wedged=((0.0, 0.01),)), make_slice(), "cdn")
        self.assertIn("wedged", text)

    def test_renders_with_inaccessible_interval(self) -> None:
        text = self._write(make_result(inaccessible=((0.07, 0.08),)), make_slice(), "cdn")
        self.assertIn("inaccessible", text)

    def test_renders_with_convergence_off(self) -> None:
        text = self._write(make_result(converged=None), None, "cdn")
        self.assertIn("off", text)

    def test_renders_non_converged_with_reasons(self) -> None:
        text = self._write(make_result(converged=False), None, "cdn")
        self.assertIn("NOT converged", text)
        self.assertIn("mesh dQ", text)

    def test_slice_none_omits_the_field_section(self) -> None:
        with_slice = self._write(make_result(), make_slice(), "cdn")
        without = self._write(make_result(), None, "cdn")
        self.assertIn("Field slice", with_slice)
        self.assertNotIn("Field slice", without)

    def test_the_band_sentence_is_present(self) -> None:
        text = self._write(make_result(), None, "cdn")
        self.assertIn("band", text)

    def test_magnets_table_lists_every_magnet(self) -> None:
        text = self._write(make_result(), None, "cdn")
        self.assertIn("mag:N42:+z", text)
        self.assertIn("mag:N42:-z", text)
        self.assertIn("#o1.1", text)

    def test_convergence_table_present(self) -> None:
        text = self._write(make_result(), None, "cdn")
        self.assertIn("Convergence", text)
        self.assertIn("pose grid", text)

    def test_cylindrical_report_uses_torque_units(self) -> None:
        text = self._write(make_result(slider=False, threshold=0.02), make_slice(), "cdn")
        self.assertIn("N m", text)

    def test_embed_inlines_plotly_and_is_large(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = report.write_report(make_result(), make_slice(), d, plotly="embed")
            text = path.read_text(encoding="utf-8")
            self.assertFalse(_has_cdn_script(text), "embed must not load plotly from the CDN")
            self.assertGreater(path.stat().st_size, 1_000_000)

    def test_cdn_references_the_cdn_and_is_small(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = report.write_report(make_result(), make_slice(), d, plotly="cdn")
            text = path.read_text(encoding="utf-8")
            self.assertTrue(_has_cdn_script(text), "cdn must load plotly from the CDN")
            self.assertLess(path.stat().st_size, 200_000)

    def test_plotly_js_appears_exactly_once(self) -> None:
        # Two figures (curves + slice); the bundle/CDN tag must appear once.
        with tempfile.TemporaryDirectory() as d:
            text = report.write_report(make_result(), make_slice(), d, plotly="cdn").read_text(encoding="utf-8")
            self.assertEqual(len(_CDN_SCRIPT.findall(text)), 1)


class WriteFieldTests(unittest.TestCase):
    def test_embed_field_inlines_plotly(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = report.write_field(make_slice(), d, plotly="embed")
            self.assertEqual(path.name, report.FIELD_NAME)
            text = path.read_text(encoding="utf-8")
            self.assertFalse(_has_cdn_script(text))
            self.assertGreater(path.stat().st_size, 1_000_000)

    def test_cdn_field_references_cdn(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = report.write_field(make_slice(), d, plotly="cdn")
            text = path.read_text(encoding="utf-8")
            self.assertTrue(_has_cdn_script(text))
            self.assertLess(path.stat().st_size, 200_000)

    def test_field_creates_missing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            nested = Path(d) / "a" / "b"
            path = report.write_field(make_slice(), nested, plotly="cdn")
            self.assertTrue(path.is_file())

    def test_field_with_no_outlines(self) -> None:
        sl = make_slice()
        bare = FieldSlice(plane=sl.plane, u=sl.u, v=sl.v, B=sl.B, outlines=())
        with tempfile.TemporaryDirectory() as d:
            path = report.write_field(bare, d, plotly="cdn")
            self.assertTrue(path.is_file())

    def test_field_with_zero_field_has_no_arrows_but_renders(self) -> None:
        # An all-zero B must not divide by zero when scaling the quiver.
        sl = make_slice()
        zero = FieldSlice(plane=sl.plane, u=sl.u, v=sl.v, B=np.zeros_like(sl.B), outlines=sl.outlines)
        with tempfile.TemporaryDirectory() as d:
            path = report.write_field(zero, d, plotly="cdn")
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
