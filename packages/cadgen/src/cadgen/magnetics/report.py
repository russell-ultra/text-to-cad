"""``sweep.json`` and the plotly ``report.html`` (lane A4).

``report.html`` (``include_plotlyjs=True`` by default so it opens offline at
~3.5 MB; ``plotly="cdn"`` for the ~50 kB form), top to bottom:

1. Verdict banner: tendency, ``Q +- u``, threshold and ``exceeds_threshold``
   (with the ``null`` case worded), ``admissible``, equilibria, convergence
   status and reasons; the sentence that rest under friction is a BAND around
   each stable equilibrium, not a point.
2. ``Q`` and ``U`` against ``q`` with equilibria marked, inaccessible AND wedged
   intervals hatched, the ``--at`` pose marked.
3. Field slice at ``--at``: ``|B|`` heatmap, in-plane arrows, magnet outlines.
4. Magnets table (ref, label, shape, polarization, position, moving).
5. Convergence table (axis, levels, delta).

``sweep.json`` is ``types.sweep_result_to_json`` written compactly
(``separators=(",", ":")``); it is a FILE, so indentation would be allowed, but
the stdout payload budget applies to anything a caller might print, and one
serialisation is simpler than two.

Every function below is a stub for lane A4: ``raise NotImplementedError("lane A4")``.
No plotly/numpy import at module scope -- use ``cadgen.magnetics._deps``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from cadgen.magnetics.types import FieldSlice, SweepResult  # noqa: F401 - part of the contract

__all__ = ["JSON_NAME", "REPORT_NAME", "FIELD_NAME", "PlotlyMode", "write_json", "write_report", "write_field"]

JSON_NAME = "sweep.json"
REPORT_NAME = "report.html"
FIELD_NAME = "field.html"

PlotlyMode = Literal["embed", "cdn"]


def write_json(result: SweepResult, out_dir: Path | str) -> Path:
    """Write ``<out_dir>/sweep.json`` (compact ``sweep_result_to_json``) and return its path.

    Creates ``out_dir``. The document round-trips through ``json.loads`` with
    ``schema == "magnetics-report/1"`` and a ``q_unit``.
    """
    raise NotImplementedError("lane A4")


def write_report(
    result: SweepResult,
    slice: FieldSlice | None,
    out_dir: Path | str,
    plotly: PlotlyMode = "embed",
) -> Path:
    """Write ``<out_dir>/report.html`` with the layout in the module docstring; return its path.

    ``slice`` ``None`` omits section 3. ``plotly="embed"`` passes
    ``include_plotlyjs=True`` (no ``https://cdn.plot.ly`` in the output, > 1 MB);
    ``"cdn"`` passes ``include_plotlyjs="cdn"`` (< 200 kB). Renders with 0 or
    many equilibria, ``exceeds_threshold=None``, ``tendency="indeterminate"``
    and wedged intervals.
    """
    raise NotImplementedError("lane A4")


def write_field(slice: FieldSlice, out_dir: Path | str, plotly: PlotlyMode = "embed") -> Path:
    """Write ``<out_dir>/field.html`` -- section 3 of the report on its own -- for
    ``cadgen magnetics field``; return its path. Same ``plotly`` semantics as
    :func:`write_report`.
    """
    raise NotImplementedError("lane A4")
