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

Every function draws its dependencies through :mod:`cadgen.magnetics._deps`; no
plotly/numpy import lives at module scope, so ``import cadgen.magnetics.report``
succeeds with the ``magnetics`` extra absent.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Literal

from cadgen.magnetics import _deps
from cadgen.magnetics.types import FieldSlice, SweepResult, sweep_result_to_json

__all__ = ["JSON_NAME", "REPORT_NAME", "FIELD_NAME", "PlotlyMode", "write_json", "write_report", "write_field"]

JSON_NAME = "sweep.json"
REPORT_NAME = "report.html"
FIELD_NAME = "field.html"

PlotlyMode = Literal["embed", "cdn"]

#: Colour for hatched inaccessible spans; wedged spans get their own colour so a
#: reader can tell a clearance gap from a DOF that never reaches equilibrium.
_INACCESSIBLE_COLOUR = "#d62728"
_WEDGED_COLOUR = "#9467bd"
_AT_COLOUR = "#2ca02c"
_Q_COLOUR = "#1f77b4"
_U_COLOUR = "#ff7f0e"


# --------------------------------------------------------------------------- json


def write_json(result: SweepResult, out_dir: Path | str) -> Path:
    """Write ``<out_dir>/sweep.json`` (compact ``sweep_result_to_json``) and return its path.

    Creates ``out_dir``. The document round-trips through ``json.loads`` with
    ``schema == "magnetics-report/1"`` and a ``q_unit``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / JSON_NAME
    payload = json.dumps(sweep_result_to_json(result), separators=(",", ":"))
    path.write_text(payload, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- report


def write_report(
    result: SweepResult,
    slice: FieldSlice | None,
    out_dir: Path | str,
    plotly: PlotlyMode = "embed",
) -> Path:
    """Write ``<out_dir>/report.html`` with the layout in the module docstring; return its path.

    ``slice`` ``None`` omits section 3. ``plotly="embed"`` passes
    ``include_plotlyjs=True`` (the plotly bundle is inlined, > 1 MB, no CDN
    ``<script src>``); ``"cdn"`` passes ``include_plotlyjs="cdn"`` (< 200 kB, a
    CDN ``<script src>``). Renders with 0 or many equilibria,
    ``exceeds_threshold=None``, ``tendency="indeterminate"`` and wedged
    intervals.
    """
    data = sweep_result_to_json(result)
    q_unit = data["q_unit"]

    sections: list[tuple[str, Any]] = []
    sections.append(("html", _verdict_banner_html(data)))
    sections.append(("fig", _curves_figure(result, data, q_unit)))
    if slice is not None:
        sections.append(("html", _heading_html("Field slice")))
        sections.append(("fig", _slice_figure(slice)))
    sections.append(("html", _magnets_table_html(data["magnets"])))
    sections.append(("html", _convergence_table_html(data["convergence"])))

    document = _render_document("magnetics report", sections, plotly)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / REPORT_NAME
    path.write_text(document, encoding="utf-8")
    return path


def write_field(slice: FieldSlice, out_dir: Path | str, plotly: PlotlyMode = "embed") -> Path:
    """Write ``<out_dir>/field.html`` -- section 3 of the report on its own -- for
    ``cadgen magnetics field``; return its path. Same ``plotly`` semantics as
    :func:`write_report`.
    """
    sections: list[tuple[str, Any]] = [
        ("html", _heading_html("Field slice")),
        ("fig", _slice_figure(slice)),
    ]
    document = _render_document("magnetics field", sections, plotly)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / FIELD_NAME
    path.write_text(document, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- figures


def _curves_figure(result: SweepResult, data: dict[str, Any], q_unit: str) -> Any:
    """``Q(q)`` over ``U(q)``: equilibria markers, hatched inaccessible/wedged
    spans, the ``--at`` verdict pose."""
    _deps.plotly()  # gate the extra before importing submodules
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    qs = [s.q for s in result.samples]
    Qs = [s.Q for s in result.samples]
    # U_J is None at an inaccessible pose; None leaves a gap in the trace.
    Us = [s.U_J for s in result.samples]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("generalized force Q(q)", "potential energy U(q)"),
    )
    fig.add_trace(
        go.Scatter(x=qs, y=Qs, mode="lines", name="Q", line={"color": _Q_COLOUR}),
        row=1,
        col=1,
    )
    fig.add_hline(y=0.0, line={"color": "#888", "width": 1}, row=1, col=1)
    fig.add_trace(
        go.Scatter(x=qs, y=Us, mode="lines", name="U", line={"color": _U_COLOUR}, connectgaps=False),
        row=2,
        col=1,
    )

    # Equilibria: stable filled, unstable hollow, on the Q curve (Q == 0 there).
    stable = [e for e in result.equilibria if e.stable]
    unstable = [e for e in result.equilibria if not e.stable]
    if stable:
        fig.add_trace(
            go.Scatter(
                x=[e.q for e in stable],
                y=[0.0] * len(stable),
                mode="markers",
                name="stable equilibrium",
                marker={"symbol": "circle", "size": 11, "color": _Q_COLOUR},
            ),
            row=1,
            col=1,
        )
    if unstable:
        fig.add_trace(
            go.Scatter(
                x=[e.q for e in unstable],
                y=[0.0] * len(unstable),
                mode="markers",
                name="unstable equilibrium",
                marker={
                    "symbol": "circle-open",
                    "size": 11,
                    "color": _Q_COLOUR,
                    "line": {"width": 2, "color": _Q_COLOUR},
                },
            ),
            row=1,
            col=1,
        )

    # Hatched spans: inaccessible (clearance) and wedged (no held-DOF zero).
    for lo, hi in data["inaccessible"]:
        _add_span(fig, go, lo, hi, _INACCESSIBLE_COLOUR, "inaccessible")
    for lo, hi in data["wedged"]:
        _add_span(fig, go, lo, hi, _WEDGED_COLOUR, "wedged")

    # The verdict pose.
    at_q = data["verdict"]["at_q"]
    fig.add_vline(
        x=at_q,
        line={"color": _AT_COLOUR, "dash": "dash", "width": 2},
        annotation_text="--at",
        annotation_position="top",
    )

    fig.update_xaxes(title_text=f"q [{q_unit}]", row=2, col=1)
    fig.update_yaxes(title_text=f"Q [{_generalized_unit(data)}]", row=1, col=1)
    fig.update_yaxes(title_text="U [J]", row=2, col=1)
    fig.update_layout(height=640, margin={"t": 60, "b": 50, "l": 70, "r": 30}, showlegend=True)
    return fig


def _add_span(fig: Any, go: Any, lo: float, hi: float, colour: str, label: str) -> None:
    """A labelled translucent band across both rows for the excluded ``[lo, hi]``.

    plotly layout shapes carry no pattern fill, so an excluded interval reads as
    a translucent colour block with its kind in the corner rather than a literal
    cross-hatch.
    """
    fig.add_vrect(
        x0=lo,
        x1=hi,
        fillcolor=colour,
        opacity=0.15,
        line_width=0,
        layer="below",
        annotation_text=label,
        annotation_position="top left",
    )


def _slice_figure(slice: FieldSlice) -> Any:
    """``|B|`` heatmap, in-plane quiver, magnet outlines on the slice plane."""
    _deps.plotly()
    np = _deps.numpy()
    import plotly.figure_factory as ff
    import plotly.graph_objects as go

    u = np.asarray(slice.u, dtype=float)
    v = np.asarray(slice.v, dtype=float)
    B = np.asarray(slice.B, dtype=float)  # (N, N, 3) world components

    plane = slice.plane
    u_axis = np.asarray(plane["u_axis"], dtype=float)
    v_axis = np.asarray(plane["v_axis"], dtype=float)

    Bmag = np.linalg.norm(B, axis=-1)  # (N, N)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=u,
            y=v,
            z=Bmag,
            colorscale="Viridis",
            colorbar={"title": "|B| [T]"},
            name="|B|",
        )
    )

    # In-plane components: project the world field onto the plane's u/v axes.
    Bu = B @ u_axis  # (N, N)
    Bv = B @ v_axis
    uu, vv = np.meshgrid(u, v)  # row index -> v, col index -> u
    # z[j, i] is at (x=u[i], y=v[j]); the projections share that indexing.
    stride = max(1, u.shape[0] // 16)
    sel = np.s_[::stride, ::stride]
    x_q = uu[sel].ravel()
    y_q = vv[sel].ravel()
    u_q = Bu[sel].ravel()
    v_q = Bv[sel].ravel()
    span = float(max(u.max() - u.min(), v.max() - v.min())) or 1.0
    mag = float(np.max(np.hypot(u_q, v_q)))
    if mag > 0:
        scale = 0.04 * span / mag
        quiver = ff.create_quiver(x_q, y_q, u_q, v_q, scale=scale, line={"color": "#ffffff", "width": 1})
        for trace in quiver.data:
            trace.showlegend = False
            fig.add_trace(trace)

    # Magnet outlines: closed polygons in (u, v) metres.
    for k, outline in enumerate(slice.outlines):
        poly = np.asarray(outline, dtype=float)
        if poly.size == 0:
            continue
        closed = np.vstack([poly, poly[:1]])
        fig.add_trace(
            go.Scatter(
                x=closed[:, 0],
                y=closed[:, 1],
                mode="lines",
                line={"color": "#ffffff", "width": 2},
                name="magnet" if k == 0 else None,
                showlegend=(k == 0),
                hoverinfo="skip",
            )
        )

    fig.update_xaxes(title_text=f"u [m]  ({plane.get('spec', '')})", constrain="domain")
    fig.update_yaxes(title_text="v [m]", scaleanchor="x", scaleratio=1)
    fig.update_layout(height=560, margin={"t": 40, "b": 50, "l": 70, "r": 30})
    return fig


# --------------------------------------------------------------------------- html blocks


def _render_document(title: str, sections: list[tuple[str, Any]], plotly: PlotlyMode) -> str:
    """Assemble the full HTML page.

    The FIRST figure carries the plotly.js payload (``include_plotlyjs=True`` for
    embed, ``"cdn"`` otherwise); every later figure gets ``False`` so the bundle
    or CDN tag appears exactly once. There is always at least one figure.
    """
    import plotly.io as pio

    include_first: Any = True if plotly == "embed" else "cdn"
    body_parts: list[str] = []
    seen_fig = False
    for kind, payload in sections:
        if kind == "html":
            body_parts.append(payload)
        else:  # kind == "fig"
            include = include_first if not seen_fig else False
            seen_fig = True
            body_parts.append(
                pio.to_html(payload, include_plotlyjs=include, full_html=False, default_width="100%")
            )
    body = "\n".join(body_parts)
    return _PAGE_TEMPLATE.format(title=html.escape(title), body=body)


def _heading_html(text: str) -> str:
    return f'<h2 class="section">{html.escape(text)}</h2>'


def _verdict_banner_html(data: dict[str, Any]) -> str:
    v = data["verdict"]
    q_unit = data["q_unit"]
    gen_unit = _generalized_unit(data)
    tendency = v["tendency"]

    if tendency == "indeterminate":
        tendency_txt = "indeterminate (|Q| below the pose-grid uncertainty)"
        tendency_class = "warn"
    else:
        tendency_txt = f"tends toward {tendency}"
        tendency_class = "ok"

    threshold = v["threshold"]
    exceeds = v["exceeds_threshold"]
    if threshold is None:
        thresh_txt = "no friction threshold supplied"
    elif exceeds is None:
        thresh_txt = (
            f"threshold {threshold:g} {gen_unit}: <em>indeterminate</em> "
            f"(the band |Q| ± u straddles it)"
        )
    elif exceeds:
        thresh_txt = f"|Q| exceeds the {threshold:g} {gen_unit} threshold (motion expected)"
    else:
        thresh_txt = f"|Q| stays under the {threshold:g} {gen_unit} threshold (friction can hold)"

    admissible = v["admissible"]
    adm_txt = (
        "admissible: the tendency stays within the mate limits and accessible poses"
        if admissible
        else "NOT admissible: the tendency points outside the limits or into an inaccessible interval"
    )
    adm_class = "ok" if admissible else "warn"

    eqs = data["equilibria"]
    if eqs:
        parts = []
        for e in eqs:
            kind = "stable" if e["stable"] else "unstable"
            parts.append(f"{e['q']:g} {q_unit} ({kind})")
        eq_txt = f"{len(eqs)} equilibri{'um' if len(eqs) == 1 else 'a'}: " + ", ".join(parts)
    else:
        eq_txt = "no equilibria in range"

    conv = data["convergence"]
    converged = conv["converged"]
    if converged is None:
        conv_txt = "convergence protocol off (finest level, one pass)"
        conv_class = "warn"
    elif converged:
        conv_txt = "converged"
        conv_class = "ok"
    else:
        reasons = conv["reasons"]
        detail = ("; ".join(reasons)) if reasons else "a refinement axis moved the answer beyond the tolerance"
        conv_txt = f"NOT converged: {detail}"
        conv_class = "warn"

    rows = [
        f'<div class="v-line {tendency_class}"><strong>Verdict at q = {v["at_q"]:g} {html.escape(q_unit)}:</strong> '
        f'{html.escape(tendency_txt)}, Q = {v["Q"]:g} ± {v["uncertainty"]:g} {html.escape(gen_unit)}</div>',
        f'<div class="v-line">{thresh_txt}</div>',
        f'<div class="v-line {adm_class}">{html.escape(adm_txt)}</div>',
        f'<div class="v-line">{html.escape(eq_txt)}</div>',
        f'<div class="v-line {conv_class}">{html.escape(conv_txt)}</div>',
        '<div class="v-note">Rest position under friction is a <em>band</em> around each stable '
        "equilibrium, not a point: any pose whose |Q| stays below the friction threshold can hold.</div>",
    ]
    return '<div class="banner">' + "\n".join(rows) + "</div>"


def _magnets_table_html(magnets: list[dict[str, Any]]) -> str:
    head = "<tr><th>ref</th><th>label</th><th>shape</th><th>polarization_world_T</th><th>position_m</th><th>moving</th></tr>"
    body_rows = []
    for m in magnets:
        moving = m["moving"]
        moving_txt = "—" if moving is None else ("yes" if moving else "no")
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(str(m['ref']))}</td>"
            f"<td>{html.escape(str(m['label']))}</td>"
            f"<td>{html.escape(str(m['shape']))}</td>"
            f"<td>{_fmt_vec(m['polarization_world_T'])}</td>"
            f"<td>{_fmt_vec(m['position_m'])}</td>"
            f"<td>{moving_txt}</td>"
            "</tr>"
        )
    return (
        _heading_html("Magnets")
        + '<table class="grid"><thead>'
        + head
        + "</thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )


def _convergence_table_html(conv: dict[str, Any]) -> str:
    head = "<tr><th>axis</th><th>levels</th><th>&Delta;</th><th>note</th></tr>"
    rows = []

    mesh = conv.get("mesh")
    if mesh is not None:
        rows.append(_conv_row("mesh (elements)", mesh.get("levels"), mesh.get("dQ"), f"eps = {mesh.get('eps')!r}"))
    eps = conv.get("eps")
    if eps is not None:
        rows.append(_conv_row("eps (m)", eps.get("levels"), eps.get("dQ"), f"meshing = {eps.get('meshing')!r}"))
    grid = conv.get("pose_grid")
    if grid is not None:
        note = f"equilibria agree: {grid.get('equilibria_agree')}; dU_max = {_fmt_num(grid.get('dU_max'))}"
        rows.append(_conv_row("pose grid (samples)", grid.get("samples"), grid.get("dQ"), note))

    if not rows:
        rows.append('<tr><td colspan="4">convergence protocol off — finest level, one pass</td></tr>')

    converged = conv.get("converged")
    status = "off" if converged is None else ("converged" if converged else "NOT converged")
    reasons = conv.get("reasons") or []
    reason_txt = (" — " + html.escape("; ".join(reasons))) if reasons else ""
    caption = f'<div class="v-line">status: {status}{reason_txt}</div>'

    return (
        _heading_html("Convergence")
        + '<table class="grid"><thead>'
        + head
        + "</thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + caption
    )


def _conv_row(axis: str, levels: Any, dq: Any, note: str) -> str:
    return (
        "<tr>"
        f"<td>{html.escape(axis)}</td>"
        f"<td>{html.escape(_fmt_levels(levels))}</td>"
        f"<td>{_fmt_num(dq)}</td>"
        f"<td>{html.escape(note)}</td>"
        "</tr>"
    )


# --------------------------------------------------------------------------- formatting


def _generalized_unit(data: dict[str, Any]) -> str:
    """The unit of ``Q`` and the friction threshold: N for a translation DOF,
    N m for a rotation. Keyed off the reported ``q_unit`` (``DOF_UNITS``)."""
    return "N m" if data["q_unit"] == "rad" else "N"


def _fmt_vec(vec: Any) -> str:
    return "(" + ", ".join(_fmt_num(x) for x in vec) + ")"


def _fmt_num(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{x:g}"
    return html.escape(str(x))


def _fmt_levels(levels: Any) -> str:
    if levels is None:
        return "—"
    if isinstance(levels, (list, tuple)):
        return ", ".join(_fmt_num(x) for x in levels)
    return str(levels)


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0 auto; max-width: 1000px; padding: 24px; line-height: 1.4; }}
h1 {{ font-size: 1.5rem; margin: 0 0 16px; }}
h2.section {{ font-size: 1.15rem; margin: 28px 0 8px; border-bottom: 1px solid #8884; padding-bottom: 4px; }}
.banner {{ border: 1px solid #8884; border-radius: 8px; padding: 14px 16px; background: #8881; }}
.v-line {{ margin: 4px 0; }}
.v-note {{ margin: 8px 0 0; font-size: 0.9rem; opacity: 0.85; }}
.v-line.ok strong, .v-line.ok {{ }}
.v-line.warn {{ color: #b8860b; }}
table.grid {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 4px 0; }}
table.grid th, table.grid td {{ border: 1px solid #8884; padding: 4px 8px; text-align: left; }}
table.grid th {{ background: #8882; }}
</style>
</head>
<body>
<h1>{title}</h1>
{body}
</body>
</html>
"""
