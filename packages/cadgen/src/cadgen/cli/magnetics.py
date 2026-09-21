"""``cadgen magnetics`` -- permanent-magnet mechanism analysis on a cadgen model.

    cadgen magnetics inspect <model.step> [--mate NAME] [--json]
    cadgen magnetics sweep   <model.step> [--mate NAME] [--dof MATE.travel|MATE.turn]
                             [--hold VALUE | --relax] [--samples N] [--friction-N F | --friction-Nm T]
                             [--at Q] [--convergence off|quick|full] [--converge-tol TOL]
                             [--out DIR] [--plotly embed|cdn] [--json]
    cadgen magnetics field   <model.step> [--mate NAME] [--slice x=MM|y=MM|z=MM|mate[:UP]]
                             [--grid N] [--at Q] [--out DIR] [--plotly embed|cdn] [--json]

A leaf occurrence named ``mag:<grade>:<direction>`` is a magnet; a ``slider``,
``revolute`` or ``cylindrical`` mate declared through ``@step(kinematics=...)``
is the degree of freedom. ``inspect`` lists what the model declares; ``sweep``
computes force, torque and potential energy along the mate and writes
``sweep.json`` + ``report.html``; ``field`` samples ``B`` on one plane.

Exit codes: 0 = analysis ran (``converged: false`` warns on stderr and still
exits 0); 1 = analysis error (extra not installed, no magnets, a bad
declaration, a stale sidecar, an unusable mate); 2 = usage error (argparse rows
of the validation matrix: ``--samples < 3``, ``--grid < 8``, bad ``--slice`` or
``--dof`` syntax, ``--hold`` with ``--relax``, both friction flags).

Stream contract: the RESULT is on stdout (``--json`` is ONE compact line);
narration and errors are on stderr. The stdout payload is O(1): the sample
table lives in ``sweep.json``, never on stdout.

Import light: this module and ``cadgen.magnetics.types`` are stdlib-only, so
``--help`` answers with the ``magnetics`` extra absent and without the CAD
kernel. Physics and the kernel are imported inside the verb bodies.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from cadgen.magnetics.types import (
    MagneticsError,
    MateError,
    SceneError,
    SweepConfig,
    sweep_result_to_json,
)

DEFAULT_PROG = "cadgen magnetics"

#: mm per metre / degrees per radian: model units enter cadgen.magnetics only
#: through scene.load, so the CLI converts the model-unit flags (--at, --hold)
#: to SI here, at the one other model-unit boundary.
_MM_PER_M = 1e3
MM_TO_M = 1e-3


class _UsageError(Exception):
    """A validation-matrix row that needs the mate to detect, but is a usage
    error (exit 2), not an analysis error (exit 1).

    argparse handles the rows expressible as a single ``type=`` callable or a
    cross-flag check; these (``--dof`` for the wrong mate kind, ``--at`` or
    ``--hold`` outside the mate's limits, a friction flag mismatched to the DOF)
    can only be judged once ``select_mate`` has resolved the mate. ``main``
    turns this into ``parser.error`` so it exits 2 like the argparse rows.
    """

MIN_SAMPLES = 3
MIN_GRID = 8
DEFAULT_SAMPLES = 201
DEFAULT_GRID = 120
CONVERGENCE_MODES = ("off", "quick", "full")
PLOTLY_MODES = ("embed", "cdn")
CYLINDRICAL_SUB_DOFS = ("travel", "turn")

# `x=12.5`, `y=-3`, `z=0`, `mate`, `mate:x`. The value is a model-unit (mm)
# coordinate; `up` is one of the model axes.
_SLICE_RE = re.compile(
    r"^(?:(?P<axis>[xyz])=(?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
    r"|mate(?::(?P<up>[xyz]))?)$"
)
# `<mate>.travel` / `<mate>.turn`: a mate name has no dots (cadgen.kinematics).
_DOF_RE = re.compile(r"^(?P<mate>[^.\s]+)\.(?P<sub>travel|turn)$")

_EXIT_CODES = (
    "exit codes: 0 analysis ran (converged: false still exits 0, with a warning on stderr); "
    "1 analysis error (message on stderr); 2 usage error"
)


# ------------------------------------------------------------------ argparse types
#
# Each `type=` callable IS one argparse-only row of the validation matrix: argparse
# prints `argument --flag: <message>` and exits 2, so the message names the value and
# the callable's flag is named by argparse.


def _int_at_least(minimum: int, flag: str) -> Callable[[str], int]:
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{flag} must be an integer >= {minimum}, got {text!r}") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"{flag} must be >= {minimum}, got {value}")
        return value

    return parse


def _float_flag(flag: str, *, minimum: float | None = None, exclusive: bool = False) -> Callable[[str], float]:
    def parse(text: str) -> float:
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{flag} must be a number, got {text!r}") from None
        if value != value:  # nan
            raise argparse.ArgumentTypeError(f"{flag} must be a number, got {text!r}")
        if minimum is not None:
            if exclusive and value <= minimum:
                raise argparse.ArgumentTypeError(f"{flag} must be > {minimum:g}, got {text}")
            if not exclusive and value < minimum:
                raise argparse.ArgumentTypeError(f"{flag} must be >= {minimum:g}, got {text}")
        return value

    return parse


def _slice_spec(text: str) -> str:
    if _SLICE_RE.match(text.strip()) is None:
        raise argparse.ArgumentTypeError(
            f"--slice must be x=<mm>, y=<mm>, z=<mm>, mate or mate:<x|y|z>, got {text!r}"
        )
    return text.strip()


def _dof_spec(text: str) -> str:
    if _DOF_RE.match(text.strip()) is None:
        raise argparse.ArgumentTypeError(
            f"--dof must be <mate>.travel or <mate>.turn (a cylindrical mate's sub-DOF), got {text!r}"
        )
    return text.strip()


# ------------------------------------------------------------------ parser


def _add_common(sub: argparse.ArgumentParser, *, out: bool) -> None:
    sub.add_argument("step", help="the generated .step (its <name>.step.json sidecar carries the mates)")
    sub.add_argument(
        "--mate",
        metavar="NAME",
        help="the slider, revolute or cylindrical mate to drive (default: the model's sole mate)",
    )
    if out:
        sub.add_argument(
            "--out",
            metavar="DIR",
            help="output directory (default: <step>-magnetics/ beside the STEP)",
        )
        sub.add_argument(
            "--plotly",
            choices=PLOTLY_MODES,
            default="embed",
            help="embed plotly.js in the HTML (offline, ~3.5 MB; default) or load it from the CDN (~50 kB)",
        )
    sub.add_argument("--json", action="store_true", help="one compact JSON line on stdout")


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog or DEFAULT_PROG,
        description=(
            "Permanent-magnet mechanism analysis: force, torque and potential energy along a "
            "cadgen-declared mate, from leaves named mag:<grade>:<direction>. Needs the "
            'magnetics extra: pip install "cadgen[magnetics]".'
        ),
        epilog=_EXIT_CODES,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser(
        "inspect",
        help="list the magnets (shape, polarization, position) and the mate a model declares",
        description=(
            "Per magnet: ref, label, shape class (cuboid | cylinder | mesh), polarization_world_T, "
            "position_m, orientation_quat and moving (null unless --mate selects a mate)."
        ),
        epilog=_EXIT_CODES,
    )
    _add_common(inspect, out=False)

    sweep = sub.add_parser(
        "sweep",
        help="sweep the mate: Q(q), U(q), equilibria, verdict, convergence -> sweep.json + report.html",
        description=(
            "Samples the mate across its limits, projects the magnetic force/torque onto the DOF, "
            "integrates the potential energy, finds equilibria and their stability, checks clearance "
            "per pose, runs the convergence protocol and writes sweep.json and report.html to --out."
        ),
        epilog=_EXIT_CODES,
    )
    _add_common(sweep, out=True)
    sweep.add_argument(
        "--dof",
        metavar="MATE.travel|MATE.turn",
        type=_dof_spec,
        help="the sub-DOF a cylindrical mate sweeps (required for that kind; invalid for others)",
    )
    held = sweep.add_mutually_exclusive_group()
    held.add_argument(
        "--hold",
        metavar="VALUE",
        type=_float_flag("--hold"),
        default=0.0,
        help="hold a cylindrical mate's other sub-DOF at VALUE (mm or degrees; default 0)",
    )
    held.add_argument(
        "--relax",
        action="store_true",
        help="instead of --hold, set the other sub-DOF where its own generalized force vanishes at each pose",
    )
    sweep.add_argument(
        "--samples",
        metavar="N",
        type=_int_at_least(MIN_SAMPLES, "--samples"),
        default=DEFAULT_SAMPLES,
        help=f"poses across the limits (>= {MIN_SAMPLES}; default {DEFAULT_SAMPLES})",
    )
    friction = sweep.add_mutually_exclusive_group()
    friction.add_argument(
        "--friction-N",
        dest="friction_N",
        metavar="F",
        type=_float_flag("--friction-N", minimum=0.0),
        help="force threshold in newtons the verdict compares |Q| against (translation DOF)",
    )
    friction.add_argument(
        "--friction-Nm",
        dest="friction_Nm",
        metavar="T",
        type=_float_flag("--friction-Nm", minimum=0.0),
        help="torque threshold in newton-metres the verdict compares |Q| against (rotation DOF)",
    )
    sweep.add_argument(
        "--at",
        metavar="Q",
        type=_float_flag("--at"),
        help="the verdict pose, in the mate's model units (mm or degrees; default: the lower limit)",
    )
    sweep.add_argument(
        "--convergence",
        choices=CONVERGENCE_MODES,
        default="quick",
        help="refinement protocol: off (finest level once), quick (mesh {20,50} + grid; default), full",
    )
    sweep.add_argument(
        "--converge-tol",
        dest="converge_tol",
        metavar="TOL",
        type=_float_flag("--converge-tol", minimum=0.0, exclusive=True),
        default=0.02,
        help="relative delta below which a refinement axis counts as converged (default 0.02)",
    )

    field = sub.add_parser(
        "field",
        help="sample |B| and in-plane arrows on one plane at one pose -> field.html",
        description=(
            "Sources are the magnets posed at --at. The plane is axis-normal (x=|y=|z=<mm>) or the "
            "plane containing the mate axis (mate[:<up>], up defaulting to the bounding box's shortest axis)."
        ),
        epilog=_EXIT_CODES,
    )
    _add_common(field, out=True)
    field.add_argument(
        "--slice",
        metavar="x=MM|y=MM|z=MM|mate[:UP]",
        type=_slice_spec,
        default="mate",
        help="the plane (default: mate)",
    )
    field.add_argument(
        "--grid",
        metavar="N",
        type=_int_at_least(MIN_GRID, "--grid"),
        default=DEFAULT_GRID,
        help=f"samples per side (>= {MIN_GRID}; default {DEFAULT_GRID})",
    )
    field.add_argument(
        "--at",
        metavar="Q",
        type=_float_flag("--at"),
        default=0.0,
        help="the pose, in the mate's model units (default 0 = the artifact as written)",
    )
    return parser


# ------------------------------------------------------------------ output helpers


def _emit_json(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _narrate(text: str) -> None:
    sys.stderr.write(text.rstrip("\n") + "\n")


def _fail(message: str) -> int:
    sys.stderr.write(f"cadgen magnetics: {message}\n")
    return 1


def _out_dir(step: Path, out: str | None) -> Path:
    return Path(out) if out else step.with_name(f"{step.stem}-magnetics")


# ------------------------------------------------------------------ model <-> SI
#
# `MateSpec.limits` are SI (m, rad); the flags `--at`/`--hold` are MODEL units
# (mm for a translation, degrees for a rotation). Whether a limits key is
# angular is fixed by the key -- `turn` is always rad, `travel` always m -- and
# for the single-DOF `value` key by the mate kind (revolute rotates, slider
# translates).


def _is_angular(mate: Any, key: str) -> bool:
    if key == "turn":
        return True
    if key == "travel":
        return False
    return mate.kind == "revolute"  # the "value" key


def _model_to_si(mate: Any, key: str, value: float) -> float:
    return math.radians(value) if _is_angular(mate, key) else value * MM_TO_M


def _si_to_model(mate: Any, key: str, value_si: float) -> float:
    return math.degrees(value_si) if _is_angular(mate, key) else value_si * _MM_PER_M


def _within(value: float, lo: float, hi: float) -> bool:
    """``value`` in the closed ``[lo, hi]`` with a range-scaled float tolerance."""
    tol = 1e-9 * max(abs(hi - lo), 1.0)
    return lo - tol <= value <= hi + tol


def _model_unit(mate: Any, key: str) -> str:
    return "degrees" if _is_angular(mate, key) else "mm"


def _mate_summary(mate: Any) -> dict[str, Any] | None:
    if mate is None:
        return None
    return {
        "name": mate.name,
        "kind": mate.kind,
        "childId": mate.child_id,
        "axis": list(mate.dir),
        "origin_m": list(mate.origin_m),
        "limits": {key: list(value) for key, value in mate.limits.items()},
        "q_unit": mate.q_unit,
    }


def _magnet_summary(magnet: Any) -> dict[str, Any]:
    return {
        "ref": magnet.ref,
        "label": magnet.label,
        "shape": magnet.shape,
        "polarization_world_T": list(magnet.polarization_world_T),
        "position_m": list(magnet.position_m),
        "orientation_quat": list(magnet.orientation_quat),
        "moving": magnet.moving,
    }


# ------------------------------------------------------------------ verbs
#
# Bodies orchestrate the cadgen.magnetics modules (imported inside, so --help never
# pays for them and lane A5's tests can patch `cadgen.magnetics.scene.load`,
# `kinematics.select_mate` and `physics.sweep` on their modules). Lane A5 finishes
# the error mapping and messages; the skeleton here fixes the shape.


def _cmd_inspect(args: argparse.Namespace) -> int:
    from cadgen.magnetics import scene as scene_mod

    step = Path(args.step)
    mag_scene = scene_mod.load(step, args.mate)
    payload = {
        "step": str(step),
        "mate": _mate_summary(mag_scene.mate),
        "magnets": [_magnet_summary(m) for m in mag_scene.magnets],
    }
    if args.json:
        _emit_json(payload)
        return 0
    mate = mag_scene.mate
    mate_text = f"{mate.name} ({mate.kind})" if mate is not None else "none selected"
    sys.stdout.write(f"magnets {len(mag_scene.magnets)}  mate {mate_text}\n")
    for m in mag_scene.magnets:
        moving = "-" if m.moving is None else ("moving" if m.moving else "fixed")
        j = ",".join(f"{v:.3g}" for v in m.polarization_world_T)
        p = ",".join(f"{v:.4g}" for v in m.position_m)
        sys.stdout.write(f"{m.ref}  {m.label}  {m.shape}  J=[{j}] T  at [{p}] m  {moving}\n")
    return 0


def _resolve_dof(mate: Any, dof: str | None) -> tuple[str, str | None]:
    """``(swept_key, held_key)`` for the mate, mapping the mate-dependent ``--dof``
    rows of the validation matrix to exit 2.

    A cylindrical mate routes through ``kinematics.sweep_dof`` (which raises
    :class:`MateError` naming the mate and its sub-DOFs when ``--dof`` is missing
    or names a sub-DOF the mate lacks); the CLI turns that into a usage error. A
    single-DOF mate sweeps ``value`` and rejects ``--dof`` outright.
    """
    from cadgen.magnetics import kinematics as kinematics_mod

    if mate.kind == "cylindrical":
        try:
            return kinematics_mod.sweep_dof(mate, dof)
        except MateError as exc:
            raise _UsageError(str(exc)) from exc
    if dof is not None:
        raise _UsageError(
            f"--dof {dof!r} is only valid for a cylindrical mate; {mate.name!r} is a {mate.kind} mate"
        )
    return "value", None


def _prepare_sweep(args: argparse.Namespace, mate: Any) -> tuple[str, str | None]:
    """Validate the flags that only make sense against a resolved mate.

    Returns ``(swept_key, held_key)``. Raises :class:`_UsageError` (exit 2) for
    the mate-dependent matrix rows: ``--dof`` for the wrong kind, a friction
    flag mismatched to the DOF, ``--at`` outside the swept limits, ``--hold``
    outside the held sub-DOF's limits.
    """
    swept_key, held_key = _resolve_dof(mate, args.dof)

    angular = _is_angular(mate, swept_key)
    if args.friction_N is not None and angular:
        raise _UsageError(
            "--friction-N is a force threshold for a translation DOF, but this sweep drives a "
            f"rotation ({args.dof or mate.name}); use --friction-Nm"
        )
    if args.friction_Nm is not None and not angular:
        raise _UsageError(
            "--friction-Nm is a torque threshold for a rotation DOF, but this sweep drives a "
            f"translation ({args.dof or mate.name}); use --friction-N"
        )

    if args.at is not None:
        at_si = _model_to_si(mate, swept_key, args.at)
        lo, hi = mate.limits[swept_key]
        if not _within(at_si, lo, hi):
            unit = _model_unit(mate, swept_key)
            lo_m, hi_m = _si_to_model(mate, swept_key, lo), _si_to_model(mate, swept_key, hi)
            raise _UsageError(
                f"--at {args.at:g} is outside the mate's limits [{lo_m:g}, {hi_m:g}] {unit}"
            )

    if held_key is not None and not args.relax:
        hold_si = _model_to_si(mate, held_key, args.hold)
        lo, hi = mate.limits[held_key]
        if not _within(hold_si, lo, hi):
            unit = _model_unit(mate, held_key)
            lo_m, hi_m = _si_to_model(mate, held_key, lo), _si_to_model(mate, held_key, hi)
            raise _UsageError(
                f"--hold {args.hold:g} is outside {mate.name}.{held_key}'s limits "
                f"[{lo_m:g}, {hi_m:g}] {unit}"
            )

    return swept_key, held_key


def _sweep_config(args: argparse.Namespace, mate: Any, swept_key: str, held_key: str | None) -> SweepConfig:
    friction = args.friction_N if args.friction_N is not None else args.friction_Nm
    at_si = None if args.at is None else _model_to_si(mate, swept_key, args.at)
    hold_si = 0.0 if held_key is None else _model_to_si(mate, held_key, args.hold)
    return SweepConfig(
        samples=int(args.samples),
        convergence=args.convergence,
        converge_tol=float(args.converge_tol),
        friction=friction,
        at=at_si,
        dof=args.dof,
        hold=hold_si,
        relax=bool(args.relax),
    )


def _cmd_sweep(args: argparse.Namespace) -> int:
    from cadgen.magnetics import kinematics as kinematics_mod
    from cadgen.magnetics import physics as physics_mod
    from cadgen.magnetics import report as report_mod
    from cadgen.magnetics import scene as scene_mod

    step = Path(args.step)
    out_dir = _out_dir(step, args.out)

    mate = kinematics_mod.select_mate(step, args.mate)
    swept_key, held_key = _prepare_sweep(args, mate)
    cfg = _sweep_config(args, mate, swept_key, held_key)
    mag_scene = scene_mod.load(step, mate.name)
    _narrate(f"[cadgen] magnetics sweep {step.name}: mate {mate.name} ({mate.kind}), {cfg.samples} poses, convergence {cfg.convergence}")
    result = physics_mod.sweep(mag_scene, cfg)
    at_q = result.verdict.at_q
    # The report's field section is the mate-plane slice at the verdict pose. A
    # degenerate plane (e.g. the mate axis parallel to the shortest bbox axis
    # with no usable fallback) is not a sweep failure -- the report simply omits
    # the slice. `at_q` is already SI (physics produced it), so no conversion.
    try:
        slice_ = physics_mod.field(mag_scene, "mate", grid=DEFAULT_GRID, q=at_q)
    except SceneError as exc:
        _narrate(f"[cadgen] note: no field slice for the report ({exc}); the sweep is unaffected")
        slice_ = None
    json_path = report_mod.write_json(result, out_dir)
    report_path = report_mod.write_report(result, slice_, out_dir, plotly=args.plotly)

    if result.convergence.converged is False:
        _narrate(
            "[cadgen] warning: not converged "
            f"({', '.join(result.convergence.reasons) or 'no reason recorded'}); the verdict's uncertainty band is wide"
        )

    document = sweep_result_to_json(result)
    if args.json:
        # O(1) on stdout: the per-pose table stays in sweep.json.
        summary = {key: value for key, value in document.items() if key != "samples"}
        summary["ok"] = True
        summary["out"] = {"json": str(json_path), "report": str(report_path)}
        _emit_json(summary)
        return 0
    verdict = result.verdict
    threshold = "no threshold" if verdict.threshold is None else f"threshold {verdict.threshold:g}"
    exceeds = "n/a" if verdict.exceeds_threshold is None else str(verdict.exceeds_threshold).lower()
    sys.stdout.write(
        f"verdict at q={verdict.at_q:g} {document['q_unit']}: {verdict.tendency}  "
        f"Q={verdict.Q:.4g} +- {verdict.uncertainty:.2g}  {threshold}  exceeds {exceeds}  "
        f"admissible {str(verdict.admissible).lower()}\n"
    )
    sys.stdout.write(
        f"equilibria {len(result.equilibria)}  inaccessible {len(result.inaccessible)}  "
        f"wedged {len(result.wedged)}  converged {str(result.convergence.converged).lower()}\n"
    )
    sys.stdout.write(f"json {json_path}\nreport {report_path}\n")
    return 0


def _cmd_field(args: argparse.Namespace) -> int:
    from cadgen.magnetics import kinematics as kinematics_mod
    from cadgen.magnetics import physics as physics_mod
    from cadgen.magnetics import report as report_mod
    from cadgen.magnetics import scene as scene_mod

    step = Path(args.step)
    out_dir = _out_dir(step, args.out)
    # A mate is needed to pose the sources at --at, and to parse a mate-plane
    # slice. An axis-normal slice at the artifact as written (--at 0, no --mate)
    # needs none, so field works on a model with no mate.
    needs_mate = args.slice.startswith("mate") or args.mate is not None or args.at != 0.0
    mate = None
    if needs_mate:
        mate = kinematics_mod.select_mate(step, args.mate)
    mag_scene = scene_mod.load(step, mate.name if mate is not None else None)

    # --at is MODEL units. field carries no --dof, so a cylindrical mate poses
    # along its translation (travel); a slider/revolute along its single DOF.
    q_si = 0.0
    if mate is not None:
        pose_key = "travel" if mate.kind == "cylindrical" else "value"
        q_si = _model_to_si(mate, pose_key, args.at)

    _narrate(f"[cadgen] magnetics field {step.name}: slice {args.slice}, grid {args.grid}, at {args.at:g}")
    slice_ = physics_mod.field(mag_scene, args.slice, grid=int(args.grid), q=q_si)
    field_path = report_mod.write_field(slice_, out_dir, plotly=args.plotly)

    from cadgen.magnetics import _deps

    np = _deps.numpy()
    b_max = float(np.linalg.norm(np.asarray(slice_.B), axis=-1).max())
    if args.json:
        _emit_json(
            {
                "ok": True,
                "plane": slice_.plane,
                "grid": int(args.grid),
                "at": float(args.at),
                "B_max_T": b_max,
                "out": {"field": str(field_path)},
            }
        )
        return 0
    sys.stdout.write(f"slice {args.slice}  grid {args.grid}  |B| max {b_max:.4g} T\nfield {field_path}\n")
    return 0


# ------------------------------------------------------------------ main


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Cross-flag rows of the matrix that argparse cannot express as one type."""
    if args.command == "sweep" and args.relax and args.dof is None:
        parser.error("--relax needs --dof <mate>.travel|<mate>.turn: only a cylindrical mate has a sub-DOF to relax")


def _run(args: argparse.Namespace) -> int:
    from cadgen._internal.source_sidecar import SidecarBindingError, SidecarSchemaError

    try:
        if args.command == "inspect":
            return _cmd_inspect(args)
        if args.command == "sweep":
            return _cmd_sweep(args)
        if args.command == "field":
            return _cmd_field(args)
    except MagneticsError as exc:
        # Every analysis error is a MagneticsError subclass (dependency gate,
        # scene, mate, physics): exit 1 with the domain module's message, which
        # already names the leaf, mate or fix. _UsageError is deliberately NOT a
        # MagneticsError, so a mate-dependent usage row escapes to `main`.
        return _fail(str(exc))
    except (SidecarBindingError, SidecarSchemaError) as exc:
        # cadgen's own binding error: the sidecar is missing, stale (documentHash
        # mismatch) or an old schema. Its message already names the fix.
        return _fail(str(exc))
    return 2


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = build_parser(prog)
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])
    _validate(parser, args)
    try:
        return _run(args)
    except _UsageError as exc:
        # A validation-matrix row that needed the mate to detect: exit 2, like
        # the argparse rows, via the same error path.
        parser.error(str(exc))  # raises SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
