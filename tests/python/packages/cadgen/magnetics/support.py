"""Fixtures for the magnetics suites: tiny model scripts, a builder, a hand-built pair.

Every model here is the test's own (nothing under ``models/`` is read): the
text is written into a fresh :class:`IsolatedCadRoots` workspace with its own
store and built with ``generate_step_targets``. Magnets are declared by the
``mag:<grade>:<direction>`` naming convention -- the INSTANCE name of
``compound_from_instances`` or the joined tokens of ``label_shape(shape, "mag",
"N42", "+z")`` -- and the mate through ``@step(kinematics=...)``. A model
function takes no parameters and one file declares one model, so the channel's
polarity is a template parameter, not a function argument.

Geometry is in millimetres (cadgen models); :func:`hand_built_coaxial_pair` is
in metres (magpylib), matching what ``scene.load`` must produce from
``TWO_CUBE_MODEL``.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from tests.python.support.cad_test_roots import IsolatedCadRoots
from tests.python.support.paths import add_repo_path

add_repo_path("packages/cadgen/src")

__all__ = [
    "MAGNET_MM",
    "COAXIAL_SPACING_MM",
    "N42_REMANENCE_T",
    "TWO_CUBE_MODEL",
    "CHANNEL_MODEL_TEMPLATE",
    "CHANNEL_POLARITIES",
    "TILTED_CUBE_MODEL",
    "OBLIQUE_PRISM_MODEL",
    "channel_model",
    "build_model",
    "hand_built_coaxial_pair",
]

#: 1/4" cube magnets, the fixture the design is built around.
MAGNET_MM = 6.35
#: Centre spacing of the two-cube regression pair.
COAXIAL_SPACING_MM = 30.0
#: |J| the grade table assigns to N42.
N42_REMANENCE_T = 1.30

# Two coaxial 6.35 mm N42 cubes along z, 30 mm centre to centre, opposite
# polarization (+z fixed at the origin, -z on the slider). The slider mate
# carries the second cube along +z from the authored placement; its parent and
# child are the GROUP compounds (`base`, `slider`), whose plain labels are
# sidecar targets -- the magnet names themselves carry `+`/`-` and are never
# `#label` aliases.
TWO_CUBE_MODEL = f"""\
import cadgen
from cadgen import compound_from_instances, step
from cadgen import build123d as bd

SIDE = {MAGNET_MM}
SPACING = {COAXIAL_SPACING_MM}

KINEMATICS = {{
    "mates": [
        cadgen.slider("travel", parent="#base", child="#slider",
                      origin=(0, 0, 0), direction=(0, 0, 1), limits=(0, 20)),
    ],
}}


@step(kinematics=KINEMATICS)
def two_cube():
    cube = bd.Box(SIDE, SIDE, SIDE)
    base = compound_from_instances("base", [(cube, bd.Location(), "mag:N42:+z")])
    slider = compound_from_instances("slider", [(cube, bd.Location((0, 0, SPACING)), "mag:N42:-z")])
    return bd.Compound(children=[base, slider])


if __name__ == "__main__":
    two_cube()
"""

#: The channel template's polarity parameter.
CHANNEL_POLARITIES = ("aligned", "alternating", "one_flipped")

# A small channel: two rows of three 6.35 mm cubes tilted 45 degrees about y
# (leaning toward +x), a plate under them, and one slider cube between the rows
# riding a slider mate along x. `__POLARITY__` is substituted by
# :func:`channel_model`; a model function takes no parameters, so each
# arrangement is its own file.
CHANNEL_MODEL_TEMPLATE = f"""\
import cadgen
from cadgen import compound_from_instances, label_shape, step
from cadgen import build123d as bd

MAGNET_MM = {MAGNET_MM}
PITCH_MM = 12.0
GAP_MM = 10.0
TILT_DEG = 45.0
N_PER_ROW = 3
CHANNEL_LEN_MM = PITCH_MM * (N_PER_ROW - 1)
POLARITY = __POLARITY__

KINEMATICS = {{
    "mates": [
        cadgen.slider("travel", parent="#channel", child="#slider",
                      origin=(0, 0, 0), direction=(1, 0, 0), limits=(0, CHANNEL_LEN_MM)),
    ],
}}


def _row_polarity(row, index):
    if POLARITY == "alternating":
        return "+z" if index % 2 == 0 else "-z"
    if POLARITY == "one_flipped" and row == "row_a" and index == N_PER_ROW // 2:
        return "-z"
    return "+z"


def _row(name, y):
    cube = bd.Box(MAGNET_MM, MAGNET_MM, MAGNET_MM)
    return compound_from_instances(name, [
        (cube, bd.Pos(i * PITCH_MM, y, 0) * bd.Rot(0, TILT_DEG, 0), "mag:N42:" + _row_polarity(name, i))
        for i in range(N_PER_ROW)
    ])


def _group(label, parts):
    return bd.Compound(obj=list(parts), children=list(parts), label=label)


@step(kinematics=KINEMATICS)
def channel():
    y = GAP_MM / 2 + MAGNET_MM / 2
    plate = label_shape(
        bd.Pos(CHANNEL_LEN_MM / 2, 0, -(MAGNET_MM / 2 + 2.0)) * bd.Box(CHANNEL_LEN_MM + 2 * MAGNET_MM, 2 * y + MAGNET_MM, 2.0),
        "plate",
    )
    body = _group("channel", [plate, _row("row_a", y), _row("row_b", -y)])
    slider = compound_from_instances("slider", [(bd.Box(MAGNET_MM, MAGNET_MM, MAGNET_MM), bd.Location(), "mag:N42:+z")])
    return bd.Compound(children=[body, slider])


if __name__ == "__main__":
    channel()
"""

# One cube rotated 45 degrees about x and placed off the origin: the body frame
# is not the world frame and the AABB is not the dimension. Placed as an
# INSTANCE so the occurrence transform carries the placement (rotation + the
# (5, 0, 0) translation, which is the placed centroid) for the transform
# contract test. No mate. A bare root shape -- labelled part or single-instance
# group -- loses its name in the STEP, which is why the magnet sits under a
# plain root compound here and below (the shape every real assembly has).
TILTED_CUBE_MODEL = f"""\
from cadgen import compound_from_instances, step
from cadgen import build123d as bd

SIDE = {MAGNET_MM}


@step
def tilted_cube():
    cube = bd.Box(SIDE, SIDE, SIDE)
    tilted = compound_from_instances("tilted", [(cube, bd.Pos(5, 0, 0) * bd.Rot(45, 0, 0), "mag:N42:+z")])
    return bd.Compound(children=[tilted])


if __name__ == "__main__":
    tilted_cube()
"""

# A parallelepiped: six planar faces, NOT mutually orthogonal, so it must fall
# through the cuboid check to a TriangularMesh. Named through `label_shape`'s
# joined tokens (the standalone-part spelling of the convention). No mate.
OBLIQUE_PRISM_MODEL = f"""\
from cadgen import label_shape, step
from cadgen import build123d as bd

SIDE = {MAGNET_MM}


@step
def oblique_prism():
    prism = label_shape(bd.extrude(bd.Rectangle(SIDE, SIDE), amount=SIDE, dir=(0.4, 0, 1)), "mag", "N42", "+z")
    return bd.Compound(children=[prism])


if __name__ == "__main__":
    oblique_prism()
"""


def channel_model(polarity: str) -> str:
    """The channel model text for one arrangement."""
    if polarity not in CHANNEL_POLARITIES:
        raise ValueError(f"polarity must be one of {CHANNEL_POLARITIES}, got {polarity!r}")
    return CHANNEL_MODEL_TEMPLATE.replace("__POLARITY__", repr(polarity))


def build_model(testcase: unittest.TestCase, name: str, text: str) -> Path:
    """Write ``text`` as ``<name>.py`` in a fresh isolated workspace, build it, return the ``.step``.

    The workspace, its cwd switch and its ``CADGEN_CACHE_DIR`` are torn down by
    ``testcase``'s cleanups. Raises ``AssertionError`` when the build exits non-zero.
    """
    roots = IsolatedCadRoots(testcase, prefix="magnetics-")
    tempdir = roots.temporary_cad_directory(prefix=f"{name}-")
    testcase.addCleanup(tempdir.cleanup)
    script = Path(tempdir.name) / f"{name}.py"
    script.write_text(text, encoding="utf-8")

    from cadgen.catalog import StepImportOptions
    from cadgen.generation import generate_step_targets

    code = generate_step_targets(
        [str(script)],
        step_options=StepImportOptions(),
        force=True,
        verbose=False,
    )
    if code != 0:
        raise AssertionError(f"building {script} exited {code}")
    step = script.with_suffix(".step")
    if not step.is_file():
        raise AssertionError(f"building {script} wrote no {step.name}")
    return step


def hand_built_coaxial_pair() -> tuple[Any, Any]:
    """``(fixed, moving)`` magpylib Cuboids matching ``TWO_CUBE_MODEL`` at rest, in metres.

    6.35 mm N42 cubes, coaxial along z, 30 mm centre spacing; the fixed one at
    the origin polarized +z, the moving one at ``z = 0.030`` polarized -z. The
    regression test compares ``Q`` through cadgen against these to 0.1 %.
    """
    import magpylib as magpy

    side = MAGNET_MM * 1e-3
    fixed = magpy.magnet.Cuboid(
        polarization=(0.0, 0.0, N42_REMANENCE_T), dimension=(side, side, side), position=(0.0, 0.0, 0.0)
    )
    moving = magpy.magnet.Cuboid(
        polarization=(0.0, 0.0, -N42_REMANENCE_T),
        dimension=(side, side, side),
        position=(0.0, 0.0, COAXIAL_SPACING_MM * 1e-3),
    )
    return fixed, moving
