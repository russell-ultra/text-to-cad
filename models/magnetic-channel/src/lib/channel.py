"""Parametric magnet channel: two tilted rows and one sliding magnet.

The sample the magnetics design is built around. Two rows of 1/4" (6.35 mm) N42
cube magnets, each cube tilted 45 degrees about y (leaning toward +x), sit on a
plate either side of a gap. A single magnet rides that gap along +x on a
``slider`` mate -- the pin-slot the question "would it be pushed to one end of
the channel?" is asked of. ``build_channel`` places the rows; the polarity of
the row cubes is the only thing that changes between the three arrangements:

- ``aligned``      -- every row magnet polarized +z (all attract/repel alike);
- ``alternating``  -- +z / -z along each row (a Halbach-ish ripple);
- ``one_flipped``  -- one middle magnet of ``row_a`` reversed (a defect).

Every leaf that is a magnet is named ``mag:N42:<dir>`` through the INSTANCE-name
argument of :func:`compound_from_instances`, the convention the magnetics tool
reads (see ``skills/magnetics``). The plate is plain structure and carries no
``mag:`` name, so the tool ignores it.

All lengths here are millimetres and all angles degrees -- cadgen model units.
``cadgen.magnetics`` converts to SI at ``scene.load``; nothing in this module
needs to know that.
"""

from __future__ import annotations

import cadgen
from cadgen import build123d as bd
from cadgen import compound_from_instances, label_shape

#: 1/4" cube magnets -- the physical stock the sample is sized for.
MAGNET_MM = 6.35
#: Centre-to-centre spacing of the cubes along a row.
PITCH_MM = 12.0
#: Clear gap between the two rows; the slider magnet rides down its middle.
GAP_MM = 10.0
#: Lean of every row cube about y (toward +x).
TILT_DEG = 45.0
#: Cubes per row. Two rows plus the one slider magnet -> 13 ``mag:`` leaves.
N_PER_ROW = 6
#: Travel span of the slider along +x: the centres of the first and last cubes.
CHANNEL_LEN_MM = PITCH_MM * (N_PER_ROW - 1)
#: Plate thickness under the rows.
PLATE_THICK_MM = 2.0
#: Half-range of the yaw sub-DOF on the cylindrical variant (about the channel
#: axis), in degrees. The pin-slot's real yaw freedom is a Phase 3 measurement.
YAW_LIMIT_DEG = 30.0

#: The polarity arrangements ``build_channel`` accepts, one model file each.
POLARITIES = ("aligned", "alternating", "one_flipped")


def _check_polarity(polarity: str) -> None:
    if polarity not in POLARITIES:
        raise ValueError(f"polarity must be one of {POLARITIES}, got {polarity!r}")


def _row_polarity(polarity: str, row: str, index: int) -> str:
    """The ``+z``/``-z`` magnetization axis of one row cube."""
    if polarity == "alternating":
        return "+z" if index % 2 == 0 else "-z"
    if polarity == "one_flipped" and row == "row_a" and index == N_PER_ROW // 2:
        return "-z"
    return "+z"


def _row(polarity: str, name: str, y: float):
    """One tilted row of ``N_PER_ROW`` magnet cubes, named by the convention."""
    cube = bd.Box(MAGNET_MM, MAGNET_MM, MAGNET_MM)
    return compound_from_instances(
        name,
        [
            (
                cube,
                bd.Pos(i * PITCH_MM, y, 0) * bd.Rot(0, TILT_DEG, 0),
                "mag:N42:" + _row_polarity(polarity, name, i),
            )
            for i in range(N_PER_ROW)
        ],
    )


def _group(label: str, parts):
    """A labelled compound whose ``label`` is a resolvable ``#`` mate ref."""
    return bd.Compound(obj=list(parts), children=list(parts), label=label)


def build_channel(polarity: str):
    """The channel assembly for one ``polarity`` arrangement.

    A root ``Compound`` with two children: the ``#channel`` body (plate plus the
    two tilted rows) and the ``#slider`` magnet at rest (q = 0). Both labels are
    the mate's parent/child refs; see :func:`slider_kinematics`.
    """
    _check_polarity(polarity)
    y = GAP_MM / 2 + MAGNET_MM / 2
    plate = label_shape(
        bd.Pos(CHANNEL_LEN_MM / 2, 0, -(MAGNET_MM / 2 + PLATE_THICK_MM / 2))
        * bd.Box(CHANNEL_LEN_MM + 2 * MAGNET_MM, 2 * y + MAGNET_MM, PLATE_THICK_MM),
        "plate",
    )
    body = _group("channel", [plate, _row(polarity, "row_a", y), _row(polarity, "row_b", -y)])
    slider = compound_from_instances(
        "slider",
        [(bd.Box(MAGNET_MM, MAGNET_MM, MAGNET_MM), bd.Location(), "mag:N42:+z")],
    )
    return bd.Compound(children=[body, slider])


def slider_kinematics() -> dict:
    """A single ``travel`` slider carrying ``#slider`` along +x over the channel."""
    return {
        "mates": [
            cadgen.slider(
                "travel",
                parent="#channel",
                child="#slider",
                origin=(0, 0, 0),
                direction=(1, 0, 0),
                limits=(0, CHANNEL_LEN_MM),
            ),
        ],
    }


def yaw_kinematics() -> dict:
    """A ``travel`` cylindrical mate: the slider travels +x AND rolls about it.

    The ``turn`` sub-DOF is the rotational freedom a real pin-slot may or may not
    permit. ``cadgen magnetics sweep ... --dof travel.travel --relax`` sweeps the
    travel and, at each pose, sets ``turn`` where its own generalized torque
    vanishes -- the yaw curve the design asks to compare against a bench
    measurement.
    """
    return {
        "mates": [
            cadgen.cylindrical(
                "travel",
                parent="#channel",
                child="#slider",
                origin=(0, 0, 0),
                direction=(1, 0, 0),
                limits={"travel": (0, CHANNEL_LEN_MM), "turn": (-YAW_LIMIT_DEG, YAW_LIMIT_DEG)},
            ),
        ],
    }
