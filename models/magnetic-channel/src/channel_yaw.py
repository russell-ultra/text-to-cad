"""Yaw variant: the aligned channel on a cylindrical (travel + turn) mate.

Same geometry as ``channel_aligned``, but the slider rides a ``cylindrical``
mate so it can both travel along +x and roll about that axis. This is the model
for the pin-slot yaw question:

    cadgen magnetics sweep STEP/channel_yaw.step --dof travel.travel --relax

sweeps the travel and, at each pose, relaxes ``turn`` to where its generalized
torque vanishes -- the yaw curve to compare against a bench measurement.
"""

from __future__ import annotations

from cadgen import step

from lib.channel import build_channel, yaw_kinematics


@step(out="../STEP/channel_yaw.step", kinematics=yaw_kinematics())
def channel_yaw():
    return build_channel("aligned")


if __name__ == "__main__":
    channel_yaw()
