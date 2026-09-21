"""Aligned channel: every row magnet polarized +z."""

from __future__ import annotations

from cadgen import step

from lib.channel import build_channel, slider_kinematics


@step(out="../STEP/channel_aligned.step", kinematics=slider_kinematics())
def channel_aligned():
    return build_channel("aligned")


if __name__ == "__main__":
    channel_aligned()
