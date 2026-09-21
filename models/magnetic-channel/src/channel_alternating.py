"""Alternating channel: row magnets alternate +z / -z along each row."""

from __future__ import annotations

from cadgen import step

from lib.channel import build_channel, slider_kinematics


@step(out="../STEP/channel_alternating.step", kinematics=slider_kinematics())
def channel_alternating():
    return build_channel("alternating")


if __name__ == "__main__":
    channel_alternating()
