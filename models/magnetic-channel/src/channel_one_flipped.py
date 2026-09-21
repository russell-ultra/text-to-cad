"""One-flipped channel: the middle magnet of row_a is reversed (a defect)."""

from __future__ import annotations

from cadgen import step

from lib.channel import build_channel, slider_kinematics


@step(out="../STEP/channel_one_flipped.step", kinematics=slider_kinematics())
def channel_one_flipped():
    return build_channel("one_flipped")


if __name__ == "__main__":
    channel_one_flipped()
