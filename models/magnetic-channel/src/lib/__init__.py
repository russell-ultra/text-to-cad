"""Shared geometry and kinematics for the magnetic-channel sample.

The three arrangement models (``channel_aligned``, ``channel_alternating``,
``channel_one_flipped``) and the yaw variant (``channel_yaw``) all build their
geometry with :func:`lib.channel.build_channel` and declare their mate with the
kinematics helpers in the same module, so the arrangements differ only by their
magnet polarity and (for the yaw variant) their mate kind.
"""
