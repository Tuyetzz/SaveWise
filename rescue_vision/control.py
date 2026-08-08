"""Proportional turn control with a deadband, plus differential mixing. PRD 6.8.

Pure module. No I/O, no clock.

Sign convention (PRD Appendix B): positive turn == rover rotates clockwise /
to its right, and turn takes the SAME sign as bearing. If the physical rover
turns away from people, swap the motor pin pairs in MOTOR_PINS -- do not negate
KP, which would leave this convention lying to the next reader.
"""

from __future__ import annotations

import math

from rescue_vision.config import Config
from rescue_vision.types import Command, TrackState


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def turn_command(bearing_deg: float, cfg: Config) -> float:
    """Deadband proportional control toward frame centre."""
    if abs(bearing_deg) <= cfg.deadband_deg:
        return 0.0
    turn = clamp(cfg.kp * bearing_deg, -1.0, 1.0)
    if 0.0 < abs(turn) < cfg.min_turn:
        # Below the stiction floor a DC motor buzzes without turning, while
        # the controller believes it commanded motion.
        turn = math.copysign(cfg.min_turn, turn)
    return turn


def drive_command(target: TrackState | None, turn: float, cfg: Config) -> float:
    """Turn in place first, then advance. PRD 6.8."""
    if target is None:
        return 0.0
    if abs(target.bearing_deg) > cfg.deadband_deg:
        return 0.0
    # Only an estimate we trust may hold the rover back. A distance already
    # flagged invalid must not gate behaviour on a number known to be wrong.
    if target.distance_valid and target.distance_m < cfg.stop_distance_m:
        return 0.0
    return cfg.approach_speed


def compute_command(target: TrackState | None, cfg: Config) -> Command:
    """Full command for one frame."""
    if target is None:
        return Command(turn=0.0, drive=0.0)
    turn = turn_command(target.bearing_deg, cfg)
    return Command(turn=turn, drive=drive_command(target, turn, cfg))


def mix(turn: float, forward: float) -> tuple[float, float]:
    """Differential mixing.

    Clamp AFTER mixing so a simultaneous turn+forward request cannot silently
    saturate one side.
    """
    return (
        clamp(forward + turn, -1.0, 1.0),
        clamp(forward - turn, -1.0, 1.0),
    )
