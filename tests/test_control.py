import pytest

from rescue_vision.config import Config
from rescue_vision.control import (
    clamp,
    compute_command,
    drive_command,
    mix,
    turn_command,
)
from rescue_vision.types import BBox, TrackState

CFG = Config()


def target(bearing_deg=0.0, distance_m=5.0, distance_valid=True):
    return TrackState(
        track_id=1,
        bbox=BBox(300.0, 100.0, 360.0, 400.0),
        confidence=0.9,
        bearing_deg=bearing_deg,
        distance_m=distance_m,
        distance_valid=distance_valid,
        confirmed=True,
    )


def test_clamp_bounds_both_ways():
    assert clamp(5.0, -1.0, 1.0) == 1.0
    assert clamp(-5.0, -1.0, 1.0) == -1.0
    assert clamp(0.5, -1.0, 1.0) == 0.5


def test_turn_is_zero_inside_the_deadband():
    """No deadband and no derivative term means the rover oscillates forever."""
    assert turn_command(0.0, CFG) == 0.0
    assert turn_command(4.9, CFG) == 0.0
    assert turn_command(-4.9, CFG) == 0.0


def test_turn_takes_the_same_sign_as_bearing():
    """PRD Appendix B. Getting this backwards looks identical to a detection bug."""
    assert turn_command(25.0, CFG) > 0.0
    assert turn_command(-25.0, CFG) < 0.0


def test_turn_is_proportional_to_error_above_the_stiction_floor():
    assert turn_command(25.0, CFG) == pytest.approx(0.5)


def test_small_error_is_lifted_to_min_turn_to_beat_stiction():
    """A DC motor at 8 percent duty cycle buzzes rather than turning."""
    raw = CFG.kp * 6.0  # 0.12, below MIN_TURN of 0.25
    assert raw < CFG.min_turn
    assert turn_command(6.0, CFG) == pytest.approx(CFG.min_turn)
    assert turn_command(-6.0, CFG) == pytest.approx(-CFG.min_turn)


def test_turn_is_clamped_to_unit_range():
    assert turn_command(1000.0, CFG) == 1.0
    assert turn_command(-1000.0, CFG) == -1.0


def test_turn_never_oscillates_across_a_bearing_sweep():
    """Sweeping the error toward zero must decrease |turn| monotonically and
    land at exactly zero, never overshooting into the opposite sign."""
    previous = 1.1
    for bearing in [40.0, 30.0, 20.0, 10.0, 6.0, 4.0, 0.0]:
        t = turn_command(bearing, CFG)
        assert t >= 0.0
        assert t <= previous
        previous = t
    assert turn_command(0.0, CFG) == 0.0


def test_turn_is_antisymmetric():
    for bearing in [3.0, 6.0, 12.0, 25.0, 60.0]:
        assert turn_command(bearing, CFG) == pytest.approx(-turn_command(-bearing, CFG))


def test_no_target_means_no_drive():
    assert drive_command(None, turn=0.0, cfg=CFG) == 0.0


def test_no_drive_while_still_turning():
    """Turn in place first -- easier to debug and looks more deliberate."""
    assert drive_command(target(bearing_deg=30.0), turn=0.6, cfg=CFG) == 0.0


def test_no_drive_when_closer_than_stop_distance():
    t = target(bearing_deg=0.0, distance_m=1.0)
    assert drive_command(t, turn=0.0, cfg=CFG) == 0.0


def test_approaches_when_centred_and_far_enough():
    t = target(bearing_deg=0.0, distance_m=5.0)
    assert drive_command(t, turn=0.0, cfg=CFG) == pytest.approx(CFG.approach_speed)


def test_approaches_when_distance_is_invalid_but_centred():
    """Never gate motion on a distance already flagged untrustworthy."""
    t = target(bearing_deg=0.0, distance_valid=False, distance_m=0.1)
    assert drive_command(t, turn=0.0, cfg=CFG) == pytest.approx(CFG.approach_speed)


def test_compute_command_with_no_target_is_a_full_stop():
    cmd = compute_command(None, CFG)
    assert cmd.turn == 0.0
    assert cmd.drive == 0.0


def test_compute_command_turns_toward_an_off_centre_target():
    cmd = compute_command(target(bearing_deg=-20.0), CFG)
    assert cmd.turn < 0.0
    assert cmd.drive == 0.0


def test_mix_pure_forward_drives_both_sides_equally():
    assert mix(turn=0.0, forward=0.5) == (0.5, 0.5)


def test_mix_pure_right_turn_is_opposite_on_each_side():
    left, right = mix(turn=0.5, forward=0.0)
    assert left == 0.5
    assert right == -0.5


def test_mix_clamps_after_mixing_not_before():
    """Clamping before mixing would let one side silently saturate."""
    left, right = mix(turn=0.8, forward=0.8)
    assert left == 1.0
    assert right == pytest.approx(0.0)
