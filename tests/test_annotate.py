import numpy as np

from rescue_vision.annotate import draw_overlay, format_label
from rescue_vision.types import BBox, Command, TrackState


def track(track_id=1, confirmed=True):
    return TrackState(
        track_id=track_id,
        bbox=BBox(100.0, 50.0, 200.0, 350.0),
        confidence=0.90,
        bearing_deg=-12.4,
        distance_m=3.2,
        distance_valid=True,
        confirmed=confirmed,
        display_confidence=0.90,
    )


def test_overlay_returns_a_new_array_and_leaves_the_input_untouched():
    frame = np.zeros((480, 640, 3), np.uint8)
    out = draw_overlay(frame, [track()], 1, Command(-0.31, 0.0), 12.0)
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)
    assert frame.sum() == 0


def test_overlay_draws_something_for_each_track():
    frame = np.zeros((480, 640, 3), np.uint8)
    second = TrackState(
        track_id=2,
        bbox=BBox(400.0, 60.0, 480.0, 300.0),
        confidence=0.8,
        bearing_deg=10.0,
        distance_m=4.0,
        distance_valid=True,
        confirmed=True,
        display_confidence=0.8,
    )
    one = draw_overlay(frame, [track(1)], 1, Command(0.0, 0.0), 12.0)
    two = draw_overlay(frame, [track(1), second], 1, Command(0.0, 0.0), 12.0)
    assert two.sum() > one.sum()


def test_overlay_handles_no_tracks():
    frame = np.zeros((480, 640, 3), np.uint8)
    out = draw_overlay(frame, [], None, Command(0.0, 0.0), 12.0)
    assert out.shape == frame.shape


def test_overlay_handles_a_box_at_the_very_top_of_frame():
    """The label must not be drawn at a negative y coordinate."""
    frame = np.zeros((480, 640, 3), np.uint8)
    t = track()
    t.bbox = BBox(10.0, 0.0, 80.0, 200.0)
    out = draw_overlay(frame, [t], 1, Command(0.0, 0.0), 12.0)
    assert out.shape == frame.shape


def test_label_includes_the_confidence_score():
    """Judges and operators need to see how sure the detector is."""
    label = format_label(track(1))
    assert "0.90" in label
    assert "#1" in label
    assert "person" in label


def test_label_uses_the_sampled_confidence_not_the_live_one():
    """Amendment A: the on-screen number is held steady at 1 Hz."""
    t = track()
    t.confidence = 0.41
    t.display_confidence = 0.87
    assert "0.87" in format_label(t)
    assert "0.41" not in format_label(t)


def test_label_shows_the_reason_when_distance_is_untrustworthy():
    t = track(1)
    t.distance_valid = False
    t.invalid_reason = "not_upright"
    label = format_label(t)
    assert "not_upright" in label
    assert "3.2m" not in label
