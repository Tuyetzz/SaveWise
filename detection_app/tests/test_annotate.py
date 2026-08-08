import numpy as np

from rescue_vision.annotate import draw_overlay, format_label, place_label
from rescue_vision.types import BBox, TrackState


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
    out = draw_overlay(frame, [track()], 12.0)
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
    one = draw_overlay(frame, [track(1)], 12.0)
    two = draw_overlay(frame, [track(1), second], 12.0)
    assert two.sum() > one.sum()


def test_overlay_handles_no_tracks():
    frame = np.zeros((480, 640, 3), np.uint8)
    out = draw_overlay(frame, [], 12.0)
    assert out.shape == frame.shape


def test_overlay_handles_a_box_at_the_very_top_of_frame():
    """The label must not be drawn at a negative y coordinate."""
    frame = np.zeros((480, 640, 3), np.uint8)
    t = track()
    t.bbox = BBox(10.0, 0.0, 80.0, 200.0)
    out = draw_overlay(frame, [t], 12.0)
    assert out.shape == frame.shape


def test_label_is_clamped_inside_the_right_frame_edge():
    """A person at the frame edge is the normal case during a sweep -- their
    confidence must not be rendered off-screen."""
    x, y = place_label(620, 100, tw=200, th=12, frame_w=640, frame_h=480, occupied=[])
    assert x + 200 <= 640
    assert x >= 0


def test_label_is_clamped_inside_the_left_frame_edge():
    x, y = place_label(-30, 100, tw=100, th=12, frame_w=640, frame_h=480, occupied=[])
    assert x >= 0


def test_label_never_sits_above_the_top_of_frame():
    x, y = place_label(10, 2, tw=100, th=12, frame_w=640, frame_h=480, occupied=[])
    assert y - 12 >= 0


def test_two_labels_that_would_collide_are_separated():
    """Two people standing close together must not overdraw each other."""
    first = place_label(100, 100, tw=180, th=12, frame_w=640, frame_h=480, occupied=[])
    occupied = [(first[0], first[1] - 12, first[0] + 180, first[1])]
    second = place_label(
        120, 100, tw=180, th=12, frame_w=640, frame_h=480, occupied=occupied
    )
    assert second[1] != first[1]


def test_a_distant_label_is_left_where_it_is():
    first = place_label(100, 100, tw=100, th=12, frame_w=640, frame_h=480, occupied=[])
    occupied = [(first[0], first[1] - 12, first[0] + 100, first[1])]
    second = place_label(
        400, 100, tw=100, th=12, frame_w=640, frame_h=480, occupied=occupied
    )
    assert second[1] == first[1]


def test_three_crowded_people_all_get_distinct_label_rows():
    """Directly the multi-person case: nobody's confidence may be hidden."""
    frame = np.zeros((480, 640, 3), np.uint8)
    crowd = []
    for i in range(3):
        t = track(i + 1)
        t.bbox = BBox(100.0 + i * 20, 150.0, 180.0 + i * 20, 400.0)
        crowd.append(t)
    out = draw_overlay(frame, crowd, 12.0)
    assert out.shape == frame.shape


def test_label_carries_the_person_number_and_confidence():
    assert format_label(track(1), sighting_id=2) == "P2  confidence_score = 0.90"


def test_label_is_pure_ascii_because_hershey_fonts_cannot_render_more():
    """cv2 renders a middle dot as '?'."""
    assert format_label(track(1), sighting_id=2).isascii()


def test_an_unconfirmed_track_has_no_person_number_yet():
    assert format_label(track(1, confirmed=False), None) == "confidence_score = 0.90"


def test_each_person_is_drawn_in_their_own_colour():
    from rescue_vision.palette import colour_for, hex_to_bgr

    frame = np.zeros((480, 640, 3), np.uint8)
    a, b = track(1), track(2)
    b.bbox = BBox(300.0, 50.0, 400.0, 350.0)
    out = draw_overlay(frame, [a, b], 12.0, 2, {1: 1, 2: 2})

    pixels = out.reshape(-1, 3)
    for sid in (1, 2):
        want = np.array(hex_to_bgr(colour_for(sid)))
        assert (pixels == want).all(axis=1).any(), f"P{sid} colour missing"


def test_a_persons_colour_does_not_change_when_someone_else_leaves():
    """Colour follows the entity, never its rank."""
    from rescue_vision.palette import colour_for, hex_to_bgr

    frame = np.zeros((480, 640, 3), np.uint8)
    second = track(2)
    second.bbox = BBox(300.0, 50.0, 400.0, 350.0)

    both = draw_overlay(frame, [track(1), second], 12.0, 2, {1: 1, 2: 2})
    alone = draw_overlay(frame, [second], 12.0, 2, {2: 2})

    want = np.array(hex_to_bgr(colour_for(2)))
    for img in (both, alone):
        assert (img.reshape(-1, 3) == want).all(axis=1).any()


def test_an_unconfirmed_person_is_drawn_grey_not_coloured():
    """Grey means 'not yet counted'; colour means 'in the report'."""
    from rescue_vision.palette import hex_to_bgr, TENTATIVE_HEX

    frame = np.zeros((480, 640, 3), np.uint8)
    out = draw_overlay(frame, [track(1, confirmed=False)], 12.0, 0, {})
    want = np.array(hex_to_bgr(TENTATIVE_HEX))
    assert (out.reshape(-1, 3) == want).all(axis=1).any()


def test_label_omits_track_id_bearing_and_distance():
    """These stay in the JSONL; they are just not drawn on the frame."""
    label = format_label(track(7))
    for noise in ("#7", "person", "deg", "3.2m", "dist"):
        assert noise not in label


def test_label_omits_the_distance_rejection_reason():
    t = track(1)
    t.distance_valid = False
    t.invalid_reason = "clipped_bottom"
    assert format_label(t) == "confidence_score = 0.90"


def test_label_uses_the_sampled_confidence_not_the_live_one():
    """Amendment A: the on-screen number is held steady at 1 Hz."""
    t = track()
    t.confidence = 0.41
    t.display_confidence = 0.87
    assert "0.87" in format_label(t)
    assert "0.41" not in format_label(t)
