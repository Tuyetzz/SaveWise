import math

import pytest

from rescue_vision.config import Config
from rescue_vision.geometry import (
    bearing_deg,
    distance_m,
    distance_validity,
    focal_px,
    iou,
    measure,
)
from rescue_vision.types import BBox

CFG = Config()


def test_bearing_is_zero_at_frame_centre():
    assert bearing_deg(320.0, 640, 53.5) == pytest.approx(0.0)


def test_bearing_is_negative_left_of_centre():
    """PRD Appendix B: negative == person is LEFT of centre."""
    assert bearing_deg(100.0, 640, 53.5) < 0.0


def test_bearing_is_positive_right_of_centre():
    assert bearing_deg(540.0, 640, 53.5) > 0.0


def test_bearing_at_frame_edge_is_half_the_hfov():
    """The tan form must return exactly +/-HFOV/2 at the frame edges."""
    assert bearing_deg(640.0, 640, 53.5) == pytest.approx(53.5 / 2, abs=1e-6)
    assert bearing_deg(0.0, 640, 53.5) == pytest.approx(-53.5 / 2, abs=1e-6)


def test_bearing_is_antisymmetric_about_centre():
    left = bearing_deg(320.0 - 150.0, 640, 53.5)
    right = bearing_deg(320.0 + 150.0, 640, 53.5)
    assert left == pytest.approx(-right)


def test_tan_form_differs_from_linear_approximation_mid_frame():
    """Guards against someone 'simplifying' back to the linear form.

    The two forms agree exactly at the centre and at both edges -- that is
    forced by construction, since the tan form is defined to hit +/-HFOV/2 at
    the edges. They diverge in BETWEEN, peaking around the middle of each half.
    So probe at cx = 480, three quarters across a 640 px frame, where the
    linear approximation under-reads by roughly 0.8 degrees.
    """
    linear = ((480.0 / 640) - 0.5) * 53.5
    assert abs(bearing_deg(480.0, 640, 53.5) - linear) > 0.5


def test_tan_and_linear_forms_agree_at_centre_and_edges():
    """Pins the property that makes the previous test probe mid-frame."""
    for cx in (0.0, 320.0, 640.0):
        linear = ((cx / 640) - 0.5) * 53.5
        assert bearing_deg(cx, 640, 53.5) == pytest.approx(linear, abs=1e-6)


def test_focal_px_matches_the_pinhole_formula():
    expected = 640 / (2 * math.tan(math.radians(53.5) / 2))
    assert focal_px(640, 53.5) == pytest.approx(expected)


def test_distance_of_a_person_filling_the_frame_height():
    f = focal_px(640, 53.5)
    assert distance_m(480.0, f, 1.7) == pytest.approx(1.7 * f / 480.0)


def test_distance_halves_when_bbox_height_doubles():
    f = focal_px(640, 53.5)
    assert distance_m(200.0, f, 1.7) == pytest.approx(2 * distance_m(400.0, f, 1.7))


def test_distance_of_zero_height_bbox_is_infinite_not_a_crash():
    assert math.isinf(distance_m(0.0, 500.0, 1.7))


def test_distance_invalid_when_bbox_clipped_at_bottom():
    bbox = BBox(300.0, 100.0, 340.0, 479.0)
    valid, reason = distance_validity(bbox, 640, 480, 3.0, CFG)
    assert valid is False
    assert reason == "clipped_bottom"


def test_distance_invalid_when_bbox_clipped_at_top():
    bbox = BBox(300.0, 1.0, 340.0, 300.0)
    valid, reason = distance_validity(bbox, 640, 480, 3.0, CFG)
    assert valid is False
    assert reason == "clipped_top"


def test_distance_invalid_for_lying_down_person():
    """A prone person is the actual rescue target -- flag, never trust."""
    bbox = BBox(100.0, 200.0, 400.0, 260.0)  # 300 wide, 60 tall -> ratio 0.2
    valid, reason = distance_validity(bbox, 640, 480, 3.0, CFG)
    assert valid is False
    assert reason == "not_upright"


def test_distance_invalid_when_bbox_too_short():
    bbox = BBox(300.0, 200.0, 320.0, 230.0)  # 30 px tall, below the 40 px floor
    valid, reason = distance_validity(bbox, 640, 480, 3.0, CFG)
    assert valid is False
    assert reason == "bbox_too_small"


def test_distance_invalid_when_out_of_physical_range():
    bbox = BBox(300.0, 100.0, 340.0, 300.0)
    valid, reason = distance_validity(bbox, 640, 480, 99.0, CFG)
    assert valid is False
    assert reason == "implausible_distance"


def test_distance_valid_for_upright_fully_visible_person():
    bbox = BBox(300.0, 100.0, 360.0, 400.0)  # 60x300, ratio 5.0, no edge contact
    valid, reason = distance_validity(bbox, 640, 480, 3.0, CFG)
    assert valid is True
    assert reason is None


def test_measure_combines_bearing_and_distance():
    bbox = BBox(100.0, 100.0, 160.0, 400.0)
    m = measure(bbox, 640, 480, CFG)
    assert m.bearing_deg < 0.0  # left of centre
    assert m.distance_m > 0.0
    assert m.distance_valid is True


def test_iou_of_identical_boxes_is_one():
    b = BBox(0.0, 0.0, 10.0, 10.0)
    assert iou(b, b) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert iou(BBox(0.0, 0.0, 10.0, 10.0), BBox(50.0, 50.0, 60.0, 60.0)) == 0.0


def test_iou_of_half_overlapping_boxes():
    a = BBox(0.0, 0.0, 10.0, 10.0)
    b = BBox(5.0, 0.0, 15.0, 10.0)
    assert iou(a, b) == pytest.approx(50.0 / 150.0)
