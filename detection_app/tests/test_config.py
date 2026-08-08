import dataclasses

import pytest

from rescue_vision.config import Config
from rescue_vision.types import BBox


def test_config_defaults_match_prd_appendix_a():
    c = Config()
    assert c.hfov_deg == 53.5
    assert c.assumed_human_height_m == 1.7
    assert c.person_class_id == 0
    assert c.scan_conf == 0.25
    assert c.confirm_conf == 0.45
    assert c.n_confirm == 2


def test_config_carries_no_control_or_motor_constants():
    """The rover drives itself. Nothing here may command motion."""
    fields = set(Config().__dataclass_fields__)
    for gone in (
        "kp",
        "deadband_deg",
        "min_turn",
        "target_hold",
        "search_hold",
        "stop_distance_m",
        "approach_speed",
        "watchdog_timeout",
    ):
        assert gone not in fields


def test_displayed_confidence_is_sampled_once_per_second():
    """Amendment A: a number redrawn at 10 Hz is unreadable."""
    assert Config().confidence_sample_interval == 1.0


def test_config_is_immutable():
    c = Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.scan_conf = 0.5


def test_bbox_geometry_properties():
    b = BBox(10.0, 20.0, 40.0, 120.0)
    assert b.width == 30.0
    assert b.height == 100.0
    assert b.cx == 25.0
    assert b.cy == 70.0
    assert b.area == 3000.0
    assert b.aspect_ratio == pytest.approx(100.0 / 30.0)


def test_bbox_aspect_ratio_of_zero_width_is_zero_not_a_crash():
    assert BBox(5.0, 5.0, 5.0, 50.0).aspect_ratio == 0.0
