import numpy as np

from rescue_vision.detector import ScriptedDetector
from rescue_vision.types import BBox, RawDetection

FRAME = np.zeros((480, 640, 3), np.uint8)


def test_scripted_detector_replays_its_script():
    d1 = [RawDetection(BBox(0.0, 0.0, 10.0, 30.0), 0.9, 1)]
    d2 = [RawDetection(BBox(5.0, 0.0, 15.0, 30.0), 0.9, 1)]
    det = ScriptedDetector([d1, d2])
    assert det.scan(FRAME) == d1
    assert det.scan(FRAME) == d2


def test_scripted_detector_returns_nothing_past_the_end_of_its_script():
    det = ScriptedDetector([[RawDetection(BBox(0.0, 0.0, 10.0, 30.0), 0.9, 1)]])
    det.scan(FRAME)
    assert det.scan(FRAME) == []


def test_scripted_confirm_echoes_the_last_scan_boxes():
    boxes = [RawDetection(BBox(0.0, 0.0, 10.0, 30.0), 0.9, 1)]
    det = ScriptedDetector([boxes])
    det.scan(FRAME)
    assert det.confirm(FRAME) == [boxes[0].bbox]


def test_scripted_confirm_returns_nothing_when_confirm_all_is_false():
    boxes = [RawDetection(BBox(0.0, 0.0, 10.0, 30.0), 0.9, 1)]
    det = ScriptedDetector([boxes], confirm_all=False)
    det.scan(FRAME)
    assert det.confirm(FRAME) == []


def test_confirm_is_skipped_when_the_scan_pass_found_nothing():
    """PRD 6.5 escalation rule -- no candidates means no confirm pass."""
    det = ScriptedDetector([])
    assert det.should_confirm(has_candidates=False, now=100.0) is False


def test_confirm_runs_on_the_first_candidate_frame():
    det = ScriptedDetector([])
    assert det.should_confirm(has_candidates=True, now=100.0) is True


def test_confirm_is_rate_limited_to_protect_fps():
    det = ScriptedDetector([], confirm_min_interval=0.15)
    assert det.should_confirm(True, now=100.0) is True
    assert det.should_confirm(True, now=100.05) is False
    assert det.should_confirm(True, now=100.20) is True


def test_a_declined_confirm_does_not_reset_the_rate_limit_clock():
    """Otherwise a busy scene could starve the confirm pass indefinitely."""
    det = ScriptedDetector([], confirm_min_interval=0.15)
    assert det.should_confirm(True, now=100.0) is True
    for t in [100.02, 100.05, 100.10, 100.14]:
        assert det.should_confirm(True, now=t) is False
    assert det.should_confirm(True, now=100.16) is True
