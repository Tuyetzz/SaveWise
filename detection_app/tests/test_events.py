import json

import pytest

from rescue_vision.config import Config
from rescue_vision.events import RawEventWriter, build_event
from rescue_vision.types import BBox, TrackState

CFG = Config()


def track(track_id=3, bearing_deg=-12.4, distance_m=3.2, distance_valid=True):
    return TrackState(
        track_id=track_id,
        bbox=BBox(312.0, 118.0, 466.0, 502.0),
        confidence=0.87,
        bearing_deg=bearing_deg,
        distance_m=distance_m,
        distance_valid=distance_valid,
        confirmed=True,
    )


def test_raw_event_carries_the_v2_schema():
    ev = build_event(track(), frame_index=412, timestamp=1.5)
    assert ev["schema"] == "rescue.detection.v2"
    assert ev["frame_index"] == 412
    assert ev["track_id"] == 3
    assert ev["bbox_xyxy"] == [312, 118, 466, 502]
    assert ev["bearing_deg"] == pytest.approx(-12.4)


def test_raw_event_has_no_command_or_target_fields():
    """This subsystem no longer steers the rover, so v1's fields are gone."""
    ev = build_event(track(), 1, 0.0)
    for gone in ("turn_command", "drive_command", "is_target"):
        assert gone not in ev


def test_invalid_distance_is_nulled_and_explained():
    t = track(distance_valid=False)
    t.invalid_reason = "not_upright"
    ev = build_event(t, 1, 0.0)
    assert ev["distance_valid"] is False
    assert ev["invalid_reason"] == "not_upright"
    assert ev["distance_m"] is None


def test_logged_confidence_is_the_live_value_not_the_sampled_one():
    """Amendment A: displays are held at 1 Hz, logs stay per-frame."""
    t = track()
    t.confidence = 0.41
    t.display_confidence = 0.87
    assert build_event(t, 1, 0.0)["confidence"] == pytest.approx(0.41)


def test_writer_emits_one_json_object_per_line(tmp_path):
    writer = RawEventWriter(tmp_path / "events.jsonl", CFG)
    writer.emit([build_event(track(), 1, 0.0)])
    writer.emit([build_event(track(), 2, 0.0)])
    writer.close()

    lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["frame_index"] == 1
    assert json.loads(lines[1])["frame_index"] == 2


def test_writer_creates_its_parent_directory(tmp_path):
    writer = RawEventWriter(tmp_path / "nested" / "events.jsonl", CFG)
    writer.emit([build_event(track(), 1, 0.0)])
    writer.close()
    assert (tmp_path / "nested" / "events.jsonl").exists()
