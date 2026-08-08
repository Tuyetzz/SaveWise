import json

import numpy as np
import pytest

from rescue_vision.config import Config
from rescue_vision.events import EventWriter, build_event
from rescue_vision.types import BBox, Command, TrackState

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


def test_event_matches_the_prd_schema():
    ev = build_event(
        track=track(),
        target_id=3,
        command=Command(turn=-0.31, drive=0.0),
        frame_index=412,
        timestamp=1754640000.123,
        annotated_frame="detections/frame_000412.jpg",
    )
    assert ev["schema"] == "rescue.detection.v1"
    assert ev["frame_index"] == 412
    assert ev["track_id"] == 3
    assert ev["bbox_xyxy"] == [312, 118, 466, 502]
    assert ev["bearing_deg"] == pytest.approx(-12.4)
    assert ev["is_target"] is True
    assert ev["turn_command"] == pytest.approx(-0.31)
    assert ev["annotated_frame"] == "detections/frame_000412.jpg"


def test_non_target_track_is_reported_but_not_flagged():
    ev = build_event(track(track_id=9), 3, Command(0.0, 0.0), 1, 0.0, None)
    assert ev["is_target"] is False
    assert ev["annotated_frame"] is None


def test_commands_are_repeated_on_every_row():
    """A consumer reading a single line must have everything it needs."""
    ev = build_event(track(track_id=9), 3, Command(-0.5, 0.3), 1, 0.0, None)
    assert ev["turn_command"] == pytest.approx(-0.5)
    assert ev["drive_command"] == pytest.approx(0.3)


def test_invalid_distance_is_reported_with_its_reason():
    t = track(distance_valid=False)
    t.invalid_reason = "not_upright"
    ev = build_event(t, 3, Command(0.0, 0.0), 1, 0.0, None)
    assert ev["distance_valid"] is False
    assert ev["invalid_reason"] == "not_upright"
    assert ev["distance_m"] is None


def test_logged_confidence_is_the_live_value_not_the_sampled_one():
    """Amendment A: displays are held at 1 Hz, logs stay per-frame."""
    t = track()
    t.confidence = 0.41
    t.display_confidence = 0.87
    ev = build_event(t, 3, Command(0.0, 0.0), 1, 0.0, None)
    assert ev["confidence"] == pytest.approx(0.41)


def test_emit_writes_one_json_object_per_line(tmp_path):
    writer = EventWriter(tmp_path / "events.jsonl", tmp_path / "frames", CFG)
    writer.emit([build_event(track(), 3, Command(0.0, 0.0), 1, 0.0, None)])
    writer.emit([build_event(track(), 3, Command(0.0, 0.0), 2, 0.0, None)])
    writer.close()

    lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["frame_index"] == 1
    assert json.loads(lines[1])["frame_index"] == 2


def test_frame_saving_is_rate_limited_per_track(tmp_path):
    """Saved frames are the one unbounded item in the 16 GB disk budget."""
    writer = EventWriter(tmp_path / "e.jsonl", tmp_path / "frames", CFG)
    assert writer.should_save_frame(track_id=1, now=100.0) is True
    writer.save_frame(np.zeros((10, 10, 3), np.uint8), 1, 0, now=100.0)
    assert writer.should_save_frame(track_id=1, now=100.5) is False
    assert writer.should_save_frame(track_id=1, now=101.5) is True
    writer.close()


def test_different_tracks_have_independent_rate_limits(tmp_path):
    writer = EventWriter(tmp_path / "e.jsonl", tmp_path / "frames", CFG)
    writer.save_frame(np.zeros((10, 10, 3), np.uint8), 1, 0, now=100.0)
    assert writer.should_save_frame(track_id=2, now=100.1) is True
    writer.close()


def test_save_frame_returns_a_relative_path_and_writes_a_file(tmp_path):
    writer = EventWriter(tmp_path / "e.jsonl", tmp_path / "frames", CFG)
    rel = writer.save_frame(np.zeros((20, 20, 3), np.uint8), 7, 412, now=1.0)
    writer.close()
    assert rel is not None
    assert "412" in rel
    assert (tmp_path / rel).exists()


def test_save_frame_returns_none_when_saving_is_disabled(tmp_path):
    cfg = Config(save_frames=False)
    writer = EventWriter(tmp_path / "e.jsonl", tmp_path / "frames", cfg)
    assert writer.save_frame(np.zeros((10, 10, 3), np.uint8), 1, 0, now=1.0) is None
    writer.close()


def test_disk_cap_deletes_oldest_frames_first(tmp_path):
    """Cap set below one frame, so every save is immediately evicted."""
    cfg = Config(max_output_dir_mb=0)
    frames = tmp_path / "frames"
    writer = EventWriter(tmp_path / "e.jsonl", frames, cfg)
    for i in range(5):
        writer.save_frame(np.zeros((50, 50, 3), np.uint8), 1, i, now=float(i) * 2)
    writer.enforce_disk_cap()
    writer.close()
    assert len(list(frames.glob("*.jpg"))) == 0


def test_frames_survive_under_a_generous_cap(tmp_path):
    frames = tmp_path / "frames"
    writer = EventWriter(tmp_path / "e.jsonl", frames, CFG)
    for i in range(3):
        writer.save_frame(np.zeros((50, 50, 3), np.uint8), 1, i, now=float(i) * 2)
    writer.close()
    assert len(list(frames.glob("*.jpg"))) == 3
