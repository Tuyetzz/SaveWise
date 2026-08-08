import json

import numpy as np
import pytest

from rescue_vision.config import Config
from rescue_vision.detector import ScriptedDetector
from rescue_vision.events import EventWriter
from rescue_vision.pipeline import Pipeline
from rescue_vision.rover import ConsoleRover
from rescue_vision.types import BBox, RawDetection

CFG = Config(save_frames=False, n_confirm=1)
FRAME = np.zeros((480, 640, 3), np.uint8)


class FakeClock:
    """Advances 0.1 s per call so nothing depends on wall time."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 0.1
        return self.t


def person_at(cx: float, track_id: int = 1) -> RawDetection:
    """An upright, fully visible person centred on cx."""
    return RawDetection(BBox(cx - 30.0, 100.0, cx + 30.0, 400.0), 0.9, track_id)


def build(tmp_path, script, cfg=CFG):
    detector = ScriptedDetector(script)
    rover = ConsoleRover(cfg, sink=lambda _: None)
    writer = EventWriter(tmp_path / "events.jsonl", tmp_path / "frames", cfg)
    pipeline = Pipeline(detector, rover, writer, cfg, clock=FakeClock())
    return pipeline, rover, writer


def test_person_on_the_left_yields_negative_bearing_and_a_left_turn(tmp_path):
    """PRD 6.10 acceptance test: the bearing sign must be correct."""
    pipeline, rover, writer = build(tmp_path, [[person_at(100.0)]] * 3)
    result = None
    for i in range(3):
        result = pipeline.process_frame(FRAME, i)
    writer.close()
    rover.close()

    assert result.target is not None
    assert result.target.bearing_deg < 0.0
    assert result.command.turn < 0.0  # same sign as bearing


def test_person_on_the_right_yields_positive_bearing_and_a_right_turn(tmp_path):
    pipeline, rover, writer = build(tmp_path, [[person_at(540.0)]] * 3)
    result = None
    for i in range(3):
        result = pipeline.process_frame(FRAME, i)
    writer.close()
    rover.close()

    assert result.target.bearing_deg > 0.0
    assert result.command.turn > 0.0


def test_centred_person_produces_no_turn(tmp_path):
    pipeline, rover, writer = build(tmp_path, [[person_at(320.0)]] * 3)
    result = None
    for i in range(3):
        result = pipeline.process_frame(FRAME, i)
    writer.close()
    rover.close()

    assert abs(result.target.bearing_deg) <= CFG.deadband_deg
    assert result.command.turn == 0.0


def test_no_detections_commands_a_full_stop(tmp_path):
    pipeline, rover, writer = build(tmp_path, [[], [], []])
    result = None
    for i in range(3):
        result = pipeline.process_frame(FRAME, i)
    writer.close()
    rover.close()

    assert result.target is None
    assert result.command.turn == 0.0
    assert result.command.drive == 0.0


def test_exactly_one_row_per_frame_is_flagged_as_target(tmp_path):
    script = [[person_at(200.0, 1), person_at(450.0, 2)]] * 3
    pipeline, rover, writer = build(tmp_path, script)
    for i in range(3):
        pipeline.process_frame(FRAME, i)
    writer.close()
    rover.close()

    lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert lines
    by_frame: dict[int, int] = {}
    for line in lines:
        row = json.loads(line)
        assert row["schema"] == "rescue.detection.v1"
        by_frame[row["frame_index"]] = by_frame.get(row["frame_index"], 0) + int(
            row["is_target"]
        )
    assert set(by_frame.values()) == {1}


def test_unconfirmed_tracks_are_not_written_to_the_log(tmp_path):
    """One alert per confirmed track, never one per raw detection."""
    cfg = Config(save_frames=False, n_confirm=99)  # never promotes
    pipeline, rover, writer = build(tmp_path, [[person_at(320.0)]] * 3, cfg=cfg)
    for i in range(3):
        pipeline.process_frame(FRAME, i)
    writer.close()
    rover.close()
    assert (tmp_path / "events.jsonl").read_text().strip() == ""


def test_a_vanished_person_stops_the_rover_and_stops_the_log(tmp_path):
    """Found by the acceptance run: a track the detector stopped reporting used
    to linger for track_max_age_frames, fabricating detection rows and holding
    a turn command for ~3 s. PRD FR10 wants an explicit stop when nobody is
    tracked."""
    script = [[person_at(100.0)], [person_at(100.0)], [], [], []]
    pipeline, rover, writer = build(tmp_path, script)
    results = [pipeline.process_frame(FRAME, i) for i in range(5)]
    writer.close()
    rover.close()

    assert results[1].target is not None
    assert results[1].command.turn < 0.0

    for r in results[2:]:
        assert r.target is None
        assert r.command.turn == 0.0
        assert r.command.drive == 0.0
        assert r.rows == []

    logged = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().strip().splitlines()
    ]
    assert max(r["frame_index"] for r in logged) == 1


def test_a_stale_track_is_not_drawn_on_the_annotated_frame(tmp_path):
    """Drawing a box where nobody is looks like a detection bug to a judge."""
    script = [[person_at(320.0)], []]
    pipeline, rover, writer = build(tmp_path, script)
    pipeline.process_frame(FRAME, 0)
    second = pipeline.process_frame(FRAME, 1)
    writer.close()
    rover.close()
    assert second.tracks == []


def test_a_failing_frame_does_not_kill_the_run(tmp_path):
    """NFR4: log and continue."""

    class ExplodingDetector(ScriptedDetector):
        def scan(self, frame):
            if self._index == 1:
                self._index += 1
                raise RuntimeError("simulated inference failure")
            return super().scan(frame)

    detector = ExplodingDetector([[person_at(320.0)]] * 4)
    rover = ConsoleRover(CFG, sink=lambda _: None)
    writer = EventWriter(tmp_path / "e.jsonl", tmp_path / "frames", CFG)
    pipeline = Pipeline(detector, rover, writer, CFG, clock=FakeClock())

    processed = pipeline.run(iter([FRAME] * 4))
    writer.close()
    rover.close()
    assert processed == 4


def test_run_stops_the_rover_when_the_stream_ends(tmp_path):
    pipeline, rover, writer = build(tmp_path, [[person_at(100.0)]] * 3)
    pipeline.run(iter([FRAME] * 3))
    writer.close()
    assert rover.commands[-1] == (0.0, 0.0)
    rover.close()


def test_run_stops_the_rover_on_an_unhandled_source_error(tmp_path):
    """Motors must never be left running by a crash."""

    def exploding_source():
        yield FRAME
        raise RuntimeError("camera died")

    pipeline, rover, writer = build(tmp_path, [[person_at(100.0)]] * 3)
    with pytest.raises(RuntimeError):
        pipeline.run(exploding_source())
    writer.close()
    assert rover.commands[-1] == (0.0, 0.0)
    rover.close()


def test_max_frames_limits_the_run(tmp_path):
    pipeline, rover, writer = build(tmp_path, [[person_at(100.0)]] * 10)
    processed = pipeline.run(iter([FRAME] * 10), max_frames=4)
    writer.close()
    rover.close()
    assert processed == 4
