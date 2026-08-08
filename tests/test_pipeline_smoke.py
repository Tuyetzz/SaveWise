import json

import numpy as np
import pytest

from rescue_vision.config import Config
from rescue_vision.detector import ScriptedDetector
from rescue_vision.events import RawEventWriter
from rescue_vision.pipeline import Pipeline
from rescue_vision.sightings import SightingRecorder
from rescue_vision.types import BBox, RawDetection

CFG = Config(n_confirm=1)
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


def build(tmp_path, script, cfg=CFG, raw=False):
    detector = ScriptedDetector(script)
    recorder = SightingRecorder(cfg, tmp_path)
    writer = RawEventWriter(tmp_path / "events.jsonl", cfg) if raw else None
    pipeline = Pipeline(detector, recorder, cfg, writer, clock=FakeClock())
    return pipeline, recorder, writer


def sightings(tmp_path):
    path = tmp_path / "sightings.jsonl"
    if not path.exists():
        return []
    text = path.read_text().strip()
    return [json.loads(x) for x in text.splitlines()] if text else []


def test_driving_past_one_person_logs_exactly_one_sighting(tmp_path):
    """The pivot in one test: 40 frames of one person is one journey record."""
    pipeline, _, _ = build(tmp_path, [[person_at(320.0)]] * 40)
    pipeline.run(iter([FRAME] * 40))
    rows = sightings(tmp_path)
    assert len(rows) == 1
    assert rows[0]["frames_seen"] == 40
    assert rows[0]["schema"] == "rescue.sighting.v1"


def test_two_people_in_frame_log_two_sightings(tmp_path):
    script = [[person_at(200.0, 1), person_at(450.0, 2)]] * 5
    pipeline, _, _ = build(tmp_path, script)
    pipeline.run(iter([FRAME] * 5))
    assert len(sightings(tmp_path)) == 2


def test_bearing_sign_is_preserved_in_the_log(tmp_path):
    """Negative == person was to the LEFT of the rover's heading."""
    pipeline, _, _ = build(tmp_path, [[person_at(100.0)]] * 5)
    pipeline.run(iter([FRAME] * 5))
    assert sightings(tmp_path)[0]["bearing_at_peak_deg"] < 0.0


def test_a_person_on_the_right_logs_a_positive_bearing(tmp_path):
    pipeline, _, _ = build(tmp_path, [[person_at(540.0)]] * 5)
    pipeline.run(iter([FRAME] * 5))
    assert sightings(tmp_path)[0]["bearing_at_peak_deg"] > 0.0


def test_an_empty_journey_logs_nothing(tmp_path):
    pipeline, _, _ = build(tmp_path, [[], [], []])
    pipeline.run(iter([FRAME] * 3))
    assert sightings(tmp_path) == []


def test_a_brief_detection_gap_does_not_close_a_sighting(tmp_path):
    """The person is still there; the detector just missed a frame."""
    script = [[person_at(320.0)], [], [person_at(320.0)], [person_at(320.0)]]
    pipeline, _, _ = build(tmp_path, script)
    results = [pipeline.process_frame(FRAME, i) for i in range(4)]
    assert all(r.sightings_so_far == 0 for r in results)
    assert sightings(tmp_path) == []


def test_a_person_leaving_frame_closes_their_sighting_after_the_grace_period(tmp_path):
    cfg = Config(n_confirm=1, sighting_gap_s=0.25)  # FakeClock steps 0.1s
    script = [[person_at(320.0)], [person_at(320.0)], [], [], [], []]
    pipeline, _, _ = build(tmp_path, script, cfg=cfg)
    results = [pipeline.process_frame(FRAME, i) for i in range(6)]
    assert results[1].sightings_so_far == 0  # still visible
    assert results[-1].sightings_so_far == 1  # gone long enough, record written
    assert results[-1].tracks == []


def test_unconfirmed_tracks_never_reach_the_journey_log(tmp_path):
    cfg = Config(n_confirm=99)  # never promotes
    pipeline, _, _ = build(tmp_path, [[person_at(320.0)]] * 3, cfg=cfg)
    pipeline.run(iter([FRAME] * 3))
    assert sightings(tmp_path) == []


def test_one_image_is_saved_per_sighting_not_per_frame(tmp_path):
    """Retires the PRD disk risk: bounded by people seen, not frames run."""
    script = [[person_at(200.0, 1), person_at(450.0, 2)]] * 40
    pipeline, _, _ = build(tmp_path, script)
    pipeline.run(iter([FRAME] * 40))
    assert len(list((tmp_path / "sightings").glob("*.jpg"))) == 2


def test_no_raw_log_is_written_by_default(tmp_path):
    pipeline, _, _ = build(tmp_path, [[person_at(320.0)]] * 3)
    pipeline.run(iter([FRAME] * 3))
    assert not (tmp_path / "events.jsonl").exists()


def test_raw_log_records_every_frame_when_enabled(tmp_path):
    pipeline, _, writer = build(tmp_path, [[person_at(320.0)]] * 3, raw=True)
    pipeline.run(iter([FRAME] * 3))
    writer.close()
    rows = [
        json.loads(x)
        for x in (tmp_path / "events.jsonl").read_text().strip().splitlines()
    ]
    assert len(rows) == 3
    assert rows[0]["schema"] == "rescue.detection.v2"


def test_raw_log_carries_no_command_fields(tmp_path):
    """This subsystem no longer commands the rover."""
    pipeline, _, writer = build(tmp_path, [[person_at(320.0)]] * 2, raw=True)
    pipeline.run(iter([FRAME] * 2))
    writer.close()
    row = json.loads((tmp_path / "events.jsonl").read_text().strip().splitlines()[0])
    for gone in ("turn_command", "drive_command", "is_target"):
        assert gone not in row


def test_a_failing_frame_does_not_kill_the_run(tmp_path):
    """NFR4: log and continue."""

    class ExplodingDetector(ScriptedDetector):
        def scan(self, frame):
            if self._index == 1:
                self._index += 1
                raise RuntimeError("simulated inference failure")
            return super().scan(frame)

    detector = ExplodingDetector([[person_at(320.0)]] * 4)
    recorder = SightingRecorder(CFG, tmp_path)
    pipeline = Pipeline(detector, recorder, CFG, clock=FakeClock())
    assert pipeline.run(iter([FRAME] * 4)) == 4


def test_a_crash_still_finalises_the_journey_log(tmp_path):
    """Everything found before the failure must survive on disk."""

    def exploding_source():
        yield FRAME
        yield FRAME
        raise RuntimeError("camera died")

    pipeline, _, _ = build(tmp_path, [[person_at(320.0)]] * 3)
    with pytest.raises(RuntimeError):
        pipeline.run(exploding_source())
    assert len(sightings(tmp_path)) == 1


def test_max_frames_limits_the_run(tmp_path):
    pipeline, _, _ = build(tmp_path, [[person_at(100.0)]] * 10)
    assert pipeline.run(iter([FRAME] * 10), max_frames=4) == 4
