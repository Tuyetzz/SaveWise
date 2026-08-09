import cv2
import numpy as np
import pytest

from rescue_vision.config import Config
from rescue_vision.frame_source import VideoFileSource, create_frame_source

CFG = Config()


@pytest.fixture
def clip(tmp_path):
    """A 6-frame 64x48 synthetic clip."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(6):
        writer.write(np.full((48, 64, 3), i * 10, np.uint8))
    writer.release()
    return path


def test_video_file_source_reports_its_dimensions(clip):
    src = VideoFileSource(str(clip))
    assert src.width == 64
    assert src.height == 48
    src.close()


def test_video_file_source_yields_frames_then_stops(clip):
    src = VideoFileSource(str(clip))
    frames = list(src)
    src.close()
    assert len(frames) == 6
    assert frames[0].shape == (48, 64, 3)


def test_read_returns_none_at_end_of_stream(clip):
    src = VideoFileSource(str(clip))
    for _ in range(6):
        src.read()
    assert src.read() is None
    src.close()


def test_missing_file_raises_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoFileSource(str(tmp_path / "nope.mp4"))


def test_factory_selects_a_file_source_for_a_path(clip):
    src = create_frame_source(str(clip), CFG)
    assert isinstance(src, VideoFileSource)
    src.close()


def test_factory_selects_a_webcam_for_a_digit_string(monkeypatch):
    """No real camera in CI, so intercept the constructor and assert dispatch."""
    import rescue_vision.frame_source as fs

    captured = {}

    class FakeWebcam:
        def __init__(self, index, cfg):
            captured["index"] = index

    monkeypatch.setattr(fs, "WebcamSource", FakeWebcam)
    fs.create_frame_source("2", CFG)
    assert captured["index"] == 2


def test_factory_selects_the_pi_camera_for_the_picamera_spec(monkeypatch):
    import rescue_vision.frame_source as fs

    captured = {}

    class FakePiCamera:
        def __init__(self, cfg):
            captured["built"] = True

    monkeypatch.setattr(fs, "PiCameraSource", FakePiCamera)
    fs.create_frame_source("picamera", CFG)
    assert captured["built"] is True


def test_factory_selects_the_ws_source_for_a_ws_url(monkeypatch):
    import rescue_vision.frame_source as fs

    captured = {}

    class FakeWs:
        def __init__(self, url, cfg):
            captured["url"] = url

    monkeypatch.setattr(fs, "WebSocketFrameSource", FakeWs)
    fs.create_frame_source("wss://example.test/api/ws/video/feed", CFG)
    assert captured["url"] == "wss://example.test/api/ws/video/feed"


def test_factory_rejects_a_missing_path():
    with pytest.raises(FileNotFoundError):
        create_frame_source("definitely_not_here.mp4", CFG)
