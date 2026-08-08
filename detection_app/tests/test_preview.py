import urllib.request

import numpy as np
import pytest

from rescue_vision.preview import MjpegServer


@pytest.fixture
def server():
    s = MjpegServer(port=0)  # port 0 -> the OS picks a free port
    s.start()
    yield s
    s.stop()


def test_server_reports_a_url(server):
    assert server.url.startswith("http://")


def test_publish_before_any_client_connects_does_not_raise(server):
    server.publish(np.zeros((48, 64, 3), np.uint8))


def test_stream_endpoint_serves_multipart_jpeg(server):
    server.publish(np.zeros((48, 64, 3), np.uint8))
    url = f"http://127.0.0.1:{server._port}/stream"
    with urllib.request.urlopen(url, timeout=5) as resp:
        assert "multipart/x-mixed-replace" in resp.headers["Content-Type"]
        chunk = resp.read(512)
    assert b"\xff\xd8" in chunk  # JPEG start-of-image marker


def test_index_page_embeds_the_stream(server):
    url = f"http://127.0.0.1:{server._port}/"
    with urllib.request.urlopen(url, timeout=5) as resp:
        body = resp.read().decode()
    assert "/stream" in body


def test_publish_keeps_only_the_latest_frame(server):
    """A slow viewer must never back-pressure the detection loop."""
    for i in range(50):
        server.publish(np.full((48, 64, 3), i, np.uint8))
    assert server.pending_frames() <= 1
