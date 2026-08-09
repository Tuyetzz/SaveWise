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


# --- WsPublisher: annotated frames pushed to the backend relay ---


class FakeWs:
    def __init__(self):
        self.frames = []
        self.closed = False
        self.fail_next_send = False

    def send_binary(self, data):
        if self.fail_next_send:
            raise OSError("link down")
        self.frames.append(data)

    def close(self):
        self.closed = True


def test_publisher_sends_jpeg_frames():
    from rescue_vision.preview import WsPublisher

    ws = FakeWs()
    pub = WsPublisher("ws://backend/annotated", connect=lambda: ws)
    pub.publish(np.zeros((48, 64, 3), np.uint8), now=0.0)
    assert pub.sent == 1
    assert ws.frames[0].startswith(b"\xff\xd8")  # JPEG start-of-image


def test_publisher_survives_backend_being_down_and_throttles_retries():
    from rescue_vision.preview import WsPublisher

    attempts = []

    def connect():
        attempts.append(1)
        raise OSError("connection refused")

    pub = WsPublisher("ws://backend/annotated", connect=connect)
    frame = np.zeros((48, 64, 3), np.uint8)
    pub.publish(frame, now=0.0)  # tries, fails
    pub.publish(frame, now=1.0)  # inside the cooldown: no attempt
    pub.publish(frame, now=4.0)  # cooldown over: tries again
    assert len(attempts) == 2
    assert pub.sent == 0


def test_publisher_reconnects_after_a_send_failure():
    from rescue_vision.preview import WsPublisher

    sockets = []

    def connect():
        ws = FakeWs()
        sockets.append(ws)
        return ws

    pub = WsPublisher("ws://backend/annotated", connect=connect)
    frame = np.zeros((48, 64, 3), np.uint8)
    pub.publish(frame, now=0.0)
    sockets[0].fail_next_send = True
    pub.publish(frame, now=1.0)  # send fails; socket dropped
    assert sockets[0].closed
    pub.publish(frame, now=10.0)  # past cooldown: new socket, delivers
    assert len(sockets) == 2
    assert pub.sent == 2
