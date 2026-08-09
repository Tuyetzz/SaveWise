"""The rover relays, end to end over real WebSockets (TestClient).

The hubs in app.rover are module-global (one rover, one backend process), so
these tests exercise them through a minimal app that mounts only the rover
router — no whisper model, no database.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import rover


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(rover.router)
    # Reset the module-global hubs so tests don't leak state into each other.
    rover.video_hub = rover.VideoHub()
    rover.annotated_hub = rover.VideoHub()
    rover.rover_hub = rover.RoverHub()
    with TestClient(app) as c:
        yield c


def recv_until(ws, t: str, tries: int = 10) -> dict:
    """Skip unrelated frames (the agent socket also gets 0.5 s refreshes)."""
    for _ in range(tries):
        msg = json.loads(ws.receive_text())
        if msg.get("t") == t:
            return msg
    raise AssertionError(f"no {t!r} message in {tries} frames")


def test_control_sees_pi_connect_and_commands_reach_the_agent(client):
    with client.websocket_connect("/api/ws/rover/control") as control:
        first = recv_until(control, "status")
        assert first["pi_connected"] is False

        with client.websocket_connect("/api/ws/rover/agent") as agent:
            assert recv_until(control, "status")["pi_connected"] is True
            # On connect the agent learns the current command.
            assert recv_until(agent, "cmd")["cmd"] == "stop"

            control.send_text(json.dumps({"t": "cmd", "cmd": "forward"}))
            for _ in range(10):
                if recv_until(agent, "cmd")["cmd"] == "forward":
                    break
            else:
                raise AssertionError("forward never reached the agent")
            assert recv_until(control, "status")["cmd"] == "forward"

            # Switching to auto tells the Pi to hand over to the Arduino loop.
            control.send_text(json.dumps({"t": "mode", "mode": "auto"}))
            for _ in range(10):
                if recv_until(agent, "cmd")["cmd"] == "auto":
                    break
            else:
                raise AssertionError("auto never reached the agent")


def test_last_controller_leaving_stops_the_rover(client):
    with client.websocket_connect("/api/ws/rover/agent") as agent:
        recv_until(agent, "cmd")
        with client.websocket_connect("/api/ws/rover/control") as control:
            recv_until(control, "status")
            control.send_text(json.dumps({"t": "cmd", "cmd": "forward"}))
            assert recv_until(control, "status")["cmd"] == "forward"
        # Controller gone: the dead-man's switch must reset to stop.
        for _ in range(10):
            if recv_until(agent, "cmd")["cmd"] == "stop":
                break
        else:
            raise AssertionError("stop never reached the agent")
    assert rover.rover_hub.cmd == "stop"


def test_manual_commands_are_ignored_in_auto_mode(client):
    with client.websocket_connect("/api/ws/rover/control") as control:
        recv_until(control, "status")
        control.send_text(json.dumps({"t": "mode", "mode": "auto"}))
        assert recv_until(control, "status")["mode"] == "auto"
        control.send_text(json.dumps({"t": "cmd", "cmd": "forward"}))
        control.send_text(json.dumps({"t": "mode", "mode": "manual"}))
        status = recv_until(control, "status")
        assert status["mode"] == "manual"
        assert status["cmd"] == "stop"  # the forward never took


def test_video_frames_relay_from_upload_to_feed(client):
    with client.websocket_connect("/api/ws/video/feed") as feed:
        with client.websocket_connect("/api/ws/video/upload") as upload:
            upload.send_bytes(b"frame-1")
            assert feed.receive_bytes() == b"frame-1"
            upload.send_bytes(b"frame-2")
            assert feed.receive_bytes() == b"frame-2"

    status = client.get("/api/rover/status").json()
    assert status["video_frames_relayed"] == 2
    assert status["video_uploader"] is False


def test_annotated_channel_is_independent_of_the_raw_one(client):
    with (
        client.websocket_connect("/api/ws/video/feed") as raw_feed,
        client.websocket_connect("/api/ws/video/annotated/feed") as annotated_feed,
        client.websocket_connect("/api/ws/video/upload") as phone,
        client.websocket_connect("/api/ws/video/annotated/upload") as detector,
    ):
        phone.send_bytes(b"raw-frame")
        detector.send_bytes(b"boxed-frame")
        assert raw_feed.receive_bytes() == b"raw-frame"
        assert annotated_feed.receive_bytes() == b"boxed-frame"

    status = client.get("/api/rover/status").json()
    assert status["video_frames_relayed"] == 1
    assert status["annotated_frames_relayed"] == 1
