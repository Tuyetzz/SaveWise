"""Rover relays: phone video -> detection app, operator commands -> Pi.

The Pi camera is out of action (mismatched components), so the phone on the
rover is the camera: the front-end pushes JPEG frames to /api/ws/video/upload
and the detection app subscribes to /api/ws/video/feed. The backend only
relays — it never decodes a frame.

Driving works the same way in reverse: the operator UI sends commands to
/api/ws/rover/control, the Pi keeps a socket open on /api/ws/rover/agent and
mirrors whatever command is current onto GPIO pins for the Arduino (see pi/).

Both relays are latest-state, not queues: a slow subscriber skips frames, and
a Pi that reconnects gets the current mode/command immediately.
"""

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

COMMANDS = {"forward", "backward", "left", "right", "stop"}
MODES = {"manual", "auto"}


class VideoHub:
    """Latest-frame relay. The uploader overwrites `latest`; each feed client
    waits on the condition and sends whatever is newest when it wakes, so a
    slow detection pass naturally drops frames instead of building a queue."""

    def __init__(self) -> None:
        self.cond = asyncio.Condition()
        self.latest: bytes | None = None
        self.seq = 0
        self.uploader_connected = False
        self.viewers = 0

    async def publish(self, frame: bytes) -> None:
        async with self.cond:
            self.latest = frame
            self.seq += 1
            self.cond.notify_all()


# Two channels: "raw" is the phone's camera (consumed by the detection app
# and the admin console), "annotated" is the detection app's overlay — same
# frames with bounding boxes drawn on — viewed in the admin console.
video_hub = VideoHub()
annotated_hub = VideoHub()


async def _relay_upload(ws: WebSocket, hub: VideoHub) -> None:
    await ws.accept()
    hub.uploader_connected = True
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                await hub.publish(msg["bytes"])
    except WebSocketDisconnect:
        pass
    finally:
        hub.uploader_connected = False


async def _relay_feed(ws: WebSocket, hub: VideoHub) -> None:
    await ws.accept()
    hub.viewers += 1
    seen = hub.seq
    try:
        while True:
            async with hub.cond:
                await hub.cond.wait_for(lambda: hub.seq != seen)
                frame, seen = hub.latest, hub.seq
            await ws.send_bytes(frame)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.viewers -= 1


@router.websocket("/api/ws/video/upload")
async def video_upload(ws: WebSocket):
    """The phone on the rover. Binary messages are JPEG frames."""
    await _relay_upload(ws, video_hub)


@router.websocket("/api/ws/video/feed")
async def video_feed(ws: WebSocket):
    """Subscribers: the detection app (rescue_vision --source ws://...) and
    the admin console's raw view."""
    await _relay_feed(ws, video_hub)


@router.websocket("/api/ws/video/annotated/upload")
async def annotated_upload(ws: WebSocket):
    """The detection app (rescue_vision --publish ws://...): the same frames
    it consumed, with detection boxes drawn on."""
    await _relay_upload(ws, annotated_hub)


@router.websocket("/api/ws/video/annotated/feed")
async def annotated_feed(ws: WebSocket):
    """Subscribers: the admin console's detection view."""
    await _relay_feed(ws, annotated_hub)


class RoverHub:
    """Fan-out of driving state. Controllers (operator UIs) set it, agents
    (the Pi) mirror it onto GPIO. State is held here so a reconnecting Pi
    resumes the current command without operator action."""

    def __init__(self) -> None:
        self.agents: set[WebSocket] = set()
        self.controllers: set[WebSocket] = set()
        self.mode = "manual"
        self.cmd = "stop"

    def status(self) -> dict:
        return {
            "t": "status",
            "pi_connected": bool(self.agents),
            "mode": self.mode,
            "cmd": self.cmd,
        }

    def agent_command(self) -> dict:
        """What the Pi should be doing right now. In auto mode the Arduino's
        own obstacle-avoidance loop drives; manual commands are ignored."""
        return {"t": "cmd", "cmd": "auto" if self.mode == "auto" else self.cmd}

    async def _send_all(self, clients: set[WebSocket], payload: dict) -> None:
        message = json.dumps(payload)
        for ws in list(clients):
            try:
                await ws.send_text(message)
            except Exception:
                clients.discard(ws)

    async def push(self) -> None:
        await self._send_all(self.agents, self.agent_command())
        await self._send_all(self.controllers, self.status())

    async def set_cmd(self, cmd: str) -> None:
        self.cmd = cmd
        await self.push()

    async def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.cmd = "stop"  # never carry a held command across a mode switch
        await self.push()


rover_hub = RoverHub()


@router.websocket("/api/ws/rover/control")
async def rover_control(ws: WebSocket):
    """Operator UI. Sends {"t":"cmd","cmd":...} / {"t":"mode","mode":...},
    receives {"t":"status",...} whenever anything changes."""
    await ws.accept()
    rover_hub.controllers.add(ws)
    await ws.send_text(json.dumps(rover_hub.status()))
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get("t") == "cmd" and msg.get("cmd") in COMMANDS:
                if rover_hub.mode == "manual":
                    await rover_hub.set_cmd(msg["cmd"])
            elif msg.get("t") == "mode" and msg.get("mode") in MODES:
                await rover_hub.set_mode(msg["mode"])
    except WebSocketDisconnect:
        pass
    finally:
        rover_hub.controllers.discard(ws)
        # Dead-man's switch: nobody left holding a button -> stop the rover.
        if not rover_hub.controllers and rover_hub.mode == "manual":
            await rover_hub.set_cmd("stop")


@router.websocket("/api/ws/rover/agent")
async def rover_agent(ws: WebSocket):
    """The Pi. Receives the current command immediately on change (via
    RoverHub.push) AND re-sent every 0.5 s as a liveness signal — the pi app
    stops the motors if that stream goes quiet, so a dead link can never
    leave the rover driving. Anything the Pi sends back is only a heartbeat."""
    await ws.accept()
    rover_hub.agents.add(ws)
    await rover_hub._send_all(rover_hub.controllers, rover_hub.status())

    async def refresh() -> None:
        while True:
            await ws.send_text(json.dumps(rover_hub.agent_command()))
            await asyncio.sleep(0.5)

    refresh_task = asyncio.create_task(refresh())
    try:
        while True:
            await ws.receive_text()  # heartbeats; detects disconnect
    except WebSocketDisconnect:
        pass
    finally:
        refresh_task.cancel()
        rover_hub.agents.discard(ws)
        await rover_hub._send_all(rover_hub.controllers, rover_hub.status())


@router.get("/api/rover/status")
def rover_status():
    """Debugging aid for the demo: is everyone actually connected?"""
    return {
        **{k: v for k, v in rover_hub.status().items() if k != "t"},
        "controllers": len(rover_hub.controllers),
        "video_uploader": video_hub.uploader_connected,
        "video_viewers": video_hub.viewers,
        "video_frames_relayed": video_hub.seq,
        "annotated_uploader": annotated_hub.uploader_connected,
        "annotated_viewers": annotated_hub.viewers,
        "annotated_frames_relayed": annotated_hub.seq,
    }
