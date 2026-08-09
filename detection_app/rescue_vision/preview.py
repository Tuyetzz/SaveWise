"""MJPEG preview server, stdlib only.

Lets you watch the annotated feed from a laptop browser while the Pi runs
headless (Raspberry Pi OS Lite has no desktop, and PRD 6.10 forbids imshow
there). Latest-frame-wins: a slow or absent viewer never back-pressures the
detection loop.

Measured cost is one JPEG encode per frame -- ~1.3 ms on desktop, ~5-8 ms on a
Pi 5, so roughly 5-8% of a 100 ms frame budget.

This is a local convenience for the demo, not an inference dependency: PRD NFR5
(no cloud, no network needed for detection) still holds.
"""

from __future__ import annotations

import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

log = logging.getLogger(__name__)

_BOUNDARY = "frameboundary"

_INDEX = b"""<!doctype html>
<title>rescue_vision preview</title>
<style>body{background:#111;margin:0;display:grid;place-items:center;height:100vh}
img{max-width:100%;image-rendering:pixelated}</style>
<img src="/stream" alt="live detection feed">
"""


class MjpegServer:
    """Serves the most recent published frame as an MJPEG stream."""

    def __init__(self, port: int, quality: int = 80) -> None:
        self._port = port
        self._quality = quality
        self._latest: bytes | None = None
        self._lock = threading.Condition()
        self._seq = 0
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        server = self  # closed over by the handler

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args) -> None:
                return  # silence per-request stderr spam

            def do_GET(self) -> None:  # noqa: N802 - stdlib naming
                if self.path.startswith("/stream"):
                    self._serve_stream()
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(_INDEX)))
                    self.end_headers()
                    self.wfile.write(_INDEX)

            def _serve_stream(self) -> None:
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
                )
                self.end_headers()
                last_seq = -1
                try:
                    while True:
                        with server._lock:
                            server._lock.wait_for(
                                lambda: server._latest is not None
                                and server._seq != last_seq,
                                timeout=5.0,
                            )
                            frame = server._latest
                            last_seq = server._seq
                        if frame is None:
                            continue
                        self.wfile.write(
                            f"--{_BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                            f"Content-Length: {len(frame)}\r\n\r\n".encode()
                        )
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass  # viewer closed the tab; not an error

        self._httpd = ThreadingHTTPServer(("0.0.0.0", self._port), Handler)
        self._httpd.daemon_threads = True
        self._port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        log.info("preview stream at %s", self.url)

    def publish(self, frame: np.ndarray) -> None:
        """Encode and store the newest frame, discarding any older one."""
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            return
        with self._lock:
            self._latest = buf.tobytes()
            self._seq += 1
            self._lock.notify_all()

    def pending_frames(self) -> int:
        """Always 0 or 1 -- there is no queue, only a latest slot."""
        with self._lock:
            return 0 if self._latest is None else 1

    @property
    def url(self) -> str:
        return f"http://{_local_ip()}:{self._port}"

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


def _local_ip() -> str:
    """Best-effort LAN address, so the printed URL works from a laptop."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # no packets sent; this just picks a route
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


class WsPublisher:
    """Pushes annotated JPEGs to the backend relay (--publish), so the admin
    console can watch the detection view — boxes and all — from anywhere.

    Best-effort by design: the backend being down must never slow or crash
    the detection loop, so publish() swallows failures and retries the
    connection at most every RETRY_S seconds.
    """

    RETRY_S = 3.0

    def __init__(self, url: str, quality: int = 70, connect=None) -> None:
        if connect is None:
            import websocket  # lazy: only --publish runs need websocket-client

            connect = lambda: websocket.create_connection(url, timeout=5)
        self._url = url
        self._quality = quality
        self._connect = connect
        self._ws = None
        self._next_attempt = 0.0
        self.sent = 0

    def publish(self, frame: np.ndarray, now: float | None = None) -> None:
        import time

        now = time.monotonic() if now is None else now
        if self._ws is None:
            if now < self._next_attempt:
                return
            try:
                self._ws = self._connect()
                log.info("publishing annotated frames to %s", self._url)
            except Exception as exc:
                self._next_attempt = now + self.RETRY_S
                log.debug("annotated publish connect failed: %s", exc)
                return
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            return
        try:
            self._ws.send_binary(buf.tobytes())
            self.sent += 1
        except Exception as exc:
            log.warning("annotated publish send failed (%s); will reconnect", exc)
            self._close_quietly()
            self._next_attempt = now + self.RETRY_S

    def _close_quietly(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def stop(self) -> None:
        self._close_quietly()
