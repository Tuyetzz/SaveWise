#!/usr/bin/env python3
"""SaveWise rover bridge: backend WebSocket -> GPIO -> Arduino.

The Pi keeps an outbound socket open to the backend's /api/ws/rover/agent
endpoint (outbound only, so it works from any network) and mirrors the
current driving command onto three GPIO pins as a 3-bit code. The Arduino
(pi/arduino/RoverRemote) reads those pins and drives the motors.

Command codes on (BIT2 BIT1 BIT0):
    0 stop | 1 forward | 2 backward | 3 left | 4 right | 5 auto
    "auto" hands control to the Arduino's own obstacle-avoidance loop.
    The Arduino treats 6 and 7 (pins floating high via its pullups, i.e.
    Pi off or cable out) as stop.

Failsafe layers:
  - the backend re-sends the current command every 0.5 s; if nothing arrives
    for WATCHDOG_S the motors are stopped locally,
  - any disconnect or crash stops the motors before reconnecting,
  - the Arduino stops on the all-high code it sees if the Pi dies entirely.

Runs with gpiozero, which works on the Pi 5 (RPi.GPIO and pigpio do NOT —
see detection_app/CLAUDE.md). Use --dry-run to test off-Pi.
"""

import argparse
import asyncio
import json
import logging
import os
import time

import websockets

log = logging.getLogger("rover_pi")

DEFAULT_URL = os.environ.get(
    "ROVER_AGENT_URL", "wss://hackathon.marcusnguyen.dev/api/ws/rover/agent"
)

# BCM pin numbers wired to the Arduino's CMD inputs (see pi/README.md).
BIT_PINS = (17, 27, 22)  # bit0, bit1, bit2

CODES = {"stop": 0, "forward": 1, "backward": 2, "left": 3, "right": 4, "auto": 5}

WATCHDOG_S = 2.0  # no server message for this long -> stop
HEARTBEAT_S = 5.0  # keeps the socket warm through proxies
RECONNECT_S = 3.0


class GpioBus:
    """Three output pins holding the current command code."""

    def __init__(self) -> None:
        from gpiozero import DigitalOutputDevice  # lazy: Pi only

        self._bits = [DigitalOutputDevice(pin, initial_value=False) for pin in BIT_PINS]

    def write(self, code: int) -> None:
        # Not atomic — the Arduino double-reads with a settle delay, so a
        # transient mixed code between these three writes is harmless.
        for i, bit in enumerate(self._bits):
            bit.value = bool((code >> i) & 1)

    def close(self) -> None:
        self.write(CODES["stop"])
        for bit in self._bits:
            bit.close()


class DryRunBus:
    """--dry-run: log instead of touching GPIO."""

    def write(self, code: int) -> None:
        log.info("GPIO would be %s (code %d)", format(code, "03b"), code)

    def close(self) -> None:
        self.write(CODES["stop"])


class Rover:
    def __init__(self, bus) -> None:
        self._bus = bus
        self._cmd = "stop"
        self._bus.write(CODES["stop"])

    @property
    def cmd(self) -> str:
        return self._cmd

    def apply(self, cmd: str) -> None:
        if cmd not in CODES:
            log.warning("ignoring unknown command %r", cmd)
            return
        if cmd != self._cmd:
            log.info("command: %s -> %s", self._cmd, cmd)
            self._cmd = cmd
            self._bus.write(CODES[cmd])

    def stop(self) -> None:
        self.apply("stop")


async def session(url: str, rover: Rover) -> None:
    """One connection's lifetime: receive commands, heartbeat, watchdog."""
    # Aggressive pings: a dead base station should surface in seconds, not
    # the library's ~40 s default.
    async with websockets.connect(url, ping_interval=2, ping_timeout=3) as ws:
        log.info("connected to %s", url)
        last_msg = time.monotonic()

        async def receive() -> None:
            nonlocal last_msg
            async for raw in ws:
                last_msg = time.monotonic()
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if msg.get("t") == "cmd":
                    rover.apply(str(msg.get("cmd")))

        async def heartbeat() -> None:
            while True:
                await ws.send(json.dumps({"t": "heartbeat", "cmd": rover.cmd}))
                await asyncio.sleep(HEARTBEAT_S)

        async def watchdog() -> None:
            while True:
                await asyncio.sleep(0.2)
                if rover.cmd != "stop" and time.monotonic() - last_msg > WATCHDOG_S:
                    log.warning("no server traffic for %.1fs — stopping", WATCHDOG_S)
                    rover.stop()

        tasks = [
            asyncio.create_task(receive()),
            asyncio.create_task(heartbeat()),
            asyncio.create_task(watchdog()),
        ]
        try:
            # receive() returning or any task raising ends the session.
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()


async def run(url: str, rover: Rover) -> None:
    while True:
        try:
            await session(url, rover)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("connection lost (%s)", exc)
        rover.stop()  # never keep driving while offline
        log.info("reconnecting in %.0fs...", RECONNECT_S)
        await asyncio.sleep(RECONNECT_S)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"agent WebSocket URL (or env ROVER_AGENT_URL; default {DEFAULT_URL})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="log GPIO writes instead of driving pins (test off-Pi)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    bus = DryRunBus() if args.dry_run else GpioBus()
    rover = Rover(bus)
    try:
        asyncio.run(run(args.url, rover))
    except KeyboardInterrupt:
        pass
    finally:
        bus.close()
        log.info("stopped; GPIO released")


if __name__ == "__main__":
    main()
