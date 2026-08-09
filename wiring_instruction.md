# Rover bring-up: flash the Arduino, wire the Pi 5, drive from the console

The complete bench procedure: one sketch to flash, four wires to connect, then
a layer-by-layer test so any failure points at exactly one place. Written for
the **Raspberry Pi 5** — gpiozero/lgpio only; `RPi.GPIO` and `pigpio` do not
work on it.

> ⚠️ **Before anything:** prop the chassis on a box so the wheels spin in the
> air. The rover stays off the floor until the final step.

## The command path you are building

```
Admin Console (/control)
      │  WebSocket
      ▼
backend  /api/ws/rover/control ─► /api/ws/rover/agent
      │  WebSocket (outbound from the Pi)
      ▼
pi/rover_pi.py  ─►  GPIO 17 / 27 / 22  (3-bit command code)
      │
      ▼
Arduino A0 / A1 / A2  ─►  RoverRemote.ino  ─►  MDD3A  ─►  motors

codes: 0 stop · 1 fwd · 2 back · 3 left · 4 right · 5 auto · 6–7 stop (failsafe)
```

## Part 1 — Flash the Arduino

1. Install the [Arduino IDE](https://www.arduino.cc/en/software) on any laptop.
2. **File → Open** → `pi/arduino/RoverRemote/RoverRemote.ino`. This is the only
   thing that gets flashed — it contains manual driving, the old autonomous
   mode, and the ultrasonic safety stop. It replaces `RoverTestV2.ino`.
3. Plug the Arduino in over USB.
   - **Tools → Board**: the same board used for RoverTestV2 (almost certainly
     *Arduino Uno*).
   - **Tools → Port**: the port that appeared (`COM…` on Windows,
     `/dev/ttyACM0` on Linux).
4. Click **Upload**, wait for "Done uploading."
5. Open **Tools → Serial Monitor** at **9600 baud**. Expect:

   ```
   RoverRemote ready — waiting for Pi commands
   command code: 7
   ```

   **Code 7 is correct**: nothing is wired to A0–A2 yet, so the Arduino's
   pullups read all-high = stop. Unplugged fails safe. If motors try to run
   instead, stop and re-check the motor wiring.

### Alternative: flash from the Pi itself

Useful once the Arduino rides on the rover plugged into the Pi's USB — retune
`DRIVE_SPEED` / `TURN_SPEED` without unplugging anything:

```bash
sudo apt install arduino-cli
arduino-cli core install arduino:avr
arduino-cli compile -b arduino:avr:uno pi/arduino/RoverRemote
arduino-cli upload  -b arduino:avr:uno -p /dev/ttyACM0 pi/arduino/RoverRemote
arduino-cli monitor -p /dev/ttyACM0 -c 9600   # replaces the Serial Monitor
```

## Part 2 — Wire Pi 5 → Arduino (power off first)

Four jumpers, female (Pi header) → male (Arduino). All four physical pins sit
in one row on the inner column of the 40-pin header — pin 1 is the corner pin
nearest the SD card slot; odd numbers are the inner row.

| Pi (BCM) | Pi physical pin | → Arduino | Role |
| -------- | --------------- | --------- | ---- |
| GND      | **9**           | GND       | common ground — **required** |
| GPIO17   | **11**          | A0        | command bit 0 |
| GPIO27   | **13**          | A1        | command bit 1 |
| GPIO22   | **15**          | A2        | command bit 2 |

> ⚠️ **One-way only.** Pi → Arduino inputs is safe (3.3 V is a valid HIGH for a
> 5 V Arduino). **Never connect an Arduino output to a Pi pin** — 5 V will
> damage the Pi's 3.3 V GPIO.

- Motor power is unchanged: battery → MDD3A, exactly as wired for RoverTestV2.
- The Arduino can be powered from one of the Pi's USB-A ports (~50 mA). That
  also shares ground through the cable — wire the GND jumper anyway.
- Pi 5 power: it wants 5 V/5 A USB-C. On the rover use a good USB-C PD power
  bank; a low-voltage warning may throttle the CPU but GPIO still works.

## Part 3 — Pi software

```bash
cd ~/projects/SaveWise/pi
python3 -m venv --system-site-packages .venv   # keeps apt's gpiozero/lgpio visible
.venv/bin/pip install -r requirements.txt
```

**Pi 5 specifics:** GPIO moved to the RP1 chip, so `RPi.GPIO`, `pigpio`, and
the old `/sys/class/gpio` interface are all dead on this board. The bridge uses
gpiozero + lgpio, which work. If gpiozero ever reports "no default pin
factory":

```bash
sudo apt install python3-lgpio python3-gpiozero
```

For hand-poking pins from the shell, the modern tool is `pinctrl`
(e.g. `pinctrl set 17 op dh` drives GPIO17 high).

## Part 4 — Test layer by layer

### 4a — Pins → Arduino (no network involved)

Serial Monitor open, then on the Pi:

```bash
python3 -c "
from gpiozero import DigitalOutputDevice
import time
bits = [DigitalOutputDevice(p) for p in (17, 27, 22)]
for code in (1, 3, 4, 2, 0):        # fwd, left, right, back, stop
    for i, b in enumerate(bits): b.value = bool((code >> i) & 1)
    print('sent', code); time.sleep(2)
"
```

The Serial Monitor should print `command code: 1, 3, 4, 2, 0` in step, wheels
spinning accordingly.

- Codes wrong or stuck at 7 → **wiring** (ground jumper first, then swapped
  signal wires).
- Codes right but no motion → **motor side** (battery, MDD3A) — the command
  path is fine.

### 4b — Backend → Pi bridge

```bash
.venv/bin/python rover_pi.py
# or, over plain LAN instead of the nginx proxy:
.venv/bin/python rover_pi.py --url ws://<server-ip>:8000/api/ws/rover/agent
```

Expect `connected to wss://…` in its log, the **"pi" dot green** in the Admin
Console, and `"pi_connected": true` from `GET /api/rover/status`. (The backend
must have been restarted since the rover routes were added — it deliberately
runs without auto-reload.)

### 4c — End to end

1. Open `/control` on the laptop, hold **▲** — pi log shows
   `command: stop -> forward`, serial shows `command code: 1`, wheels spin.
   Release → everything stops.
2. Toggle **Autonomous** — serial shows `command code: 5` and the Arduino runs
   its own obstacle-avoidance loop. **STOP** drops back to manual.
3. **Now** put the rover on the floor.

## Troubleshooting

| Symptom | Where to look |
| ------- | ------------- |
| Serial stuck at code 7 | Pi pins not reaching the Arduino: ground jumper first, then the three signal wires and their order. |
| Codes change, motors silent | Motor battery, MDD3A power, motor wiring — the command path is fine. |
| A direction is reversed | Swap that motor's two wires at the MDD3A, or swap its pin pair in the sketch's PIN DEFINITIONS and re-flash. Test forward first, then turns. |
| "pi" dot stays grey | `rover_pi.py` log — wrong `--url`, or the backend wasn't restarted (check that `/api/rover/status` exists). |
| Forward refuses to drive | Something < 20 cm in front of the middle ultrasonic — that's the safety gate working as designed. |
| Rover keeps moving after a drop-out | Shouldn't happen: the backend re-sends every 0.5 s, the Pi watchdog stops after 2 s of silence, and the pullups read stop if the Pi dies. If it does, make sure the pi app is the current code. |
