"use client";

import type { RoverCmd, RoverMode, RoverStatus } from "../useRover";

function Dot({ on }: { on: boolean }) {
  return (
    <span
      className={`h-2 w-2 rounded-full ${on ? "bg-emerald-400" : "bg-zinc-600"}`}
    />
  );
}

// Press-and-hold: direction on pointer-down, stop on release/leave/cancel so
// a finger sliding off the pad never leaves the rover driving.
function PadButton({
  cmd,
  label,
  active,
  disabled,
  onHold,
  onRelease,
}: {
  cmd: RoverCmd;
  label: string;
  active: boolean;
  disabled: boolean;
  onHold: (cmd: RoverCmd) => void;
  onRelease: () => void;
}) {
  return (
    <button
      disabled={disabled}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        onHold(cmd);
      }}
      onPointerUp={onRelease}
      onPointerCancel={onRelease}
      onContextMenu={(e) => e.preventDefault()}
      className={`flex h-14 select-none items-center justify-center rounded-lg border text-xl transition-colors ${
        active
          ? "border-emerald-500 bg-emerald-600/30 text-emerald-200"
          : "border-zinc-700 bg-zinc-900 text-zinc-300"
      } enabled:active:bg-emerald-600/40 disabled:opacity-30`}
      style={{ touchAction: "none" }}
      aria-label={cmd}
    >
      {label}
    </button>
  );
}

export default function DrivePanel({
  status,
  sendCmd,
  setMode,
}: {
  status: RoverStatus;
  sendCmd: (cmd: RoverCmd) => void;
  setMode: (mode: RoverMode) => void;
}) {
  const manual = status.mode === "manual";
  const padDisabled = !status.connected || !manual;
  const isActive = (cmd: RoverCmd) => manual && status.cmd === cmd;

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-3">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-xs font-bold uppercase tracking-widest text-zinc-400">
          Drive
        </h2>
        <div className="flex items-center gap-3 text-[10px] uppercase tracking-widest text-zinc-500">
          <span className="flex items-center gap-1.5">
            <Dot on={status.connected} /> link
          </span>
          <span className="flex items-center gap-1.5">
            <Dot on={status.piConnected} /> pi
          </span>
        </div>
      </div>

      <div className="mb-3 grid grid-cols-2 gap-1 rounded-lg border border-zinc-800 bg-zinc-950/50 p-1">
        {(["manual", "auto"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            disabled={!status.connected}
            className={`rounded-md px-3 py-1.5 text-xs font-semibold transition-colors disabled:opacity-40 ${
              status.mode === m
                ? "bg-zinc-100 text-zinc-950"
                : "text-zinc-400 hover:bg-zinc-800"
            }`}
          >
            {m === "manual" ? "Manual" : "Autonomous"}
          </button>
        ))}
      </div>

      <div className="mx-auto grid max-w-64 grid-cols-3 gap-1.5">
        <div />
        <PadButton
          cmd="forward"
          label="▲"
          active={isActive("forward")}
          disabled={padDisabled}
          onHold={sendCmd}
          onRelease={() => sendCmd("stop")}
        />
        <div />
        <PadButton
          cmd="left"
          label="◀"
          active={isActive("left")}
          disabled={padDisabled}
          onHold={sendCmd}
          onRelease={() => sendCmd("stop")}
        />
        <button
          onClick={() => {
            if (!manual) setMode("manual"); // emergency: also drops out of auto
            sendCmd("stop");
          }}
          disabled={!status.connected}
          className="flex h-14 select-none items-center justify-center rounded-lg border border-red-800 bg-red-950/40 text-xs font-bold text-red-300 transition-colors enabled:active:bg-red-900 disabled:opacity-30"
        >
          STOP
        </button>
        <PadButton
          cmd="right"
          label="▶"
          active={isActive("right")}
          disabled={padDisabled}
          onHold={sendCmd}
          onRelease={() => sendCmd("stop")}
        />
        <div />
        <PadButton
          cmd="backward"
          label="▼"
          active={isActive("backward")}
          disabled={padDisabled}
          onHold={sendCmd}
          onRelease={() => sendCmd("stop")}
        />
        <div />
      </div>

      <p className="mt-3 text-[10px] leading-relaxed text-zinc-600">
        {status.mode === "auto"
          ? "Autonomous: the Arduino's obstacle-avoidance loop is driving. STOP switches back to manual."
          : "Hold a direction to drive; releasing stops. Commands go via the base station to the Pi, which signals the Arduino over GPIO."}
        {!status.piConnected && status.connected && (
          <span className="text-amber-400"> Rover Pi is offline.</span>
        )}
      </p>
    </div>
  );
}
