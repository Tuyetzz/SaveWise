"use client";

import Nav from "../Nav";
import { useRoverControl } from "../useRover";
import DrivePanel from "./DrivePanel";
import VideoFeed from "./VideoFeed";

// The admin side of the rover: watch the phone's relayed camera (raw, or the
// detection app's view with bounding boxes) and drive. The phone itself stays
// on /rover as pure input — camera and interview mic.
export default function AdminConsole() {
  const { status, sendCmd, setMode } = useRoverControl();

  return (
    <main className="flex min-h-screen flex-col bg-zinc-950 font-sans text-zinc-100">
      <Nav
        active="control"
        status={
          <div className="flex items-center gap-2 rounded-full border border-zinc-800 px-3 py-1 text-xs text-zinc-300">
            <span
              className={`h-2 w-2 rounded-full ${
                status.piConnected
                  ? "bg-emerald-400"
                  : status.connected
                    ? "bg-amber-400 animate-pulse"
                    : "bg-zinc-500"
              }`}
            />
            {status.piConnected
              ? status.mode === "auto"
                ? "Rover autonomous"
                : "Rover ready"
              : status.connected
                ? "Waiting for rover Pi…"
                : "Connecting…"}
          </div>
        }
      />

      <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-4 p-4 md:flex-row">
        {/* live camera — raw, or annotated with detection boxes */}
        <section className="flex-1">
          <VideoFeed />
          <p className="mt-2 text-[11px] leading-relaxed text-zinc-500">
            Live from the phone mounted on the rover, via the backend relay.
            Detection view shows the person-detection pipeline&apos;s output —
            bounding boxes, confidence, FPS — and lags the raw camera by a
            beat; switch to raw for the lowest-latency driving view. The
            camera itself is started from the phone&apos;s Rover Console page.
          </p>
        </section>

        {/* drive controls */}
        <aside className="w-full shrink-0 md:w-80">
          <DrivePanel status={status} sendCmd={sendCmd} setMode={setMode} />
        </aside>
      </div>
    </main>
  );
}
