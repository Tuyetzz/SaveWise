"use client";

import { useRef } from "react";

// The phone mounted on the rover IS the camera — pure input. There is no
// preview here: watching happens in the Admin Console, which shows the
// relayed feed (raw or with detection boxes). The hidden <video> element
// below still exists because frame capture needs somewhere to decode the
// camera stream; it is kept mounted and playing, just invisible.
export default function CameraPanel({
  streaming,
  camError,
  startStream,
  stopStream,
}: {
  streaming: boolean;
  camError: string | null;
  startStream: (el: HTMLVideoElement) => Promise<void>;
  stopStream: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);

  return (
    <section className="w-full shrink-0 md:w-80">
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-3">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-xs font-bold uppercase tracking-widest text-zinc-400">
            Camera uplink
          </h2>
          <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-zinc-500">
            <span
              className={`h-2 w-2 rounded-full ${
                streaming ? "bg-emerald-400 animate-pulse" : "bg-zinc-600"
              }`}
            />
            {streaming ? "streaming" : "off"}
          </div>
        </div>

        {/* capture surface only — intentionally invisible, never display:none
            (some browsers pause a display:none video, which would freeze the
            uplink) */}
        <video
          ref={videoRef}
          muted
          playsInline
          aria-hidden
          className="pointer-events-none absolute h-px w-px opacity-0"
        />

        <button
          onClick={() =>
            streaming
              ? stopStream()
              : videoRef.current && startStream(videoRef.current)
          }
          className={`w-full rounded-lg px-4 py-2 text-sm font-semibold transition-colors ${
            streaming
              ? "border border-red-800 text-red-400 hover:bg-red-950"
              : "bg-emerald-600 hover:bg-emerald-500"
          }`}
        >
          {streaming ? "Stop camera" : "Start camera"}
        </button>
        {camError && <p className="mt-2 text-xs text-red-400">{camError}</p>}
        <p className="mt-2 text-[10px] leading-relaxed text-zinc-600">
          This phone is the rover&apos;s camera and microphone — the live view
          (including detection boxes) is in the Admin Console. Keep this page
          open and the screen on while the rover drives.
        </p>
      </div>
    </section>
  );
}
