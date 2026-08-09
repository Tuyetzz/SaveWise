"use client";

import { useEffect, useRef, useState } from "react";
import { wsBase } from "../triageDisplay";

// Live view of the rover, straight off the backend relay. Two channels:
//   detection — frames the detection app publishes back with bounding boxes
//               drawn on (rescue_vision --publish ...)
//   raw       — the phone's camera as uploaded, lowest latency
// JPEG-per-message, rendered by swapping an <img>'s blob URL.
const STALE_MS = 3000;

type Channel = "detection" | "raw";

const CHANNEL_PATH: Record<Channel, string> = {
  detection: "/api/ws/video/annotated/feed",
  raw: "/api/ws/video/feed",
};

export default function VideoFeed() {
  const imgRef = useRef<HTMLImageElement>(null);
  const [channel, setChannel] = useState<Channel>("detection");
  const [connected, setConnected] = useState(false);
  const [live, setLive] = useState(false);
  const lastFrameRef = useRef(0);

  useEffect(() => {
    let closed = false;
    let ws: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | null = null;
    let currentUrl: string | null = null;
    lastFrameRef.current = 0;
    setLive(false);

    const connect = () => {
      if (closed) return;
      ws = new WebSocket(`${wsBase()}${CHANNEL_PATH[channel]}`);
      ws.binaryType = "arraybuffer";
      ws.onopen = () => setConnected(true);
      ws.onmessage = (e) => {
        if (!(e.data instanceof ArrayBuffer) || !imgRef.current) return;
        const url = URL.createObjectURL(
          new Blob([e.data], { type: "image/jpeg" }),
        );
        imgRef.current.src = url;
        if (currentUrl) URL.revokeObjectURL(currentUrl);
        currentUrl = url;
        lastFrameRef.current = Date.now();
        setLive(true);
      };
      ws.onclose = () => {
        setConnected(false);
        setLive(false);
        if (!closed) retry = setTimeout(connect, 2000);
      };
      ws.onerror = () => ws?.close();
    };
    connect();

    // The relay never pushes "the sender stopped" — infer it from silence.
    const staleTimer = setInterval(() => {
      if (Date.now() - lastFrameRef.current > STALE_MS) setLive(false);
    }, 1000);

    return () => {
      closed = true;
      if (retry) clearTimeout(retry);
      clearInterval(staleTimer);
      ws?.close();
      if (currentUrl) URL.revokeObjectURL(currentUrl);
    };
  }, [channel]);

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <div className="grid grid-cols-2 gap-1 rounded-lg border border-zinc-800 bg-zinc-950/50 p-1">
          {(["detection", "raw"] as const).map((c) => (
            <button
              key={c}
              onClick={() => setChannel(c)}
              className={`rounded-md px-3 py-1 text-xs font-semibold transition-colors ${
                channel === c
                  ? "bg-zinc-100 text-zinc-950"
                  : "text-zinc-400 hover:bg-zinc-800"
              }`}
            >
              {c === "detection" ? "Detection view" : "Raw camera"}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-zinc-500">
          <span
            className={`h-2 w-2 rounded-full ${
              live ? "bg-emerald-400 animate-pulse" : "bg-zinc-600"
            }`}
          />
          {live ? "live" : "no feed"}
        </div>
      </div>

      <div className="relative overflow-hidden rounded-lg border border-zinc-800 bg-black">
        {/* eslint-disable-next-line @next/next/no-img-element -- blob URLs swapped at stream rate; next/image can't help here */}
        <img
          ref={imgRef}
          alt={
            channel === "detection"
              ? "Live rover camera with detection boxes"
              : "Live rover camera"
          }
          className={`aspect-video w-full object-contain ${live ? "" : "opacity-20"}`}
        />
        {!live && (
          <p className="absolute inset-0 flex items-center justify-center px-6 text-center text-sm text-zinc-500">
            {!connected
              ? "Connecting to the video relay…"
              : channel === "detection"
                ? "No detection frames — is rescue_vision running with --publish, and is the phone camera on?"
                : "No frames — start the camera on the rover phone (Rover Console page)"}
          </p>
        )}
      </div>
    </div>
  );
}
