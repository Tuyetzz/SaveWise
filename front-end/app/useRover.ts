"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { wsBase } from "./triageDisplay";

// Driving commands are press-and-hold: the pad sends the direction on
// pointer-down and "stop" on release, and the backend relays only the latest
// state to the Pi. "auto" is not sendable from the pad — it is what the Pi
// runs when the mode is switched to autonomous (Arduino obstacle-avoidance).
export type RoverCmd = "forward" | "backward" | "left" | "right" | "stop";
export type RoverMode = "manual" | "auto";

export type RoverStatus = {
  connected: boolean; // this browser's control socket
  piConnected: boolean; // the Pi's agent socket, as reported by the backend
  mode: RoverMode;
  cmd: string; // last command the backend accepted (echoed to all operators)
};

// Phone camera -> backend relay -> detection app + drive console. Modest rate
// and size on purpose: subscribers drop stale frames anyway, and the phone is
// also running the interview audio loop.
const STREAM = {
  fps: 8,
  maxWidth: 640,
  jpegQuality: 0.6,
  // Skip a frame rather than queue it if the uplink falls behind.
  maxBufferedBytes: 256 * 1024,
};

/** Operator side: the drive console's link to the rover (via the backend).
 * Lives on /control — the phone on the rover never opens this socket, so
 * closing the console page is what trips the backend's dead-man stop. */
export function useRoverControl() {
  const [status, setStatus] = useState<RoverStatus>({
    connected: false,
    piConnected: false,
    mode: "manual",
    cmd: "stop",
  });

  const controlRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const closedRef = useRef(false);

  useEffect(() => {
    closedRef.current = false;
    const connect = () => {
      if (closedRef.current) return;
      const ws = new WebSocket(`${wsBase()}/api/ws/rover/control`);
      controlRef.current = ws;
      ws.onopen = () => setStatus((s) => ({ ...s, connected: true }));
      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data as string);
        if (msg.t === "status")
          setStatus((s) => ({
            ...s,
            piConnected: !!msg.pi_connected,
            mode: msg.mode === "auto" ? "auto" : "manual",
            cmd: String(msg.cmd ?? "stop"),
          }));
      };
      ws.onclose = () => {
        setStatus((s) => ({ ...s, connected: false, piConnected: false }));
        if (!closedRef.current)
          reconnectRef.current = setTimeout(connect, 2000);
      };
      ws.onerror = () => ws.close();
    };
    connect();
    return () => {
      closedRef.current = true;
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      controlRef.current?.close();
      controlRef.current = null;
    };
  }, []);

  const sendCmd = useCallback((cmd: RoverCmd) => {
    const ws = controlRef.current;
    if (ws && ws.readyState === WebSocket.OPEN)
      ws.send(JSON.stringify({ t: "cmd", cmd }));
  }, []);

  const setMode = useCallback((mode: RoverMode) => {
    const ws = controlRef.current;
    if (ws && ws.readyState === WebSocket.OPEN)
      ws.send(JSON.stringify({ t: "mode", mode }));
  }, []);

  return { status, sendCmd, setMode };
}

/** Rover side: the phone's camera pushed up to the backend relay.
 * Lives on /rover, next to the triage interview. */
export function useCameraUplink() {
  const [streaming, setStreaming] = useState(false);
  const [camError, setCamError] = useState<string | null>(null);

  const videoWsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sendTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const videoElRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const encodingRef = useRef(false);

  const stopStream = useCallback(() => {
    if (sendTimerRef.current) clearInterval(sendTimerRef.current);
    sendTimerRef.current = null;
    videoWsRef.current?.close();
    videoWsRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoElRef.current) videoElRef.current.srcObject = null;
    setStreaming(false);
  }, []);

  const startStream = useCallback(
    async (videoEl: HTMLVideoElement) => {
      setCamError(null);
      videoElRef.current = videoEl;
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: { ideal: "environment" }, // rear camera on the phone
            width: { ideal: STREAM.maxWidth },
            height: { ideal: 480 },
          },
        });
      } catch (err) {
        setCamError(`Camera access failed: ${err}`);
        return;
      }
      streamRef.current = stream;
      videoEl.srcObject = stream;
      try {
        await videoEl.play();
      } catch {
        // autoplay quirks — the muted playsInline element usually still starts
      }

      const ws = new WebSocket(`${wsBase()}/api/ws/video/upload`);
      ws.binaryType = "arraybuffer";
      videoWsRef.current = ws;
      ws.onclose = () => {
        // Relay gone mid-stream: tear down so the button offers a restart.
        if (videoWsRef.current === ws) {
          setCamError("Video uplink closed — is the backend running?");
          stopStream();
        }
      };

      if (!canvasRef.current) canvasRef.current = document.createElement("canvas");
      const canvas = canvasRef.current;

      sendTimerRef.current = setInterval(() => {
        if (
          ws.readyState !== WebSocket.OPEN ||
          encodingRef.current ||
          ws.bufferedAmount > STREAM.maxBufferedBytes ||
          videoEl.videoWidth === 0
        )
          return;
        const scale = Math.min(1, STREAM.maxWidth / videoEl.videoWidth);
        canvas.width = Math.round(videoEl.videoWidth * scale);
        canvas.height = Math.round(videoEl.videoHeight * scale);
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
        encodingRef.current = true;
        canvas.toBlob(
          async (blob) => {
            try {
              if (blob && ws.readyState === WebSocket.OPEN)
                ws.send(await blob.arrayBuffer());
            } finally {
              encodingRef.current = false;
            }
          },
          "image/jpeg",
          STREAM.jpegQuality,
        );
      }, 1000 / STREAM.fps);

      setStreaming(true);
    },
    [stopStream],
  );

  useEffect(() => stopStream, [stopStream]);

  return { streaming, camError, startStream, stopStream };
}
