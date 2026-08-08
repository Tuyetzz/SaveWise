"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export const FIELD_ORDER = [
  "trapped",
  "breathing",
  "respiratory_rate",
  "radial_pulse_present",
  "obeys_commands",
  "can_walk",
  "injuries",
  "people_in_building",
  "others_last_seen",
] as const;

export type FieldName = (typeof FIELD_ORDER)[number];
export type FieldValue = boolean | number | string | null;
export type Fields = Record<FieldName, FieldValue>;

export const FIELD_LABELS: Record<FieldName, string> = {
  trapped: "Trapped",
  breathing: "Breathing",
  respiratory_rate: "Respiratory rate",
  radial_pulse_present: "Radial pulse",
  obeys_commands: "Alert / obeys",
  can_walk: "Can walk",
  injuries: "Injuries",
  people_in_building: "People inside",
  others_last_seen: "Others last seen",
};

export type FeedEntry =
  | { kind: "question"; text: string; seq: number }
  | { kind: "answer"; text: string; seq: number }
  | { kind: "info"; text: string }
  | { kind: "error"; text: string };

export type Phase =
  | "idle"
  | "connecting"
  | "asking" // question audio is playing
  | "listening" // mic armed, waiting for the casualty
  | "processing" // segment sent, waiting for transcript/fields
  | "complete"
  | "ended"; // disconnected or ended early

export type MicMode = "auto" | "hold";

const emptyFields = (): Fields =>
  Object.fromEntries(FIELD_ORDER.map((f) => [f, null])) as Fields;

// --- client-side endpointing (the server never runs VAD) ---
const VAD = {
  chunk: 512, // samples per worklet post @16 kHz = 32 ms
  startRms: 0.015, // voiced above this
  startChunks: 3, // ~96 ms of voice to open a segment
  endChunks: 28, // ~900 ms of silence to close it
  preRollChunks: 10, // ~320 ms kept from before the trigger
  minVoicedChunks: 8, // ~256 ms — discard blips
  maxSeconds: 19.5, // server rejects > 20 s
  // Barge-in while the question is playing: needs to stand clear of the
  // ambient average (EMA) so speaker bleed doesn't trigger it.
  bargeFloor: 0.02,
  bargeEmaFactor: 3,
  bargeChunks: 4, // ~128 ms of sustained voice interrupts playback
};

// No answer for this long after a question finishes -> ask the server to
// repeat. The server allows 3 asks total, then marks the case no-response.
const NO_RESPONSE_MS = 5000;

const WORKLET_SOURCE = `
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() { super(); this.parts = []; this.len = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      this.parts.push(new Float32Array(ch));
      this.len += ch.length;
      if (this.len >= ${VAD.chunk}) {
        const out = new Float32Array(this.len);
        let o = 0;
        for (const p of this.parts) { out.set(p, o); o += p.length; }
        this.port.postMessage(out, [out.buffer]);
        this.parts = []; this.len = 0;
      }
    }
    return true;
  }
}
registerProcessor("capture", CaptureProcessor);
`;

function concatFloat32(parts: Float32Array[]): Float32Array {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

function toPcm16(samples: Float32Array, fromRate: number): ArrayBuffer {
  let f32 = samples;
  if (fromRate !== 16000) {
    const n = Math.round((samples.length * 16000) / fromRate);
    const resampled = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const x = (i * (samples.length - 1)) / (n - 1);
      const lo = Math.floor(x);
      const hi = Math.min(lo + 1, samples.length - 1);
      resampled[i] = samples[lo] + (samples[hi] - samples[lo]) * (x - lo);
    }
    f32 = resampled;
  }
  const pcm = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return pcm.buffer;
}

export function defaultServerUrl(): string {
  if (typeof window === "undefined") return "";
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.hostname}:8000/ws/converse`;
}

export function useInterview() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [feed, setFeed] = useState<FeedEntry[]>([]);
  const [fields, setFields] = useState<Fields>(emptyFields);
  const [level, setLevel] = useState(0);
  const [interviewId, setInterviewId] = useState<string | null>(null);
  const [micMode, setMicMode] = useState<MicMode>("auto");

  const wsRef = useRef<WebSocket | null>(null);
  const playCtxRef = useRef<AudioContext | null>(null);
  const nextPlayTimeRef = useRef(0);
  const questionSampleRateRef = useRef(24000);
  const listenTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const discardAudioRef = useRef(false); // true after barge-in, until next question
  const bargeRef = useRef({ ema: 0.005, count: 0 });

  const captureCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const phaseRef = useRef<Phase>("idle");
  const micModeRef = useRef<MicMode>("auto");
  const holdingRef = useRef(false);
  phaseRef.current = phase;
  micModeRef.current = micMode;

  // VAD state lives in refs — it runs at audio rate, not render rate.
  const vad = useRef({
    inSpeech: false,
    voiced: 0,
    silent: 0,
    preRoll: [] as Float32Array[],
    segment: [] as Float32Array[],
  });

  const push = useCallback((entry: FeedEntry) => {
    setFeed((f) => [...f, entry]);
  }, []);

  const clearSilenceTimer = useCallback(() => {
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
  }, []);

  const armSilenceTimer = useCallback(() => {
    clearSilenceTimer();
    silenceTimerRef.current = setTimeout(() => {
      const ws = wsRef.current;
      if (phaseRef.current !== "listening" || !ws || ws.readyState !== WebSocket.OPEN)
        return;
      ws.send(JSON.stringify({ t: "repeat" }));
      setLevel(0);
      setPhase("asking");
    }, NO_RESPONSE_MS);
  }, [clearSilenceTimer]);

  const stopPlayback = useCallback(() => {
    for (const src of activeSourcesRef.current) {
      try {
        src.stop();
      } catch {
        // already ended
      }
    }
    activeSourcesRef.current.clear();
    nextPlayTimeRef.current = 0;
  }, []);

  const sendSegment = useCallback(
    (parts: Float32Array[]) => {
      const ws = wsRef.current;
      const ctx = captureCtxRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN || !ctx || parts.length === 0)
        return;
      clearSilenceTimer();
      ws.send(toPcm16(concatFloat32(parts), ctx.sampleRate));
      setLevel(0);
      setPhase("processing");
    },
    [clearSilenceTimer],
  );

  const resetVad = useCallback(() => {
    const v = vad.current;
    v.inSpeech = false;
    v.voiced = 0;
    v.silent = 0;
    v.preRoll = [];
    v.segment = [];
  }, []);

  const onMicChunk = useCallback(
    (chunk: Float32Array) => {
      let sum = 0;
      for (let i = 0; i < chunk.length; i++) sum += chunk[i] * chunk[i];
      const rms = Math.sqrt(sum / chunk.length);

      // Hold-to-talk: buffer while the button is down, page sends on release.
      if (micModeRef.current === "hold") {
        if (holdingRef.current && phaseRef.current === "listening") {
          vad.current.segment.push(chunk);
          setLevel(Math.min(1, rms * 12));
        }
        return;
      }

      const v = vad.current;

      // Barge-in: while the question is playing, sustained voice well above
      // the ambient average interrupts playback and starts the answer segment.
      if (phaseRef.current === "asking") {
        const b = bargeRef.current;
        v.preRoll.push(chunk);
        if (v.preRoll.length > VAD.preRollChunks) v.preRoll.shift();
        const threshold = Math.max(VAD.bargeFloor, b.ema * VAD.bargeEmaFactor);
        b.count = rms > threshold ? b.count + 1 : 0;
        b.ema = b.ema * 0.95 + rms * 0.05;
        if (b.count >= VAD.bargeChunks) {
          b.count = 0;
          if (listenTimerRef.current) clearTimeout(listenTimerRef.current);
          discardAudioRef.current = true;
          stopPlayback();
          v.inSpeech = true;
          v.segment = [...v.preRoll];
          v.voiced = VAD.bargeChunks;
          v.silent = 0;
          setPhase("listening");
        }
        return;
      }

      if (phaseRef.current !== "listening") return; // half-duplex gate
      setLevel(Math.min(1, rms * 12));

      const voiced = rms > VAD.startRms;
      if (!v.inSpeech) {
        v.preRoll.push(chunk);
        if (v.preRoll.length > VAD.preRollChunks) v.preRoll.shift();
        v.voiced = voiced ? v.voiced + 1 : 0;
        if (v.voiced >= VAD.startChunks) {
          v.inSpeech = true;
          v.segment = [...v.preRoll];
          v.silent = 0;
          clearSilenceTimer(); // they are speaking — no repeat needed
        }
        return;
      }

      v.segment.push(chunk);
      if (voiced) {
        v.voiced++;
        v.silent = 0;
      } else {
        v.silent++;
      }
      const seconds = (v.segment.length * VAD.chunk) / 16000;
      if (v.silent >= VAD.endChunks || seconds > VAD.maxSeconds) {
        const enough = v.voiced >= VAD.minVoicedChunks;
        const segment = v.segment;
        resetVad();
        if (enough) sendSegment(segment);
        else armSilenceTimer(); // blip discarded — keep the repeat clock running
      }
    },
    [armSilenceTimer, clearSilenceTimer, resetVad, sendSegment, stopPlayback],
  );

  const startMic = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    streamRef.current = stream;
    let ctx: AudioContext;
    try {
      ctx = new AudioContext({ sampleRate: 16000 });
    } catch {
      ctx = new AudioContext(); // some browsers refuse the rate; we resample on send
    }
    captureCtxRef.current = ctx;
    const workletUrl = URL.createObjectURL(
      new Blob([WORKLET_SOURCE], { type: "application/javascript" }),
    );
    await ctx.audioWorklet.addModule(workletUrl);
    URL.revokeObjectURL(workletUrl);
    const node = new AudioWorkletNode(ctx, "capture");
    node.port.onmessage = (e) => onMicChunk(e.data as Float32Array);
    ctx.createMediaStreamSource(stream).connect(node);
  }, [onMicChunk]);

  const playChunk = useCallback((buf: ArrayBuffer) => {
    const ctx = playCtxRef.current;
    if (!ctx || discardAudioRef.current) return; // barged in — drop the rest
    const pcm = new Int16Array(buf);
    if (pcm.length === 0) return;
    const rate = questionSampleRateRef.current;
    const audioBuf = ctx.createBuffer(1, pcm.length, rate);
    const ch = audioBuf.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
    const src = ctx.createBufferSource();
    src.buffer = audioBuf;
    src.connect(ctx.destination);
    const at = Math.max(ctx.currentTime + 0.02, nextPlayTimeRef.current);
    activeSourcesRef.current.add(src);
    src.onended = () => activeSourcesRef.current.delete(src);
    src.start(at);
    nextPlayTimeRef.current = at + audioBuf.duration;
  }, []);

  const cleanup = useCallback(() => {
    if (listenTimerRef.current) clearTimeout(listenTimerRef.current);
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    wsRef.current?.close();
    wsRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    captureCtxRef.current?.close();
    captureCtxRef.current = null;
    playCtxRef.current?.close();
    playCtxRef.current = null;
  }, []);

  const start = useCallback(
    async (serverUrl: string) => {
      setFeed([]);
      setFields(emptyFields());
      setInterviewId(null);
      resetVad();
      setPhase("connecting");
      playCtxRef.current = new AudioContext();
      nextPlayTimeRef.current = 0;
      try {
        await startMic();
      } catch (err) {
        push({ kind: "error", text: `Microphone access failed: ${err}` });
        cleanup();
        setPhase("idle");
        return;
      }

      const ws = new WebSocket(serverUrl);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        push({ kind: "info", text: "Connected. Interview starting…" });
        ws.send(JSON.stringify({ t: "start" }));
      };

      ws.onmessage = (e) => {
        if (e.data instanceof ArrayBuffer) {
          playChunk(e.data);
          return;
        }
        const msg = JSON.parse(e.data as string);
        switch (msg.t) {
          case "question":
            questionSampleRateRef.current = msg.sample_rate ?? 24000;
            discardAudioRef.current = false;
            bargeRef.current.count = 0;
            clearSilenceTimer();
            if ((msg.attempt ?? 1) > 1) {
              push({
                kind: "info",
                text: `No response — asking again (attempt ${msg.attempt} of 3).`,
              });
            } else {
              push({ kind: "question", text: msg.text, seq: msg.seq });
            }
            setPhase("asking");
            break;
          case "question_end": {
            if (discardAudioRef.current) break; // barged in — already listening
            const ctx = playCtxRef.current;
            const waitMs = ctx
              ? Math.max(0, (nextPlayTimeRef.current - ctx.currentTime) * 1000) + 150
              : 0;
            listenTimerRef.current = setTimeout(() => {
              resetVad();
              setPhase("listening");
              armSilenceTimer();
            }, waitMs);
            break;
          }
          case "transcript":
            push({ kind: "answer", text: msg.text, seq: msg.seq });
            break;
          case "fields":
            setFields((prev) => ({ ...emptyFields(), ...prev, ...msg.known }));
            break;
          case "complete":
            setInterviewId(msg.interview_id);
            setPhase("complete");
            push({ kind: "info", text: `Interview complete (${msg.interview_id}).` });
            cleanup();
            break;
          case "no_response":
            setInterviewId(msg.interview_id);
            setPhase("ended");
            push({
              kind: "error",
              text: `No response after 3 attempts — case marked no-response (${msg.interview_id}).`,
            });
            cleanup();
            break;
          case "error":
            push({ kind: "error", text: msg.message });
            if (phaseRef.current === "processing") {
              setPhase("listening");
              armSilenceTimer();
            }
            break;
        }
      };

      ws.onerror = () => {
        push({ kind: "error", text: "WebSocket error — is the backend running?" });
      };

      ws.onclose = () => {
        if (phaseRef.current !== "complete" && phaseRef.current !== "idle") {
          setPhase((p) => (p === "complete" ? p : "ended"));
        }
      };
    },
    [armSilenceTimer, cleanup, clearSilenceTimer, playChunk, push, resetVad, startMic],
  );

  const end = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN)
      ws.send(JSON.stringify({ t: "end" }));
    cleanup();
    setPhase("ended");
    push({ kind: "info", text: "Interview ended." });
  }, [cleanup, push]);

  // Hold-to-talk: page calls these from the button.
  const holdStart = useCallback(() => {
    if (phaseRef.current !== "listening") return;
    holdingRef.current = true;
    clearSilenceTimer();
    vad.current.segment = [];
  }, [clearSilenceTimer]);

  const holdEnd = useCallback(() => {
    if (!holdingRef.current) return;
    holdingRef.current = false;
    const segment = vad.current.segment;
    vad.current.segment = [];
    setLevel(0);
    if (segment.length > 6) sendSegment(segment);
    else armSilenceTimer();
  }, [armSilenceTimer, sendSegment]);

  useEffect(() => cleanup, [cleanup]);

  return {
    phase,
    feed,
    fields,
    level,
    interviewId,
    micMode,
    setMicMode,
    start,
    end,
    holdStart,
    holdEnd,
  };
}
