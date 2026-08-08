"use client";

import { useEffect, useRef, useState } from "react";
import Nav from "../Nav";
import {
  FIELD_LABELS,
  FIELD_ORDER,
  type FieldValue,
  type Phase,
  defaultServerUrl,
  useInterview,
} from "../useInterview";

const PHASE_INFO: Record<Phase, { label: string; dot: string }> = {
  idle: { label: "Ready", dot: "bg-zinc-500" },
  connecting: { label: "Connecting…", dot: "bg-amber-400 animate-pulse" },
  asking: { label: "Speaking question", dot: "bg-sky-400 animate-pulse" },
  listening: { label: "Listening", dot: "bg-emerald-400 animate-pulse" },
  processing: { label: "Processing answer", dot: "bg-amber-400 animate-pulse" },
  complete: { label: "Complete", dot: "bg-emerald-500" },
  ended: { label: "Ended", dot: "bg-zinc-500" },
};

function ValueBadge({ value }: { value: FieldValue }) {
  if (value === null)
    return (
      <span className="rounded border border-amber-500/50 px-2 py-0.5 text-xs font-semibold tracking-wide text-amber-400">
        UNKNOWN
      </span>
    );
  const text = typeof value === "boolean" ? (value ? "YES" : "NO") : String(value);
  return (
    <span
      className="max-w-36 truncate rounded bg-emerald-500/15 px-2 py-0.5 text-xs font-semibold tracking-wide text-emerald-300"
      title={text}
    >
      {text}
    </span>
  );
}

export default function RoverConsole() {
  const {
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
  } = useInterview();

  const [serverUrl, setServerUrl] = useState("");
  useEffect(() => setServerUrl(defaultServerUrl()), []);

  const feedRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    feedRef.current?.scrollTo({ top: feedRef.current.scrollHeight, behavior: "smooth" });
  }, [feed]);

  const running = phase !== "idle" && phase !== "complete" && phase !== "ended";
  const knownCount = FIELD_ORDER.filter((f) => fields[f] !== null).length;

  return (
    <main className="flex min-h-screen flex-col bg-zinc-950 font-sans text-zinc-100">
      <Nav
        active="rover"
        status={
          <div className="flex items-center gap-2 rounded-full border border-zinc-800 px-3 py-1 text-xs text-zinc-300">
            <span className={`h-2 w-2 rounded-full ${PHASE_INFO[phase].dot}`} />
            {PHASE_INFO[phase].label}
          </div>
        }
      />

      <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-4 p-4 md:flex-row">
        {/* conversation */}
        <section className="flex min-h-[50vh] flex-1 flex-col rounded-xl border border-zinc-800 bg-zinc-900/60">
          <div
            ref={feedRef}
            className="flex-1 space-y-3 overflow-y-auto p-4"
            style={{ maxHeight: "calc(100vh - 230px)" }}
          >
            {feed.length === 0 && (
              <div className="pt-14 text-center">
                <p className="text-3xl">🛰️</p>
                <p className="mt-2 text-sm text-zinc-400">
                  When the rover reaches a survivor, press{" "}
                  <span className="font-semibold text-zinc-200">Start interview</span>.
                </p>
                <p className="mt-1 text-xs text-zinc-600">
                  Questions play as audio · answers are voice-detected automatically
                </p>
              </div>
            )}
            {feed.map((entry, i) => {
              if (entry.kind === "question")
                return (
                  <div key={i} className="max-w-[85%]">
                    <p className="mb-0.5 text-[10px] uppercase tracking-widest text-sky-500">
                      Question {entry.seq}
                    </p>
                    <div className="rounded-lg rounded-tl-none border border-sky-900/60 bg-sky-950/40 px-3 py-2 text-sm">
                      {entry.text}
                    </div>
                  </div>
                );
              if (entry.kind === "answer")
                return (
                  <div key={i} className="ml-auto max-w-[85%] text-right">
                    <p className="mb-0.5 text-[10px] uppercase tracking-widest text-emerald-500">
                      Heard
                    </p>
                    <div className="inline-block rounded-lg rounded-tr-none border border-emerald-900/60 bg-emerald-950/40 px-3 py-2 text-left text-sm">
                      {entry.text}
                    </div>
                  </div>
                );
              return (
                <p
                  key={i}
                  className={`text-center text-xs ${
                    entry.kind === "error" ? "text-red-400" : "text-zinc-500"
                  }`}
                >
                  {entry.text}
                </p>
              );
            })}
          </div>

          {/* mic / controls bar */}
          <div className="border-t border-zinc-800 p-3">
            {!running ? (
              <div className="flex flex-col gap-2 sm:flex-row">
                <input
                  value={serverUrl}
                  onChange={(e) => setServerUrl(e.target.value)}
                  spellCheck={false}
                  className="flex-1 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 font-mono text-xs text-zinc-300 outline-none focus:border-zinc-500"
                />
                <button
                  onClick={() => start(serverUrl)}
                  className="rounded-lg bg-red-600 px-5 py-2 text-sm font-semibold transition-colors hover:bg-red-500"
                >
                  {phase === "complete" || phase === "ended"
                    ? "New interview"
                    : "Start interview"}
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-3">
                {/* level meter */}
                <div className="h-2 flex-1 overflow-hidden rounded bg-zinc-800">
                  <div
                    className={`h-full transition-[width] duration-75 ${
                      phase === "listening" ? "bg-emerald-400" : "bg-zinc-600"
                    }`}
                    style={{ width: `${Math.round(level * 100)}%` }}
                  />
                </div>
                {micMode === "hold" && (
                  <button
                    onPointerDown={holdStart}
                    onPointerUp={holdEnd}
                    onPointerLeave={holdEnd}
                    disabled={phase !== "listening"}
                    className="select-none rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold enabled:active:bg-emerald-400 disabled:opacity-40"
                  >
                    Hold to talk
                  </button>
                )}
                <button
                  onClick={() =>
                    setMicMode((m) => (m === "auto" ? "hold" : "auto"))
                  }
                  className="rounded-lg border border-zinc-700 px-3 py-2 text-xs text-zinc-300 transition-colors hover:border-zinc-500"
                >
                  {micMode === "auto" ? "Auto voice" : "Push to talk"}
                </button>
                <button
                  onClick={end}
                  className="rounded-lg border border-red-800 px-3 py-2 text-xs text-red-400 transition-colors hover:bg-red-950"
                >
                  End
                </button>
              </div>
            )}
          </div>
        </section>

        {/* field state */}
        <aside className="w-full shrink-0 md:w-72">
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
            <div className="mb-3 flex items-baseline justify-between">
              <h2 className="text-xs font-bold uppercase tracking-widest text-zinc-400">
                Clinical fields
              </h2>
              <span className="text-xs tabular-nums text-zinc-500">
                {knownCount}/{FIELD_ORDER.length}
              </span>
            </div>
            <ul className="space-y-2">
              {FIELD_ORDER.map((f) => (
                <li
                  key={f}
                  className="flex items-center justify-between rounded-lg border border-zinc-800/80 bg-zinc-950/50 px-3 py-2"
                >
                  <span className="text-sm text-zinc-300">{FIELD_LABELS[f]}</span>
                  <ValueBadge value={fields[f]} />
                </li>
              ))}
            </ul>
            <p className="mt-3 text-[11px] leading-relaxed text-zinc-500">
              Unknown never defaults to a benign value — it escalates in
              triage. Extraction is field-only; question order is deterministic.
            </p>
            {interviewId && (
              <p className="mt-2 font-mono text-[11px] text-zinc-500">
                interview_id: {interviewId}
              </p>
            )}
          </div>
        </aside>
      </div>
    </main>
  );
}
