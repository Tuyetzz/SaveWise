"use client";

// Assessment Reasoning Card: one panel showing the whole decision chain in
// pipeline order — heard (raw speech) → extracted (LLM) → classified
// (deterministic START rules) → ranked (queue position). The visual boundary
// between the LLM section and the rules section is the point of the panel.

import { useCallback, useEffect, useRef, useState } from "react";
import { FIELD_LABELS, FIELD_ORDER, type FieldName } from "./useInterview";
import { apiBase, CAT, wsBase, type Category } from "./triageDisplay";

type FieldValue = boolean | number | string | null;

interface Reasoning {
  case: {
    id: string;
    status: string;
    started_at: string;
    ended_at: string | null;
    workflow: string;
  };
  assessment: {
    version: number;
    source: string;
    created_at: string | null;
    fields: Record<
      FieldName,
      { value: FieldValue; known: boolean; changed_this_turn: boolean }
    >;
    chief_complaint: string | null;
    completeness_pct: number;
    unknown_fields: FieldName[];
  };
  transcript: {
    turns: {
      index: number;
      question_text: string | null;
      transcript: string | null;
      audio_duration_ms: number | null;
    }[];
  };
  decision: {
    category: Category;
    previous_category: Category | null;
    changed_category: boolean;
    rule_fired: string;
    trace: {
      step: number;
      phase: "gate" | "rule" | "score";
      condition: string;
      evaluated: Record<string, unknown>;
      result: boolean;
      effect?: string;
      points?: number;
    }[];
    unknown_escalated: boolean;
    unknown_gates: FieldName[];
    overridden: boolean;
    displayed_category: Category;
  };
  ranking: {
    previous_position: number | null;
    current_position: number;
    delta: number | null;
    queue_size: number;
    urgency_score: number;
    moved_ahead_of: string[];
  };
}

function fmtValue(v: FieldValue): string {
  if (v === null || v === undefined) return "unknown";
  if (typeof v === "boolean") return v ? "yes" : "NO";
  return String(v);
}

function fmtEvaluated(evaluated: Record<string, unknown>): string {
  return Object.entries(evaluated)
    .map(([k, v]) => {
      const shown = Array.isArray(v)
        ? v.length
          ? v.join(", ")
          : "none"
        : v === null
          ? "unknown"
          : String(v);
      return `${k} = ${shown}`;
    })
    .join(" · ");
}

function SectionHeader({
  n,
  title,
  chip,
  chipTone,
}: {
  n: number;
  title: string;
  chip: string;
  chipTone: string;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="font-mono text-xs text-zinc-600">{n}</span>
      <h3 className="text-xs font-bold uppercase tracking-widest text-zinc-300">
        {title}
      </h3>
      <span
        className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${chipTone}`}
      >
        {chip}
      </span>
    </div>
  );
}

function FlowArrow() {
  return <p className="py-1 text-center text-zinc-700">▼</p>;
}

export default function ReasoningPanel({
  caseId,
  onClose,
}: {
  caseId: string;
  onClose: () => void;
}) {
  const [data, setData] = useState<Reasoning | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Bumped when a live update arrives while the panel is open, so changed
  // fields and the position remount and replay their flash animation.
  const [liveVersion, setLiveVersion] = useState(0);
  const prevRef = useRef<Reasoning | null>(null);

  const load = useCallback(async () => {
    try {
      const r = await fetch(`${apiBase()}/api/cases/${caseId}/reasoning`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const next: Reasoning = await r.json();
      const prev = prevRef.current;
      if (
        prev &&
        (prev.assessment.version !== next.assessment.version ||
          prev.ranking.current_position !== next.ranking.current_position ||
          prev.decision.displayed_category !== next.decision.displayed_category)
      ) {
        setLiveVersion((v) => v + 1);
      }
      prevRef.current = next;
      setData(next);
      setError(null);
    } catch (e) {
      setError(`Could not load reasoning: ${e}`);
    }
  }, [caseId]);

  // Initial load + live updates over /ws/dashboard, with slow-poll fallback.
  useEffect(() => {
    prevRef.current = null;
    setData(null);
    load();
    let ws: WebSocket | null = null;
    let closed = false;
    const connect = () => {
      ws = new WebSocket(`${wsBase()}/api/ws/dashboard`);
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.t === "case_updated" && msg.case_id === caseId) load();
        } catch {
          // ignore malformed frames
        }
      };
      ws.onclose = () => {
        if (!closed) setTimeout(connect, 2000);
      };
    };
    connect();
    const poll = setInterval(load, 5000);
    return () => {
      closed = true;
      ws?.close();
      clearInterval(poll);
    };
  }, [caseId, load]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const gateTrace = data?.decision.trace.filter((e) => e.phase === "gate") ?? [];
  const ruleTrace = data?.decision.trace.filter((e) => e.phase === "rule") ?? [];
  const scoreTrace =
    data?.decision.trace.filter((e) => e.phase === "score") ?? [];

  return (
    <>
      <div
        className="fixed inset-0 z-40 bg-black/60"
        onClick={onClose}
        aria-hidden
      />
      <aside
        className="reasoning-slide-in fixed inset-y-0 right-0 z-50 w-full max-w-lg overflow-y-auto border-l border-zinc-800 bg-zinc-950 text-zinc-100 shadow-2xl"
        role="dialog"
        aria-label={`Reasoning for case ${caseId}`}
      >
        <div className="sticky top-0 z-10 flex items-center gap-2 border-b border-zinc-800 bg-zinc-950/95 px-4 py-3 backdrop-blur">
          <h2 className="text-sm font-bold tracking-wide">Why this triage?</h2>
          <span className="font-mono text-xs text-zinc-500">{caseId}</span>
          {data && (
            <span className="text-[10px] uppercase tracking-wider text-zinc-600">
              assessment v{data.assessment.version}
            </span>
          )}
          <button
            onClick={onClose}
            className="ml-auto rounded-md border border-zinc-700 px-2 py-1 text-xs text-zinc-400 transition-colors hover:border-zinc-500"
          >
            ✕ Close
          </button>
        </div>

        {error && (
          <p className="m-4 rounded-lg border border-red-800 bg-red-950/40 px-3 py-2 text-xs text-red-300">
            {error}
          </p>
        )}
        {!data && !error && (
          <p className="m-4 text-xs text-zinc-500">Loading reasoning…</p>
        )}

        {data && (
          <div className="space-y-1 p-4">
            {/* 1 · HEARD */}
            <section className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-3">
              <SectionHeader
                n={1}
                title="Heard"
                chip="raw audio → whisper"
                chipTone="border-zinc-700 text-zinc-400"
              />
              <p className="mt-1 text-[10px] text-zinc-600">
                Verbatim transcripts — disfluencies preserved on purpose.
              </p>
              <ol className="mt-2 space-y-2.5">
                {data.transcript.turns.map((t) => (
                  <li key={t.index} className="text-xs">
                    <p className="text-sky-400">{t.question_text}</p>
                    <div className="mt-0.5 flex items-baseline gap-2">
                      <p className="min-w-0 flex-1 text-zinc-200">
                        ↳ “{t.transcript}”
                      </p>
                      {t.audio_duration_ms !== null && (
                        <span className="shrink-0 rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-zinc-500">
                          {(t.audio_duration_ms / 1000).toFixed(1)}s
                        </span>
                      )}
                    </div>
                  </li>
                ))}
                {data.transcript.turns.length === 0 && (
                  <p className="text-xs text-zinc-500">
                    No answers recorded — survivor never responded.
                  </p>
                )}
              </ol>
            </section>

            <FlowArrow />

            {/* 2 · EXTRACTED — LLM side of the boundary */}
            <section className="rounded-xl border border-fuchsia-500/30 bg-fuchsia-500/5 p-3">
              <SectionHeader
                n={2}
                title="Extracted"
                chip="🤖 LLM output — fields only"
                chipTone="border-fuchsia-500/40 text-fuchsia-300"
              />
              <p className="mt-1 text-[10px] text-zinc-500">
                The model turns messy speech into these values and nothing
                else. It never sees or picks a triage category.
              </p>
              <div className="mt-2 grid grid-cols-2 gap-1.5">
                {FIELD_ORDER.map((name) => {
                  const f = data.assessment.fields[name];
                  const changed = f?.changed_this_turn;
                  return (
                    <div
                      key={`${name}-v${data.assessment.version}-${liveVersion}`}
                      className={`rounded-md px-2 py-1.5 ${
                        !f?.known
                          ? "border border-amber-500/40 bg-amber-500/10"
                          : "border border-zinc-800 bg-zinc-900/60"
                      } ${changed ? "reasoning-flash border-sky-400/60" : ""}`}
                    >
                      <p className="text-[10px] uppercase tracking-wider text-zinc-500">
                        {FIELD_LABELS[name]}
                        {changed && (
                          <span className="ml-1 text-sky-400">● new</span>
                        )}
                      </p>
                      <p
                        className={`mt-0.5 text-xs ${
                          !f?.known
                            ? "font-semibold text-amber-400"
                            : f?.value === false
                              ? "font-semibold text-red-400"
                              : "text-zinc-200"
                        }`}
                      >
                        {!f?.known ? "? unknown" : fmtValue(f.value)}
                      </p>
                    </div>
                  );
                })}
              </div>
              <p className="mt-2 text-[10px] tabular-nums text-zinc-500">
                {data.assessment.completeness_pct}% of fields known
                {data.assessment.chief_complaint &&
                  ` · chief complaint: ${data.assessment.chief_complaint}`}
              </p>
            </section>

            {/* THE BOUNDARY — model above, code below */}
            <div className="flex items-center gap-3 py-2">
              <span className="h-px flex-1 bg-zinc-700" />
              <span className="text-[10px] font-semibold uppercase tracking-widest text-zinc-400">
                model stops here — below is deterministic code
              </span>
              <span className="h-px flex-1 bg-zinc-700" />
            </div>

            {/* 3 · CLASSIFIED — deterministic side */}
            <section className="rounded-xl border border-sky-500/30 bg-sky-500/5 p-3">
              <SectionHeader
                n={3}
                title="Classified"
                chip="⚙ deterministic START rules — no model"
                chipTone="border-sky-500/40 text-sky-300"
              />
              <div
                key={`cat-${data.decision.category}-${liveVersion}`}
                className="reasoning-flash mt-2 flex flex-wrap items-center gap-2"
              >
                {data.decision.previous_category &&
                  data.decision.changed_category && (
                    <>
                      <span
                        className={`rounded px-2 py-0.5 text-[11px] font-bold tracking-wider opacity-50 ${CAT[data.decision.previous_category].badge}`}
                      >
                        {CAT[data.decision.previous_category].icon}{" "}
                        {CAT[data.decision.previous_category].label}
                      </span>
                      <span className="text-zinc-500">→</span>
                    </>
                  )}
                <span
                  className={`rounded px-2 py-0.5 text-[11px] font-bold tracking-wider ${CAT[data.decision.category].badge}`}
                >
                  {CAT[data.decision.category].icon}{" "}
                  {CAT[data.decision.category].label}
                </span>
                <span className="font-mono text-[11px] text-zinc-400">
                  rule: {data.decision.rule_fired}
                </span>
              </div>
              {data.decision.overridden && (
                <p className="mt-1.5 rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 text-[11px] text-zinc-400">
                  ⚑ Responder override active — queue shows{" "}
                  <span className="font-semibold uppercase">
                    {data.decision.displayed_category}
                  </span>
                  ; the trace below is the automatic classification.
                </p>
              )}

              {data.decision.unknown_escalated &&
                data.decision.unknown_gates.map((g) => (
                  <p
                    key={g}
                    className="mt-1.5 rounded-md border border-amber-500/50 bg-amber-500/10 px-2 py-1 text-[11px] font-semibold text-amber-300"
                  >
                    ⚠ {FIELD_LABELS[g] ?? g} unassessable — escalated to
                    IMMEDIATE. Unknown is never assumed benign.
                  </p>
                ))}

              <p className="mb-1 mt-3 text-[10px] uppercase tracking-widest text-zinc-500">
                Gates checked
              </p>
              <ol className="space-y-1">
                {gateTrace.map((e) => {
                  const value = Object.values(e.evaluated)[0];
                  const unknown = value === null;
                  return (
                    <li
                      key={e.step}
                      className="flex items-baseline gap-2 text-[11px]"
                    >
                      <span
                        className={
                          e.result
                            ? "font-bold text-red-400"
                            : unknown
                              ? "font-bold text-amber-400"
                              : "text-emerald-400"
                        }
                      >
                        {e.result ? "✗" : unknown ? "?" : "✓"}
                      </span>
                      <span className="font-mono text-zinc-300">
                        {e.condition}
                      </span>
                      <span className="ml-auto text-right text-zinc-500">
                        {e.result
                          ? (e.effect ?? "failed")
                          : unknown
                            ? "unknown"
                            : "pass"}
                      </span>
                    </li>
                  );
                })}
              </ol>

              <p className="mb-1 mt-3 text-[10px] uppercase tracking-widest text-zinc-500">
                Rules, in precedence order — first match wins
              </p>
              <ol className="space-y-1">
                {ruleTrace.map((e) => (
                  <li key={e.step} className="text-[11px]">
                    <div className="flex items-baseline gap-2">
                      <span
                        className={
                          e.result
                            ? "font-bold text-sky-300"
                            : "text-zinc-600"
                        }
                      >
                        {e.result ? "▶" : "·"}
                      </span>
                      <span
                        className={`font-mono ${e.result ? "text-zinc-100" : "text-zinc-500"}`}
                      >
                        {e.condition}
                      </span>
                      <span
                        className={`ml-auto text-right ${e.result ? "font-semibold text-sky-300" : "text-zinc-600"}`}
                      >
                        {e.result ? e.effect : "no match"}
                      </span>
                    </div>
                    <p className="ml-5 font-mono text-[10px] text-zinc-600">
                      {fmtEvaluated(e.evaluated)}
                    </p>
                  </li>
                ))}
              </ol>

              <p className="mb-1 mt-3 text-[10px] uppercase tracking-widest text-zinc-500">
                Urgency score — additive, explainable
              </p>
              <ol className="space-y-0.5">
                {scoreTrace
                  .filter((e) => e.result)
                  .map((e) => (
                    <li
                      key={e.step}
                      className="flex items-baseline gap-2 text-[11px]"
                    >
                      <span className="w-10 shrink-0 text-right font-mono tabular-nums text-zinc-300">
                        +{e.points}
                      </span>
                      <span className="text-zinc-400">
                        {e.condition}
                        {e.effect ? ` — ${e.effect}` : ""}
                      </span>
                    </li>
                  ))}
                <li className="mt-1 flex items-baseline gap-2 border-t border-zinc-800 pt-1 text-[11px]">
                  <span className="w-10 shrink-0 text-right font-mono font-bold tabular-nums text-zinc-100">
                    {data.ranking.urgency_score}
                  </span>
                  <span className="font-semibold text-zinc-300">total</span>
                </li>
              </ol>
            </section>

            <FlowArrow />

            {/* 4 · RANKED */}
            <section className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-3">
              <SectionHeader
                n={4}
                title="Ranked"
                chip="queue effect"
                chipTone="border-zinc-700 text-zinc-400"
              />
              <div
                key={`pos-${data.ranking.current_position}-${liveVersion}`}
                className="reasoning-flash mt-2 flex items-baseline gap-3"
              >
                <p className="text-2xl font-bold tabular-nums">
                  {data.ranking.previous_position !== null &&
                    data.ranking.previous_position !==
                      data.ranking.current_position && (
                      <span className="text-zinc-600">
                        #{data.ranking.previous_position}{" "}
                        <span className="text-base">→</span>{" "}
                      </span>
                    )}
                  #{data.ranking.current_position}
                  <span className="text-sm font-normal text-zinc-500">
                    {" "}
                    of {data.ranking.queue_size}
                  </span>
                </p>
                {data.ranking.delta !== null && data.ranking.delta !== 0 && (
                  <span
                    className={`text-sm font-bold tabular-nums ${
                      data.ranking.delta > 0 ? "text-red-400" : "text-emerald-400"
                    }`}
                  >
                    {data.ranking.delta > 0
                      ? `▲ up ${data.ranking.delta}`
                      : `▼ down ${-data.ranking.delta}`}
                  </span>
                )}
              </div>
              <p className="mt-1 text-[11px] tabular-nums text-zinc-500">
                urgency score {data.ranking.urgency_score}
              </p>
              {data.ranking.moved_ahead_of.length > 0 && (
                <div className="mt-2">
                  <p className="text-[10px] uppercase tracking-widest text-zinc-500">
                    Now ahead of
                  </p>
                  <div className="mt-1 flex flex-wrap gap-1.5">
                    {data.ranking.moved_ahead_of.map((id) => (
                      <span
                        key={id}
                        className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[11px] text-zinc-300"
                      >
                        {id}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </section>
          </div>
        )}
      </aside>
    </>
  );
}
