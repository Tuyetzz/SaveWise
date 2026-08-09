"use client";

import { useCallback, useEffect, useState } from "react";
import Nav from "./Nav";
import ReasoningPanel from "./ReasoningPanel";
import { apiBase, CAT, CATEGORY_ORDER, type Category } from "./triageDisplay";
import { type FieldName } from "./useInterview";

type Workflow = "outstanding" | "dispatched" | "rescued";

interface CaseSummary {
  id: string;
  status: string;
  started_at: string;
  last_heard_at: string;
  turns: number;
  fields: Partial<Record<FieldName, boolean | number | string | null>>;
  category: Category;
  auto_category: Category;
  overridden: boolean;
  workflow: Workflow;
  notes_count: number;
  latest_note: string | null;
  score: number;
  reasons: string[];
  unknown_gates: string[];
}

interface TimelineEntry {
  seq: number;
  question: string | null;
  answer: string | null;
  fields_after: Record<string, boolean | number | string | null> | null;
}

interface CaseEvent {
  kind: string;
  value: string | null;
  note: string | null;
  created_at: string;
}

interface CaseDetail {
  timeline: TimelineEntry[];
  events: CaseEvent[];
}

interface Dashboard {
  cases: CaseSummary[];
  counts: Record<Category, number>;
  active: number;
  no_response: number;
  rescued: number;
  people_reported: number;
  others_unaccounted: number;
  unaccounted: {
    case_id: string;
    others: number;
    last_seen: string | null;
    category: Category;
  }[];
}

const WORKFLOW_CHIP: Record<Workflow, string> = {
  outstanding: "",
  dispatched: "border-sky-500/40 text-sky-300",
  rescued: "border-emerald-500/40 text-emerald-300",
};

const STATUS_CHIP: Record<string, string> = {
  active: "text-sky-300 border-sky-500/40",
  complete: "text-zinc-400 border-zinc-700",
  abandoned: "text-zinc-500 border-zinc-700",
  no_response: "text-fuchsia-300 border-fuchsia-500/40",
};

function ago(iso: string): string {
  const s = Math.max(0, Math.round((Date.now() - new Date(iso + "Z").getTime()) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
}

function Tile({ value, label, tone }: { value: number | string; label: string; tone?: string }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 px-4 py-3 transition-colors hover:border-zinc-700">
      <p className={`text-2xl font-bold tabular-nums ${tone ?? "text-zinc-100"}`}>{value}</p>
      <p className="mt-0.5 text-[11px] uppercase tracking-widest text-zinc-500">{label}</p>
    </div>
  );
}

function vitalsChips(c: CaseSummary): { text: string; known: boolean }[] {
  const f = c.fields;
  const chip = (label: string, v: boolean | number | string | null | undefined) => ({
    text: `${label}: ${v === null || v === undefined ? "?" : typeof v === "boolean" ? (v ? "yes" : "NO") : v}`,
    known: v !== null && v !== undefined,
  });
  return [
    chip("breathing", f.breathing),
    chip("resp", f.respiratory_rate),
    chip("pulse", f.radial_pulse_present),
    chip("alert", f.obeys_commands),
    chip("walk", f.can_walk),
    chip("trapped", f.trapped),
  ];
}

function ActionButton({
  onClick,
  disabled,
  children,
  tone = "border-zinc-700 text-zinc-300 hover:border-zinc-500",
}: {
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
  tone?: string;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`rounded-md border px-2 py-1 text-[11px] font-semibold transition-colors disabled:opacity-30 ${tone}`}
    >
      {children}
    </button>
  );
}

function CaseRow({
  c,
  onChanged,
  onOpenReasoning,
}: {
  c: CaseSummary;
  onChanged: () => void;
  onOpenReasoning: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [detail, setDetail] = useState<CaseDetail | null>(null);
  const [noteDraft, setNoteDraft] = useState("");
  const cat = CAT[c.category];
  const rescued = c.workflow === "rescued";

  const loadDetail = useCallback(async () => {
    try {
      const r = await fetch(`${apiBase()}/api/cases/${c.id}`);
      setDetail(await r.json());
    } catch {
      // row header already shows everything vital
    }
  }, [c.id]);

  const toggle = () => {
    const next = !open;
    setOpen(next);
    if (next && !detail) loadDetail();
  };

  const act = async (body: { kind: string; value?: string; note?: string }) => {
    try {
      await fetch(`${apiBase()}/api/cases/${c.id}/events`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      onChanged();
      if (open) loadDetail();
    } catch {
      // next poll re-syncs either way
    }
  };

  const rank = CATEGORY_ORDER.indexOf(c.category);
  const escalate = () =>
    rank > 0 && act({ kind: "override", value: CATEGORY_ORDER[rank - 1] });
  const deescalate = () =>
    rank < CATEGORY_ORDER.length - 1 &&
    act({ kind: "override", value: CATEGORY_ORDER[rank + 1] });

  return (
    <div
      className={`rounded-xl border border-zinc-800 border-l-4 bg-zinc-900/60 transition-colors hover:border-zinc-700 ${cat.edge} ${rescued ? "opacity-55" : ""}`}
    >
      <button onClick={onOpenReasoning} className="block w-full px-4 pt-3 text-left">
        <div className="flex flex-wrap items-center gap-2">
          <span className={`rounded px-2 py-0.5 text-[11px] font-bold tracking-wider ${cat.badge}`}>
            {cat.icon} {cat.label}
          </span>
          {c.overridden && (
            <span className="text-[10px] uppercase tracking-wider text-zinc-500">
              manual · auto: {c.auto_category}
            </span>
          )}
          <span className="font-mono text-xs text-zinc-400">{c.id}</span>
          <span
            className={`rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wider ${STATUS_CHIP[c.status] ?? "text-zinc-400 border-zinc-700"}`}
          >
            {c.status === "active" ? "● live" : c.status.replace("_", " ")}
          </span>
          {c.workflow !== "outstanding" && (
            <span
              className={`rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wider ${WORKFLOW_CHIP[c.workflow]}`}
            >
              {c.workflow === "dispatched" ? "🚑 dispatched" : "✓ rescued"}
            </span>
          )}
          <span className="ml-auto flex items-center gap-2 text-[11px] tabular-nums text-zinc-500">
            score {c.score} · {ago(c.last_heard_at)}
            <span className="text-zinc-600">⚖ why?</span>
          </span>
        </div>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {vitalsChips(c).map((v) => (
            <span
              key={v.text}
              className={`rounded px-1.5 py-0.5 font-mono text-[11px] ${
                v.known ? "bg-zinc-800 text-zinc-300" : "bg-amber-500/10 text-amber-400"
              }`}
            >
              {v.text}
            </span>
          ))}
          {typeof c.fields.injuries === "string" && (
            <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[11px] text-zinc-300">
              {c.fields.injuries}
            </span>
          )}
        </div>
        <p className="mt-1.5 text-xs text-zinc-500">{c.reasons.join(" · ")}</p>
        {c.latest_note && (
          <p className="mt-1 text-xs text-zinc-400">
            📝 {c.latest_note}
            {c.notes_count > 1 && (
              <span className="text-zinc-600"> · {c.notes_count} notes</span>
            )}
          </p>
        )}
      </button>

      {/* responder actions */}
      <div className="flex flex-wrap items-center gap-1.5 px-4 pb-3 pt-2">
        <ActionButton onClick={escalate} disabled={rank === 0}>
          ▲ Escalate
        </ActionButton>
        <ActionButton onClick={deescalate} disabled={rank === CATEGORY_ORDER.length - 1}>
          ▼ De-escalate
        </ActionButton>
        {c.overridden && (
          <ActionButton onClick={() => act({ kind: "override", value: "auto" })}>
            ↺ Auto
          </ActionButton>
        )}
        <span className="mx-1 h-4 w-px bg-zinc-800" />
        {c.workflow === "outstanding" && (
          <ActionButton
            onClick={() => act({ kind: "workflow", value: "dispatched" })}
            tone="border-sky-700 text-sky-300 hover:border-sky-500"
          >
            🚑 Dispatch
          </ActionButton>
        )}
        {c.workflow !== "rescued" ? (
          <ActionButton
            onClick={() => act({ kind: "workflow", value: "rescued" })}
            tone="border-emerald-700 text-emerald-300 hover:border-emerald-500"
          >
            ✓ Rescued
          </ActionButton>
        ) : (
          <ActionButton onClick={() => act({ kind: "workflow", value: "outstanding" })}>
            ↺ Reopen
          </ActionButton>
        )}
        <span className="mx-1 h-4 w-px bg-zinc-800" />
        <ActionButton onClick={toggle}>
          {open ? "▾" : "▸"} Log
        </ActionButton>
      </div>

      {open && (
        <div className="space-y-3 border-t border-zinc-800 px-4 py-3">
          <div>
            <p className="mb-2 text-[10px] uppercase tracking-widest text-zinc-500">
              Interview replay
            </p>
            {!detail ? (
              <p className="text-xs text-zinc-500">Loading replay…</p>
            ) : (
              <ol className="space-y-2">
                {detail.timeline.map((t) => (
                  <li key={t.seq} className="text-xs">
                    <p className="text-sky-400">{t.question}</p>
                    <p className="mt-0.5 text-zinc-300">↳ “{t.answer}”</p>
                  </li>
                ))}
                {detail.timeline.length === 0 && (
                  <p className="text-xs text-zinc-500">
                    No answers recorded — survivor never responded.
                  </p>
                )}
              </ol>
            )}
          </div>

          <div>
            <p className="mb-2 text-[10px] uppercase tracking-widest text-zinc-500">
              Responder log
            </p>
            <ol className="space-y-1">
              {(detail?.events ?? []).map((e, i) => (
                <li key={i} className="text-xs text-zinc-400">
                  <span className="font-mono text-[10px] text-zinc-600">
                    {ago(e.created_at)}
                  </span>{" "}
                  {e.kind === "note"
                    ? `📝 ${e.note}`
                    : e.kind === "override"
                      ? e.value === "auto"
                        ? "↺ override cleared — back to auto triage"
                        : `⚑ manually set to ${e.value?.toUpperCase()}`
                      : `→ ${e.value}`}
                </li>
              ))}
              {(detail?.events ?? []).length === 0 && (
                <p className="text-xs text-zinc-600">No responder actions yet.</p>
              )}
            </ol>
            <form
              className="mt-2 flex gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                if (noteDraft.trim()) {
                  act({ kind: "note", note: noteDraft });
                  setNoteDraft("");
                }
              }}
            >
              <input
                value={noteDraft}
                onChange={(e) => setNoteDraft(e.target.value)}
                placeholder="Add note — e.g. “K9 unit on site, need crane”"
                className="flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-200 outline-none placeholder:text-zinc-600 focus:border-zinc-500"
              />
              <ActionButton onClick={() => {}} disabled={!noteDraft.trim()}>
                Add
              </ActionButton>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

export default function CommandDashboard() {
  const [data, setData] = useState<Dashboard | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<number>(0);
  const [reasoningId, setReasoningId] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const r = await fetch(`${apiBase()}/api/dashboard`);
      setData(await r.json());
      setError(null);
      setUpdatedAt(Date.now());
    } catch (e) {
      setError(`Backend unreachable: ${e}`);
    }
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, [load]);

  return (
    <main className="min-h-screen bg-zinc-950 font-sans text-zinc-100">
      <Nav
        active="dashboard"
        status={
          <span className="flex items-center gap-2 text-[11px] text-zinc-500">
            <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
            live · {updatedAt ? `${Math.round((Date.now() - updatedAt) / 1000)}s` : "…"}
          </span>
        }
      />

      <div className="mx-auto max-w-6xl space-y-4 p-4">
        {error && (
          <p className="rounded-lg border border-red-800 bg-red-950/40 px-3 py-2 text-xs text-red-300">
            {error}
          </p>
        )}

        {data && (
          <>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-7">
              <Tile value={data.counts.immediate} label="▲ Immediate" tone="text-red-400" />
              <Tile value={data.counts.delayed} label="◆ Delayed" tone="text-amber-300" />
              <Tile value={data.counts.minor} label="● Minor" tone="text-emerald-300" />
              <Tile value={data.no_response} label="No response" tone="text-fuchsia-300" />
              <Tile value={data.people_reported} label="Souls reported" />
              <Tile
                value={data.others_unaccounted}
                label="Unaccounted"
                tone={data.others_unaccounted > 0 ? "text-amber-300" : undefined}
              />
              <Tile value={data.rescued} label="✓ Rescued" tone="text-emerald-300" />
            </div>

            <div className="flex flex-col gap-4 lg:flex-row">
              <section className="min-w-0 flex-1 space-y-2">
                <div className="flex items-baseline justify-between">
                  <h2 className="text-xs font-bold uppercase tracking-widest text-zinc-400">
                    Cases by severity
                  </h2>
                  <span className="text-[11px] text-zinc-600">
                    deterministic START ranking · most urgent first
                  </span>
                </div>
                {data.cases.map((c) => (
                  <CaseRow
                    key={c.id}
                    c={c}
                    onChanged={load}
                    onOpenReasoning={() => setReasoningId(c.id)}
                  />
                ))}
                {data.cases.length === 0 && (
                  <div className="rounded-xl border border-dashed border-zinc-800 bg-zinc-900/40 p-10 text-center">
                    <p className="text-3xl">📡</p>
                    <p className="mt-2 text-sm text-zinc-400">
                      No cases yet — deploy a rover and start an interview.
                    </p>
                  </div>
                )}
              </section>

              <aside className="w-full shrink-0 lg:w-80">
                <h2 className="mb-2 text-xs font-bold uppercase tracking-widest text-zinc-400">
                  Potentially missing people
                </h2>
                <div className="space-y-2">
                  {data.unaccounted.map((u) => (
                    <div
                      key={u.case_id}
                      className="rounded-xl border border-zinc-800 bg-zinc-900/60 px-3 py-2 transition-colors hover:border-zinc-700"
                    >
                      <p className="text-sm text-zinc-200">
                        <span className="font-bold tabular-nums">{u.others}</span>{" "}
                        {u.others === 1 ? "person" : "people"} unaccounted
                      </p>
                      <p className="mt-0.5 text-xs text-zinc-400">
                        {u.last_seen ? `Last seen: ${u.last_seen}` : "Location unknown"}
                      </p>
                      <p className="mt-1 font-mono text-[10px] text-zinc-600">
                        reported by case {u.case_id}
                      </p>
                    </div>
                  ))}
                  {data.unaccounted.length === 0 && (
                    <p className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4 text-center text-xs text-zinc-500">
                      No unaccounted people reported.
                    </p>
                  )}
                </div>
                <p className="mt-3 text-[11px] leading-relaxed text-zinc-600">
                  Counts come from survivor reports (&ldquo;how many people were
                  inside?&rdquo;). Severity is deterministic START logic — unknown
                  fields escalate, never assume benign. No LLM in this ranking.
                </p>
              </aside>
            </div>
          </>
        )}

        {reasoningId && (
          <ReasoningPanel
            caseId={reasoningId}
            onClose={() => setReasoningId(null)}
          />
        )}

        <footer className="border-t border-zinc-900 pt-3 text-center text-[11px] text-zinc-700">
          SaveWise · voice triage rover network · hackathon proof of concept —
          one rover live today, built for a swarm
        </footer>
      </div>
    </main>
  );
}
