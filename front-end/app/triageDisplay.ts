// Shared triage display constants — used by the dashboard and the
// reasoning panel. Always icon + label, never color alone.

export type Category = "immediate" | "delayed" | "minor";

export const CATEGORY_ORDER: Category[] = ["immediate", "delayed", "minor"];

export const CAT: Record<
  Category,
  { label: string; icon: string; badge: string; edge: string }
> = {
  immediate: {
    label: "IMMEDIATE",
    icon: "▲",
    badge: "bg-red-500/15 text-red-400 border border-red-500/40",
    edge: "border-l-red-500",
  },
  delayed: {
    label: "DELAYED",
    icon: "◆",
    badge: "bg-amber-400/15 text-amber-300 border border-amber-400/40",
    edge: "border-l-amber-400",
  },
  minor: {
    label: "MINOR",
    icon: "●",
    badge: "bg-emerald-500/15 text-emerald-300 border border-emerald-500/40",
    edge: "border-l-emerald-500",
  },
};

export function apiBase(): string {
  if (typeof window === "undefined") return "";
  return `${window.location.protocol}//${window.location.hostname}:8000`;
}

export function wsBase(): string {
  if (typeof window === "undefined") return "";
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.hostname}:8000`;
}
