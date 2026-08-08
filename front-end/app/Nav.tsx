"use client";

import Link from "next/link";
import type { ReactNode } from "react";

const TABS = [
  { href: "/", key: "dashboard", label: "Command Dashboard" },
  { href: "/rover", key: "rover", label: "Rover Console" },
] as const;

export type TabKey = (typeof TABS)[number]["key"];

export default function Nav({
  active,
  status,
}: {
  active: TabKey;
  status?: ReactNode;
}) {
  return (
    <header className="sticky top-0 z-10 border-b border-zinc-800 bg-zinc-950/90 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center gap-4 px-4 py-2.5">
        <Link href="/" className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-red-600 text-lg font-black text-white shadow-[0_0_14px_rgba(220,38,38,0.45)]">
            +
          </div>
          <div className="leading-tight">
            <p className="text-sm font-extrabold tracking-wide">SaveWise</p>
            <p className="text-[10px] uppercase tracking-[0.18em] text-zinc-500">
              rescue intelligence
            </p>
          </div>
        </Link>

        <nav className="ml-4 flex items-center gap-1 rounded-lg border border-zinc-800 bg-zinc-900/60 p-1">
          {TABS.map((t) => (
            <Link
              key={t.key}
              href={t.href}
              className={`rounded-md px-3 py-1.5 text-xs font-semibold transition-colors ${
                active === t.key
                  ? "bg-zinc-100 text-zinc-950"
                  : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
              }`}
            >
              {t.label}
            </Link>
          ))}
        </nav>

        <div className="ml-auto flex items-center gap-3">{status}</div>
      </div>
    </header>
  );
}
