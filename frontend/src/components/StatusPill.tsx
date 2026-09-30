/**
 * Status pill.
 *
 * `tone` is derived from meaning, never chosen for looks. The mapping is fixed
 * across the dashboard so an operator can read colour without reading text:
 *
 * - `ok`      — the subsystem works
 * - `pending` — the capability does not exist yet (a phase boundary, not a fault)
 * - `danger`  — a genuine failure
 * - `neutral` — informational metadata
 */

import type { ReactNode } from "react";

export type Tone = "ok" | "pending" | "danger" | "neutral" | "accent";

export interface StatusPillProps {
  tone?: Tone;
  /** Render a leading status dot. */
  dot?: boolean;
  /** Animate the dot, for a genuinely in-flight condition. */
  pulse?: boolean;
  title?: string;
  children: ReactNode;
}

const TONE_CLASS: Record<Tone, string> = {
  ok: "pill--ok",
  pending: "pill--pending",
  danger: "pill--danger",
  accent: "pill--accent",
  neutral: "",
};

export function StatusPill({
  tone = "neutral",
  dot = false,
  pulse = false,
  title,
  children,
}: StatusPillProps): JSX.Element {
  const classes = ["pill", TONE_CLASS[tone]].filter(Boolean).join(" ");
  return (
    <span className={classes} title={title}>
      {dot && <span className={`pill__dot${pulse ? " pill__dot--pulse" : ""}`} />}
      {children}
    </span>
  );
}

/** Badge naming the development phase that will deliver a capability. */
export function PhaseBadge({ phase }: { phase: number | null }): JSX.Element {
  if (phase === null) {
    return <StatusPill tone="pending">Not implemented</StatusPill>;
  }
  return (
    <StatusPill tone="pending" title={`Delivered in phase ${phase}`}>
      Phase {phase}
    </StatusPill>
  );
}
