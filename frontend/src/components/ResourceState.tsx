/**
 * Renderers for the non-success states of a {@link Resource}.
 *
 * Each state gets a visually distinct treatment on purpose. The failure this
 * guards against is the one where "the request failed", "this feature does not
 * exist yet" and "the roads are empty" all render as the same blank panel — at
 * which point the dashboard is actively misleading rather than merely unhelpful.
 */

import type { ReactNode } from "react";
import type { Resource } from "../hooks/useApiResource";

/** Placeholder shown while a request is in flight. */
export function LoadingState({ rows = 3 }: { rows?: number }): JSX.Element {
  return (
    <div className="skeleton-stack" role="status" aria-live="polite">
      <span className="visually-hidden">Loading…</span>
      {Array.from({ length: rows }, (_, index) => (
        <div
          key={index}
          className="skeleton"
          style={{ width: `${100 - index * 12}%` }}
          aria-hidden="true"
        />
      ))}
    </div>
  );
}

/**
 * Shown when the backend reports 501 — the capability belongs to a later phase.
 *
 * Deliberately *not* styled as an error. This is the system correctly reporting
 * a boundary of what has been built, and presenting it in red would train an
 * operator to ignore red.
 */
export function UnavailableState({
  message,
  phase,
  plannedCapabilities,
}: {
  message: string;
  phase: number | null;
  /** What this panel will show once the phase lands. */
  plannedCapabilities?: string;
}): JSX.Element {
  return (
    <div className="state">
      <div className="state__icon state__icon--pending" aria-hidden="true">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 8v5" strokeLinecap="round" />
          <circle cx="12" cy="16.5" r="0.6" fill="currentColor" stroke="none" />
          <circle cx="12" cy="12" r="9" />
        </svg>
      </div>
      <p className="state__title">
        {phase === null ? "Not implemented yet" : `Not implemented yet — phase ${phase}`}
      </p>
      <p className="state__message">{message}</p>
      {plannedCapabilities !== undefined && (
        <p className="state__hint">{plannedCapabilities}</p>
      )}
    </div>
  );
}

/** Shown when the request genuinely failed. */
export function ErrorState({
  message,
  unreachable,
  onRetry,
}: {
  message: string;
  unreachable: boolean;
  onRetry?: () => void;
}): JSX.Element {
  return (
    <div className="state" role="alert">
      <div className="state__icon state__icon--danger" aria-hidden="true">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 8v4.5" strokeLinecap="round" />
          <circle cx="12" cy="16.5" r="0.6" fill="currentColor" stroke="none" />
          <path d="M10.3 3.9 2.5 17.4a2 2 0 0 0 1.7 3h15.6a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" strokeLinejoin="round" />
        </svg>
      </div>
      <p className="state__title">
        {unreachable ? "Backend unreachable" : "Request failed"}
      </p>
      <p className="state__message">{message}</p>
      {unreachable && (
        <p className="state__hint">
          Start the API with <code>uvicorn app.main:app --reload</code> from the{" "}
          <code>backend/</code> directory.
        </p>
      )}
      {onRetry !== undefined && (
        <button type="button" className="button" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

/** Shown when a request succeeded and the result is genuinely, verifiably empty. */
export function EmptyState({
  title,
  message,
}: {
  title: string;
  message: string;
}): JSX.Element {
  return (
    <div className="state">
      <div className="state__icon" aria-hidden="true">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="3.5" y="5" width="17" height="14" rx="2" />
          <path d="M3.5 10h17" />
        </svg>
      </div>
      <p className="state__title">{title}</p>
      <p className="state__message">{message}</p>
    </div>
  );
}

/**
 * Render a resource, delegating the success case to `children`.
 *
 * Exhaustive over the union, so adding a state to {@link Resource} is a compile
 * error here rather than a silently unrendered case.
 */
export function ResourceView<T>({
  resource,
  onRetry,
  loadingRows,
  plannedCapabilities,
  children,
}: {
  resource: Resource<T>;
  onRetry?: () => void;
  loadingRows?: number;
  plannedCapabilities?: string;
  children: (data: T) => ReactNode;
}): JSX.Element {
  switch (resource.state) {
    case "loading":
      return <LoadingState {...(loadingRows !== undefined ? { rows: loadingRows } : {})} />;
    case "unavailable":
      return (
        <UnavailableState
          message={resource.message}
          phase={resource.phase}
          {...(plannedCapabilities !== undefined ? { plannedCapabilities } : {})}
        />
      );
    case "error":
      return (
        <ErrorState
          message={resource.message}
          unreachable={resource.unreachable}
          {...(onRetry !== undefined ? { onRetry } : {})}
        />
      );
    case "success":
      return <>{children(resource.data)}</>;
  }
}
