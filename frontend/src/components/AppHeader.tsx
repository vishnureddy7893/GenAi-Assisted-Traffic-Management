/**
 * Application header with live backend connection status.
 *
 * The connection pill reports two independent facts, because merging them would
 * have to misstate one: whether the backend is reachable, and whether an
 * end-to-end decision pipeline exists. Through Phase 4 those are "yes" and "no".
 */

import type { Resource } from "../hooks/useApiResource";
import type { HealthResponse } from "../api/types";
import { StatusPill } from "./StatusPill";

function ConnectionPill({ resource }: { resource: Resource<HealthResponse> }): JSX.Element {
  switch (resource.state) {
    case "loading":
      return (
        <StatusPill dot pulse tone="neutral">
          Connecting…
        </StatusPill>
      );
    case "error":
      return (
        <StatusPill dot tone="danger" title={resource.message}>
          {resource.unreachable ? "Backend offline" : "Backend error"}
        </StatusPill>
      );
    case "unavailable":
      return (
        <StatusPill dot tone="pending">
          Unavailable
        </StatusPill>
      );
    case "success":
      return (
        <StatusPill dot tone="ok" title={`Environment: ${resource.data.environment}`}>
          Backend connected
        </StatusPill>
      );
  }
}

export function AppHeader({
  health,
}: {
  health: Resource<HealthResponse>;
}): JSX.Element {
  const data = health.state === "success" ? health.data : null;

  return (
    <header className="header">
      <div className="header__inner">
        <div className="header__mark" aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="7" cy="7" r="2.4" />
            <circle cx="17" cy="17" r="2.4" />
            <path d="M7 9.4V15a2 2 0 0 0 2 2h5.6" strokeLinecap="round" />
            <path d="M17 14.6V9a2 2 0 0 0-2-2H9.4" strokeLinecap="round" />
          </svg>
        </div>

        <div className="header__titles">
          <h1 className="header__title">GenAI Traffic Management</h1>
          <span className="header__subtitle">
            Multi-junction traffic and emergency decision support
          </span>
        </div>

        <div className="header__spacer" />

        <div className="header__meta">
          {data !== null && (
            <StatusPill tone="accent" title={data.implemented_phase_name}>
              {data.implemented_phase_name}
            </StatusPill>
          )}
          {data !== null && (
            <StatusPill
              tone={data.pipeline_ready ? "ok" : "pending"}
              title={
                data.pipeline_ready
                  ? "An end-to-end decision can be produced."
                  : `${data.pipeline_blockers.length} subsystem(s) not implemented yet.`
              }
            >
              Pipeline {data.pipeline_ready ? "ready" : "not ready"}
            </StatusPill>
          )}
          {data !== null && (
            <StatusPill title="Application version">v{data.app_version}</StatusPill>
          )}
          <ConnectionPill resource={health} />
        </div>
      </div>
    </header>
  );
}
