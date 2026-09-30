/**
 * Per-subsystem status, read from `/api/health`.
 *
 * This is the panel that keeps the rest of the dashboard honest: it states
 * plainly which subsystems exist and which do not, so an empty panel elsewhere
 * is always attributable to a named, unimplemented component rather than being
 * read as an observation about traffic.
 */

import type { ComponentHealth, ComponentStatus, HealthResponse } from "../api/types";
import type { Resource } from "../hooks/useApiResource";
import { Panel } from "./Panel";
import { ResourceView } from "./ResourceState";
import { StatusPill, type Tone } from "./StatusPill";

const STATUS_TONE: Record<ComponentStatus, Tone> = {
  ok: "ok",
  not_implemented: "pending",
  not_configured: "pending",
  degraded: "pending",
  failed: "danger",
};

const STATUS_LABEL: Record<ComponentStatus, string> = {
  ok: "Operational",
  not_implemented: "Not implemented",
  not_configured: "Not configured",
  degraded: "Degraded",
  failed: "Failed",
};

function ComponentRow({ component }: { component: ComponentHealth }): JSX.Element {
  return (
    <div className="row">
      <span className="row__name">{component.name}</span>
      <span className="row__detail">{component.detail ?? "—"}</span>
      <span className="row__aside">
        <StatusPill tone={STATUS_TONE[component.status]} dot>
          {component.phase !== null && component.status === "not_implemented"
            ? `Phase ${component.phase}`
            : STATUS_LABEL[component.status]}
        </StatusPill>
      </span>
    </div>
  );
}

export function SystemStatusPanel({
  health,
  onRetry,
}: {
  health: Resource<HealthResponse>;
  onRetry: () => void;
}): JSX.Element {
  const implemented =
    health.state === "success"
      ? health.data.components.filter((c) => c.status === "ok").length
      : null;
  const total = health.state === "success" ? health.data.components.length : null;

  return (
    <Panel
      title="System status"
      subtitle="Live subsystem health reported by the backend"
      flush
      aside={
        implemented !== null && total !== null ? (
          <StatusPill title="Subsystems reporting as operational">
            {implemented}/{total} operational
          </StatusPill>
        ) : undefined
      }
    >
      <div style={{ padding: health.state === "success" ? 0 : 16 }}>
        <ResourceView resource={health} onRetry={onRetry} loadingRows={5}>
          {(data) => (
            <div className="rows">
              {data.components.map((component) => (
                <ComponentRow key={component.name} component={component} />
              ))}
            </div>
          )}
        </ResourceView>
      </div>
    </Panel>
  );
}
