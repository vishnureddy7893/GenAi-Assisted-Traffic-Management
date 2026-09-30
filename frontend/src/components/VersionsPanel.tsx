/**
 * Version pins recorded into every decision.
 *
 * Displayed because a recommendation is only auditable if the versions that
 * produced it are recoverable. Code, schema, safety rules and route weights are
 * pinned independently so a behaviour change can be attributed to the component
 * that actually caused it.
 *
 * `model_versions` is empty until Phase 10, and is rendered as explicitly empty
 * rather than omitted — "no model contributed" is information.
 */

import type { VersionsResponse } from "../api/types";
import type { Resource } from "../hooks/useApiResource";
import { Panel } from "./Panel";
import { ResourceView } from "./ResourceState";
import { StatusPill } from "./StatusPill";

export function VersionsPanel({
  versions,
  onRetry,
}: {
  versions: Resource<VersionsResponse>;
  onRetry: () => void;
}): JSX.Element {
  return (
    <Panel
      title="Version pins"
      subtitle="Recorded into every decision record for replay"
      aside={<StatusPill tone="ok" dot>Tracked</StatusPill>}
    >
      <ResourceView resource={versions} onRetry={onRetry} loadingRows={4}>
        {(data) => {
          const modelCount = Object.keys(data.model_versions).length;
          return (
            <div className="kv">
              <span className="kv__key">Application</span>
              <span className="kv__value">{data.app_version}</span>

              <span className="kv__key">Domain schema</span>
              <span className="kv__value">{data.schema_version}</span>

              <span className="kv__key">Safety rules</span>
              <span className="kv__value">{data.safety_rules_version}</span>

              <span className="kv__key">Route weights</span>
              <span className="kv__value">{data.route_weights_version}</span>

              <span className="kv__key">Models</span>
              <span className="kv__value">
                {modelCount === 0
                  ? "none — no model trained or loaded"
                  : Object.entries(data.model_versions)
                      .map(([role, version]) => `${role}=${version}`)
                      .join(", ")}
              </span>
            </div>
          );
        }}
      </ResourceView>
    </Panel>
  );
}
