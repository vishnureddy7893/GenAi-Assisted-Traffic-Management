/**
 * Development roadmap, read from the backend.
 *
 * Sourced from `/api/meta/phases` rather than duplicated here, so the dashboard
 * and the API can never disagree about what has been built. Every "not
 * implemented" label elsewhere on this page refers to a phase in this list.
 */

import type { PhasesResponse } from "../api/types";
import type { Resource } from "../hooks/useApiResource";
import { Panel } from "./Panel";
import { ResourceView } from "./ResourceState";
import { StatusPill } from "./StatusPill";

export function PhasePanel({
  phases,
  onRetry,
}: {
  phases: Resource<PhasesResponse>;
  onRetry: () => void;
}): JSX.Element {
  return (
    <Panel
      title="Development roadmap"
      subtitle="Implementation status reported by the backend"
      flush
      aside={
        phases.state === "success" ? (
          <StatusPill tone="accent">
            {phases.data.phases.filter((p) => p.implemented).length}/
            {phases.data.phases.length} complete
          </StatusPill>
        ) : undefined
      }
    >
      <div style={{ padding: phases.state === "success" ? 0 : 16 }}>
        <ResourceView resource={phases} onRetry={onRetry} loadingRows={6}>
          {(data) => (
            <div>
              {data.phases.map((phase) => (
                <div
                  key={phase.phase}
                  className={phase.implemented ? "phase phase--done" : "phase"}
                >
                  <span className="phase__num">{phase.phase}</span>
                  <span>
                    <span className="phase__name">
                      {phase.name}
                      {phase.implemented && (
                        <>
                          {" "}
                          <StatusPill tone="ok">complete</StatusPill>
                        </>
                      )}
                    </span>
                    <p className="phase__summary">{phase.summary}</p>
                  </span>
                </div>
              ))}
            </div>
          )}
        </ResourceView>
      </div>
    </Panel>
  );
}
