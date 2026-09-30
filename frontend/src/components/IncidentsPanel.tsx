/**
 * Incidents — what is disrupting the network, and how far it reaches.
 *
 * The distinction this panel exists to keep visible is blocked versus affected.
 * A blocked segment is pruned from the route graph before any search happens; an
 * affected one is priced and stays usable. Collapsing them in the UI would
 * misrepresent the safety model that the backend is careful to enforce: a jammed
 * road is slow, a blocked road is impassable, and an operator needs to see which
 * one they just declared.
 *
 * Segment choices come from the loaded network, so an incident cannot be
 * declared against a road that does not exist. The backend validates this too
 * and would reject it — offering only real ids just means the operator finds out
 * before submitting rather than after.
 *
 * Lifecycle moves are offered from `permitted_next_statuses`, which the backend
 * returns per incident. The panel does not know the transition table, and must
 * not: a terminal incident simply comes back with no permitted moves and the
 * controls disappear.
 */

import { useCallback, useEffect, useState } from "react";

import { apiGet, apiPost, endpoints } from "../api/client";
import type {
  Incident,
  IncidentListResponse,
  IncidentResponse,
  IncidentSeverity,
  IncidentStatus,
  IncidentType,
  RoadNetwork,
} from "../api/types";
import type { Resource } from "../hooks/useApiResource";
import { Panel } from "./Panel";
import { EmptyState } from "./ResourceState";
import { StatusPill, type Tone } from "./StatusPill";

/** Presentation only; the option lists come from the backend's enums. */
const INCIDENT_TYPES: IncidentType[] = [
  "accident",
  "fire",
  "road_blockage",
  "stalled_vehicle",
  "flooding",
  "construction",
  "debris",
  "protest_or_event",
  "signal_failure",
  "structural_failure",
  "other",
];

const SEVERITIES: IncidentSeverity[] = ["minor", "moderate", "major", "critical"];

const STATUS_TONE: Record<IncidentStatus, Tone> = {
  reported: "pending",
  confirmed: "danger",
  responding: "danger",
  contained: "pending",
  cleared: "ok",
  false_alarm: "neutral",
};

function IncidentRow({
  entry,
  onTransition,
}: {
  entry: { incident: Incident; permitted: IncidentStatus[] };
  onTransition: (id: string, status: IncidentStatus) => void;
}): JSX.Element {
  const { incident, permitted } = entry;
  return (
    <div className="inc" data-testid={`incident-${incident.incident_id}`}>
      <div className="source__head">
        <span className="inc__type">
          {incident.incident_type.replace(/_/g, " ")}
        </span>
        <StatusPill tone={STATUS_TONE[incident.status]} dot>
          {incident.status.replace(/_/g, " ")}
        </StatusPill>
      </div>

      <p className="inc__where">
        {incident.location.segment_id !== null && (
          <>segment {incident.location.segment_id}</>
        )}
        {incident.location.junction_id !== null && (
          <>junction {incident.location.junction_id}</>
        )}
        {" · "}severity {incident.severity}
      </p>

      {incident.description !== null && (
        <p className="inc__desc">{incident.description}</p>
      )}

      <div className="inc__effects">
        {incident.blocks_segment_ids.length > 0 && (
          <span className="inc__effect">
            <StatusPill tone="danger">blocked</StatusPill>
            {incident.blocks_segment_ids.join(", ")}
          </span>
        )}
        {incident.affected_segment_ids.length > 0 && (
          <span className="inc__effect">
            <StatusPill tone="pending">degraded</StatusPill>
            {incident.affected_segment_ids.join(", ")}
          </span>
        )}
        {incident.exclusion_zone !== null &&
          incident.exclusion_zone.excluded_segment_ids.length > 0 && (
            <span className="inc__effect">
              <StatusPill tone="danger">exclusion zone</StatusPill>
              {incident.exclusion_zone.excluded_segment_ids.join(", ")}
            </span>
          )}
        {incident.blocks_segment_ids.length === 0 &&
          incident.affected_segment_ids.length === 0 &&
          incident.exclusion_zone === null && (
            <span className="inc__effect inc__effect--none">
              No segment is blocked or degraded by this incident.
            </span>
          )}
      </div>

      {permitted.length > 0 && (
        <div className="inc__moves">
          {permitted.map((status) => (
            <button
              key={status}
              type="button"
              className="btn btn--small"
              onClick={() => onTransition(incident.incident_id, status)}
            >
              Mark {status.replace(/_/g, " ")}
            </button>
          ))}
        </div>
      )}
      {permitted.length === 0 && (
        <p className="inc__terminal">
          {incident.status.replace(/_/g, " ")} is terminal — a recurrence is a new
          incident with its own id.
        </p>
      )}
    </div>
  );
}

export function IncidentsPanel({
  network,
  onIncidentsChanged,
}: {
  network: Resource<RoadNetwork>;
  /** Route costs and blockages change when an incident does. */
  onIncidentsChanged: () => void;
}): JSX.Element {
  const loaded = network.state === "success" ? network.data : null;
  const [list, setList] = useState<IncidentListResponse | null>(null);
  const [permitted, setPermitted] = useState<Record<string, IncidentStatus[]>>({});
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(false);

  const [type, setType] = useState<IncidentType>("accident");
  const [severity, setSeverity] = useState<IncidentSeverity>("major");
  const [segment, setSegment] = useState("");
  const [effect, setEffect] = useState<"blocks" | "affects">("blocks");
  const [description, setDescription] = useState("");

  const refresh = useCallback(async () => {
    try {
      const body = await apiGet<IncidentListResponse>(endpoints.incidents);
      setList(body);
      const next: Record<string, IncidentStatus[]> = {};
      await Promise.all(
        body.incidents.map(async (incident) => {
          try {
            const detail = await apiGet<IncidentResponse>(
              endpoints.incident(incident.incident_id),
            );
            next[incident.incident_id] = detail.permitted_next_statuses;
          } catch {
            next[incident.incident_id] = [];
          }
        }),
      );
      setPermitted(next);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh, network.state]);

  const declare = useCallback(async () => {
    if (!segment) return;
    setBusy(true);
    setError(null);
    try {
      await apiPost<IncidentResponse>(endpoints.incidents, {
        incident_type: type,
        severity,
        location: { segment_id: segment },
        ...(effect === "blocks"
          ? { blocks_segment_ids: [segment] }
          : { affected_segment_ids: [segment] }),
        ...(description ? { description } : {}),
        reported_by: "operator.dashboard",
      });
      setDescription("");
      setOpen(false);
      await refresh();
      onIncidentsChanged();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }, [type, severity, segment, effect, description, refresh, onIncidentsChanged]);

  const transition = useCallback(
    async (id: string, status: IncidentStatus) => {
      setError(null);
      try {
        await apiPost<IncidentResponse>(endpoints.incidentStatus(id), { status });
        await refresh();
        onIncidentsChanged();
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    },
    [refresh, onIncidentsChanged],
  );

  return (
    <Panel
      title="Incidents"
      subtitle="Disruptions expressed over the network, blocking or degrading segments"
      aside={
        list !== null ? (
          <StatusPill tone={list.active_count > 0 ? "danger" : "pending"} dot={list.active_count > 0}>
            {list.active_count} active / {list.count} declared
          </StatusPill>
        ) : undefined
      }
    >
      {loaded === null ? (
        <EmptyState
          title="No road network is loaded"
          message="An incident names the segments it blocks, and those ids mean nothing until a network is loaded. Load one first."
        />
      ) : (
        <>
          <div className="inc__actions">
            <button
              type="button"
              className="btn"
              onClick={() => setOpen((value) => !value)}
            >
              {open ? "Cancel" : "Declare incident"}
            </button>
          </div>

          {open && (
            <div className="inc__form">
              <label className="net__field">
                <span>Type</span>
                <select value={type} onChange={(e) => setType(e.target.value as IncidentType)}>
                  {INCIDENT_TYPES.map((item) => (
                    <option key={item} value={item}>
                      {item.replace(/_/g, " ")}
                    </option>
                  ))}
                </select>
              </label>

              <label className="net__field">
                <span>Severity</span>
                <select
                  value={severity}
                  onChange={(e) => setSeverity(e.target.value as IncidentSeverity)}
                >
                  {SEVERITIES.map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
              </label>

              <label className="net__field">
                <span>Road segment</span>
                <select value={segment} onChange={(e) => setSegment(e.target.value)}>
                  <option value="">Select a segment</option>
                  {loaded.segments.map((item) => (
                    <option key={item.segment_id} value={item.segment_id}>
                      {item.segment_id} ({item.from_junction_id} → {item.to_junction_id})
                    </option>
                  ))}
                </select>
              </label>

              <label className="net__field">
                <span>Effect</span>
                <select
                  value={effect}
                  onChange={(e) =>
                    setEffect(e.target.value === "affects" ? "affects" : "blocks")
                  }
                >
                  <option value="blocks">Blocks — impassable, pruned before search</option>
                  <option value="affects">Degrades — passable, priced as a penalty</option>
                </select>
              </label>

              <label className="net__field net__field--wide">
                <span>Description (optional)</span>
                <input
                  type="text"
                  value={description}
                  maxLength={1000}
                  onChange={(e) => setDescription(e.target.value)}
                />
              </label>

              <button
                type="button"
                className="btn"
                onClick={() => void declare()}
                disabled={busy || !segment}
              >
                {busy ? "Declaring…" : "Declare"}
              </button>
            </div>
          )}

          {error !== null && <p className="section__error">{error}</p>}

          {list !== null && list.incidents.length === 0 && (
            <EmptyState
              title="No incident has been declared"
              message={list.note}
            />
          )}

          {list?.incidents.map((incident) => (
            <IncidentRow
              key={incident.incident_id}
              entry={{
                incident,
                permitted: permitted[incident.incident_id] ?? [],
              }}
              onTransition={(id, status) => void transition(id, status)}
            />
          ))}
        </>
      )}
    </Panel>
  );
}
