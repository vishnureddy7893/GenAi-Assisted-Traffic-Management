/**
 * Road network — loading it, and what it contains.
 *
 * Loading is an explicit operator action, mirroring the backend. No network is
 * loaded until someone chooses one, and this panel offers only the files the
 * backend says exist rather than a list written here. A map is exactly the kind
 * of input whose provenance should be a decision, not a default.
 *
 * The panel also states what loading *costs*: declared incidents and emergencies
 * are discarded, because their segment ids mean something different on a
 * different map. The backend reports how many were dropped and the panel repeats
 * it, so the loss is visible rather than silent.
 *
 * Junctions, segments and facilities are rendered from the loaded network. There
 * is no fallback list, no example topology and no placeholder hospital: with no
 * network loaded, the panel says so.
 */

import { useCallback, useEffect, useState } from "react";

import { apiGet, apiPost, endpoints } from "../api/client";
import type {
  BlockedSegmentsResponse,
  LoadNetworkResponse,
  NetworkStatusResponse,
  RoadNetwork,
} from "../api/types";
import type { Resource } from "../hooks/useApiResource";
import { Panel } from "./Panel";
import { EmptyState } from "./ResourceState";
import { StatusPill } from "./StatusPill";

type Tab = "junctions" | "segments" | "facilities";

export function NetworkPanel({
  network,
  onNetworkChanged,
}: {
  network: Resource<RoadNetwork>;
  /** Called after a load, so sibling panels refetch rather than showing a stale map. */
  onNetworkChanged: () => void;
}): JSX.Element {
  const loaded = network.state === "success" ? network.data : null;
  const [status, setStatus] = useState<NetworkStatusResponse | null>(null);
  const [blocked, setBlocked] = useState<BlockedSegmentsResponse | null>(null);
  const [choice, setChoice] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("junctions");

  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await apiGet<NetworkStatusResponse>(endpoints.networkStatus));
    } catch {
      setStatus(null);
    }
  }, []);

  const refreshBlocked = useCallback(async () => {
    try {
      setBlocked(await apiGet<BlockedSegmentsResponse>(endpoints.networkBlocked));
    } catch {
      // 404 while no network is loaded is the normal state, not a fault.
      setBlocked(null);
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
    void refreshBlocked();
  }, [refreshStatus, refreshBlocked, network.state]);

  const load = useCallback(async () => {
    if (!choice) return;
    setBusy(true);
    setError(null);
    setNote(null);
    try {
      const result = await apiPost<LoadNetworkResponse>(endpoints.loadNetwork, {
        relative_path: choice,
      });
      setNote(result.note);
      onNetworkChanged();
      await refreshStatus();
      await refreshBlocked();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }, [choice, onNetworkChanged, refreshStatus, refreshBlocked]);

  const blockedSet = new Set(blocked?.blocked_segment_ids ?? []);
  const degradedSet = new Set(blocked?.degraded_segment_ids ?? []);

  return (
    <Panel
      title="Road network"
      subtitle="The surveyed graph every incident, dispatch and route is expressed over"
      aside={
        loaded !== null ? (
          <StatusPill tone="ok" dot>
            {loaded.network_id} v{loaded.version}
          </StatusPill>
        ) : (
          <StatusPill tone="pending">no network loaded</StatusPill>
        )
      }
    >
      <div className="net__controls">
        <label className="net__field">
          <span>Network file</span>
          <select
            value={choice}
            onChange={(event) => setChoice(event.target.value)}
            disabled={(status?.available_files.length ?? 0) === 0}
          >
            <option value="">
              {(status?.available_files.length ?? 0) === 0
                ? "No network files found"
                : "Select a file"}
            </option>
            {status?.available_files.map((file) => (
              <option key={file} value={file}>
                {file}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="btn"
          onClick={() => void load()}
          disabled={busy || !choice}
        >
          {busy ? "Loading…" : "Load network"}
        </button>
      </div>

      {status !== null && <p className="section__note">{status.note}</p>}
      {note !== null && <p className="section__note net__loaded">{note}</p>}
      {error !== null && <p className="section__error">{error}</p>}

      {loaded === null ? (
        <EmptyState
          title="No road network is loaded"
          message="Junctions, road segments and facilities appear here once a network is loaded. Nothing is shown in the meantime, because an example topology is indistinguishable from a real one at a glance."
        />
      ) : (
        <>
          <div className="net__summary">
            <span>{loaded.junctions.length} junctions</span>
            <span>{loaded.segments.length} road segments</span>
            <span>{loaded.facilities.length} facilities</span>
            <span>{loaded.driving_side}-hand traffic</span>
            {blocked !== null && (
              <span className={blocked.blocked_segment_ids.length > 0 ? "net__alert" : ""}>
                {blocked.blocked_segment_ids.length} blocked ·{" "}
                {blocked.degraded_segment_ids.length} degraded
              </span>
            )}
          </div>

          <div className="net__tabs" role="tablist">
            {(["junctions", "segments", "facilities"] as Tab[]).map((name) => (
              <button
                key={name}
                type="button"
                role="tab"
                aria-selected={tab === name}
                className={tab === name ? "net__tab net__tab--on" : "net__tab"}
                onClick={() => setTab(name)}
              >
                {name}
              </button>
            ))}
          </div>

          {tab === "junctions" && (
            <div className="net__rows">
              {loaded.junctions.map((junction) => (
                <div key={junction.junction_id} className="net__row">
                  <span className="net__id">{junction.junction_id}</span>
                  <span className="net__name">{junction.name ?? "unnamed"}</span>
                  <StatusPill tone="neutral">{junction.kind}</StatusPill>
                  <span className="net__meta">
                    {junction.approaches.length} approaches ·{" "}
                    {junction.movements.length} movements
                  </span>
                </div>
              ))}
            </div>
          )}

          {tab === "segments" && (
            <div className="net__rows">
              {loaded.segments.map((segment) => {
                const isBlocked = blockedSet.has(segment.segment_id);
                const isDegraded = degradedSet.has(segment.segment_id);
                return (
                  <div key={segment.segment_id} className="net__row">
                    <span className="net__id">{segment.segment_id}</span>
                    <span className="net__name">
                      {segment.from_junction_id} → {segment.to_junction_id}
                    </span>
                    {isBlocked && <StatusPill tone="danger">blocked</StatusPill>}
                    {!isBlocked && isDegraded && (
                      <StatusPill tone="pending">degraded</StatusPill>
                    )}
                    <span className="net__meta">
                      {segment.length_m.value.toFixed(0)} m · {segment.lane_count} lanes
                      {segment.free_flow_speed_kph !== null && (
                        <> · {segment.free_flow_speed_kph.value.toFixed(0)} km/h free flow</>
                      )}
                    </span>
                  </div>
                );
              })}
            </div>
          )}

          {tab === "facilities" &&
            (loaded.facilities.length === 0 ? (
              <EmptyState
                title="This network declares no facilities"
                message="A facility is an emergency destination — a hospital, fire station or police station — attached to an access junction. Without one, a dispatch must name a junction directly."
              />
            ) : (
              <div className="net__rows">
                {loaded.facilities.map((facility) => (
                  <div key={facility.facility_id} className="net__row">
                    <span className="net__id">{facility.facility_id}</span>
                    <span className="net__name">{facility.name}</span>
                    <StatusPill tone="accent">{facility.facility_type}</StatusPill>
                    <span className="net__meta">
                      via {facility.access_junction_id}
                      {facility.accepts_emergency_types.length > 0 && (
                        <> · accepts {facility.accepts_emergency_types.join(", ")}</>
                      )}
                    </span>
                  </div>
                ))}
              </div>
            ))}
        </>
      )}
    </Panel>
  );
}
