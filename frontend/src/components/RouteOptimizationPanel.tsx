/**
 * Route optimization.
 *
 * The operator picks an origin, a destination and a mode; the backend ranks the
 * admissible routes and says why the winner won. Every number on this panel
 * comes from the response — there is no cost arithmetic, no ranking and no
 * reason-writing in this file, because a second implementation of the cost
 * function living in the browser is a second implementation that can disagree
 * with the one that made the decision.
 *
 * There is deliberately no approve or execute control. Human approval and
 * simulated execution belong to the Decision Engine and Simulation phases, and
 * putting a button here would let an operator act on a recommendation before
 * the step that exists to check it.
 *
 * Absence is rendered as absence. A route with no estimated travel time shows
 * "unavailable" and the reason, never a dash that could be read as zero.
 */

import { useCallback, useEffect, useState } from "react";

import { apiGet, apiPost, endpoints } from "../api/client";
import type {
  EmergencyScenariosResponse,
  MappingStatusResponse,
  RankedRoute,
  RoadNetwork,
  RouteOptimizationResult,
  RouteReason,
} from "../api/types";
import type { Resource } from "../hooks/useApiResource";
import { Panel } from "./Panel";
import { EmptyState } from "./ResourceState";
import { StatusPill, type Tone } from "./StatusPill";

/** Presentation copy for reason codes. Unknown codes render under their raw id. */
const REASON_COPY: Record<string, string> = {
  only_admissible_route: "Only admissible route",
  lower_total_cost: "Lower total cost",
  lower_travel_time: "Lower estimated travel time",
  lower_congestion_cost: "Lower traffic cost",
  lower_incident_cost: "Less incident disruption",
  shorter_distance: "Shorter distance",
  better_evidence_coverage: "Better observed",
  no_hard_blocked_segments: "No hard-blocked segments",
  travel_time_unavailable_ranked_by_distance: "Ranked by distance (no travel time)",
};

const EVIDENCE_TONE: Record<string, Tone> = {
  mapped_classified: "ok",
  mapped_stale: "pending",
  mapped_unclassified: "pending",
  unmapped: "neutral",
};

function seconds(value: number | null): string {
  return value === null ? "unavailable" : `${value.toFixed(1)} s`;
}

function ReasonRow({ reason }: { reason: RouteReason }): JSX.Element {
  const label = REASON_COPY[reason.code] ?? reason.code;
  return (
    <li className="route__reason">
      <span className="route__reason-label">{label}</span>
      {reason.value !== null && (
        <span className="route__reason-figures">
          {reason.value.toFixed(1)}
          {reason.comparison_value !== null && (
            <> vs {reason.comparison_value.toFixed(1)}</>
          )}
        </span>
      )}
      <p className="route__reason-detail">{reason.detail}</p>
    </li>
  );
}

function RouteCard({
  route,
  isRecommended,
}: {
  route: RankedRoute;
  isRecommended: boolean;
}): JSX.Element {
  const cost = route.candidate.cost;
  return (
    <div className={isRecommended ? "route route--best" : "route"}>
      <div className="source__head">
        <span className="route__title">
          {isRecommended ? "Recommended route" : `Alternative ${route.rank}`}
        </span>
        <StatusPill tone={isRecommended ? "ok" : "neutral"}>
          {totalCost(route).toFixed(1)} s cost
        </StatusPill>
      </div>

      <p className="route__sequence">{route.junction_sequence.join(" → ")}</p>
      <p className="route__segments">{route.segment_sequence.join(" · ")}</p>

      <dl className="route__figures">
        <div>
          <dt>Distance</dt>
          <dd>{(route.total_distance_m / 1000).toFixed(2)} km</dd>
        </div>
        <div>
          <dt>Estimated travel time</dt>
          <dd className={route.travel_time_available ? "" : "route__absent"}>
            {seconds(route.estimated_travel_time_s)}
          </dd>
        </div>
        <div>
          <dt>Traffic cost</dt>
          <dd>{cost.congestion_penalty_s.toFixed(1)} s</dd>
        </div>
        <div>
          <dt>Uncertainty cost</dt>
          <dd>{cost.uncertainty_penalty_s.toFixed(1)} s</dd>
        </div>
        <div>
          <dt>Incident cost</dt>
          <dd>{cost.incident_penalty_s.toFixed(1)} s</dd>
        </div>
        <div>
          <dt>Observed segments</dt>
          <dd>
            {route.mapped_segment_count} of {route.segment_sequence.length}
          </dd>
        </div>
      </dl>

      {route.evidence.length > 0 && (
        <div className="route__evidence">
          {route.evidence.map((item) => (
            <span key={item.segment_id} className="route__evidence-item">
              <StatusPill tone={EVIDENCE_TONE[item.status] ?? "neutral"}>
                {item.segment_id}
              </StatusPill>
              <span className="route__evidence-status">
                {item.status.replace(/_/g, " ")}
                {item.status !== "unmapped" && ` · ${item.congestion_level}`}
              </span>
            </span>
          ))}
        </div>
      )}

      {route.candidate.warnings.map((warning) => (
        <p key={warning} className="route__warning">
          {warning}
        </p>
      ))}

      {route.candidate.emergency_assumptions.map((assumption) => (
        <p key={assumption} className="route__assumption">
          {assumption}
        </p>
      ))}
    </div>
  );
}

/** Total is a derived sum the backend already publishes term by term. */
function totalCost(route: RankedRoute): number {
  const cost = route.candidate.cost;
  return (
    cost.base_travel_time_s +
    cost.congestion_penalty_s +
    cost.incident_penalty_s +
    cost.safety_penalty_s +
    cost.accessibility_penalty_s +
    cost.turn_penalty_s +
    cost.uncertainty_penalty_s +
    cost.distance_penalty_s
  );
}

/** What the simulation stage needs in order to evaluate what was recommended. */
export interface OptimizedRoute {
  originJunctionId: string | null;
  destinationJunctionId: string | null;
  destinationFacilityId: string | null;
  emergencyId: string | null;
  mode: "normal" | "emergency";
  hasRoutes: boolean;
}

export function RouteOptimizationPanel({
  network,
  scenarios,
  onOptimized,
}: {
  network: Resource<RoadNetwork>;
  /** Optional: lets a declared response be routed from its own record. */
  scenarios?: Resource<EmergencyScenariosResponse>;
  /**
   * Called with what was routed, so the simulation stage can evaluate the same
   * request. It is handed the *request*, not the result: the simulator asks the
   * backend for the routes itself, so the two can never be looking at
   * candidates produced by different snapshots.
   */
  onOptimized?: (route: OptimizedRoute) => void;
}): JSX.Element {
  const loaded = network.state === "success" ? network.data : null;
  const declared =
    scenarios !== undefined && scenarios.state === "success"
      ? scenarios.data.scenarios
      : [];
  const [origin, setOrigin] = useState("");
  const [destination, setDestination] = useState("");
  const [mode, setMode] = useState<"normal" | "emergency">("normal");
  const [emergencyId, setEmergencyId] = useState("");
  const [mapping, setMapping] = useState<MappingStatusResponse | null>(null);
  const [result, setResult] = useState<RouteOptimizationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Evidence coverage is worth showing before a route is requested: an operator
  // seeing "0 of 2 segments observed" understands the uncertainty cost that will
  // dominate the ranking, instead of discovering it in the breakdown afterwards.
  useEffect(() => {
    void (async () => {
      try {
        setMapping(await apiGet<MappingStatusResponse>(endpoints.routeMapping));
      } catch {
        setMapping(null);
      }
    })();
  }, [network.state]);

  const canOptimize = emergencyId !== "" || (origin !== "" && destination !== "");

  const optimize = useCallback(async () => {
    if (!canOptimize) return;
    setBusy(true);
    setError(null);
    try {
      // Routing a declared response reads its origin, destination and vehicle
      // type from its own record rather than restating them, so the two cannot
      // disagree.
      const isFacility = destination.startsWith("facility:");
      const body =
        emergencyId !== ""
          ? { emergency_id: emergencyId }
          : {
              mode,
              origin_junction_id: origin,
              ...(isFacility
                ? { destination_facility_id: destination.slice("facility:".length) }
                : { destination_junction_id: destination }),
            };
      const optimized = await apiPost<RouteOptimizationResult>(
        endpoints.optimizeRoute,
        body,
      );
      setResult(optimized);
      onOptimized?.({
        originJunctionId: emergencyId !== "" ? null : origin,
        destinationJunctionId:
          emergencyId !== "" || isFacility ? null : destination,
        destinationFacilityId:
          emergencyId === "" && isFacility
            ? destination.slice("facility:".length)
            : null,
        emergencyId: emergencyId !== "" ? emergencyId : null,
        mode: emergencyId !== "" ? "emergency" : mode,
        hasRoutes: optimized.status === "routes_found",
      });
    } catch (caught) {
      setResult(null);
      setError(caught instanceof Error ? caught.message : String(caught));
      onOptimized?.({
        originJunctionId: null,
        destinationJunctionId: null,
        destinationFacilityId: null,
        emergencyId: null,
        mode: "normal",
        hasRoutes: false,
      });
    } finally {
      setBusy(false);
    }
  }, [origin, destination, mode, emergencyId, canOptimize, onOptimized]);

  return (
    <Panel
      title="Route optimization"
      subtitle="Ranked admissible routes, with the reasons the winner won"
      aside={
        result ? (
          <StatusPill tone={result.status === "routes_found" ? "ok" : "danger"}>
            {result.status === "routes_found"
              ? `${result.routes.length} route(s)`
              : "no admissible route"}
          </StatusPill>
        ) : undefined
      }
    >
      {loaded === null ? (
        <EmptyState
          title="No road network is loaded"
          message="Routes are computed over a surveyed network. Load one before optimizing."
        />
      ) : (
        <>
          {declared.length > 0 && (
            <label className="route__field route__field--wide">
              <span>Declared emergency response</span>
              <select
                value={emergencyId}
                onChange={(event) => setEmergencyId(event.target.value)}
              >
                <option value="">Route manually (no declared response)</option>
                {declared.map((scenario) => (
                  <option key={scenario.emergency_id} value={scenario.emergency_id}>
                    {scenario.vehicle.callsign ?? scenario.vehicle.vehicle_id} ·{" "}
                    {scenario.vehicle.vehicle_type} · {scenario.priority} ·{" "}
                    {scenario.status}
                  </option>
                ))}
              </select>
            </label>
          )}

          <div className="route__controls">
            <label className="route__field">
              <span>Origin</span>
              <select
                value={origin}
                onChange={(e) => setOrigin(e.target.value)}
                disabled={emergencyId !== ""}
              >
                <option value="">Select a junction</option>
                {loaded.junctions.map((junction) => (
                  <option key={junction.junction_id} value={junction.junction_id}>
                    {junction.name ?? junction.junction_id}
                  </option>
                ))}
              </select>
            </label>

            <label className="route__field">
              <span>Destination</span>
              <select
                value={destination}
                onChange={(e) => setDestination(e.target.value)}
                disabled={emergencyId !== ""}
              >
                <option value="">Select a destination</option>
                {loaded.facilities.map((facility) => (
                  <option
                    key={facility.facility_id}
                    value={`facility:${facility.facility_id}`}
                  >
                    {facility.name} ({facility.facility_type})
                  </option>
                ))}
                {loaded.junctions.map((junction) => (
                  <option key={junction.junction_id} value={junction.junction_id}>
                    {junction.name ?? junction.junction_id}
                  </option>
                ))}
              </select>
            </label>

            <label className="route__field">
              <span>Mode</span>
              <select
                value={emergencyId !== "" ? "emergency" : mode}
                onChange={(e) =>
                  setMode(e.target.value === "emergency" ? "emergency" : "normal")
                }
                disabled={emergencyId !== ""}
              >
                <option value="normal">Normal traffic</option>
                <option value="emergency">Emergency vehicle</option>
              </select>
            </label>

            <button
              type="button"
              className="route__go"
              onClick={() => void optimize()}
              disabled={busy || !canOptimize}
            >
              {busy ? "Optimizing…" : "Optimize route"}
            </button>
          </div>

          {mapping !== null && (
            <p className="route__mapping">
              <StatusPill tone={mapping.loaded ? "ok" : "pending"}>
                {mapping.loaded
                  ? `${mapping.bound_segment_ids.length} segment(s) observed`
                  : "no observation mapping"}
              </StatusPill>{" "}
              {mapping.note}
            </p>
          )}

          {error !== null && <p className="route__error">{error}</p>}

          {result !== null && result.status === "no_admissible_route" && (
            <div className="route__refusal">
              <p className="route__refusal-title">
                No admissible route. Nothing is recommended in its place.
              </p>
              {result.rejections.map((rejection) => (
                <div key={`${rejection.reason}-${rejection.message}`}>
                  <StatusPill tone="danger">
                    {rejection.reason.replace(/_/g, " ")}
                  </StatusPill>
                  <p className="route__reason-detail">{rejection.message}</p>
                  {rejection.segment_ids.length > 0 && (
                    <p className="route__segments">
                      Segments: {rejection.segment_ids.join(", ")}
                    </p>
                  )}
                </div>
              ))}
            </div>
          )}

          {result !== null && result.recommendation !== null && (
            <div className="route__why">
              <p className="route__why-title">Why this route</p>
              <ul className="route__reasons">
                {result.recommendation.reasons.map((reason) => (
                  <ReasonRow key={reason.code} reason={reason} />
                ))}
              </ul>
              {result.recommendation.cost_margin_s !== null && (
                <p className="route__margin">
                  Margin over the runner-up:{" "}
                  {result.recommendation.cost_margin_s.toFixed(1)} s
                </p>
              )}
            </div>
          )}

          {result?.routes.map((route) => (
            <RouteCard
              key={route.candidate.route_id}
              route={route}
              isRecommended={
                route.candidate.route_id ===
                result.recommendation?.selected_route_id
              }
            />
          ))}

          {result !== null && result.pruned_segment_ids.length > 0 && (
            <p className="route__pruned">
              Excluded by hard constraints before search:{" "}
              {result.pruned_segment_ids.join(", ")}
            </p>
          )}

          {result?.warnings.map((warning) => (
            <p key={warning} className="route__warning">
              {warning}
            </p>
          ))}

          {result !== null && (
            <p className="route__provenance">
              {result.algorithm} v{result.algorithm_version} · network{" "}
              {result.network_id} v{result.network_version}
              {result.mapping_id !== null && ` · mapping ${result.mapping_id}`}
            </p>
          )}
        </>
      )}
    </Panel>
  );
}
