/**
 * Emergency scenario panel.
 *
 * Scenarios are rendered from `/api/emergency/scenarios`, and declared through
 * `/api/emergency/declare`. This file defines no emergency, no vehicle, no
 * destination and no route — if the collection is empty, the panel says nothing
 * has been declared and explains why, using the backend's own note.
 *
 * Declaration is offered only when a network is loaded, because a response is
 * anchored to junctions and facilities that must exist on a map. Origins and
 * destinations are drawn from that network, never from a list written here.
 *
 * Lifecycle moves come from `permitted_next_statuses` on each response. The
 * panel does not know the transition table and must not: a completed response
 * simply returns no permitted moves and its controls disappear.
 *
 * The strictest panel on the dashboard: an emergency recommendation is the one
 * output of this system a person might act on, so it must never display a
 * sample or demonstration value that could be mistaken for a real one. The
 * recommendation field list below is presentation scaffolding — labels and
 * descriptions of what a completed recommendation will contain — rendered with
 * explicit empty markers and never with illustrative numbers.
 */

import { useCallback, useEffect, useState } from "react";

import { apiGet, apiPost, endpoints } from "../api/client";
import type {
  EmergencyPriority,
  EmergencyResponseBody,
  EmergencyScenario,
  EmergencyScenariosResponse,
  EmergencyStatus,
  EmergencyVehicleType,
  RoadNetwork,
} from "../api/types";
import type { Resource } from "../hooks/useApiResource";
import { Panel } from "./Panel";
import { EmptyState, ResourceView } from "./ResourceState";
import { StatusPill, type Tone } from "./StatusPill";

/** Option lists mirroring the backend enums. Values, not invented data. */
const VEHICLE_TYPES: EmergencyVehicleType[] = [
  "ambulance",
  "fire_engine",
  "police",
  "rescue",
  "other",
];

const PRIORITIES: EmergencyPriority[] = ["P1_immediate", "P2_urgent", "P3_routine"];

/**
 * Labels for the fields a completed recommendation will carry.
 *
 * Static *presentation* text, not domain data: no value, id or quantity appears
 * here, and nothing in this list is presented as something the system knows.
 */
const RECOMMENDATION_FIELDS: ReadonlyArray<readonly [string, string]> = [
  ["Emergency type", "Responder class and priority tier"],
  ["Destination", "Facility or incident location"],
  ["Recommended route", "Junction sequence over admissible segments"],
  ["Predicted arrival", "Emergency traversal model, with stated assumptions"],
  ["Cost breakdown", "Travel time plus penalties, all in equivalent seconds"],
  ["Safety precautions", "Derived from the applied constraint set"],
  ["Expected outcome", "Baseline versus recommended, from simulation"],
];

/** Priority tiers that warrant visual emphasis, keyed by the API's own values. */
function priorityTone(priority: string): Tone {
  const normalised = priority.toLowerCase();
  if (normalised.startsWith("p1")) return "danger";
  if (normalised.startsWith("p2")) return "pending";
  return "neutral";
}

function destinationLabel(scenario: EmergencyScenario): string {
  const { destination } = scenario;
  if (destination.label !== null && destination.label !== "") return destination.label;
  if (destination.facility_id !== null) return destination.facility_id;
  if (destination.incident_id !== null) return destination.incident_id;
  const position = destination.position;
  if (position?.junction_id != null) return position.junction_id;
  if (position?.segment_id != null) return position.segment_id;
  return "unspecified";
}

function positionLabel(scenario: EmergencyScenario): string {
  const { current_position: position } = scenario;
  if (position.junction_id !== null) return position.junction_id;
  if (position.segment_id !== null) return position.segment_id;
  return "unknown";
}

function ScenarioRow({ scenario }: { scenario: EmergencyScenario }): JSX.Element {
  return (
    <div className="row" data-testid={`scenario-${scenario.emergency_id}`}>
      <span className="row__name">
        {scenario.vehicle.callsign ?? scenario.vehicle.vehicle_id}
      </span>
      <span className="row__detail">
        {scenario.vehicle.vehicle_type} · at {positionLabel(scenario)} →{" "}
        {destinationLabel(scenario)} · {scenario.status}
      </span>
      <span className="row__aside">
        <StatusPill tone={priorityTone(scenario.priority)} dot>
          {scenario.priority}
        </StatusPill>
      </span>
    </div>
  );
}

function EmptyRecommendation(): JSX.Element {
  return (
    <div className="rows">
      {RECOMMENDATION_FIELDS.map(([label, description]) => (
        <div className="row" key={label}>
          <span className="row__name">{label}</span>
          <span className="row__detail">{description}</span>
          <span className="row__aside" aria-label={`${label}: no value`}>
            <span className="chip">—</span>
          </span>
        </div>
      ))}
    </div>
  );
}

/**
 * Declaration form.
 *
 * Every option comes from the loaded network. The backend validates the pairing
 * again — an ambulance may not be dispatched to a fire station — and rejects it
 * with a reason, which is surfaced verbatim rather than pre-empted here.
 */
function DeclareForm({
  network,
  onDeclared,
}: {
  network: RoadNetwork;
  onDeclared: () => void;
}): JSX.Element {
  const [vehicleType, setVehicleType] = useState<EmergencyVehicleType>("ambulance");
  const [callsign, setCallsign] = useState("");
  const [priority, setPriority] = useState<EmergencyPriority>("P1_immediate");
  const [origin, setOrigin] = useState("");
  const [destination, setDestination] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const declare = useCallback(async () => {
    if (!origin || !destination) return;
    setBusy(true);
    setError(null);
    try {
      const isFacility = destination.startsWith("facility:");
      await apiPost<EmergencyResponseBody>(endpoints.declareEmergency, {
        vehicle: {
          vehicle_id: `UNIT-${Date.now().toString().slice(-6)}`,
          vehicle_type: vehicleType,
          ...(callsign ? { callsign } : {}),
        },
        priority,
        current_position: { junction_id: origin },
        destination: isFacility
          ? { facility_id: destination.slice("facility:".length) }
          : { position: { junction_id: destination } },
        declared_by: "operator.dashboard",
      });
      setCallsign("");
      onDeclared();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }, [vehicleType, callsign, priority, origin, destination, onDeclared]);

  return (
    <div className="inc__form">
      <label className="net__field">
        <span>Responder</span>
        <select
          value={vehicleType}
          onChange={(e) => setVehicleType(e.target.value as EmergencyVehicleType)}
        >
          {VEHICLE_TYPES.map((item) => (
            <option key={item} value={item}>
              {item.replace(/_/g, " ")}
            </option>
          ))}
        </select>
      </label>

      <label className="net__field">
        <span>Priority</span>
        <select
          value={priority}
          onChange={(e) => setPriority(e.target.value as EmergencyPriority)}
        >
          {PRIORITIES.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
      </label>

      <label className="net__field">
        <span>Origin junction</span>
        <select value={origin} onChange={(e) => setOrigin(e.target.value)}>
          <option value="">Select a junction</option>
          {network.junctions.map((junction) => (
            <option key={junction.junction_id} value={junction.junction_id}>
              {junction.name ?? junction.junction_id}
            </option>
          ))}
        </select>
      </label>

      <label className="net__field">
        <span>Destination</span>
        <select value={destination} onChange={(e) => setDestination(e.target.value)}>
          <option value="">Select a destination</option>
          {network.facilities.map((facility) => (
            <option key={facility.facility_id} value={`facility:${facility.facility_id}`}>
              {facility.name} ({facility.facility_type})
            </option>
          ))}
          {network.junctions.map((junction) => (
            <option key={junction.junction_id} value={junction.junction_id}>
              {junction.name ?? junction.junction_id}
            </option>
          ))}
        </select>
      </label>

      <label className="net__field">
        <span>Callsign (optional)</span>
        <input
          type="text"
          value={callsign}
          maxLength={50}
          onChange={(e) => setCallsign(e.target.value)}
        />
      </label>

      <button
        type="button"
        className="btn"
        onClick={() => void declare()}
        disabled={busy || !origin || !destination}
      >
        {busy ? "Declaring…" : "Declare response"}
      </button>

      {error !== null && <p className="section__error">{error}</p>}
    </div>
  );
}

/** Lifecycle controls for one declared response. */
function LifecycleControls({
  emergencyId,
  permitted,
  onChanged,
}: {
  emergencyId: string;
  permitted: EmergencyStatus[];
  onChanged: () => void;
}): JSX.Element | null {
  const [error, setError] = useState<string | null>(null);

  const move = useCallback(
    async (status: EmergencyStatus) => {
      setError(null);
      try {
        await apiPost<EmergencyResponseBody>(endpoints.emergencyStatus(emergencyId), {
          status,
        });
        onChanged();
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    },
    [emergencyId, onChanged],
  );

  if (permitted.length === 0) {
    return (
      <p className="inc__terminal">
        This response is finished — a further response is a new declaration with
        its own response-time clock.
      </p>
    );
  }

  return (
    <div className="inc__moves">
      {permitted.map((status) => (
        <button
          key={status}
          type="button"
          className="btn btn--small"
          onClick={() => void move(status)}
        >
          Mark {status.replace(/_/g, " ")}
        </button>
      ))}
      {error !== null && <p className="section__error">{error}</p>}
    </div>
  );
}

export function EmergencyPanel({
  scenarios,
  onRetry,
  network,
  onEmergencyChanged,
}: {
  scenarios: Resource<EmergencyScenariosResponse>;
  onRetry: () => void;
  /** Optional: when a network is loaded, declaration and lifecycle are offered. */
  network?: Resource<RoadNetwork>;
  onEmergencyChanged?: () => void;
}): JSX.Element {
  const data = scenarios.state === "success" ? scenarios.data : null;
  const hasScenarios = data !== null && data.scenarios.length > 0;
  const loadedNetwork =
    network !== undefined && network.state === "success" ? network.data : null;

  const [open, setOpen] = useState(false);
  const [permitted, setPermitted] = useState<Record<string, EmergencyStatus[]>>({});

  const refreshAll = useCallback(() => {
    onRetry();
    onEmergencyChanged?.();
  }, [onRetry, onEmergencyChanged]);

  // Permitted moves are per response and known only to the backend, so they are
  // fetched rather than derived from the status on the scenario row.
  useEffect(() => {
    if (data === null || data.scenarios.length === 0) {
      setPermitted({});
      return;
    }
    let cancelled = false;
    void (async () => {
      const next: Record<string, EmergencyStatus[]> = {};
      await Promise.all(
        data.scenarios.map(async (scenario) => {
          try {
            const detail = await apiGet<EmergencyResponseBody>(
              endpoints.emergency(scenario.emergency_id),
            );
            next[scenario.emergency_id] = detail.permitted_next_statuses;
          } catch {
            next[scenario.emergency_id] = [];
          }
        }),
      );
      if (!cancelled) setPermitted(next);
    })();
    return () => {
      cancelled = true;
    };
  }, [data]);

  return (
    <Panel
      title="Emergency scenario"
      subtitle="Declared response, candidate routes and safety precautions"
      flush
      aside={
        data !== null ? (
          <StatusPill tone={data.active_count > 0 ? "danger" : "pending"} dot={data.active_count > 0}>
            {data.active_count} active / {data.count} declared
          </StatusPill>
        ) : undefined
      }
    >
      {network !== undefined && (
        <div className="emg__declare">
          {loadedNetwork === null ? (
            <p className="section__note">
              No road network is loaded, so no response can be declared: a dispatch
              is anchored to junctions and facilities that must exist on a map.
            </p>
          ) : (
            <>
              <button
                type="button"
                className="btn"
                onClick={() => setOpen((value) => !value)}
              >
                {open ? "Cancel" : "Declare emergency response"}
              </button>
              {open && (
                <DeclareForm
                  network={loadedNetwork}
                  onDeclared={() => {
                    setOpen(false);
                    refreshAll();
                  }}
                />
              )}
            </>
          )}
        </div>
      )}

      <div style={{ padding: hasScenarios ? 0 : 16 }}>
        <ResourceView
          resource={scenarios}
          onRetry={onRetry}
          loadingRows={3}
          plannedCapabilities={
            "An emergency is a declared event with a destination and a priority — " +
            "never a visual detection alone. Once declared, routing prunes segments " +
            "blocked by hard safety constraints, then ranks the remaining routes by " +
            "risk-adjusted arrival time."
          }
        >
          {(response) =>
            response.scenarios.length === 0 ? (
              <EmptyState title="No emergency declared" message={response.note} />
            ) : (
              <div className="rows">
                {response.scenarios.map((scenario) => (
                  <div key={scenario.emergency_id}>
                    <ScenarioRow scenario={scenario} />
                    {network !== undefined && (
                      <div className="emg__lifecycle">
                        <LifecycleControls
                          emergencyId={scenario.emergency_id}
                          permitted={permitted[scenario.emergency_id] ?? []}
                          onChanged={refreshAll}
                        />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )
          }
        </ResourceView>
      </div>

      {!hasScenarios && (
        <>
          <div className="rule-group__title">Recommendation fields — currently empty</div>
          <EmptyRecommendation />
        </>
      )}

      <div style={{ padding: 16, borderTop: "1px solid var(--border)" }}>
        <div className="banner">
          <span className="banner__icon" aria-hidden="true">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="9" />
              <path d="M12 11v5.5" strokeLinecap="round" />
              <circle cx="12" cy="7.8" r="0.6" fill="currentColor" stroke="none" />
            </svg>
          </span>
          <span>
            <strong>Routes are produced by the Route Optimization panel below.</strong>{" "}
            A declared response can be routed there by selecting it, which reads its
            origin, destination and vehicle type from its own record. Expected-outcome
            and improvement figures still do not exist: those require simulation, and
            none will be estimated for display.
          </span>
        </div>
      </div>
    </Panel>
  );
}
