/**
 * Traffic simulation: baseline against recommended.
 *
 * The panel sits directly beneath route optimization because that is the real
 * dependency: there is nothing to simulate until a route has been recommended,
 * and an operator who has not run the optimizer should be told that rather than
 * shown an empty chart.
 *
 * Four kinds of number, four treatments
 * -------------------------------------
 * Every figure here is tagged with where it came from — simulated, derived,
 * observed or unavailable — because the whole risk of a simulation panel is
 * that a modelled outcome reads like a measurement once it is on a screen. An
 * unavailable value renders as the word "unavailable" and its reason, never as
 * a dash that could be skimmed as zero.
 *
 * No arithmetic happens in this file. Deltas, improvements and totals all come
 * from the backend, because a second implementation in the browser is a second
 * implementation that can disagree with the one that produced the decision.
 *
 * There is deliberately no approve, dispatch or apply control. The simulation
 * is the step that exists so a person can check a recommendation before acting
 * on it; a button here would remove exactly that check.
 */

import { useCallback, useEffect, useState } from "react";

import { apiGet, apiPost, endpoints } from "../api/client";
import type {
  Measurement,
  RoadNetwork,
  RouteSimulationResult,
  SimulationComparison,
  SimulationConfigResponse,
  SimulationLinkResult,
  SimulationRun,
} from "../api/types";
import type { Resource } from "../hooks/useApiResource";
import { Panel } from "./Panel";
import { EmptyState } from "./ResourceState";
import { StatusPill, type Tone } from "./StatusPill";

/** What the route optimization stage hands down so a route can be simulated. */
export interface SimulationContext {
  originJunctionId: string | null;
  destinationJunctionId: string | null;
  destinationFacilityId: string | null;
  emergencyId: string | null;
  mode: "normal" | "emergency";
  /** Whether the optimizer actually produced something to compare. */
  hasRoutes: boolean;
}

/** One operator-stated assumption, as the request carries it. */
interface Assumption {
  segment_id: string;
  demand_veh_per_hour: string;
  capacity_veh_per_hour: string;
  note: string;
}

const SOURCE_LABEL: Record<string, string> = {
  simulated: "simulated",
  derived: "derived",
  observed: "observed",
  static: "surveyed",
  assumed: "assumed",
  predicted: "predicted",
};

const BASIS_TONE: Record<string, Tone> = {
  surveyed: "ok",
  observed: "ok",
  stated: "pending",
  unavailable: "neutral",
};

const BETTER_COPY: Record<string, string> = {
  alternative: "The recommended route arrives sooner",
  baseline: "The baseline route arrives sooner",
  no_difference: "No difference between the two routes",
  unknown: "Not comparable",
};

/** A measured/modelled value with its origin, or an explicit absence. */
function Figure({
  label,
  measurement,
  digits = 1,
  suffix = "",
  absentReason,
}: {
  label: string;
  measurement: Measurement | null;
  digits?: number;
  suffix?: string;
  absentReason?: string;
}): JSX.Element {
  if (measurement === null) {
    return (
      <div>
        <dt>{label}</dt>
        <dd className="route__absent" title={absentReason}>
          unavailable
        </dd>
      </div>
    );
  }
  const source = measurement.provenance.source;
  return (
    <div>
      <dt>{label}</dt>
      <dd>
        {measurement.value.toFixed(digits)}
        {suffix}{" "}
        <span className={`sim__tag sim__tag--${source}`}>
          {SOURCE_LABEL[source] ?? source}
        </span>
      </dd>
    </div>
  );
}

function ArmCard({
  arm,
  highlight,
}: {
  arm: RouteSimulationResult;
  highlight: boolean;
}): JSX.Element {
  return (
    <div className={highlight ? "route route--best" : "route"}>
      <div className="source__head">
        <span className="route__title">{arm.label}</span>
        <StatusPill tone={arm.status === "simulated" ? "ok" : "danger"}>
          {arm.role}
        </StatusPill>
      </div>

      <p className="route__sequence">{arm.junction_sequence.join(" → ")}</p>
      <p className="route__segments">{arm.segment_ids.join(" · ")}</p>

      {arm.status === "unavailable" ? (
        <div className="route__refusal">
          <p className="route__refusal-title">
            No travel time for this route. Nothing was estimated in its place.
          </p>
          {arm.rejections.map((rejection) => (
            <div key={`${rejection.reason}-${rejection.message}`}>
              <StatusPill tone="danger">
                {rejection.reason.replace(/_/g, " ")}
              </StatusPill>
              <p className="route__reason-detail">{rejection.message}</p>
            </div>
          ))}
        </div>
      ) : (
        <dl className="route__figures">
          <Figure label="Travel time" measurement={arm.travel_time_s} suffix=" s" />
          <Figure
            label="Free-flow reference"
            measurement={arm.free_flow_travel_time_s}
            suffix=" s"
          />
          <Figure label="Delay" measurement={arm.delay_s} suffix=" s" />
          <Figure
            label="Completion time"
            measurement={arm.completion_time_s}
            suffix=" s"
          />
          <Figure
            label="Mean speed"
            measurement={arm.mean_speed_kph}
            suffix=" km/h"
          />
          <Figure
            label="Distance"
            measurement={arm.total_distance_m}
            digits={0}
            suffix=" m"
          />
        </dl>
      )}

      {arm.legs.length > 0 && (
        <table className="sim__legs">
          <thead>
            <tr>
              <th>Segment</th>
              <th>Enters</th>
              <th>Leaves</th>
              <th>Queue delay</th>
            </tr>
          </thead>
          <tbody>
            {arm.legs.map((leg) => (
              <tr key={leg.segment_id}>
                <td>{leg.segment_id}</td>
                <td>{leg.entry_time_s.toFixed(1)} s</td>
                <td>{leg.exit_time_s.toFixed(1)} s</td>
                <td className={leg.queue_modelled ? "" : "route__absent"}>
                  {leg.queue_modelled
                    ? `${(leg.queue_delay_s ?? 0).toFixed(1)} s`
                    : "not modelled"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {arm.warnings.map((warning) => (
        <p key={warning} className="route__warning">
          {warning}
        </p>
      ))}
    </div>
  );
}

function ComparisonBlock({
  comparison,
}: {
  comparison: SimulationComparison;
}): JSX.Element {
  if (!comparison.comparable) {
    return (
      <div className="route__refusal">
        <p className="route__refusal-title">
          No improvement figure exists for this pair.
        </p>
        <p className="route__reason-detail">{comparison.reason}</p>
      </div>
    );
  }
  return (
    <div className="route__why">
      <p className="route__why-title">{BETTER_COPY[comparison.better]}</p>
      <dl className="route__figures">
        <Figure
          label="Travel-time difference"
          measurement={comparison.travel_time_delta_s}
          suffix=" s"
        />
        <Figure
          label="Delay difference"
          measurement={comparison.delay_delta_s}
          suffix=" s"
        />
        <Figure
          label="Distance difference"
          measurement={comparison.distance_delta_m}
          digits={0}
          suffix=" m"
        />
        <div>
          <dt>Improvement</dt>
          <dd
            className={comparison.improvement_pct === null ? "route__absent" : ""}
          >
            {comparison.improvement_pct === null
              ? "unavailable"
              : `${comparison.improvement_pct.toFixed(1)} %`}{" "}
            {comparison.improvement_pct !== null && (
              <span className="sim__tag sim__tag--simulated">simulated</span>
            )}
          </dd>
        </div>
      </dl>
    </div>
  );
}

function LinkTable({ links }: { links: SimulationLinkResult[] }): JSX.Element {
  return (
    <table className="sim__links">
      <thead>
        <tr>
          <th>Segment</th>
          <th>Demand</th>
          <th>Capacity</th>
          <th>Peak queue</th>
          <th>Throughput</th>
        </tr>
      </thead>
      <tbody>
        {links.map((link) => (
          <tr key={link.segment_id}>
            <td>{link.segment_id}</td>
            <td>
              {link.demand_veh_per_hour === null ? (
                <span className="route__absent">unavailable</span>
              ) : (
                <>
                  {link.demand_veh_per_hour.toFixed(0)} veh/h{" "}
                  <StatusPill tone={BASIS_TONE[link.demand_basis] ?? "neutral"}>
                    {link.demand_basis}
                  </StatusPill>
                </>
              )}
            </td>
            <td>
              {link.capacity_veh_per_hour === null ? (
                <span className="route__absent">unavailable</span>
              ) : (
                <>
                  {link.capacity_veh_per_hour.toFixed(0)} veh/h{" "}
                  <StatusPill tone={BASIS_TONE[link.capacity_basis] ?? "neutral"}>
                    {link.capacity_basis}
                  </StatusPill>
                </>
              )}
            </td>
            <td className={link.peak_queue_veh === null ? "route__absent" : ""}>
              {link.peak_queue_veh === null
                ? "not modelled"
                : `${link.peak_queue_veh.value.toFixed(1)} veh`}
            </td>
            <td className={link.throughput_veh === null ? "route__absent" : ""}>
              {link.throughput_veh === null
                ? "not modelled"
                : `${link.throughput_veh.value.toFixed(0)} veh`}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function SimulationPanel({
  network,
  context,
}: {
  network: Resource<RoadNetwork>;
  /** Supplied by the route optimization panel once it has produced a route. */
  context: SimulationContext | null;
}): JSX.Element {
  const loaded = network.state === "success" ? network.data : null;
  const [config, setConfig] = useState<SimulationConfigResponse | null>(null);
  const [assumptions, setAssumptions] = useState<Assumption[]>([]);
  const [draftSegment, setDraftSegment] = useState("");
  const [draftDemand, setDraftDemand] = useState("");
  const [draftCapacity, setDraftCapacity] = useState("");
  const [draftNote, setDraftNote] = useState("");
  const [run, setRun] = useState<SimulationRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    void (async () => {
      try {
        setConfig(
          await apiGet<SimulationConfigResponse>(endpoints.simulationConfig),
        );
      } catch {
        setConfig(null);
      }
    })();
  }, []);

  const addAssumption = useCallback(() => {
    if (draftSegment === "" || draftNote.trim() === "") return;
    setAssumptions((current) => [
      ...current.filter((item) => item.segment_id !== draftSegment),
      {
        segment_id: draftSegment,
        demand_veh_per_hour: draftDemand,
        capacity_veh_per_hour: draftCapacity,
        note: draftNote,
      },
    ]);
    setDraftSegment("");
    setDraftDemand("");
    setDraftCapacity("");
    setDraftNote("");
  }, [draftSegment, draftDemand, draftCapacity, draftNote]);

  const compare = useCallback(async () => {
    if (context === null) return;
    setBusy(true);
    setError(null);
    try {
      const body = {
        ...(context.emergencyId !== null
          ? { emergency_id: context.emergencyId }
          : {
              mode: context.mode,
              origin_junction_id: context.originJunctionId,
              ...(context.destinationFacilityId !== null
                ? { destination_facility_id: context.destinationFacilityId }
                : { destination_junction_id: context.destinationJunctionId }),
            }),
        segment_inputs: assumptions.map((item) => ({
          segment_id: item.segment_id,
          note: item.note,
          ...(item.demand_veh_per_hour !== ""
            ? { demand_veh_per_hour: Number(item.demand_veh_per_hour) }
            : {}),
          ...(item.capacity_veh_per_hour !== ""
            ? { capacity_veh_per_hour: Number(item.capacity_veh_per_hour) }
            : {}),
        })),
      };
      setRun(await apiPost<SimulationRun>(endpoints.simulationCompare, body));
    } catch (caught) {
      setRun(null);
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }, [context, assumptions]);

  const baseline = run?.routes.find((item) => item.role === "baseline") ?? null;
  const recommended =
    run?.routes.find((item) => item.role === "recommended") ?? null;

  return (
    <Panel
      title="Traffic simulation"
      subtitle="Baseline against recommended, over one horizon and one set of assumptions"
      aside={
        run ? (
          <StatusPill tone={run.status === "completed" ? "ok" : "danger"}>
            {run.status === "completed" ? run.run_id : "unavailable"}
          </StatusPill>
        ) : undefined
      }
    >
      {loaded === null ? (
        <EmptyState
          title="No road network is loaded"
          message="A simulation runs over a surveyed network. Load one before simulating."
        />
      ) : context === null || !context.hasRoutes ? (
        <EmptyState
          title="No route has been optimized yet"
          message={
            "There is nothing to simulate until the optimizer has produced a route. " +
            "Run a route optimization above, then return here to compare it against " +
            "the baseline."
          }
        />
      ) : (
        <>
          <p className="sim__modality">
            Everything below is the <strong>modelled outcome of a hypothetical
            scenario</strong>, not a measurement of any road. Each figure is
            tagged with where it came from, and a value nothing supports reads as
            <em> unavailable</em> rather than as zero.
          </p>

          {config !== null && (
            <p className="route__mapping">
              <StatusPill tone="neutral">
                {config.model_name} v{config.model_version}
              </StatusPill>{" "}
              {config.horizon_s.toFixed(0)} s horizon at {config.step_s.toFixed(0)} s
              steps · deterministic, no random component
            </p>
          )}

          <div className="sim__assumptions">
            <p className="route__why-title">Stated assumptions</p>
            <p className="sim__hint">
              A segment with no demand and no capacity is reported unmodelled, not
              simulated against a default. Every assumption must say why it is
              being made, and the reason travels into the result.
            </p>

            <div className="route__controls">
              <label className="route__field">
                <span>Segment</span>
                <select
                  value={draftSegment}
                  onChange={(event) => setDraftSegment(event.target.value)}
                >
                  <option value="">Select a segment</option>
                  {loaded.segments.map((segment) => (
                    <option key={segment.segment_id} value={segment.segment_id}>
                      {segment.segment_id}
                    </option>
                  ))}
                </select>
              </label>
              <label className="route__field">
                <span>Demand (veh/h)</span>
                <input
                  type="number"
                  min="0"
                  value={draftDemand}
                  onChange={(event) => setDraftDemand(event.target.value)}
                />
              </label>
              <label className="route__field">
                <span>Capacity (veh/h)</span>
                <input
                  type="number"
                  min="1"
                  value={draftCapacity}
                  onChange={(event) => setDraftCapacity(event.target.value)}
                />
              </label>
              <label className="route__field">
                <span>Why</span>
                <input
                  type="text"
                  value={draftNote}
                  onChange={(event) => setDraftNote(event.target.value)}
                  placeholder="Justification for this assumption"
                />
              </label>
              <button
                type="button"
                className="route__go"
                onClick={addAssumption}
                disabled={draftSegment === "" || draftNote.trim() === ""}
              >
                Add assumption
              </button>
            </div>

            {assumptions.length > 0 && (
              <ul className="sim__assumption-list">
                {assumptions.map((item) => (
                  <li key={item.segment_id}>
                    <StatusPill tone="pending">{item.segment_id}</StatusPill>{" "}
                    {item.demand_veh_per_hour !== "" &&
                      `${item.demand_veh_per_hour} veh/h demand `}
                    {item.capacity_veh_per_hour !== "" &&
                      `${item.capacity_veh_per_hour} veh/h capacity `}
                    <span className="sim__hint">{item.note}</span>
                    <button
                      type="button"
                      className="sim__remove"
                      onClick={() =>
                        setAssumptions((current) =>
                          current.filter(
                            (other) => other.segment_id !== item.segment_id,
                          ),
                        )
                      }
                    >
                      Remove
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <button
            type="button"
            className="route__go"
            onClick={() => void compare()}
            disabled={busy}
          >
            {busy ? "Simulating…" : "Simulate baseline vs recommended"}
          </button>

          {error !== null && <p className="route__error">{error}</p>}

          {run !== null && run.status === "unavailable" && (
            <div className="route__refusal">
              <p className="route__refusal-title">
                Nothing could be simulated. No outcome is reported in its place.
              </p>
              {run.rejections.map((rejection) => (
                <div key={`${rejection.reason}-${rejection.message}`}>
                  <StatusPill tone="danger">
                    {rejection.reason.replace(/_/g, " ")}
                  </StatusPill>
                  <p className="route__reason-detail">{rejection.message}</p>
                </div>
              ))}
            </div>
          )}

          {run?.comparison && <ComparisonBlock comparison={run.comparison} />}

          {baseline && <ArmCard arm={baseline} highlight={false} />}
          {recommended && (
            <ArmCard
              arm={recommended}
              highlight={run?.comparison?.better === "alternative"}
            />
          )}

          {run?.network && (
            <div className="sim__network">
              <p className="route__why-title">Network over the horizon</p>
              <dl className="route__figures">
                <Figure
                  label="Throughput (modelled links)"
                  measurement={run.network.modelled_throughput_veh}
                  digits={0}
                  suffix=" veh"
                />
                <Figure
                  label="Peak queue (modelled links)"
                  measurement={run.network.modelled_peak_queue_veh}
                  suffix=" veh"
                />
                <Figure
                  label="Total delay (modelled links)"
                  measurement={run.network.modelled_total_delay_veh_s}
                  digits={0}
                  suffix=" veh·s"
                />
                <div>
                  <dt>Coverage</dt>
                  <dd>
                    {run.network.modelled_link_count} modelled ·{" "}
                    {run.network.unmodelled_link_count} not modelled
                  </dd>
                </div>
              </dl>
              <p className="route__reason-detail">{run.network.note}</p>
              {run.links.length > 0 && <LinkTable links={run.links} />}
            </div>
          )}

          {run?.warnings.map((warning) => (
            <p key={warning} className="route__warning">
              {warning}
            </p>
          ))}

          {run !== null && (
            <p className="route__provenance">
              {run.model_name} v{run.model_version} · network {run.network_id} v
              {run.network_version} · inputs{" "}
              {run.scenario.input_fingerprint.slice(0, 12)} · identical inputs
              reproduce this run exactly
            </p>
          )}
        </>
      )}
    </Panel>
  );
}
