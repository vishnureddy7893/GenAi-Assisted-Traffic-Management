/**
 * Site-to-junction binding — a proposal, and the operator who accepts it.
 *
 * The distinction this panel exists to keep visible is **nearest is not
 * watched**. The backend measures the distance from a camera's surveyed
 * coordinate to every junction on the map and offers the closest one. That is
 * evidence, and it is routinely wrong in ways only a person can catch: a mast
 * between two junctions is nearest to one and pointed at the other, and a
 * coordinate typed one digit out is nearest to a junction on a different road.
 *
 * So the candidate is rendered as a question. Nothing in this panel binds
 * anything until the operator presses Confirm, the button says what it will do,
 * and the reasons the proposal might be wrong are shown next to it rather than
 * buried — including when the distance is inside the site's own position
 * accuracy, at which point the ranking is noise and the panel says so.
 *
 * Choosing a junction other than the nearest is allowed and is the case that
 * most needs a record, so it requires a written reason. The backend enforces
 * that too; disabling the button early just means the operator finds out before
 * submitting rather than after.
 *
 * Once a junction is confirmed, a second stage appears: which *directed segment*
 * the camera watches. The same discipline, one step harder. At a junction the
 * two directions of every road lie on the same line, so the camera bearing can
 * narrow the choice to a line and can never narrow it to a direction of travel.
 * The bearing offset is therefore shown as a column, the list stays ordered by
 * segment id so no row is presented as the answer, and segments sharing a line
 * are called out as indistinguishable.
 *
 * Nothing in this panel implies a confirmed binding changes a route. It does
 * not, and the backend says so in `route_cost_note`, which is rendered rather
 * than paraphrased.
 */

import { useCallback, useEffect, useState } from "react";

import { ApiError, apiGet, apiPost, endpoints } from "../api/client";
import type {
  CandidateResponse,
  ConfirmedBindingResponse,
  ConfirmedSiteBinding,
  JunctionCandidate,
  ObservationSite,
  RoadNetwork,
  SegmentCandidate,
  SegmentCandidatesResponse,
  SiteListResponse,
} from "../api/types";
import type { Resource } from "../hooks/useApiResource";
import { EffectiveMappingSummary } from "./EffectiveMappingSummary";
import { Panel } from "./Panel";
import { EmptyState } from "./ResourceState";
import { StatusPill, type Tone } from "./StatusPill";

/** Metres, rendered at the precision the measurement actually supports. */
function metres(value: number): string {
  return value >= 100 ? `${Math.round(value)} m` : `${value.toFixed(1)} m`;
}

function siteLabel(site: ObservationSite): string {
  return site.label ?? site.observation_area_id;
}

/** Degrees, at the precision a compass bearing actually supports. */
function degrees(value: number): string {
  return `${Math.round(value)}\u00B0`;
}

const STATUS_TONE: Record<string, Tone> = {
  candidate_found: "ok",
  out_of_range: "pending",
  no_positioned_junction: "pending",
};

function CandidateRow({
  entry,
  emphasis,
}: {
  entry: JunctionCandidate;
  emphasis: boolean;
}): JSX.Element {
  return (
    <li
      className={emphasis ? "cand cand--nearest" : "cand"}
      data-testid={`candidate-${entry.junction_id}`}
    >
      <span className="cand__id">{entry.junction_id}</span>
      {entry.junction_name !== null && (
        <span className="cand__name">{entry.junction_name}</span>
      )}
      <span className="cand__distance">{metres(entry.distance_m)}</span>
      {entry.within_site_accuracy && (
        <StatusPill tone="pending">within position accuracy</StatusPill>
      )}
    </li>
  );
}

function SegmentRow({
  entry,
  selected,
  onSelect,
}: {
  entry: SegmentCandidate;
  selected: boolean;
  onSelect: (segmentId: string) => void;
}): JSX.Element {
  return (
    <li className="seg" data-testid={`segment-${entry.segment_id}`}>
      <label className="seg__pick">
        <input
          type="radio"
          name="segment-choice"
          value={entry.segment_id}
          checked={selected}
          onChange={() => onSelect(entry.segment_id)}
          aria-label={`${entry.segment_id} ${entry.direction} toward ${entry.other_junction_id}`}
        />
        <span className="seg__id">{entry.segment_id}</span>
      </label>
      <StatusPill tone={entry.direction === "incoming" ? "accent" : "neutral"}>
        {entry.direction}
      </StatusPill>
      <span className="seg__arm">
        {entry.direction === "incoming" ? "from" : "toward"} {entry.other_junction_id}
      </span>
      {entry.segment_name !== null && (
        <span className="seg__name">{entry.segment_name}</span>
      )}
      {entry.approach_id !== null && (
        <span className="seg__approach">approach {entry.approach_id}</span>
      )}
      <span className="seg__bearing">
        {entry.camera_bearing_offset_degrees !== null
          ? `${degrees(entry.camera_bearing_offset_degrees)} off camera`
          : "no bearing evidence"}
      </span>
    </li>
  );
}

export function SiteBindingPanel({
  network,
}: {
  network: Resource<RoadNetwork>;
}): JSX.Element {
  const loaded = network.state === "success" ? network.data : null;

  const [sites, setSites] = useState<ObservationSite[] | null>(null);
  const [selected, setSelected] = useState<string>("");
  const [candidate, setCandidate] = useState<CandidateResponse | null>(null);
  const [junction, setJunction] = useState<string>("");
  const [notes, setNotes] = useState("");
  const [segments, setSegments] = useState<SegmentCandidatesResponse | null>(null);
  const [segment, setSegment] = useState<string>("");
  const [segmentNotes, setSegmentNotes] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Sites are listed regardless of whether a network is loaded: a camera's
  // location is recorded independently of any map, and saying "no sites" when
  // the real problem is "no map" would send the operator to the wrong place.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const body = await apiGet<SiteListResponse>(endpoints.sites);
        if (!cancelled) setSites(body.sites);
      } catch (caught) {
        if (!cancelled) {
          setError(caught instanceof ApiError ? caught.message : String(caught));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  /**
   * Fetch the segment stage for a site whose junction is confirmed.
   *
   * Reads only. The chosen segment is seeded from what is already on record and
   * is otherwise left empty — never from the closest bearing, which would be
   * the panel quietly making the decision it exists to put to a person.
   */
  const loadSegments = useCallback(
    async (areaId: string, confirmed: ConfirmedSiteBinding | null) => {
      if (confirmed === null) {
        setSegments(null);
        setSegment("");
        return;
      }
      try {
        const body = await apiGet<SegmentCandidatesResponse>(
          endpoints.siteSegments(areaId),
        );
        setSegments(body);
        setSegment(body.confirmed.segment_id ?? "");
      } catch (caught) {
        setError(caught instanceof ApiError ? caught.message : String(caught));
      }
    },
    [],
  );

  /** Fetch the proposal for one site. Reads only; it creates nothing. */
  const loadCandidate = useCallback(
    async (areaId: string) => {
      if (!areaId || loaded === null) return;
      setBusy(true);
      try {
        const body = await apiGet<CandidateResponse>(endpoints.siteCandidate(areaId));
        setCandidate(body);
        // Pre-selecting the nearest junction is a convenience, never a decision:
        // nothing is sent until Confirm is pressed.
        setJunction(body.candidate.nearest?.junction_id ?? "");
        await loadSegments(areaId, body.confirmed);
      } catch (caught) {
        setError(caught instanceof ApiError ? caught.message : String(caught));
      } finally {
        setBusy(false);
      }
    },
    [loaded, loadSegments],
  );

  const selectSite = useCallback(
    async (areaId: string) => {
      setSelected(areaId);
      setCandidate(null);
      setOutcome(null);
      setError(null);
      setNotes("");
      setJunction("");
      setSegments(null);
      setSegment("");
      setSegmentNotes("");
      await loadCandidate(areaId);
    },
    [loadCandidate],
  );

  const confirm = useCallback(async () => {
    if (!selected || !junction) return;
    setBusy(true);
    setError(null);
    try {
      const body = await apiPost<ConfirmedBindingResponse>(
        endpoints.siteBinding(selected),
        {
          junction_id: junction,
          confirm: true,
          confirmed_by: "operator.dashboard",
          ...(notes ? { notes } : {}),
        },
      );
      setOutcome(body.note);
      // Re-read the proposal so the panel now shows the binding in force.
      // The outcome message is deliberately not cleared by this refresh.
      await loadCandidate(selected);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }, [selected, junction, notes, loadCandidate]);

  const confirmSegment = useCallback(async () => {
    if (!selected || !segment || segments === null) return;
    setBusy(true);
    setError(null);
    try {
      const body = await apiPost<ConfirmedBindingResponse>(
        endpoints.siteSegmentBinding(selected),
        {
          junction_id: segments.segments.junction_id,
          segment_id: segment,
          confirm: true,
          confirmed_by: "operator.dashboard",
          ...(segmentNotes ? { notes: segmentNotes } : {}),
        },
      );
      setOutcome(body.note);
      await loadCandidate(selected);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }, [selected, segment, segments, segmentNotes, loadCandidate]);

  const nearest = candidate?.candidate.nearest ?? null;
  const overriding =
    nearest !== null && junction !== "" && junction !== nearest.junction_id;
  const canConfirm =
    candidate !== null &&
    junction !== "" &&
    !busy &&
    (!overriding || notes.trim().length > 0);

  return (
    <Panel
      title="Camera to junction"
      subtitle="Which intersection each observation site watches — proposed by distance, decided by you"
      aside={
        candidate !== null ? (
          <StatusPill tone={STATUS_TONE[candidate.candidate.status] ?? "neutral"} dot>
            {candidate.candidate.status.replace(/_/g, " ")}
          </StatusPill>
        ) : undefined
      }
    >
      {loaded === null ? (
        <EmptyState
          title="No road network is loaded"
          message="A binding names a junction, and those ids mean nothing until a network is loaded. Load one first."
        />
      ) : sites !== null && sites.length === 0 ? (
        <EmptyState
          title="No observation site has been recorded"
          message="Record where a camera is — on upload, or with POST /api/sites — and its nearest junction can then be proposed."
        />
      ) : (
        <>
          <label className="net__field">
            <span>Observation site</span>
            <select
              value={selected}
              onChange={(event) => void selectSite(event.target.value)}
              aria-label="Observation site"
            >
              <option value="">Select a site…</option>
              {(sites ?? []).map((site) => (
                <option key={site.observation_area_id} value={site.observation_area_id}>
                  {siteLabel(site)}
                </option>
              ))}
            </select>
          </label>

          {candidate !== null && (
            <div className="bind" data-testid="candidate-result">
              <p className="bind__site">
                {siteLabel(candidate.site)} ·{" "}
                {candidate.site.location.latitude.toFixed(5)},{" "}
                {candidate.site.location.longitude.toFixed(5)}
                {candidate.site.accuracy_m !== null && (
                  <> · ±{metres(candidate.site.accuracy_m)}</>
                )}
              </p>

              {nearest !== null ? (
                <p className="bind__nearest" data-testid="nearest-junction">
                  Nearest junction <strong>{nearest.junction_id}</strong>
                  {nearest.junction_name !== null && <> ({nearest.junction_name})</>}
                  {" at "}
                  <strong>{metres(nearest.distance_m)}</strong>
                </p>
              ) : (
                <p className="bind__nearest">
                  No junction on this map carries a surveyed position, so no
                  distance could be measured.
                </p>
              )}

              <p className="bind__note">{candidate.note}</p>

              {!candidate.candidate.is_discriminating && (
                <p className="bind__warn" data-testid="not-discriminating">
                  This ranking is not decisive: the distance is within the
                  position&rsquo;s own uncertainty, so proximity alone cannot
                  identify the junction.
                </p>
              )}

              {candidate.candidate.warnings.map((warning) => (
                <p key={warning} className="bind__warn">
                  {warning}
                </p>
              ))}

              {candidate.candidate.runners_up.length > 0 && (
                <ul className="cand__list">
                  {nearest !== null && <CandidateRow entry={nearest} emphasis />}
                  {candidate.candidate.runners_up.map((entry) => (
                    <CandidateRow
                      key={entry.junction_id}
                      entry={entry}
                      emphasis={false}
                    />
                  ))}
                </ul>
              )}

              {candidate.confirmed !== null && (
                <p className="bind__confirmed" data-testid="existing-binding">
                  Already bound to {candidate.confirmed.junction_id} by{" "}
                  {candidate.confirmed.confirmation.confirmed_by}. Confirming
                  again replaces that decision.
                </p>
              )}

              <label className="net__field">
                <span>Junction to confirm</span>
                <select
                  value={junction}
                  onChange={(event) => setJunction(event.target.value)}
                  aria-label="Junction to confirm"
                >
                  <option value="">Select a junction…</option>
                  {loaded.junctions.map((item) => (
                    <option key={item.junction_id} value={item.junction_id}>
                      {item.junction_id}
                      {item.name !== null ? ` — ${item.name}` : ""}
                    </option>
                  ))}
                </select>
              </label>

              {overriding && (
                <label className="net__field net__field--wide">
                  <span>Why this junction rather than the nearest?</span>
                  <input
                    value={notes}
                    onChange={(event) => setNotes(event.target.value)}
                    aria-label="Why this junction rather than the nearest?"
                    placeholder="e.g. the camera faces south across the flyover"
                  />
                </label>
              )}

              <button
                type="button"
                className="btn"
                disabled={!canConfirm}
                onClick={() => void confirm()}
              >
                {busy ? "Confirming…" : "Confirm binding"}
              </button>

              <p className="bind__disclaimer">
                Nothing is bound until you confirm. A nearest-junction result is
                proximity evidence, not a statement about what the camera is
                pointed at.
              </p>
            </div>
          )}

          {segments !== null && (
            <div className="bind bind--segment" data-testid="segment-stage">
              <p className="bind__stage">
                Step 2 — which directed road does it watch?
              </p>
              <p className="bind__site" data-testid="segment-junction">
                Junction {segments.segments.junction_id}
                {segments.segments.junction_name !== null && (
                  <> ({segments.segments.junction_name})</>
                )}{" "}
                confirmed by {segments.confirmed.confirmation.confirmed_by}
              </p>

              {segments.confirmed.segment_id !== null && (
                <p className="bind__confirmed" data-testid="existing-segment">
                  Already bound to {segments.confirmed.segment_id} (
                  {segments.confirmed.segment_direction}). Confirming another
                  replaces that.
                </p>
              )}

              {!segments.segments.bearing_evidence_available && (
                <p className="bind__warn" data-testid="no-bearing-evidence">
                  No bearing evidence is available for this site, so the choice
                  rests entirely on what you know the camera sees.
                </p>
              )}

              {segments.segments.warnings.map((warning) => (
                <p key={warning} className="bind__warn">
                  {warning}
                </p>
              ))}

              {segments.segments.candidates.length > 0 ? (
                <ul className="seg__list">
                  {segments.segments.candidates.map((entry) => (
                    <SegmentRow
                      key={entry.segment_id}
                      entry={entry}
                      selected={segment === entry.segment_id}
                      onSelect={setSegment}
                    />
                  ))}
                </ul>
              ) : (
                <p className="bind__note">{segments.segments.note}</p>
              )}

              <label className="net__field net__field--wide">
                <span>
                  Why this segment? (required when the camera bearing points
                  elsewhere)
                </span>
                <input
                  value={segmentNotes}
                  onChange={(event) => setSegmentNotes(event.target.value)}
                  aria-label="Why this segment?"
                  placeholder="e.g. the lens is angled down the south arm"
                />
              </label>

              <button
                type="button"
                className="btn"
                disabled={segment === "" || busy}
                onClick={() => void confirmSegment()}
              >
                {busy ? "Confirming\u2026" : "Confirm segment"}
              </button>

              <p className="bind__disclaimer">
                Bearing says which line the camera points along, never which
                direction of travel it watches — the two directions of one road
                share that line. A confirmed segment records which road the
                evidence belongs to and does not change any route cost.
              </p>
            </div>
          )}

          {outcome !== null && (
            <p className="bind__outcome" role="status" data-testid="confirm-outcome">
              {outcome}
            </p>
          )}
          {error !== null && (
            <p className="bind__error" role="alert">
              {error}
            </p>
          )}

          <EffectiveMappingSummary />
        </>
      )}
    </Panel>
  );
}
