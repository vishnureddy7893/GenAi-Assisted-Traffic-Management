/**
 * Traffic state — the phase 5 interpretation of phase 4 measurements.
 *
 * This is the step where measurement becomes judgement, and the panel keeps the
 * two visibly apart. Everything above it on the page is what was *measured*: box
 * counts, tracks, occupancy. This section shows what the backend concluded from
 * those numbers, the signals it used, and the thresholds it used them against.
 *
 * Three things it refuses to do, because the backend refuses to do them:
 *
 * - It never renders `unknown` as a band. An unclassified verdict shows the
 *   reason the backend gave for refusing, not a neutral-looking placeholder that
 *   could be mistaken for "clear".
 * - It never hides an unavailable signal. A verdict resting on one signal out of
 *   three is weaker than one resting on all three, and the only way a reader can
 *   tell is if the missing ones are still listed with their reason.
 * - It computes nothing. The index, the band and the confidence are all read
 *   from the response; a second implementation in the browser is a second
 *   implementation that can disagree with the one that made the call.
 */

import { useCallback, useState } from "react";

import { apiPost, endpoints } from "../api/client";
import type { CongestionLevel, IntersectionTrafficState } from "../api/types";
import { StatusPill, type Tone } from "./StatusPill";

/** Presentation copy only. The bands themselves come from the API. */
const LEVEL_TONE: Record<CongestionLevel, Tone> = {
  free_flow: "ok",
  light: "ok",
  moderate: "pending",
  heavy: "danger",
  severe: "danger",
  gridlock: "danger",
  unknown: "neutral",
};

const LEVEL_LABEL: Record<CongestionLevel, string> = {
  free_flow: "Free flow",
  light: "Light",
  moderate: "Moderate",
  heavy: "Heavy",
  severe: "Severe",
  gridlock: "Gridlock",
  unknown: "Unclassified",
};

const BASIS_LABEL: Record<string, string> = {
  physical: "calibrated measurements",
  image_space: "image-space measurements",
  insufficient: "insufficient evidence",
};

function StateBody({ state }: { state: IntersectionTrafficState }): JSX.Element {
  const { congestion } = state;
  const available = congestion.signals.filter(
    (signal) => signal.availability === "available",
  );
  const missing = congestion.signals.filter(
    (signal) => signal.availability !== "available",
  );

  return (
    <div className="tstate">
      <div className="tstate__verdict">
        <StatusPill tone={LEVEL_TONE[congestion.level]} dot>
          {LEVEL_LABEL[congestion.level]}
        </StatusPill>
        <span className="tstate__index">
          {congestion.index === null ? (
            <span className="tstate__absent">no index</span>
          ) : (
            <>index {congestion.index.toFixed(4)}</>
          )}
        </span>
        <span className="tstate__basis">
          from {BASIS_LABEL[congestion.basis] ?? congestion.basis}
          {congestion.confidence !== null && (
            <> · confidence {congestion.confidence.toFixed(2)}</>
          )}
        </span>
      </div>

      {congestion.reason !== null && (
        <p className="tstate__reason">{congestion.reason}</p>
      )}

      <dl className="tstate__figures">
        <div>
          <dt>Vehicles</dt>
          <dd>{state.vehicle_count}</dd>
        </div>
        <div>
          <dt>Stationary</dt>
          <dd className={state.stationary_vehicle_count === null ? "tstate__absent" : ""}>
            {state.stationary_vehicle_count ?? "unavailable"}
          </dd>
        </div>
        <div>
          <dt>Measurement quality</dt>
          <dd>{state.quality_grade}</dd>
        </div>
        <div>
          <dt>Observation area</dt>
          <dd className="tstate__mono">{state.observation_area_id}</dd>
        </div>
      </dl>

      {available.length > 0 && (
        <div className="tstate__signals">
          <p className="tstate__signals-title">Signals used</p>
          {available.map((signal) => (
            <div key={signal.name} className="tstate__signal">
              <span className="tstate__signal-name">
                {signal.name.replace(/_/g, " ")}
              </span>
              <span className="tstate__signal-figures">
                {signal.raw_value?.toFixed(4)}
                {signal.unit !== null && ` ${signal.unit}`} → score{" "}
                {signal.score?.toFixed(3)} · weight {signal.weight}
              </span>
            </div>
          ))}
        </div>
      )}

      {missing.length > 0 && (
        <div className="tstate__signals">
          <p className="tstate__signals-title">
            Signals not available ({missing.length})
          </p>
          {missing.map((signal) => (
            <div key={signal.name} className="tstate__signal">
              <span className="tstate__signal-name tstate__absent">
                {signal.name.replace(/_/g, " ")}
              </span>
              <span className="tstate__signal-reason">{signal.reason}</span>
            </div>
          ))}
        </div>
      )}

      <p className="tstate__method">
        {congestion.method} v{congestion.method_version} · state {state.state_id}
      </p>
    </div>
  );
}

export function TrafficStateSection({
  mediaId,
  jobId,
  disabled,
  disabledReason,
}: {
  /** Present for an ingested image. */
  mediaId?: string;
  /** Present for a video, once its vision job has completed. */
  jobId?: string;
  disabled?: boolean;
  disabledReason?: string;
}): JSX.Element {
  const [state, setState] = useState<IntersectionTrafficState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const interpret = useCallback(async () => {
    const path =
      jobId !== undefined
        ? endpoints.interpretJobState(jobId)
        : mediaId !== undefined
          ? endpoints.interpretImageState(mediaId)
          : null;
    if (path === null) return;

    setBusy(true);
    setError(null);
    try {
      setState(await apiPost<IntersectionTrafficState>(path, {}));
    } catch (caught) {
      setState(null);
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }, [mediaId, jobId]);

  return (
    <section className="section">
      <div className="section__head">
        <h3 className="section__title">Traffic State</h3>
        <button
          type="button"
          className="btn btn--small"
          onClick={() => void interpret()}
          disabled={busy || disabled === true}
        >
          {busy ? "Interpreting…" : "Interpret traffic state"}
        </button>
      </div>

      <p className="section__hint">
        Turns the measurements above into a congestion band, using configured
        thresholds. Where the evidence does not support a band the answer is
        <em> unclassified with a reason</em>, never a default.
      </p>

      {disabled === true && disabledReason !== undefined && (
        <p className="section__note">{disabledReason}</p>
      )}

      {error !== null && <p className="section__error">{error}</p>}

      {state === null && error === null && !busy && (
        <p className="section__note">
          No traffic state has been interpreted for this input yet. Run the traffic
          analysis above first — interpretation reads those measurements rather
          than recomputing them.
        </p>
      )}

      {state !== null && <StateBody state={state} />}
    </section>
  );
}
