/**
 * Traffic Analysis section.
 *
 * The middle rung of a three-rung ladder, and the labels say so explicitly:
 *
 * - **Computer Vision Analysis** — "what was detected?"
 * - **Traffic Analysis** — "what measurements were derived?"  (this file)
 * - **Traffic Management** — "what action should be taken?"   (a later phase)
 *
 * Every number rendered here arrives in a `MetricValue` carrying its own
 * availability. A metric that could not be computed is shown as *unavailable
 * with the backend's stated reason* — never as a zero, a dash, or a hidden row.
 * Silently omitting an unavailable measurement would let a reader assume it was
 * measured and found to be nothing.
 *
 * Image-space values are labelled as such wherever they appear. A speed in
 * pixels per second is not a speed in km/h, and the two must never be
 * confusable at a glance.
 */

import { useCallback, useState } from "react";
import { ApiError, apiPost, endpoints } from "../api/client";
import type {
  DirectionLabel,
  MetricValue,
  QualityGrade,
  TrafficAnalytics,
} from "../api/types";
import { StatusPill, type Tone } from "./StatusPill";

/** Human-readable label for each unavailability reason code. */
const AVAILABILITY_LABEL: Record<string, string> = {
  unavailable_no_calibration: "needs calibration",
  unavailable_no_geometry: "needs road geometry",
  unavailable_insufficient_data: "insufficient data",
  unavailable_not_applicable: "not applicable",
};

const QUALITY_TONE: Record<QualityGrade, Tone> = {
  good: "ok",
  degraded: "pending",
  poor: "danger",
};

/** Image-space heading labels. Never compass directions. */
const DIRECTION_ARROW: Record<DirectionLabel, string> = {
  right: "→",
  down_right: "↘",
  down: "↓",
  down_left: "↙",
  left: "←",
  up_left: "↖",
  up: "↑",
  up_right: "↗",
};

function formatNumber(value: number, unit: string | null): string {
  const decimals = Math.abs(value) >= 100 ? 0 : Math.abs(value) >= 10 ? 1 : 3;
  const text = value.toFixed(decimals);
  return unit ? `${text} ${unit}` : text;
}

/**
 * Render one metric: its value, or an explicit statement that it is unavailable.
 *
 * The unavailable branch shows the backend's own reason rather than a generic
 * placeholder, because "needs a calibrated camera" and "clip too short" require
 * completely different responses from an operator.
 */
function Metric({
  label,
  metric,
  hint,
}: {
  label: string;
  metric: MetricValue<number | string> | undefined;
  hint?: string;
}): JSX.Element {
  if (metric === undefined) {
    return (
      <div className="row">
        <span className="row__name">{label}</span>
        <span className="row__detail" />
        <span className="row__aside">
          <span className="chip">not reported</span>
        </span>
      </div>
    );
  }

  const isAvailable = metric.availability === "available" && metric.value !== null;

  return (
    <div className="row" data-testid={`metric-${label.toLowerCase().replace(/\s+/g, "-")}`}>
      <span className="row__name">{label}</span>
      <span className="row__detail">
        {isAvailable ? (hint ?? "") : (metric.reason ?? "")}
      </span>
      <span className="row__aside">
        {isAvailable ? (
          <>
            <span className="kv__value">
              {typeof metric.value === "number"
                ? formatNumber(metric.value, metric.unit)
                : String(metric.value)}
            </span>
            {metric.is_image_space && (
              <StatusPill title="Measured in pixels or frame fractions, not physical units">
                image-space
              </StatusPill>
            )}
          </>
        ) : (
          <StatusPill tone="pending">
            {AVAILABILITY_LABEL[metric.availability] ?? "unavailable"}
          </StatusPill>
        )}
      </span>
    </div>
  );
}

function TallyRow({
  label,
  tally,
  hint,
}: {
  label: string;
  tally: { by_class: Record<string, number>; total: number; vehicles: number; pedestrians: number };
  hint: string;
}): JSX.Element {
  const entries = Object.entries(tally.by_class).sort((a, b) => b[1] - a[1]);
  return (
    <div className="cv__tally" data-testid={`tally-${label.toLowerCase().replace(/\s+/g, "-")}`}>
      <div className="cv__tally-head">
        <span className="cv__tally-label">{label}</span>
        <span className="cv__tally-total">{tally.total}</span>
      </div>
      <div className="chips">
        {entries.length === 0 ? (
          <span className="chip">none</span>
        ) : (
          entries.map(([cls, count]) => (
            <span key={cls} className="chip chip--on">
              {cls} {count}
            </span>
          ))
        )}
      </div>
      <div className="chips">
        <span className="chip">vehicles {tally.vehicles}</span>
        <span className="chip">pedestrians {tally.pedestrians}</span>
      </div>
      <p className="cv__tally-hint">{hint}</p>
    </div>
  );
}

/** Spatial distribution grid: where in the frame objects sit. */
function DistributionGrid({ analytics }: { analytics: TrafficAnalytics }): JSX.Element {
  const grid = analytics.spatial_distribution.grid_counts;
  const max = Math.max(1, ...grid.flat());
  return (
    <div className="cv__tally" data-testid="spatial-distribution">
      <div className="cv__tally-head">
        <span className="cv__tally-label">Spatial distribution</span>
      </div>
      <div
        className="dist-grid"
        style={{ gridTemplateColumns: `repeat(${analytics.spatial_distribution.grid_cols}, 1fr)` }}
        aria-label="Object counts per region of the frame"
      >
        {grid.flat().map((count, index) => (
          <span
            key={index}
            className="dist-grid__cell"
            style={{ opacity: 0.25 + 0.75 * (count / max) }}
            title={`${count} object(s)`}
          >
            {count}
          </span>
        ))}
      </div>
      <p className="cv__tally-hint">
        Object ground-contact points per region of the frame. Image-space only — it
        says nothing about lanes or approaches, which need a zone configuration.
      </p>
    </div>
  );
}

function DirectionHistogram({ analytics }: { analytics: TrafficAnalytics }): JSX.Element {
  const entries = Object.entries(analytics.direction_histogram) as Array<
    [DirectionLabel, number]
  >;
  return (
    <div className="cv__tally" data-testid="direction-histogram">
      <div className="cv__tally-head">
        <span className="cv__tally-label">Movement directions</span>
      </div>
      <div className="chips">
        {entries.length === 0 ? (
          <span className="chip">no measurable movement</span>
        ) : (
          entries.map(([direction, count]) => (
            <span key={direction} className="chip chip--on">
              {DIRECTION_ARROW[direction]} {direction.replace("_", "-")} {count}
            </span>
          ))
        )}
      </div>
      <p className="cv__tally-hint">
        Headings in <strong>image space</strong>, measured clockwise from frame-right.
        These are not compass bearings: deriving north would require the camera's
        orientation, which is not configured.
      </p>
    </div>
  );
}

function TrajectoryTable({ analytics }: { analytics: TrafficAnalytics }): JSX.Element {
  const rows = analytics.trajectories.slice(0, 12);
  return (
    <div className="rows" data-testid="trajectory-table">
      {rows.map((t) => {
        const speed = t.image_speed_px_per_s;
        const direction = t.direction_label;
        return (
          <div className="row" key={t.track_id} data-testid={`trajectory-${t.track_id}`}>
            <span className="row__name">
              #{t.track_id} {t.road_user_class}
            </span>
            <span className="row__detail">
              frames {t.first_frame_index}–{t.last_frame_index} ·{" "}
              {t.duration_s.availability === "available" && t.duration_s.value !== null
                ? `${t.duration_s.value.toFixed(2)} s`
                : "duration unavailable"}
              {t.is_stationary && " · stationary"}
            </span>
            <span className="row__aside">
              {direction.availability === "available" && direction.value !== null && (
                <span className="chip chip--on">
                  {DIRECTION_ARROW[direction.value as DirectionLabel]}
                </span>
              )}
              {speed.availability === "available" && speed.value !== null ? (
                <span className="kv__value">{speed.value.toFixed(1)} px/s</span>
              ) : (
                <StatusPill tone="pending">
                  {AVAILABILITY_LABEL[speed.availability] ?? "unavailable"}
                </StatusPill>
              )}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function QueueSection({ analytics }: { analytics: TrafficAnalytics }): JSX.Element {
  return (
    <div className="cv__tally" data-testid="queue-section">
      <div className="cv__tally-head">
        <span className="cv__tally-label">Queue estimate</span>
      </div>
      <Metric
        label="Stationary vehicles"
        metric={analytics.stationary_vehicle_count}
        hint="Low movement, sustained across frames"
      />
      <Metric label="Total queue length" metric={analytics.total_queue_length_m} />
      {analytics.queue_groups.length > 0 && (
        <div className="rows">
          {analytics.queue_groups.map((group) => (
            <div className="row" key={group.group_id} data-testid={`queue-group-${group.group_id}`}>
              <span className="row__name">Cluster {group.group_id}</span>
              <span className="row__detail">
                {group.member_count} stopped vehicle(s) · tracks{" "}
                {group.member_track_ids.join(", ")}
              </span>
              <span className="row__aside">
                {group.extent_px.availability === "available" &&
                group.extent_px.value !== null ? (
                  <span className="kv__value">{group.extent_px.value.toFixed(0)} px</span>
                ) : (
                  <span className="chip">extent unavailable</span>
                )}
              </span>
            </div>
          ))}
        </div>
      )}
      <p className="cv__tally-hint">
        A queue is estimated from <strong>movement, persistence and proximity</strong> —
        never from a headcount. A frame full of free-flowing vehicles is busy, not
        queued. Queue length in metres requires camera calibration.
      </p>
    </div>
  );
}

function ProvenancePanel({ analytics }: { analytics: TrafficAnalytics }): JSX.Element {
  const p = analytics.provenance;
  const q = analytics.quality;
  return (
    <div className="cv__tally" data-testid="analytics-provenance">
      <div className="cv__tally-head">
        <span className="cv__tally-label">Provenance &amp; quality</span>
        <StatusPill tone={QUALITY_TONE[q.grade]} dot>
          {q.grade}
        </StatusPill>
      </div>
      <div className="kv">
        <span className="kv__key">Observation area</span>
        <span className="kv__value">{analytics.observation_area_id}</span>
        <span className="kv__key">Source</span>
        <span className="kv__value">{p.source_id}</span>
        <span className="kv__key">Model</span>
        <span className="kv__value">
          {p.model.name}
          {p.model.sha256 ? ` · ${p.model.sha256.slice(0, 12)}` : ""}
        </span>
        <span className="kv__key">Tracker</span>
        <span className="kv__value">{p.tracker ?? "not applicable (single frame)"}</span>
        <span className="kv__key">Method</span>
        <span className="kv__value">
          {p.method} v{p.method_version}
        </span>
        <span className="kv__key">{p.frame_range ? "Frames" : "Frame"}</span>
        <span className="kv__value">
          {p.frame_range ? `${p.frame_range[0]}–${p.frame_range[1]}` : String(p.frame_index ?? "—")}
        </span>
        <span className="kv__key">Frames analysed</span>
        <span className="kv__value">{q.frames_analysed}</span>
        <span className="kv__key">Mean confidence</span>
        <span className="kv__value">
          {q.mean_detection_confidence !== null
            ? `${(q.mean_detection_confidence * 100).toFixed(1)}%`
            : "not reported"}
        </span>
        <span className="kv__key">Calibration</span>
        <span className="kv__value">
          {q.calibration_available ? "configured" : "not configured"}
        </span>
      </div>
      {q.warnings.length > 0 && (
        <ul className="cv__warnings" data-testid="quality-warnings">
          {q.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------
export function TrafficAnalysisSection({
  mediaId,
  jobId,
  disabled,
  disabledReason,
}: {
  /** Present for image analysis. */
  mediaId?: string;
  /** Present for video analysis, once a vision job has completed. */
  jobId?: string;
  disabled?: boolean;
  disabledReason?: string;
}): JSX.Element {
  const [analytics, setAnalytics] = useState<TrafficAnalytics | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async (): Promise<void> => {
    setBusy(true);
    setError(null);
    try {
      const path =
        jobId !== undefined
          ? endpoints.analyseJobTraffic(jobId)
          : endpoints.analyseImageTraffic(mediaId ?? "");
      setAnalytics(await apiPost<TrafficAnalytics>(path));
    } catch (caught) {
      // The backend's message states exactly what could not be measured.
      setError(caught instanceof ApiError ? caught.message : "Traffic analysis failed.");
      setAnalytics(null);
    } finally {
      setBusy(false);
    }
  }, [mediaId, jobId]);

  const canRun = !disabled && (mediaId !== undefined || jobId !== undefined);

  return (
    <section className="cv" data-testid="traffic-analysis">
      <div className="cv__heading">
        <span className="cv__title">Traffic Analysis</span>
        <StatusPill tone="accent">Measurements derived</StatusPill>
        {analytics !== null && (
          <StatusPill title="Snapshot measures one frame; sequence measures motion over time">
            {analytics.mode}
          </StatusPill>
        )}
      </div>

      <p className="cv__tally-hint">
        Computer Vision Analysis answers <em>what was detected</em>. This section answers{" "}
        <em>what measurements were derived</em>. What those measurements{" "}
        <em>mean</em> — and what to do about them — is Traffic Management, a later phase.
      </p>

      <div className="cv__actions">
        <button type="button" className="button" onClick={() => void run()} disabled={busy || !canRun}>
          {busy ? "Measuring…" : "Derive measurements"}
        </button>
        {disabled === true && (
          <span className="source__note">{disabledReason ?? "Run detection first."}</span>
        )}
      </div>

      {error !== null && (
        <p className="upload__error" role="alert" data-testid="analytics-error">
          {error}
        </p>
      )}

      {analytics === null && error === null && (
        <div className="state" data-testid="analytics-empty">
          <p className="state__title">No measurements yet</p>
          <p className="state__message">
            Derive measurements to see counts, positions, movement and queue evidence.
            Nothing is calculated until you do.
          </p>
        </div>
      )}

      {analytics !== null && (
        <div className="cv__results" data-testid="analytics-results">
          <TallyRow
            label="Detected objects"
            tally={analytics.detection_tally}
            hint={
              analytics.mode === "sequence"
                ? "Detection boxes across all analysed frames. One object appears once per frame it is visible in — this is not a vehicle count."
                : "Objects detected in this frame."
            }
          />

          {analytics.mode === "sequence" && (
            <TallyRow
              label="Tracked objects"
              tally={analytics.tracked_object_tally}
              hint="Distinct objects followed across frames. Closer to a vehicle count, but it does not account for objects entering before or leaving after the clip."
            />
          )}

          <div className="rows">
            <Metric
              label="Active tracked objects"
              metric={analytics.active_tracked_objects}
              hint="Present in the final analysed frame"
            />
            <Metric
              label="Mean movement speed"
              metric={analytics.mean_image_speed_diagonals_per_s}
              hint="Frame diagonals per second — resolution-independent"
            />
            <Metric
              label="Mean trajectory duration"
              metric={analytics.mean_trajectory_duration_s}
            />
            <Metric
              label="Arrival rate"
              metric={analytics.arrival_rate_per_minute}
              hint="Distinct objects first observed, per minute"
            />
            <Metric
              label="Image occupancy"
              metric={analytics.spatial_distribution.image_occupancy_ratio}
              hint="Fraction of frame area covered — not traffic density"
            />
            <Metric
              label="Physical speed"
              metric={analytics.mean_speed_kph}
            />
            <Metric
              label="Traffic density"
              metric={analytics.traffic_density_veh_per_km_per_lane}
            />
          </div>

          <DistributionGrid analytics={analytics} />

          {analytics.mode === "sequence" && (
            <>
              <DirectionHistogram analytics={analytics} />
              {analytics.trajectories.length > 0 && <TrajectoryTable analytics={analytics} />}
              <QueueSection analytics={analytics} />
            </>
          )}

          <ProvenancePanel analytics={analytics} />
        </div>
      )}
    </section>
  );
}
