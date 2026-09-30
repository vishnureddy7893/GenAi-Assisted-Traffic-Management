/**
 * Computer Vision Analysis.
 *
 * Displays what the perception pipeline saw: detected road users, their classes
 * and confidences, tracked identities across video frames, and the model and
 * device that produced them.
 *
 * It displays **no traffic conclusion**. No congestion level, no queue length,
 * no ETA, no route, no recommendation — those require the traffic state
 * extractor and the phases after it, and none of them exist yet.
 *
 * Counts shown here are tallies of boxes and tracks, labelled as such. A
 * stationary vehicle across sixty frames contributes sixty detections and one
 * track, which is exactly why neither number is a traffic volume.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, apiGet, apiPost, endpoints, overlayUrl } from "../api/client";
import type {
  ClassTally,
  DetectionOut,
  DetectorStatus,
  FrameListResponse,
  FrameResult,
  ImageAnalysis,
  InferenceMetadata,
  RoadUserClass,
  TrackListResponse,
  TrackSummary,
  VideoAnalysisSummary,
} from "../api/types";
import { StatusPill } from "./StatusPill";

/** Class colours, matching the backend overlay renderer so the two agree. */
const CLASS_COLOURS: Partial<Record<RoadUserClass, string>> = {
  car: "rgb(56,176,255)",
  truck: "rgb(255,140,66)",
  bus: "rgb(255,214,61)",
  motorcycle: "rgb(168,130,255)",
  bicycle: "rgb(94,224,168)",
  person: "rgb(255,99,132)",
};

function colourFor(cls: RoadUserClass): string {
  return CLASS_COLOURS[cls] ?? "rgb(180,190,205)";
}

/** Section heading required to label perception output unambiguously. */
function AnalysisHeading({ aside }: { aside?: React.ReactNode }): JSX.Element {
  return (
    <div className="cv__heading">
      <h3 className="cv__title">Computer Vision Analysis</h3>
      {aside}
    </div>
  );
}

/** Model, device and threshold provenance for a set of results. */
function InferencePanel({ inference }: { inference: InferenceMetadata }): JSX.Element {
  return (
    <div className="rows" data-testid="inference-metadata">
      <div className="row">
        <span className="row__name">Model</span>
        <span className="row__detail">{inference.model.backend}</span>
        <span className="row__aside">
          <span className="kv__value">{inference.model.name}</span>
        </span>
      </div>
      <div className="row">
        <span className="row__name">Weights checksum</span>
        <span className="row__detail">Identifies the exact bytes loaded</span>
        <span className="row__aside">
          <span className="kv__value">
            {inference.model.sha256 ? inference.model.sha256.slice(0, 12) : "—"}
          </span>
        </span>
      </div>
      <div className="row">
        <span className="row__name">Device</span>
        <span className="row__detail">{inference.device_reason}</span>
        <span className="row__aside">
          <StatusPill tone={inference.device_kind === "cuda" ? "ok" : "neutral"} dot>
            {inference.device_kind.toUpperCase()}
          </StatusPill>
        </span>
      </div>
      <div className="row">
        <span className="row__name">Inference size</span>
        <span className="row__detail">Confidence ≥ {inference.confidence_threshold}</span>
        <span className="row__aside">
          <span className="kv__value">{inference.image_size} px</span>
        </span>
      </div>
      {inference.tracker !== null && (
        <div className="row">
          <span className="row__name">Tracker</span>
          <span className="row__detail">Associates detections across frames</span>
          <span className="row__aside">
            <span className="kv__value">{inference.tracker}</span>
          </span>
        </div>
      )}
    </div>
  );
}

/** Per-class counts, explicitly labelled as detection counts. */
function TallyRow({
  tally,
  label,
  hint,
  testId,
}: {
  tally: ClassTally;
  label: string;
  hint: string;
  testId: string;
}): JSX.Element {
  const entries = Object.entries(tally.by_class) as Array<[RoadUserClass, number]>;
  return (
    <div className="cv__tally" data-testid={testId}>
      <div className="cv__tally-head">
        <span className="cv__tally-label">{label}</span>
        <span className="cv__tally-total">{tally.total}</span>
      </div>
      <p className="cv__tally-hint">{hint}</p>
      {entries.length === 0 ? (
        <p className="cv__tally-hint">No road users detected.</p>
      ) : (
        <div className="chips">
          {entries
            .sort((a, b) => b[1] - a[1])
            .map(([cls, count]) => (
              <span key={cls} className="chip chip--on" style={{ borderColor: colourFor(cls) }}>
                <span
                  className="cv__swatch"
                  style={{ background: colourFor(cls) }}
                  aria-hidden="true"
                />
                {cls} {count}
              </span>
            ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Image analysis
// ---------------------------------------------------------------------------
export function ImageVisionAnalysis({
  mediaId,
  status,
  onAssetMissing,
}: {
  mediaId: string;
  status: DetectorStatus | null;
  /** Called when the backend reports this asset does not exist.
   *
   *  The panel above owns the selection, so it is the only thing that can
   *  clear it. Showing the error here and leaving a dead asset selected would
   *  let an operator press Run detection against it indefinitely.
   *
   *  The backend's message travels with it because clearing the selection
   *  unmounts this component: without it the operator would watch the asset
   *  disappear and never learn why. */
  onAssetMissing?: ((mediaId: string, message: string) => void) | undefined;
}): JSX.Element {
  const [analysis, setAnalysis] = useState<ImageAnalysis | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Reset when a different image is loaded, so results never appear attached to
  // the wrong asset.
  useEffect(() => {
    setAnalysis(null);
    setError(null);
  }, [mediaId]);

  const run = useCallback(async (): Promise<void> => {
    setBusy(true);
    setError(null);
    try {
      const result = await apiPost<ImageAnalysis>(endpoints.analyzeImage(mediaId));
      setAnalysis(result);
    } catch (caught) {
      // The backend's real message, verbatim. A failed analysis must never be
      // presented as an analysis that found nothing.
      setError(caught instanceof ApiError ? caught.message : "Analysis failed unexpectedly.");
      setAnalysis(null);
      if (caught instanceof ApiError && caught.code === "not_found") {
        onAssetMissing?.(mediaId, caught.message);
      }
    } finally {
      setBusy(false);
    }
  }, [mediaId, onAssetMissing]);

  const unavailable = status !== null && !status.detection_available;

  return (
    <section className="cv" data-testid="image-vision-analysis">
      <AnalysisHeading
        aside={
          analysis !== null ? (
            <StatusPill tone="ok" dot>
              {analysis.confirmed_tally.total} of {analysis.detection_tally.total}{" "}
              countable
            </StatusPill>
          ) : (
            <StatusPill tone="pending">Not analysed</StatusPill>
          )
        }
      />

      <div className="cv__actions">
        <button
          type="button"
          className="button"
          onClick={() => void run()}
          disabled={busy || unavailable}
        >
          {busy ? "Analysing…" : "Run detection"}
        </button>
        {unavailable && (
          <span className="source__note">
            No detector backend is available on the server.
          </span>
        )}
      </div>

      {error !== null && (
        <p className="upload__error" role="alert" data-testid="analysis-error">
          {error}
        </p>
      )}

      {analysis === null && error === null && (
        <div className="state" data-testid="analysis-empty">
          <p className="state__title">No analysis run yet</p>
          <p className="state__message">
            Run detection to see which road users are visible in this image. Nothing
            is inferred until you do.
          </p>
        </div>
      )}

      {analysis !== null && (
        <div className="cv__results">
          <figure className="preview">
            <img
              className="preview__image"
              src={overlayUrl(mediaId, analysis.analysis_id)}
              alt="Detection overlay rendered by this application"
              data-testid="detection-overlay"
            />
            <figcaption className="preview__caption">
              Detection overlay — generated by this application from the detections below
            </figcaption>
          </figure>

          <TallyRow
            tally={analysis.detection_tally}
            label="Detected objects"
            hint="Every box the detector reported. Not a traffic volume, and not
              all of it is countable."
            testId="detection-tally"
          />

          <TallyRow
            tally={analysis.confirmed_tally}
            label="Countable road users"
            hint="The subset that passed the quality checks. This is what the
              measurement layer consumes; the rest are listed below with the
              reason they were excluded."
            testId="confirmed-tally"
          />

          {analysis.detection_tally.total > analysis.confirmed_tally.total && (
            <p className="cv__tally-hint" data-testid="excluded-note">
              {analysis.detection_tally.total - analysis.confirmed_tally.total} detection(s)
              are reported but not counted — duplicates, boxes below their class's
              confirmation threshold, or shapes no road vehicle can have. They stay
              visible so a disputed verdict can be checked rather than guessed at.
            </p>
          )}

          <DetectionTable detections={analysis.detections} />
          <InferencePanel inference={analysis.inference} />
        </div>
      )}
    </section>
  );
}

function DetectionTable({ detections }: { detections: DetectionOut[] }): JSX.Element {
  if (detections.length === 0) {
    return (
      <p className="cv__tally-hint" data-testid="detection-list-empty">
        No road users were detected in this image.
      </p>
    );
  }
  return (
    <div className="rows" data-testid="detection-list">
      {detections.slice(0, 25).map((d, index) => (
        <div
          className={d.quality === "confirmed" ? "row" : "row row--unconfirmed"}
          key={`${d.road_user_class}-${index}`}
        >
          <span className="row__name">
            <span
              className="cv__swatch"
              style={{ background: colourFor(d.road_user_class) }}
              aria-hidden="true"
            />
            {d.road_user_class}
            {d.track_id !== null && ` #${d.track_id}`}
            {d.quality !== "confirmed" && (
              <StatusPill tone={d.quality === "rejected" ? "danger" : "pending"}>
                {d.quality}
              </StatusPill>
            )}
          </span>
          <span className="row__detail">
            {Math.round(d.bbox.width)} × {Math.round(d.bbox.height)} px at (
            {Math.round(d.bbox.x1)}, {Math.round(d.bbox.y1)})
            {d.quality_reason !== null && (
              <span className="cv__quality-reason">{d.quality_reason}</span>
            )}
          </span>
          <span className="row__aside">
            <span className="kv__value">{(d.confidence * 100).toFixed(1)}%</span>
          </span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Video analysis
// ---------------------------------------------------------------------------
const POLL_MS = 1200;

export function VideoVisionAnalysis({
  mediaId,
  videoRef,
  frameWidth,
  frameHeight,
  status,
  onJobCompleted,
  onAssetMissing,
}: {
  mediaId: string;
  videoRef: React.RefObject<HTMLVideoElement>;
  frameWidth: number | null;
  frameHeight: number | null;
  status: DetectorStatus | null;
  /**
   * Called with the job id once analysis completes, so the Traffic Analysis
   * section can derive measurements from perception output that already exists
   * rather than re-running inference.
   */
  onJobCompleted?: (jobId: string) => void;
  /** Called when the backend reports this asset does not exist, so the panel
   *  that owns the selection can clear it. Carries the backend's message,
   *  which would otherwise be unmounted along with this component. */
  onAssetMissing?: ((mediaId: string, message: string) => void) | undefined;
}): JSX.Element {
  const [job, setJob] = useState<VideoAnalysisSummary | null>(null);
  const [frames, setFrames] = useState<FrameResult[]>([]);
  const [tracks, setTracks] = useState<TrackSummary[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const pollRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    setJob(null);
    setFrames([]);
    setTracks([]);
    setError(null);
  }, [mediaId]);

  // Track playback position so the overlay can show the boxes for whichever
  // frame is on screen.
  useEffect(() => {
    const element = videoRef.current;
    if (element === null) return undefined;
    const onTime = (): void => setCurrentTime(element.currentTime);
    element.addEventListener("timeupdate", onTime);
    element.addEventListener("seeked", onTime);
    return () => {
      element.removeEventListener("timeupdate", onTime);
      element.removeEventListener("seeked", onTime);
    };
  }, [videoRef, job]);

  const loadResults = useCallback(async (jobId: string): Promise<void> => {
    const [framePage, trackList] = await Promise.all([
      apiGet<FrameListResponse>(`${endpoints.jobFrames(jobId)}?limit=500`),
      apiGet<TrackListResponse>(endpoints.jobTracks(jobId)),
    ]);
    setFrames(framePage.frames);
    setTracks(trackList.tracks);
  }, []);

  useEffect(() => {
    return () => {
      if (pollRef.current !== undefined) window.clearInterval(pollRef.current);
    };
  }, []);

  const run = useCallback(async (): Promise<void> => {
    setBusy(true);
    setError(null);
    setFrames([]);
    setTracks([]);
    try {
      const started = await apiPost<VideoAnalysisSummary>(endpoints.analyzeVideo(mediaId), {});
      setJob(started);

      if (pollRef.current !== undefined) window.clearInterval(pollRef.current);
      pollRef.current = window.setInterval(() => {
        void (async () => {
          try {
            const latest = await apiGet<VideoAnalysisSummary>(endpoints.job(started.job_id));
            setJob(latest);
            if (latest.status !== "running" && latest.status !== "pending") {
              if (pollRef.current !== undefined) window.clearInterval(pollRef.current);
              pollRef.current = undefined;
              setBusy(false);
              if (latest.status === "completed") {
                await loadResults(started.job_id);
                onJobCompleted?.(started.job_id);
              }
            }
          } catch (caught) {
            if (pollRef.current !== undefined) window.clearInterval(pollRef.current);
            pollRef.current = undefined;
            setBusy(false);
            setError(caught instanceof ApiError ? caught.message : "Polling failed.");
          }
        })();
      }, POLL_MS);
    } catch (caught) {
      setBusy(false);
      setError(caught instanceof ApiError ? caught.message : "Analysis failed unexpectedly.");
      if (caught instanceof ApiError && caught.code === "not_found") {
        onAssetMissing?.(mediaId, caught.message);
      }
    }
  }, [mediaId, loadResults, onAssetMissing]);

  const unavailable = status !== null && !status.detection_available;

  // Nearest analysed frame at or before the current playback position.
  const activeFrame = frames.reduce<FrameResult | null>((best, frame) => {
    const t = frame.media_time_s ?? 0;
    if (t > currentTime + 0.001) return best;
    if (best === null) return frame;
    return t >= (best.media_time_s ?? 0) ? frame : best;
  }, null);

  return (
    <section className="cv" data-testid="video-vision-analysis">
      <AnalysisHeading
        aside={
          job !== null ? (
            <StatusPill
              tone={
                job.status === "completed" ? "ok" : job.status === "failed" ? "danger" : "pending"
              }
              dot
            >
              {job.status}
            </StatusPill>
          ) : (
            <StatusPill tone="pending">Not analysed</StatusPill>
          )
        }
      />

      <div className="cv__actions">
        <button
          type="button"
          className="button"
          onClick={() => void run()}
          disabled={busy || unavailable}
        >
          {busy ? "Analysing…" : "Run detection & tracking"}
        </button>
        {job !== null && job.status === "running" && (
          <span className="source__note">
            {job.frames_analysed} frames processed
            {job.progress !== null && ` · ${Math.round(job.progress * 100)}%`}
          </span>
        )}
        {unavailable && (
          <span className="source__note">
            No detector backend is available on the server.
          </span>
        )}
      </div>

      {error !== null && (
        <p className="upload__error" role="alert" data-testid="analysis-error">
          {error}
        </p>
      )}

      {job !== null && job.status === "failed" && job.error !== null && (
        <p className="upload__error" role="alert" data-testid="job-error">
          {job.error}
        </p>
      )}

      {job === null && error === null && (
        <div className="state" data-testid="analysis-empty">
          <p className="state__title">No analysis run yet</p>
          <p className="state__message">
            Run detection and tracking to see which road users appear in this video
            and which frames they occupy.
          </p>
        </div>
      )}

      {activeFrame !== null && frameWidth !== null && frameHeight !== null && (
        <FrameOverlay
          frame={activeFrame}
          frameWidth={frameWidth}
          frameHeight={frameHeight}
        />
      )}

      {job !== null && job.status === "completed" && (
        <div className="cv__results">
          <div className="rows" data-testid="processing-info">
            <div className="row">
              <span className="row__name">Frames processed</span>
              <span className="row__detail">Stride {job.frame_stride}</span>
              <span className="row__aside">
                <span className="kv__value">{job.frames_analysed}</span>
              </span>
            </div>
            <div className="row">
              <span className="row__name">Suspected ID switches</span>
              <span className="row__detail">Implausible jumps flagged by the tracker</span>
              <span className="row__aside">
                <span className="kv__value">{job.suspected_id_switches}</span>
              </span>
            </div>
          </div>

          <TallyRow
            tally={job.track_tally}
            label="Tracked objects"
            hint="Distinct objects tracked across frames. Not a traffic count: it excludes anything entering before or leaving after this clip."
            testId="track-tally"
          />
          <TallyRow
            tally={job.detection_tally}
            label="Total detections"
            hint="Boxes summed over every analysed frame. One object appears once per frame it is visible in."
            testId="detection-tally"
          />

          <TrackTable tracks={tracks} />
          {job.inference !== null && <InferencePanel inference={job.inference} />}
        </div>
      )}
    </section>
  );
}

/** Boxes for one frame, scaled onto the displayed video. */
function FrameOverlay({
  frame,
  frameWidth,
  frameHeight,
}: {
  frame: FrameResult;
  frameWidth: number;
  frameHeight: number;
}): JSX.Element {
  return (
    <div className="cv__overlay-wrap" data-testid="video-frame-overlay">
      <svg
        viewBox={`0 0 ${frameWidth} ${frameHeight}`}
        preserveAspectRatio="xMidYMid meet"
        className="cv__overlay-svg"
        role="img"
        aria-label={`Detections for frame ${frame.frame_index}: ${frame.detections.length} objects`}
      >
        {frame.detections.map((d, index) => (
          <g key={`${d.track_id ?? index}`}>
            <rect
              x={d.bbox.x1}
              y={d.bbox.y1}
              width={d.bbox.width}
              height={d.bbox.height}
              fill="none"
              stroke={colourFor(d.road_user_class)}
              strokeWidth={Math.max(1, frameWidth / 320)}
            />
            <text
              x={d.bbox.x1}
              y={Math.max(10, d.bbox.y1 - 3)}
              fill={colourFor(d.road_user_class)}
              fontSize={Math.max(9, frameWidth / 48)}
            >
              {d.track_id !== null ? `#${d.track_id} ` : ""}
              {d.road_user_class}
            </text>
          </g>
        ))}
      </svg>
      <p className="preview__caption">
        Frame {frame.frame_index} · {frame.detections.length} detections
        {frame.media_time_s !== null && ` · ${frame.media_time_s.toFixed(2)} s`}
      </p>
    </div>
  );
}

function TrackTable({ tracks }: { tracks: TrackSummary[] }): JSX.Element {
  if (tracks.length === 0) {
    return (
      <p className="cv__tally-hint" data-testid="track-list-empty">
        No objects were tracked in this video.
      </p>
    );
  }
  return (
    <div className="rows" data-testid="track-list">
      {tracks.slice(0, 25).map((track) => (
        <div className="row" key={track.track_id}>
          <span className="row__name">
            <span
              className="cv__swatch"
              style={{ background: colourFor(track.road_user_class) }}
              aria-hidden="true"
            />
            #{track.track_id} {track.road_user_class}
          </span>
          <span className="row__detail">
            frames {track.first_frame_index}–{track.last_frame_index} · seen{" "}
            {track.frames_tracked}×
            {track.id_switch_suspected && " · ID switch suspected"}
          </span>
          <span className="row__aside">
            <StatusPill tone={track.is_confirmed ? "ok" : "pending"}>
              {track.is_confirmed ? "confirmed" : "tentative"}
            </StatusPill>
          </span>
        </div>
      ))}
    </div>
  );
}
