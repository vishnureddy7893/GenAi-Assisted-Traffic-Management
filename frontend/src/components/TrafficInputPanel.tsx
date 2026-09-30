/**
 * Traffic input selector: Image · Video · Live (future).
 *
 * This replaces the "No source configured" placeholder with a working ingest
 * surface. Everything it displays about an ingested file was measured by the
 * backend and returned over the API — dimensions, format, frame rate, duration.
 *
 * Since Phase 3 it also hosts the Computer Vision Analysis section, which
 * reports detected and tracked road users.
 *
 * What it deliberately never displays: congestion, speed, queue length, travel
 * time, predictions, routes or recommendations. None of those exist yet. The
 * counts it does show are tallies of detection boxes and tracked objects,
 * labelled as such — a tally of boxes is not a traffic volume.
 *
 * The accepted file types come from `/api/media/formats` rather than a literal
 * in this file, so the picker cannot drift away from what the backend actually
 * validates.
 */

import { useCallback, useRef, useState } from "react";
import {
  ApiError,
  apiPost,
  apiUpload,
  endpoints,
  mediaContentUrl,
} from "../api/client";
import type {
  DetectorStatus,
  FormatsResponse,
  MediaAsset,
  MediaKind,
  PlaybackState,
} from "../api/types";
import { useApiResource } from "../hooks/useApiResource";
import { Panel } from "./Panel";
import { ErrorState, LoadingState, ResourceView } from "./ResourceState";
import { StatusPill } from "./StatusPill";
import { ImageVisionAnalysis, VideoVisionAnalysis } from "./VisionAnalysis";
import { TrafficAnalysisSection } from "./TrafficAnalysis";
import { TrafficStateSection } from "./TrafficState";

type InputMode = MediaKind | "live";

/** Tab definitions. Labels and icons are presentation; no domain data here. */
const MODES: ReadonlyArray<{ id: InputMode; label: string; future: boolean }> = [
  { id: "image", label: "Image", future: false },
  { id: "video", label: "Video", future: false },
  { id: "live", label: "Live", future: true },
];

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

/** Render a measured value, or an explicit "not reported" marker. */
function MetaRow({
  label,
  value,
}: {
  label: string;
  value: string | number | null | undefined;
}): JSX.Element {
  const missing = value === null || value === undefined || value === "";
  return (
    <div className="row">
      <span className="row__name">{label}</span>
      <span className="row__detail" />
      <span className="row__aside">
        {missing ? (
          <span className="chip" title="Not reported by the file">
            not reported
          </span>
        ) : (
          <span className="kv__value">{value}</span>
        )}
      </span>
    </div>
  );
}

/**
 * Scope notice shown on every ingested asset.
 *
 * Updated in Phase 3: detection and tracking now exist, so claiming otherwise
 * would be as inaccurate as the overclaim it was written to prevent. What is
 * still absent is the *interpretation* of what those objects mean for traffic.
 */
function RawInputNotice(): JSX.Element {
  return (
    <div className="banner" data-testid="raw-input-notice">
      <span className="banner__icon" aria-hidden="true">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="9" />
          <path d="M12 11v5.5" strokeLinecap="round" />
          <circle cx="12" cy="7.8" r="0.6" fill="currentColor" stroke="none" />
        </svg>
      </span>
      <span>
        <strong>Perception only — traffic interpretation not implemented yet.</strong>{" "}
        This file is decoded, measured, and (on request) analysed to detect and track
        road users. What those objects mean for traffic is not: no congestion level,
        no queue length, no speed, no travel time, no route and no recommendation
        exists at this phase.
      </span>
    </div>
  );
}

function acceptFor(formats: FormatsResponse | null, kind: MediaKind): string {
  if (formats === null) return "";
  return formats.formats
    .filter((f) => f.kind === kind)
    .flatMap((f) => f.extensions)
    .join(",");
}

function maxBytesFor(formats: FormatsResponse | null, kind: MediaKind): number | null {
  if (formats === null) return null;
  return kind === "image" ? formats.max_image_upload_bytes : formats.max_video_upload_bytes;
}

// ---------------------------------------------------------------------------
// Upload control
// ---------------------------------------------------------------------------
function UploadControl({
  kind,
  formats,
  onUploaded,
}: {
  kind: MediaKind;
  formats: FormatsResponse | null;
  onUploaded: (asset: MediaAsset) => void;
}): JSX.Element {
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const accept = acceptFor(formats, kind);
  const maxBytes = maxBytesFor(formats, kind);

  const handleChange = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>): Promise<void> => {
      const file = event.target.files?.[0];
      if (file === undefined) return;
      setBusy(true);
      setError(null);
      try {
        const path = kind === "image" ? endpoints.uploadImage : endpoints.uploadVideo;
        const asset = await apiUpload<MediaAsset>(path, file);
        onUploaded(asset);
      } catch (caught) {
        // The backend's validation message is shown verbatim: it states exactly
        // why the file was refused, which is more useful than a generic string.
        setError(
          caught instanceof ApiError ? caught.message : "Upload failed unexpectedly.",
        );
      } finally {
        setBusy(false);
        if (inputRef.current !== null) inputRef.current.value = "";
      }
    },
    [kind, onUploaded],
  );

  return (
    <div className="upload">
      <label className="button" htmlFor={`upload-${kind}`}>
        {busy ? "Uploading…" : `Choose ${kind} file`}
      </label>
      <input
        ref={inputRef}
        id={`upload-${kind}`}
        data-testid={`upload-input-${kind}`}
        type="file"
        accept={accept}
        disabled={busy}
        onChange={(event) => void handleChange(event)}
        className="visually-hidden"
      />
      <span className="source__note">
        {accept === "" ? "Loading accepted formats…" : `Accepts ${accept}`}
        {maxBytes !== null && ` · up to ${formatBytes(maxBytes)}`}
      </span>
      {error !== null && (
        <p className="upload__error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Image mode
// ---------------------------------------------------------------------------
function ImageMode({
  formats,
  detector,
  onMediaChanged,
}: {
  formats: FormatsResponse | null;
  detector: DetectorStatus | null;
  onMediaChanged?: ((mediaId: string | null) => void) | undefined;
}): JSX.Element {
  const [asset, setAsset] = useState<MediaAsset | null>(null);
  const [lost, setLost] = useState<string | null>(null);

  const load = useCallback(
    (next: MediaAsset) => {
      setAsset(next);
      setLost(null);
      onMediaChanged?.(next.media_id);
    },
    [onMediaChanged],
  );

  /**
   * Drop a selection the backend no longer has.
   *
   * The asset id is only ever the one the backend returned from an upload, so
   * this fires when the asset has genuinely gone: deleted, or its file removed
   * from the media directory. Keeping it selected would let an operator press
   * Run detection against a dead id indefinitely, which is the state this
   * recovers from. The failure is stated, not hidden, and the panel returns to
   * the one action that can succeed -- ingesting an image.
   */
  const forget = useCallback(
    (missingId: string, message: string) => {
      setAsset((current) =>
        current !== null && current.media_id === missingId ? null : current,
      );
      setLost(message);
      onMediaChanged?.(null);
    },
    [onMediaChanged],
  );

  return (
    <div className="ingest">
      <UploadControl kind="image" formats={formats} onUploaded={load} />

      {lost !== null && asset === null && (
        <p className="ingest__lost" role="alert" data-testid="image-asset-lost">
          {lost} The selected image is no longer held by the server, so it has
          been cleared. Ingest an image again to continue.
        </p>
      )}

      {asset === null ? (
        <div className="state">
          <p className="state__title">No image loaded</p>
          <p className="state__message">
            Choose a traffic image to ingest. It will be decoded and measured; its
            contents will not be analysed.
          </p>
        </div>
      ) : (
        <div className="ingest__loaded" data-testid="image-loaded">
          <figure className="preview">
            <img
              src={mediaContentUrl(asset.media_id)}
              alt={`Ingested traffic still: ${asset.original_filename}`}
              className="preview__image"
            />
            <figcaption className="preview__caption">
              {asset.original_filename}
            </figcaption>
          </figure>

          <div className="rows">
            <MetaRow label="Source file" value={asset.original_filename} />
            <MetaRow label="Format" value={asset.format.toUpperCase()} />
            <MetaRow label="File size" value={formatBytes(asset.size_bytes)} />
            <MetaRow
              label="Dimensions"
              value={
                asset.image === null
                  ? null
                  : `${asset.image.width} × ${asset.image.height} px`
              }
            />
            <MetaRow label="Colour mode" value={asset.image?.mode ?? null} />
            <MetaRow label="Channels" value={asset.image?.channels ?? null} />
            <MetaRow
              label="Animated"
              value={asset.image === null ? null : asset.image.is_animated ? "yes" : "no"}
            />
            {asset.image?.frame_count != null && (
              <MetaRow label="Frames" value={asset.image.frame_count} />
            )}
          </div>

          <RawInputNotice />
          <ImageVisionAnalysis
            mediaId={asset.media_id}
            status={detector}
            onAssetMissing={forget}
          />
          <TrafficAnalysisSection
            mediaId={asset.media_id}
            disabled={detector !== null && !detector.detection_available}
            disabledReason="No detector backend is available, so nothing can be measured."
          />
          <TrafficStateSection
            mediaId={asset.media_id}
            disabled={detector !== null && !detector.detection_available}
            disabledReason="No detector backend is available, so there is nothing to interpret."
          />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Video mode
// ---------------------------------------------------------------------------
function PlaybackControls({ mediaId }: { mediaId: string }): JSX.Element {
  const [state, setState] = useState<PlaybackState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(
    async (action: string, body?: unknown): Promise<void> => {
      setBusy(true);
      setError(null);
      try {
        const next = await apiPost<PlaybackState>(
          endpoints.playbackAction(mediaId, action),
          body,
        );
        setState(next);
      } catch (caught) {
        setError(caught instanceof ApiError ? caught.message : "Playback control failed.");
      } finally {
        setBusy(false);
      }
    },
    [mediaId],
  );

  return (
    <div className="playback" data-testid="playback-controls">
      <div className="playback__buttons">
        <button type="button" className="button" disabled={busy} onClick={() => void run("open")}>
          Open
        </button>
        <button type="button" className="button" disabled={busy} onClick={() => void run("read")}>
          Read frame
        </button>
        <button
          type="button"
          className="button"
          disabled={busy}
          onClick={() => void run("seek", { frame_index: 0 })}
        >
          Seek to start
        </button>
        <button type="button" className="button" disabled={busy} onClick={() => void run("reset")}>
          Reset
        </button>
        <button type="button" className="button" disabled={busy} onClick={() => void run("close")}>
          Close
        </button>
      </div>

      {error !== null && (
        <p className="upload__error" role="alert">
          {error}
        </p>
      )}

      {state !== null && (
        <div className="rows" data-testid="playback-state">
          <MetaRow label="Session" value={state.is_open ? "open" : "closed"} />
          <MetaRow label="Current frame index" value={state.current_frame_index} />
          <MetaRow label="Playback position" value={`${state.position_s.toFixed(3)} s`} />
          <MetaRow label="Frames read" value={state.frames_read} />
          {state.exhausted && <MetaRow label="End of file" value="reached" />}
        </div>
      )}
    </div>
  );
}

function VideoMode({
  formats,
  detector,
}: {
  formats: FormatsResponse | null;
  detector: DetectorStatus | null;
}): JSX.Element {
  const [asset, setAsset] = useState<MediaAsset | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [lost, setLost] = useState<string | null>(null);

  const load = useCallback((next: MediaAsset) => {
    setAsset(next);
    setJobId(null);
    setLost(null);
  }, []);

  /** Drop a selection the backend no longer has. See ImageMode for why. */
  const forget = useCallback((missingId: string, message: string) => {
    setAsset((current) =>
      current !== null && current.media_id === missingId ? null : current,
    );
    setJobId(null);
    setLost(message);
  }, []);

  return (
    <div className="ingest">
      <UploadControl kind="video" formats={formats} onUploaded={load} />

      {lost !== null && asset === null && (
        <p className="ingest__lost" role="alert" data-testid="video-asset-lost">
          {lost} The selected video is no longer held by the server, so it has
          been cleared. Ingest a video again to continue.
        </p>
      )}

      {asset === null ? (
        <div className="state">
          <p className="state__title">No video loaded</p>
          <p className="state__message">
            Choose a recorded traffic video to ingest. Its container will be decoded
            and measured; its contents will not be analysed.
          </p>
        </div>
      ) : (
        <div className="ingest__loaded" data-testid="video-loaded">
          <figure className="preview">
            {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
            <video
              ref={videoRef}
              className="preview__video"
              src={mediaContentUrl(asset.media_id)}
              controls
              preload="metadata"
              data-testid="video-preview"
            />
            <figcaption className="preview__caption">
              {asset.original_filename}
            </figcaption>
          </figure>

          <div className="rows">
            <MetaRow label="Source file" value={asset.original_filename} />
            <MetaRow label="Format" value={asset.format.toUpperCase()} />
            <MetaRow label="File size" value={formatBytes(asset.size_bytes)} />
            <MetaRow
              label="Dimensions"
              value={
                asset.video === null
                  ? null
                  : `${asset.video.width} × ${asset.video.height} px`
              }
            />
            <MetaRow
              label="Frame rate"
              value={asset.video?.fps == null ? null : `${asset.video.fps.toFixed(2)} fps`}
            />
            <MetaRow label="Frame count" value={asset.video?.frame_count ?? null} />
            <MetaRow
              label="Duration"
              value={
                asset.video?.duration_s == null
                  ? null
                  : `${asset.video.duration_s.toFixed(2)} s`
              }
            />
            <MetaRow label="Codec" value={asset.video?.codec_fourcc ?? null} />
          </div>

          <PlaybackControls mediaId={asset.media_id} />
          <RawInputNotice />
          <VideoVisionAnalysis
            mediaId={asset.media_id}
            videoRef={videoRef}
            frameWidth={asset.video?.width ?? null}
            frameHeight={asset.video?.height ?? null}
            status={detector}
            onJobCompleted={setJobId}
            onAssetMissing={forget}
          />
          <TrafficAnalysisSection
            {...(jobId !== null ? { jobId } : {})}
            disabled={jobId === null}
            disabledReason="Run detection and tracking first: measurements are derived from its output."
          />
          <TrafficStateSection
            {...(jobId !== null ? { jobId } : {})}
            disabled={jobId === null}
            disabledReason="Run detection and tracking first: interpretation reads those measurements."
          />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Live mode
// ---------------------------------------------------------------------------
function LiveMode(): JSX.Element {
  const [probe, setProbe] = useState<string | null>(null);
  const [phaseLabel, setPhaseLabel] = useState<number | null>(null);

  const check = useCallback(async (): Promise<void> => {
    try {
      await apiPost(endpoints.liveConnect);
      setProbe("The backend unexpectedly accepted a live connection.");
    } catch (caught) {
      if (caught instanceof ApiError) {
        setProbe(caught.message);
        setPhaseLabel(caught.phase);
      } else {
        setProbe("Live connection is unavailable.");
      }
    }
  }, []);

  return (
    <div className="ingest" data-testid="live-mode">
      <div className="state">
        <div className="state__icon state__icon--pending" aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 3l18 18" strokeLinecap="round" />
            <path d="M5 9a9 9 0 0 1 2.6-1.8M8.5 12.5a5 5 0 0 1 1.6-1.1M12 16.5h.01" strokeLinecap="round" />
          </svg>
        </div>
        <p className="state__title">Live traffic — future capability</p>
        <p className="state__message">
          Live ingestion is not implemented. No camera, RTSP endpoint or traffic feed
          is connected, and no live connection is simulated.
        </p>
        <p className="state__hint">
          The architecture for it is in place: the live source interface, its
          capability profile (unbounded, non-replayable, real-time) and its
          backpressure policy are all defined. Only the implementation is absent.
        </p>
        <button type="button" className="button" onClick={() => void check()}>
          Check availability
        </button>
        {probe !== null && (
          <p className="state__message" data-testid="live-probe-result">
            {phaseLabel !== null && (
              <StatusPill tone="pending">Phase {phaseLabel}</StatusPill>
            )}{" "}
            {probe}
          </p>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------
export function TrafficInputPanel({
  onMediaChanged,
}: {
  /** Called with the ingested media id, so a later stage can explain it.
   *  Optional: the panel is fully usable without a listener. */
  onMediaChanged?: ((mediaId: string | null) => void) | undefined;
} = {}): JSX.Element {
  const [mode, setMode] = useState<InputMode>("image");
  const { resource, refresh } = useApiResource<FormatsResponse>(endpoints.mediaFormats);
  const formats = resource.state === "success" ? resource.data : null;
  const detectorResource = useApiResource<DetectorStatus>(endpoints.visionStatus);
  const detector =
    detectorResource.resource.state === "success" ? detectorResource.resource.data : null;

  return (
    <Panel
      title="Traffic input"
      subtitle="Recorded image and video ingestion — raw input only"
      flush
      aside={
        detector !== null ? (
          <StatusPill tone={detector.detection_available ? "ok" : "pending"} dot>
            {detector.detection_available
              ? `Detector: ${detector.selected_backend ?? "ready"}`
              : "No detector available"}
          </StatusPill>
        ) : (
          <StatusPill tone="ok" dot>
            Recorded ingestion available
          </StatusPill>
        )
      }
    >
      <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
        <div className="tabs" role="tablist" aria-label="Traffic input type">
          {MODES.map((entry) => (
            <button
              key={entry.id}
              type="button"
              role="tab"
              id={`input-tab-${entry.id}`}
              aria-selected={mode === entry.id}
              aria-controls={`input-panel-${entry.id}`}
              className="tab"
              onClick={() => setMode(entry.id)}
            >
              {entry.label}
              {entry.future && <StatusPill tone="pending">Future</StatusPill>}
            </button>
          ))}
        </div>
      </div>

      <div
        role="tabpanel"
        id={`input-panel-${mode}`}
        aria-labelledby={`input-tab-${mode}`}
        style={{ padding: 16 }}
      >
        {mode === "live" ? (
          <LiveMode />
        ) : resource.state === "loading" ? (
          <LoadingState rows={3} />
        ) : resource.state === "error" ? (
          <ErrorState
            message={resource.message}
            unreachable={resource.unreachable}
            onRetry={refresh}
          />
        ) : (
          <ResourceView resource={resource} onRetry={refresh}>
            {() =>
              mode === "image" ? (
                <ImageMode
              formats={formats}
              detector={detector}
              onMediaChanged={onMediaChanged}
            />
              ) : (
                <VideoMode formats={formats} detector={detector} />
              )
            }
          </ResourceView>
        )}
      </div>
    </Panel>
  );
}
