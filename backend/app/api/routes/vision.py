"""Computer-vision perception API.

Phase 3 scope: **what objects are visible, and which are the same object over
time**. No endpoint here returns a traffic conclusion — no congestion level, no
queue length, no ETA, no route, no recommendation. Those require the traffic
state extractor and the phases after it.

Every response carries provenance: the source asset, the model identity
(including the checksum of the weights actually loaded), the execution device,
and the frame index and timestamp where applicable. A detection that cannot be
attributed to a specific model on a specific frame is not reportable.

When the detector cannot be loaded, these endpoints return the **real error**.
They never return an empty detection list, because "the model failed to load"
and "there are no vehicles" are opposite claims.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import Response
from pydantic import Field

from app.api.deps import (
    ImageAnalysisStoreDep,
    JobRegistryDep,
    MediaStoreDep,
    ProviderDep,
    SettingsDep,
)
from app.api.schemas import ApiModel, ErrorResponse
from app.core.errors import ResourceNotFoundError, ValidationFailedError
from app.core.logging import get_logger
from app.ingestion.media import MediaKind
from app.ingestion.playback import epoch_for
from app.vision.analysis import (
    DetectorOverrides,
    FrameResult,
    ImageAnalysis,
    TrackSummary,
    VideoAnalysisJob,
    VideoAnalysisSummary,
    analyse_image,
    new_job_id,
)
from app.vision.overlay import encode_png, render_overlay
from app.vision.quality import QualityPolicy, policy_from
from app.vision.registry import BackendStatus, backend_statuses, resolve_backend

logger = get_logger(__name__)
router = APIRouter(prefix="/vision", tags=["vision"])

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    501: {"model": ErrorResponse, "description": "No detector backend available"},
    502: {"model": ErrorResponse, "description": "Model load or inference failure"},
}


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
class DetectorStatus(ApiModel):
    """Whether perception is available, and with what.

    ``model_loaded`` reflects reality: it is only true once weights are actually
    in memory. A configured model that has never been loaded reports ``false``.
    """

    detection_available: bool
    selected_backend: str | None
    configured_model: str
    configured_device: str
    model_loaded: bool
    load_error: str | None = None
    backends: tuple[BackendStatus, ...] = ()
    confidence_threshold: float
    iou_threshold: float
    image_size: int
    max_detections: int
    quality_policy: QualityPolicy
    """The rules deciding which detections may be counted. Published because a
    vehicle count is not interpretable without them."""

    frame_stride: int
    max_frames: int
    tracker: str


@router.get("/status", response_model=DetectorStatus, summary="Detector and model status")
def detector_status(settings: SettingsDep, provider: ProviderDep) -> DetectorStatus:
    """Report perception availability without forcing a model load.

    Deliberately does not load the model: a status check that costs a
    weights load would make the dashboard expensive to open.
    """
    try:
        spec = resolve_backend(settings)
        selected = spec.name if spec else None
        available = spec is not None and spec.is_available
    except ValidationFailedError:
        selected, available = None, False

    from app.vision.bytetrack import TRACKER_NAME

    return DetectorStatus(
        detection_available=available,
        selected_backend=selected,
        configured_model=settings.detector_model,
        configured_device=settings.detector_device,
        model_loaded=provider.is_loaded,
        load_error=provider.load_error,
        backends=backend_statuses(settings),
        confidence_threshold=settings.detector_confidence_threshold,
        iou_threshold=settings.detector_iou_threshold,
        image_size=settings.detector_image_size,
        max_detections=settings.detector_max_detections,
        quality_policy=policy_from(settings),
        frame_stride=settings.video_frame_stride,
        max_frames=settings.video_max_frames,
        tracker=TRACKER_NAME,
    )


# ---------------------------------------------------------------------------
# Image analysis
# ---------------------------------------------------------------------------
class AnalyzeImageRequest(ApiModel):
    """Optional per-request detector settings.

    Every field is optional and falls back to configuration, so an existing
    caller that posts no body gets exactly the behaviour it always got.

    They exist because the right threshold is a property of the camera rather
    than of the software: a wide junction view full of distant vehicles wants a
    lower confidence and a larger input size than a close approach view, and
    finding that out requires trying it against the actual frame. Whatever is
    used is reported back in ``inference``, so a detection count is never
    separable from the settings that produced it.
    """

    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    iou_threshold: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    image_size: Annotated[int, Field(ge=64, le=4096)] | None = None
    """Inference input size. The most consequential of the four for a camera
    looking down a street, where most vehicles are small in frame."""

    max_detections: Annotated[int, Field(gt=0, le=10_000)] | None = None


@router.post(
    "/images/{media_id}/analyze",
    response_model=ImageAnalysis,
    responses=_ERROR_RESPONSES,
    summary="Run detection on an ingested image",
)
def analyze_image_endpoint(
    media_id: str,
    store: MediaStoreDep,
    provider: ProviderDep,
    settings: SettingsDep,
    analyses: ImageAnalysisStoreDep,
    request: AnalyzeImageRequest | None = None,
) -> ImageAnalysis:
    """Detect road users in a still image and cache the annotated overlay.

    Posting no body uses the configured detector. Posting thresholds runs this
    one request against them instead, on a detector built for the purpose and
    released afterwards -- the deployment's configuration is never mutated by a
    request.
    """
    asset = store.get(media_id)
    if asset.kind is not MediaKind.IMAGE:
        msg = f"Media {media_id!r} is a {asset.kind.value}, not an image."
        raise ValidationFailedError(msg, media_id=media_id)

    path = store.path_for(media_id)
    overrides = (
        DetectorOverrides(
            confidence_threshold=request.confidence_threshold,
            iou_threshold=request.iou_threshold,
            image_size=request.image_size,
            max_detections=request.max_detections,
        )
        if request is not None
        else None
    )

    with provider.with_overrides(overrides) as (detector, effective):
        analysis, frame, assessed = analyse_image(
            media_id=media_id,
            path=path,
            detector=detector,
            settings=effective,
            epoch=epoch_for(path),
        )

    # Assessed rather than raw, so an uncountable box is drawn as one.
    overlay = render_overlay(frame, assessed)
    analyses[media_id] = (analysis, encode_png(overlay))

    logger.info(
        "image analysed",
        extra={
            "media_id": media_id,
            "detections": analysis.detection_tally.total,
            "confirmed": analysis.confirmed_tally.total,
            "confirmed_vehicles": analysis.confirmed_tally.vehicles,
            "excluded": analysis.excluded_count,
            "model": analysis.inference.model.name,
            "device": analysis.inference.device_kind,
            "image_size": analysis.inference.image_size,
            "confidence_threshold": analysis.inference.confidence_threshold,
            "overridden": overrides is not None and not overrides.is_empty,
        },
    )
    return analysis


@router.get(
    "/images/{media_id}/analysis",
    response_model=ImageAnalysis,
    summary="Most recent image analysis",
)
def get_image_analysis(media_id: str, analyses: ImageAnalysisStoreDep) -> ImageAnalysis:
    """Return the cached analysis, or 404 if the image has not been analysed."""
    entry = analyses.get(media_id)
    if entry is None:
        msg = f"Image {media_id!r} has not been analysed yet."
        raise ResourceNotFoundError(msg, media_id=media_id)
    return entry[0]


@router.get("/images/{media_id}/overlay", summary="Annotated detection overlay")
def get_image_overlay(media_id: str, analyses: ImageAnalysisStoreDep) -> Response:
    """Serve the annotated PNG this application rendered.

    Generated by us from the same detection records the API returns, so the
    picture and the numbers cannot disagree.
    """
    entry = analyses.get(media_id)
    if entry is None:
        msg = f"Image {media_id!r} has not been analysed yet, so no overlay exists."
        raise ResourceNotFoundError(msg, media_id=media_id)
    return Response(
        content=entry[1],
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# Video analysis
# ---------------------------------------------------------------------------
class AnalyzeVideoRequest(ApiModel):
    """Optional per-job overrides."""

    frame_stride: Annotated[int, Field(ge=1, le=60)] | None = None
    max_frames: Annotated[int, Field(ge=1, le=10_000)] | None = None
    run_sync: bool = False
    """Run inline instead of on a worker thread. Used by tests, where a
    background thread would race the assertions."""


class JobListResponse(ApiModel):
    """Analysis jobs known to this process."""

    jobs: tuple[VideoAnalysisSummary, ...]
    count: int


class FrameListResponse(ApiModel):
    """A page of per-frame results."""

    job_id: str
    frames: tuple[FrameResult, ...]
    offset: int
    limit: int
    total: int


class TrackListResponse(ApiModel):
    """Tracked objects for one job."""

    job_id: str
    tracks: tuple[TrackSummary, ...]
    count: int


@router.post(
    "/videos/{media_id}/analyze",
    response_model=VideoAnalysisSummary,
    responses=_ERROR_RESPONSES,
    status_code=202,
    summary="Start video detection and tracking",
)
def analyze_video_endpoint(
    media_id: str,
    store: MediaStoreDep,
    provider: ProviderDep,
    settings: SettingsDep,
    jobs: JobRegistryDep,
    request: AnalyzeVideoRequest | None = None,
) -> VideoAnalysisSummary:
    """Start an analysis job over a recorded video.

    Returns 202 with the job's initial state; progress is polled from
    ``/api/vision/jobs/{job_id}``. Frames are processed incrementally, so
    memory does not scale with clip length.
    """
    asset = store.get(media_id)
    if asset.kind is not MediaKind.VIDEO:
        msg = f"Media {media_id!r} is a {asset.kind.value}, not a video."
        raise ValidationFailedError(msg, media_id=media_id)

    options = request or AnalyzeVideoRequest()
    path = store.path_for(media_id)

    job = VideoAnalysisJob(
        job_id=new_job_id(),
        media_id=media_id,
        path=path,
        provider=provider,
        settings=settings,
        epoch=epoch_for(path),
        frame_stride=options.frame_stride,
        max_frames=options.max_frames,
    )
    jobs.submit(job, run_sync=options.run_sync)
    return job.summary()


@router.get("/jobs", response_model=JobListResponse, summary="Analysis jobs")
def list_jobs(
    jobs: JobRegistryDep,
    media_id: Annotated[str | None, Query(description="Filter by media asset")] = None,
) -> JobListResponse:
    """List analysis jobs, newest first."""
    found = jobs.list_for_media(media_id) if media_id else jobs.all()
    summaries = tuple(job.summary() for job in found)
    return JobListResponse(jobs=summaries, count=len(summaries))


@router.get(
    "/jobs/{job_id}",
    response_model=VideoAnalysisSummary,
    summary="Analysis job status and results",
)
def get_job(job_id: str, jobs: JobRegistryDep) -> VideoAnalysisSummary:
    """Return status, progress and aggregate results for one job."""
    return jobs.get(job_id).summary()


@router.get(
    "/jobs/{job_id}/frames",
    response_model=FrameListResponse,
    summary="Per-frame detections and track ids",
)
def get_job_frames(
    job_id: str,
    jobs: JobRegistryDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> FrameListResponse:
    """Return a page of per-frame results.

    Paged because a long clip produces thousands of frames, and a single
    unbounded response would be both slow and useless to a client.
    """
    job = jobs.get(job_id)
    return FrameListResponse(
        job_id=job_id,
        frames=job.frames(offset, limit),
        offset=offset,
        limit=limit,
        total=job.frame_count,
    )


@router.get(
    "/jobs/{job_id}/tracks",
    response_model=TrackListResponse,
    summary="Tracked objects",
)
def get_job_tracks(job_id: str, jobs: JobRegistryDep) -> TrackListResponse:
    """Return one record per tracked object over its visible trajectory."""
    job = jobs.get(job_id)
    tracks = job.track_summaries()
    return TrackListResponse(job_id=job_id, tracks=tracks, count=len(tracks))


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=VideoAnalysisSummary,
    summary="Cancel a running analysis job",
)
def cancel_job(job_id: str, jobs: JobRegistryDep) -> VideoAnalysisSummary:
    """Request cancellation. Honoured between frames."""
    job = jobs.get(job_id)
    job.cancel()
    return job.summary()
