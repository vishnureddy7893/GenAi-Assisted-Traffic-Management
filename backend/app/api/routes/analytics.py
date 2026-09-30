"""Traffic analytics API.

Phase 4 scope: **measurements derived from perception**. Counts, positions,
directions, displacements, image-space speeds, flow over time, and queue
evidence.

What no endpoint here returns: congestion level, travel time, ETA, route,
recommendation or emergency decision. Those are interpretations of measurements,
and they belong to later phases.

Every response carries provenance and a quality assessment, and any metric that
could not be computed says so with a reason rather than arriving as zero.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter
from pydantic import Field

from app.analytics.engine import (
    ANALYTICS_SCHEMA_VERSION,
    METHOD,
    METHOD_VERSION,
    AnalyticsContext,
    ObservedFrame,
    ObservedObject,
    TrafficAnalyticsEngine,
)
from app.analytics.schemas import TrafficAnalytics
from app.api.deps import (
    AnalyticsStoreDep,
    ImageAnalysisStoreDep,
    JobRegistryDep,
    MediaStoreDep,
    ProviderDep,
    SettingsDep,
)
from app.api.schemas import ApiModel, ErrorResponse
from app.core.errors import ResourceNotFoundError, ValidationFailedError
from app.core.logging import get_logger
from app.domain.site import observation_area_id_for_media
from app.ingestion.media import MediaKind
from app.ingestion.playback import epoch_for
from app.vision.analysis import analyse_image
from app.vision.overlay import encode_png, render_overlay
from app.vision.quality import DetectionQuality

logger = get_logger(__name__)
router = APIRouter(prefix="/analytics", tags=["analytics"])

_ERRORS: dict[int | str, dict[str, object]] = {
    404: {"model": ErrorResponse, "description": "Asset or job not found"},
    422: {"model": ErrorResponse, "description": "Wrong media kind or job not finished"},
}


class AnalyticsConfigResponse(ApiModel):
    """Thresholds and method identity behind every measurement.

    Exposed so a reported number can be reproduced and argued with: a queue
    estimate means nothing without knowing what counted as "stopped".
    """

    method: str
    method_version: str
    schema_version: str
    stationary_speed_frac: float
    min_stationary_frames: int
    queue_proximity_factor: float
    flow_window_s: float
    min_track_frames: int
    calibration_configured: bool
    road_geometry_configured: bool


class GeometryOverrides(ApiModel):
    """Optional surveyed road geometry for one request.

    Supplied per-request in Phase 4 because no network mapping exists yet.
    Without these, traffic density is reported unavailable rather than guessed.
    """

    lane_count: Annotated[int, Field(ge=1, le=12)] | None = None
    segment_length_m: Annotated[float, Field(gt=0, le=10_000)] | None = None


@router.get("/config", response_model=AnalyticsConfigResponse, summary="Analytics configuration")
def analytics_config(settings: SettingsDep) -> AnalyticsConfigResponse:
    """Return the thresholds and method version behind every measurement."""
    return AnalyticsConfigResponse(
        method=METHOD,
        method_version=METHOD_VERSION,
        schema_version=ANALYTICS_SCHEMA_VERSION,
        stationary_speed_frac=settings.analytics_stationary_speed_frac,
        min_stationary_frames=settings.analytics_min_stationary_frames,
        queue_proximity_factor=settings.analytics_queue_proximity_factor,
        flow_window_s=settings.analytics_flow_window_s,
        min_track_frames=settings.analytics_min_track_frames,
        calibration_configured=False,
        road_geometry_configured=False,
    )


# ---------------------------------------------------------------------------
# Image analytics
# ---------------------------------------------------------------------------
@router.post(
    "/images/{media_id}",
    response_model=TrafficAnalytics,
    responses=_ERRORS,
    summary="Derive traffic measurements from an image",
)
def analyse_image_traffic(
    media_id: str,
    store: MediaStoreDep,
    provider: ProviderDep,
    settings: SettingsDep,
    analyses: ImageAnalysisStoreDep,
    analytics_store: AnalyticsStoreDep,
    geometry: GeometryOverrides | None = None,
) -> TrafficAnalytics:
    """Run detection if needed, then derive snapshot measurements.

    Snapshot mode: counts, positions and spatial distribution are available;
    speed, direction, flow and queue estimation are reported unavailable,
    because a single frame contains no time over which anything can move.
    """
    asset = store.get(media_id)
    if asset.kind is not MediaKind.IMAGE:
        msg = f"Media {media_id!r} is a {asset.kind.value}, not an image."
        raise ValidationFailedError(msg, media_id=media_id)

    path = store.path_for(media_id)
    detector = provider.get()

    analysis, frame, assessed = analyse_image(
        media_id=media_id,
        path=path,
        detector=detector,
        settings=settings,
        epoch=epoch_for(path),
    )
    analyses[media_id] = (analysis, encode_png(render_overlay(frame, assessed)))

    observed = ObservedFrame(
        frame_index=0,
        timestamp=analysis.detections[0].timestamp
        if analysis.detections and analysis.detections[0].timestamp
        else analysis.analysed_at,
        width=analysis.frame_width or frame.shape[1],
        height=analysis.frame_height or frame.shape[0],
        # Confirmed detections only. A measurement that counted a duplicate
        # box, a building detected as a bus, or a box on a traffic island would
        # be reporting traffic that is not there -- and every figure downstream,
        # from vehicle count to congestion band, would inherit it.
        objects=tuple(
            ObservedObject(
                road_user_class=d.road_user_class,
                confidence=d.confidence,
                x1=d.bbox.x1,
                y1=d.bbox.y1,
                x2=d.bbox.x2,
                y2=d.bbox.y2,
                track_id=None,
            )
            for d in analysis.detections
            if d.quality is DetectionQuality.CONFIRMED
        ),
    )

    result = TrafficAnalyticsEngine(settings).analyse(
        [observed],
        AnalyticsContext(
            media_id=media_id,
            source_id=analysis.source_id,
            observation_area_id=observation_area_id_for_media(media_id),
            model=analysis.inference.model,
            tracker=None,
            calibration=None,
            lane_count=geometry.lane_count if geometry else None,
            segment_length_m=geometry.segment_length_m if geometry else None,
            settings=settings,
        ),
    )
    analytics_store[media_id] = result
    logger.info(
        "image analytics computed",
        extra={
            "media_id": media_id,
            "objects": result.detection_tally.total,
            "detections_reported": analysis.detection_tally.total,
            "detections_excluded": analysis.excluded_count,
        },
    )
    return result


@router.get(
    "/images/{media_id}",
    response_model=TrafficAnalytics,
    responses=_ERRORS,
    summary="Most recent image analytics",
)
def get_image_analytics(media_id: str, analytics_store: AnalyticsStoreDep) -> TrafficAnalytics:
    """Return cached analytics, or 404 if none has been computed."""
    result = analytics_store.get(media_id)
    if result is None:
        msg = f"No traffic analytics have been computed for image {media_id!r}."
        raise ResourceNotFoundError(msg, media_id=media_id)
    return result


# ---------------------------------------------------------------------------
# Video analytics
# ---------------------------------------------------------------------------
@router.post(
    "/jobs/{job_id}",
    response_model=TrafficAnalytics,
    responses=_ERRORS,
    summary="Derive traffic measurements from a completed video job",
)
def analyse_job_traffic(
    job_id: str,
    jobs: JobRegistryDep,
    store: MediaStoreDep,
    settings: SettingsDep,
    analytics_store: AnalyticsStoreDep,
    geometry: GeometryOverrides | None = None,
) -> TrafficAnalytics:
    """Derive sequence measurements from an analysed video.

    Reads the job's stored per-frame detections and track ids rather than
    re-running inference: perception has already happened, and repeating it
    would be both slow and a second chance to disagree with itself.
    """
    job = jobs.get(job_id)
    summary = job.summary()

    if summary.status.value not in ("completed", "cancelled"):
        msg = (
            f"Job {job_id!r} is {summary.status.value}; analytics require a finished "
            "analysis. Poll the job until it completes."
        )
        raise ValidationFailedError(msg, job_id=job_id, status=summary.status.value)

    frames = job.frames(0, settings.video_max_frames)
    if not frames:
        msg = f"Job {job_id!r} produced no analysed frames, so no measurements exist."
        raise ValidationFailedError(msg, job_id=job_id)

    if summary.inference is None:
        msg = (
            f"Job {job_id!r} carries no inference metadata; refusing to report "
            "measurements without provenance."
        )
        raise ValidationFailedError(msg, job_id=job_id)

    # Frame dimensions come from the measured video container, never from the
    # model's input size: normalising pixel motion by the wrong frame size would
    # scale every image-space speed by a constant factor.
    asset = store.get(job.media_id)
    if asset.video is None:
        msg = f"Media {job.media_id!r} carries no video metadata to normalise measurements against."
        raise ValidationFailedError(msg, job_id=job_id)
    frame_width, frame_height = asset.video.width, asset.video.height

    observed = [
        ObservedFrame(
            frame_index=f.frame_index,
            timestamp=f.timestamp,
            media_time_s=f.media_time_s,
            width=frame_width,
            height=frame_height,
            objects=tuple(
                ObservedObject(
                    road_user_class=d.road_user_class,
                    confidence=d.confidence,
                    x1=d.bbox.x1,
                    y1=d.bbox.y1,
                    x2=d.bbox.x2,
                    y2=d.bbox.y2,
                    track_id=d.track_id,
                )
                for d in f.detections
                if d.quality is DetectionQuality.CONFIRMED
            ),
        )
        for f in frames
    ]

    result = TrafficAnalyticsEngine(settings).analyse(
        observed,
        AnalyticsContext(
            media_id=job.media_id,
            source_id=f"video:{job.media_id}",
            observation_area_id=observation_area_id_for_media(job.media_id),
            model=summary.inference.model,
            tracker=summary.inference.tracker,
            calibration=None,
            lane_count=geometry.lane_count if geometry else None,
            segment_length_m=geometry.segment_length_m if geometry else None,
            settings=settings,
        ),
    )
    analytics_store[f"job:{job_id}"] = result
    logger.info(
        "video analytics computed",
        extra={
            "job_id": job_id,
            "frames": len(observed),
            "tracks": result.tracked_object_tally.total,
        },
    )
    return result


@router.get(
    "/jobs/{job_id}",
    response_model=TrafficAnalytics,
    responses=_ERRORS,
    summary="Most recent analytics for a video job",
)
def get_job_analytics(job_id: str, analytics_store: AnalyticsStoreDep) -> TrafficAnalytics:
    """Return cached analytics for a job, or 404 if none has been computed."""
    result = analytics_store.get(f"job:{job_id}")
    if result is None:
        msg = f"No traffic analytics have been computed for job {job_id!r}."
        raise ResourceNotFoundError(msg, job_id=job_id)
    return result
