"""Perception orchestration: raw input -> detection -> tracking.

Where the stages meet, and where they stop. This module runs the detector over
frames and feeds the results to the tracker. It does not count traffic, estimate
queues, classify congestion or recommend anything: those belong to the traffic
state extractor and the phases after it.

The one number this module does produce -- a per-class detection count -- is a
tally of boxes in a frame, not a traffic volume. A vehicle sitting still for
sixty frames contributes sixty detections and one track, which is precisely why
counting traffic requires the tracker's identities and a counting line, and is
therefore a later phase's job rather than this one's.

Video is processed **incrementally**. Frames are pulled one at a time from the
recorded source and released, so memory is bounded by one frame regardless of
clip length.
"""

from __future__ import annotations

import datetime as dt
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from app.core.config import Settings
from app.core.errors import DataSourceError, ValidationFailedError
from app.core.logging import get_logger
from app.domain.base import DomainModel
from app.domain.enums import VEHICLE_CLASSES, VehicleClass
from app.vision.bytetrack import ByteTracker
from app.vision.detector import Detection, DetectionResult, VehicleDetector
from app.vision.device import ResolvedDevice
from app.vision.models import ModelIdentity
from app.vision.quality import (
    AssessedDetection,
    DetectionQuality,
    QualityFlag,
    QualityPolicy,
    assess,
    policy_from,
)
from app.vision.registry import create_detector
from app.vision.tracker import Track

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result schemas
# ---------------------------------------------------------------------------
class BoundingBoxOut(DomainModel):
    """Bounding box in source-image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float

    @classmethod
    def from_detection(cls, detection: Detection) -> BoundingBoxOut:
        """Project a detection's box onto the wire schema."""
        box = detection.bbox
        return cls(
            x1=box.x1,
            y1=box.y1,
            x2=box.x2,
            y2=box.y2,
            width=box.width,
            height=box.height,
        )


class DetectionOut(DomainModel):
    """One detected road user, with full provenance."""

    road_user_class: VehicleClass
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    bbox: BoundingBoxOut
    track_id: int | None = None
    """Present only for video, where tracking establishes identity across frames."""

    frame_index: Annotated[int, Field(ge=0)] | None = None
    timestamp: dt.datetime | None = None
    source_id: Annotated[str, Field(max_length=120)] | None = None
    """The data source that produced the frame, so a detection is always
    traceable back to the exact input it came from."""

    quality: DetectionQuality = DetectionQuality.CONFIRMED
    """Whether this detection may be counted as a road user. Every detection is
    reported whatever its verdict -- a filtered-away box is invisible to an
    operator trying to work out why a count is wrong."""

    quality_flags: tuple[QualityFlag, ...] = ()
    quality_reason: Annotated[str, Field(max_length=600)] | None = None
    """Why it is not confirmed, in checkable terms. Required reading whenever
    the verdict is anything other than confirmed."""

    area_fraction: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    aspect_ratio: Annotated[float, Field(ge=0.0)] | None = None
    touches_frame_edge: bool = False
    """Geometry the verdict rests on, reported so it can be checked rather than
    trusted."""

    duplicate_of_index: Annotated[int, Field(ge=0)] | None = None


class InferenceMetadata(DomainModel):
    """What produced a set of detections, and on what."""

    backend: Annotated[str, Field(min_length=1, max_length=40)]
    model: ModelIdentity
    device: Annotated[str, Field(min_length=1, max_length=120)]
    device_kind: Annotated[str, Field(min_length=1, max_length=20)]
    device_reason: Annotated[str, Field(max_length=300)]
    image_size: Annotated[int, Field(gt=0)]
    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)]
    iou_threshold: Annotated[float, Field(ge=0.0, le=1.0)]
    tracker: Annotated[str, Field(max_length=80)] | None = None


class FrameResult(DomainModel):
    """Detection and tracking output for one analysed frame."""

    frame_index: Annotated[int, Field(ge=0)]
    media_time_s: float | None = None
    timestamp: dt.datetime
    inference_ms: float | None = None
    detections: tuple[DetectionOut, ...] = ()
    active_track_ids: tuple[int, ...] = ()

    @property
    def detection_count(self) -> int:
        """Boxes in this frame. Not a traffic volume."""
        return len(self.detections)


class ClassTally(DomainModel):
    """Per-class counts.

    A tally of detections or tracks -- explicitly not a traffic measurement.
    Turning these into flow requires counting lines and a time window, which is
    the traffic state extractor's job.
    """

    by_class: dict[VehicleClass, int] = Field(default_factory=dict)
    total: Annotated[int, Field(ge=0)] = 0
    vehicles: Annotated[int, Field(ge=0)] = 0
    """Subtotal over classes that are actually vehicles, so pedestrians and
    cyclists are never silently summed into a vehicle figure."""

    @classmethod
    def from_classes(cls, classes: list[VehicleClass]) -> ClassTally:
        """Tally a list of road-user classes."""
        counts = Counter(classes)
        return cls(
            by_class=dict(counts),
            total=len(classes),
            vehicles=sum(n for cls_, n in counts.items() if cls_ in VEHICLE_CLASSES),
        )


class ImageAnalysis(DomainModel):
    """Result of one detection pass over a still image."""

    analysis_id: Annotated[str, Field(min_length=1, max_length=64)]
    media_id: Annotated[str, Field(min_length=1, max_length=64)]
    source_id: Annotated[str, Field(min_length=1, max_length=120)]
    analysed_at: dt.datetime
    inference: InferenceMetadata
    detections: tuple[DetectionOut, ...] = ()

    detection_tally: ClassTally = ClassTally()
    """Every box the detector reported, whatever its quality. Useful for tuning
    and misleading as a traffic figure -- it includes duplicates and rejects."""

    confirmed_tally: ClassTally = ClassTally()
    """The subset a measurement may count. This is the tally the analytics
    layer consumes; the difference between the two is the quality layer's
    effect, visible rather than silent."""

    quality_policy: QualityPolicy | None = None
    """The rules in force when this was assessed. A count produced under one
    policy and read against another is unreproducible."""

    frame_width: int | None = None
    frame_height: int | None = None
    overlay_available: bool = False

    @property
    def excluded_count(self) -> int:
        """How many reported detections a measurement may not count."""
        return self.detection_tally.total - self.confirmed_tally.total


class JobStatus(StrEnum):
    """Lifecycle of a video analysis job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VideoAnalysisSummary(DomainModel):
    """Status and results of a video analysis job."""

    job_id: Annotated[str, Field(min_length=1, max_length=64)]
    media_id: Annotated[str, Field(min_length=1, max_length=64)]
    status: JobStatus
    created_at: dt.datetime
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None

    frames_analysed: Annotated[int, Field(ge=0)] = 0
    frames_total_estimate: Annotated[int, Field(ge=0)] | None = None
    frame_stride: Annotated[int, Field(ge=1)] = 1
    progress: Annotated[float, Field(ge=0.0, le=1.0)] | None = None

    detection_tally: ClassTally = ClassTally()
    """Every detection across every analysed frame. One vehicle appears once per
    frame it is visible in, so this is not a count of vehicles."""

    confirmed_detection_tally: ClassTally = ClassTally()
    """The subset that passed the quality checks, across every frame."""

    track_tally: ClassTally = ClassTally()
    """Distinct tracked objects. Closer to a vehicle count, but still not one:
    it does not account for objects entering before or leaving after the clip."""

    unique_track_count: Annotated[int, Field(ge=0)] = 0
    suspected_id_switches: Annotated[int, Field(ge=0)] = 0
    inference: InferenceMetadata | None = None
    error: Annotated[str, Field(max_length=1000)] | None = None
    """The real failure message. Never replaced by empty results."""


class TrackSummary(DomainModel):
    """One tracked object over its visible trajectory."""

    track_id: int
    road_user_class: VehicleClass
    first_frame_index: Annotated[int, Field(ge=0)]
    last_frame_index: Annotated[int, Field(ge=0)]
    first_seen_at: dt.datetime
    last_seen_at: dt.datetime
    frames_tracked: Annotated[int, Field(ge=1)]
    is_confirmed: bool
    id_switch_suspected: bool = False
    mean_confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None


# ---------------------------------------------------------------------------
# Detector provider
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DetectorOverrides:
    """Per-request detector settings, for tuning against a specific frame.

    These exist because the right threshold is a property of the *camera*, not
    of the software: a wide junction view full of distant vehicles wants a lower
    confidence and a larger input size than a close approach view. Finding that
    out requires trying it, and a setting that can only be changed by editing an
    environment file and restarting cannot be tried at all.

    Every field is optional and falls back to configuration, so the deployed
    default remains the answer to "what produced this?" unless a caller
    deliberately says otherwise -- and when one does, the override travels into
    the reported provenance rather than sitting silently beside it.
    """

    confidence_threshold: float | None = None
    iou_threshold: float | None = None
    image_size: int | None = None
    max_detections: int | None = None

    @property
    def is_empty(self) -> bool:
        """Whether this asks for nothing, in which case configuration stands."""
        return all(
            value is None
            for value in (
                self.confidence_threshold,
                self.iou_threshold,
                self.image_size,
                self.max_detections,
            )
        )

    def apply_to(self, settings: Settings) -> Settings:
        """Return settings with these overrides applied.

        A copy, never a mutation: the application's settings object is shared by
        every request, and one caller's experiment must not become another
        caller's configuration.
        """
        update: dict[str, object] = {}
        if self.confidence_threshold is not None:
            update["detector_confidence_threshold"] = self.confidence_threshold
        if self.iou_threshold is not None:
            update["detector_iou_threshold"] = self.iou_threshold
        if self.image_size is not None:
            update["detector_image_size"] = self.image_size
        if self.max_detections is not None:
            update["detector_max_detections"] = self.max_detections
        return settings.model_copy(update=update) if update else settings


class DetectorProvider:
    """Lazily constructs and caches the configured detector.

    Loading YOLO weights takes on the order of a second, so a per-request load
    would dominate inference time. The detector is therefore built once and
    reused, guarded by a lock because a video job runs on a worker thread while
    an image request may arrive on another.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._detector: VehicleDetector | None = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    @property
    def load_error(self) -> str | None:
        """The last load failure, if any. Surfaced rather than swallowed."""
        return self._load_error

    @property
    def is_loaded(self) -> bool:
        """Whether a detector is currently loaded."""
        return self._detector is not None

    def get(self) -> VehicleDetector:
        """Return the loaded detector, loading it on first use."""
        with self._lock:
            if self._detector is not None:
                return self._detector
            try:
                detector = create_detector(self._settings, self._settings.models_dir)
                detector.load()
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                raise
            self._detector = detector
            self._load_error = None
            return detector

    @contextmanager
    def with_overrides(
        self, overrides: DetectorOverrides | None
    ) -> Iterator[tuple[VehicleDetector, Settings]]:
        """Yield a detector honouring ``overrides``, and the settings it used.

        With no overrides this is the cached detector and the application's own
        settings -- the common path pays nothing. With overrides it is a
        throwaway detector built for this request and closed afterwards.

        Transient rather than cached on purpose. Caching one model per distinct
        threshold combination would let a caller sweeping a parameter range pin
        a copy of the weights in memory for each step, and the weights are the
        expensive part. The cost is a reload per tuning request, which is the
        right trade for an operation a human performs a handful of times.

        The settings are yielded alongside because provenance is assembled from
        them: a result produced at a caller's threshold must not be reported
        under the deployment's.
        """
        if overrides is None or overrides.is_empty:
            yield self.get(), self._settings
            return

        effective = overrides.apply_to(self._settings)
        detector = create_detector(effective, effective.models_dir)
        detector.load()
        try:
            yield detector, effective
        finally:
            detector.close()

    def close(self) -> None:
        """Release the detector."""
        with self._lock:
            if self._detector is not None:
                self._detector.close()
            self._detector = None


def _inference_metadata(
    detector: VehicleDetector,
    settings: Settings,
    *,
    tracker: str | None = None,
) -> InferenceMetadata:
    """Assemble provenance from a loaded detector."""
    identity: ModelIdentity | None = getattr(detector, "identity", None)
    device: ResolvedDevice | None = getattr(detector, "device", None)
    info = detector.info

    if identity is None:
        msg = "Detector reports no model identity; refusing to report results without provenance."
        raise DataSourceError(msg)
    if device is None:
        msg = "Detector reports no execution device; refusing to report results without provenance."
        raise DataSourceError(msg)

    return InferenceMetadata(
        backend=identity.backend,
        model=identity,
        device=device.name,
        device_kind=device.kind.value,
        device_reason=device.reason,
        image_size=info.input_width or settings.detector_image_size,
        confidence_threshold=settings.detector_confidence_threshold,
        iou_threshold=settings.detector_iou_threshold,
        tracker=tracker,
    )


def _to_out(
    assessed: AssessedDetection,
    *,
    frame_index: int | None,
    timestamp: dt.datetime | None,
    source_id: str,
    track_id: int | None = None,
) -> DetectionOut:
    """Project an assessed detection onto the wire schema, with provenance.

    Takes the assessment rather than the bare detection so that a box can never
    reach the wire without its verdict attached: an unlabelled detection is one
    a consumer will assume is countable.
    """
    detection = assessed.detection
    return DetectionOut(
        road_user_class=detection.vehicle_class,
        confidence=detection.confidence,
        bbox=BoundingBoxOut.from_detection(detection),
        track_id=track_id,
        frame_index=frame_index,
        timestamp=timestamp,
        source_id=source_id,
        quality=assessed.quality,
        quality_flags=assessed.flags,
        quality_reason=assessed.reason,
        area_fraction=assessed.area_fraction,
        aspect_ratio=assessed.aspect_ratio,
        touches_frame_edge=assessed.touches_frame_edge,
        duplicate_of_index=assessed.duplicate_of_index,
    )


def _assign_track_ids(
    detections: tuple[Detection, ...],
    tracks: tuple[Track, ...],
) -> list[int | None]:
    """Attach track ids to the detections they were matched from.

    Matching by exact box identity: the tracker stores the detection box it
    absorbed, so equality is reliable here and avoids re-running an association
    that has already been performed.
    """
    lookup: dict[tuple[float, float, float, float], int] = {}
    for track in tracks:
        if not track.history:
            continue
        box = track.current.bbox
        lookup[(box.x1, box.y1, box.x2, box.y2)] = track.track_id

    return [
        lookup.get((d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2)) for d in detections
    ]


# ---------------------------------------------------------------------------
# Image analysis
# ---------------------------------------------------------------------------
def analyse_image(
    *,
    media_id: str,
    path: Path,
    detector: VehicleDetector,
    settings: Settings,
    epoch: dt.datetime,
) -> tuple[ImageAnalysis, Any, tuple[AssessedDetection, ...]]:
    """Run one detection pass over a still image, and assess what it produced.

    Returns the analysis record, the decoded RGB frame and the *assessed*
    detections, so the caller can render an overlay without decoding the file
    twice -- and so the overlay can draw a rejected box differently from a
    confirmed one instead of making them look alike.
    """
    from app.ingestion.image_source import RecordedImageDataSource

    source = RecordedImageDataSource(
        source_id=f"image:{media_id}", path=path, epoch=epoch
    )
    source.open()
    try:
        sample = source.read()
        if sample is None:
            msg = f"Image {path.name} yielded no frame."
            raise DataSourceError(msg, media_id=media_id)

        result: DetectionResult = detector.detect(
            sample.payload,
            source_id=source.source_id,
            sequence=sample.sequence,
            captured_at=sample.captured_at,
        )
        frame = sample.payload
    finally:
        source.close()

    policy = policy_from(settings)
    assessed = assess(
        result.detections,
        frame_width=result.frame_width or 0,
        frame_height=result.frame_height or 0,
        policy=policy,
    )

    detections = tuple(
        _to_out(
            item,
            frame_index=0,
            timestamp=result.captured_at,
            source_id=result.source_id,
        )
        for item in assessed
    )

    analysis = ImageAnalysis(
        analysis_id=uuid.uuid4().hex,
        media_id=media_id,
        source_id=result.source_id,
        analysed_at=dt.datetime.now(tz=dt.UTC),
        inference=_inference_metadata(detector, settings),
        detections=detections,
        detection_tally=ClassTally.from_classes([d.road_user_class for d in detections]),
        confirmed_tally=ClassTally.from_classes(
            [
                d.road_user_class
                for d in detections
                if d.quality is DetectionQuality.CONFIRMED
            ]
        ),
        quality_policy=policy,
        frame_width=result.frame_width,
        frame_height=result.frame_height,
        overlay_available=True,
    )
    return analysis, frame, assessed


# ---------------------------------------------------------------------------
# Video analysis
# ---------------------------------------------------------------------------
class VideoAnalysisJob:
    """A single video analysis run.

    Holds its own tracker, so two concurrent jobs cannot share track ids.
    """

    def __init__(
        self,
        *,
        job_id: str,
        media_id: str,
        path: Path,
        provider: DetectorProvider,
        settings: Settings,
        epoch: dt.datetime,
        frame_stride: int | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.job_id = job_id
        self.media_id = media_id
        self._path = path
        self._provider = provider
        self._settings = settings
        self._epoch = epoch
        self._stride = frame_stride or settings.video_frame_stride
        self._max_frames = max_frames or settings.video_max_frames

        self._status = JobStatus.PENDING
        self._created_at = dt.datetime.now(tz=dt.UTC)
        self._started_at: dt.datetime | None = None
        self._finished_at: dt.datetime | None = None
        self._error: str | None = None
        self._cancelled = threading.Event()

        self._frames: list[FrameResult] = []
        self._track_first: dict[int, tuple[int, dt.datetime, VehicleClass]] = {}
        self._track_last: dict[int, tuple[int, dt.datetime]] = {}
        self._track_hits: Counter[int] = Counter()
        self._track_conf: dict[int, list[float]] = {}
        self._track_confirmed: dict[int, bool] = {}
        self._track_switch: dict[int, bool] = {}
        self._detection_classes: list[VehicleClass] = []
        self._confirmed_classes: list[VehicleClass] = []
        # Resolved once per job rather than per frame: the policy is fixed
        # for the run, and re-reading it mid-job would let a configuration
        # change split one video's frames across two rule sets.
        self._quality_policy = policy_from(settings)
        self._inference: InferenceMetadata | None = None
        self._frames_total: int | None = None
        self._lock = threading.Lock()

    # --- state ------------------------------------------------------------
    @property
    def status(self) -> JobStatus:
        """Current lifecycle state."""
        return self._status

    def cancel(self) -> None:
        """Request cancellation; honoured between frames."""
        self._cancelled.set()

    def frames(self, offset: int = 0, limit: int = 200) -> tuple[FrameResult, ...]:
        """A slice of per-frame results."""
        with self._lock:
            return tuple(self._frames[offset : offset + limit])

    @property
    def frame_count(self) -> int:
        """Frames analysed so far."""
        with self._lock:
            return len(self._frames)

    def track_summaries(self) -> tuple[TrackSummary, ...]:
        """One record per tracked object, ordered by first appearance."""
        with self._lock:
            summaries: list[TrackSummary] = []
            for track_id, (first_idx, first_at, cls) in self._track_first.items():
                last_idx, last_at = self._track_last.get(track_id, (first_idx, first_at))
                confidences = self._track_conf.get(track_id, [])
                summaries.append(
                    TrackSummary(
                        track_id=track_id,
                        road_user_class=cls,
                        first_frame_index=first_idx,
                        last_frame_index=last_idx,
                        first_seen_at=first_at,
                        last_seen_at=last_at,
                        frames_tracked=self._track_hits[track_id],
                        is_confirmed=self._track_confirmed.get(track_id, False),
                        id_switch_suspected=self._track_switch.get(track_id, False),
                        mean_confidence=(
                            sum(confidences) / len(confidences) if confidences else None
                        ),
                    )
                )
        return tuple(sorted(summaries, key=lambda s: (s.first_frame_index, s.track_id)))

    def summary(self) -> VideoAnalysisSummary:
        """Status plus aggregate results."""
        with self._lock:
            frames_done = len(self._frames)
            classes = list(self._detection_classes)
            confirmed_classes = list(self._confirmed_classes)
            track_classes = [cls for _, _, cls in self._track_first.values()]
            switches = sum(1 for flag in self._track_switch.values() if flag)
            total_estimate = self._frames_total

        progress: float | None = None
        if total_estimate:
            planned = max(1, -(-total_estimate // self._stride))
            progress = min(1.0, frames_done / min(planned, self._max_frames))
        if self._status is JobStatus.COMPLETED:
            progress = 1.0

        return VideoAnalysisSummary(
            job_id=self.job_id,
            media_id=self.media_id,
            status=self._status,
            created_at=self._created_at,
            started_at=self._started_at,
            finished_at=self._finished_at,
            frames_analysed=frames_done,
            frames_total_estimate=total_estimate,
            frame_stride=self._stride,
            progress=progress,
            detection_tally=ClassTally.from_classes(classes),
            confirmed_detection_tally=ClassTally.from_classes(confirmed_classes),
            track_tally=ClassTally.from_classes(track_classes),
            unique_track_count=len(track_classes),
            suspected_id_switches=switches,
            inference=self._inference,
            error=self._error,
        )

    # --- execution --------------------------------------------------------
    def run(self) -> None:
        """Execute the job. Safe to call on a worker thread."""
        from app.ingestion.video_source import RecordedVideoDataSource

        self._status = JobStatus.RUNNING
        self._started_at = dt.datetime.now(tz=dt.UTC)
        started_perf = time.perf_counter()

        try:
            detector = self._provider.get()
            tracker = ByteTracker()
            self._inference = _inference_metadata(
                detector, self._settings, tracker=tracker.info.name
            )

            source = RecordedVideoDataSource(
                source_id=f"video:{self.media_id}", path=self._path, epoch=self._epoch
            )
            source.open()
            self._frames_total = source.metadata.frame_count

            try:
                raw_index = 0
                analysed = 0
                while analysed < self._max_frames:
                    if self._cancelled.is_set():
                        self._status = JobStatus.CANCELLED
                        return

                    # One frame in memory at a time. The clip is never
                    # materialised, so memory does not scale with its length.
                    sample = source.read()
                    if sample is None:
                        break

                    if raw_index % self._stride != 0:
                        raw_index += 1
                        continue

                    result = detector.detect(
                        sample.payload,
                        source_id=source.source_id,
                        sequence=sample.sequence,
                        captured_at=sample.captured_at,
                    )
                    tracks = tracker.update(
                        result.detections, timestamp=sample.captured_at
                    )
                    self._record(raw_index, sample, result, tracks)

                    raw_index += 1
                    analysed += 1
            finally:
                source.close()

            self._status = JobStatus.COMPLETED
        except Exception as exc:
            # The real message is retained. An analysis that failed must never
            # present itself as an analysis that found nothing.
            self._error = f"{type(exc).__name__}: {exc}"
            self._status = JobStatus.FAILED
            logger.error(
                "video analysis failed",
                extra={"job_id": self.job_id, "media_id": self.media_id, "error": self._error},
            )
        finally:
            self._finished_at = dt.datetime.now(tz=dt.UTC)
            logger.info(
                "video analysis finished",
                extra={
                    "job_id": self.job_id,
                    "status": self._status.value,
                    "frames": self.frame_count,
                    "elapsed_s": round(time.perf_counter() - started_perf, 2),
                },
            )

    def _record(
        self,
        frame_index: int,
        sample: Any,
        result: DetectionResult,
        tracks: tuple[Track, ...],
    ) -> None:
        """Accumulate one frame's output."""
        track_ids = _assign_track_ids(result.detections, tracks)
        assessed = assess(
            result.detections,
            frame_width=result.frame_width or 0,
            frame_height=result.frame_height or 0,
            policy=self._quality_policy,
        )
        detections = tuple(
            _to_out(
                item,
                frame_index=frame_index,
                timestamp=sample.captured_at,
                source_id=result.source_id,
                track_id=track_id,
            )
            for item, track_id in zip(assessed, track_ids, strict=True)
        )

        with self._lock:
            self._frames.append(
                FrameResult(
                    frame_index=frame_index,
                    media_time_s=sample.media_time_s,
                    timestamp=sample.captured_at,
                    inference_ms=result.inference_ms,
                    detections=detections,
                    active_track_ids=tuple(t.track_id for t in tracks),
                )
            )
            self._detection_classes.extend(d.road_user_class for d in detections)
            self._confirmed_classes.extend(
                d.road_user_class
                for d in detections
                if d.quality is DetectionQuality.CONFIRMED
            )

            for track in tracks:
                if track.track_id not in self._track_first:
                    self._track_first[track.track_id] = (
                        frame_index,
                        sample.captured_at,
                        track.vehicle_class,
                    )
                else:
                    first_idx, first_at, _ = self._track_first[track.track_id]
                    self._track_first[track.track_id] = (
                        first_idx,
                        first_at,
                        track.vehicle_class,
                    )
                self._track_last[track.track_id] = (frame_index, sample.captured_at)
                self._track_hits[track.track_id] += 1
                self._track_confirmed[track.track_id] = track.is_confirmed
                self._track_switch[track.track_id] = track.id_switch_suspected
                if track.history:
                    self._track_conf.setdefault(track.track_id, []).append(
                        track.current.confidence
                    )


class JobRegistry:
    """In-memory registry of video analysis jobs."""

    def __init__(self, max_jobs: int = 32) -> None:
        self._jobs: dict[str, VideoAnalysisJob] = {}
        self._order: list[str] = []
        self._max_jobs = max_jobs
        self._lock = threading.Lock()

    def submit(self, job: VideoAnalysisJob, *, run_sync: bool = False) -> VideoAnalysisJob:
        """Register a job and start it.

        ``run_sync`` executes inline, which is what tests use: a background
        thread would make assertions race the worker.
        """
        with self._lock:
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
            while len(self._order) > self._max_jobs:
                evicted = self._order.pop(0)
                self._jobs.pop(evicted, None)

        if run_sync:
            job.run()
        else:
            thread = threading.Thread(
                target=job.run, name=f"analysis-{job.job_id[:8]}", daemon=True
            )
            thread.start()
        return job

    def get(self, job_id: str) -> VideoAnalysisJob:
        """Return a job, or raise."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            msg = f"No analysis job with id {job_id!r}."
            raise ValidationFailedError(msg, job_id=job_id)
        return job

    def list_for_media(self, media_id: str) -> tuple[VideoAnalysisJob, ...]:
        """Jobs for one media asset, newest first."""
        with self._lock:
            jobs = [j for j in self._jobs.values() if j.media_id == media_id]
        return tuple(reversed(jobs))

    def all(self) -> tuple[VideoAnalysisJob, ...]:
        """Every registered job, newest first."""
        with self._lock:
            return tuple(self._jobs[jid] for jid in reversed(self._order) if jid in self._jobs)


def new_job_id() -> str:
    """Generate a job identifier."""
    return uuid.uuid4().hex
