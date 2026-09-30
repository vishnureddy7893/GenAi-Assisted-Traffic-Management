"""Traffic-state extraction: tracks to a JunctionObservation. Implementation is Phase 4.

This is the boundary where pixels become traffic engineering, and it is the
stage most often skipped in student projects -- which is why their vehicle counts
never map onto a junction's actual approaches.

Two inputs beyond the tracks are required, and neither is optional:

**Zone configuration.** A track has no idea which approach it belongs to. Turning
tracks into per-approach counts and queues needs counting lines and queue
polygons drawn in image space and bound to approach ids. Without that binding,
the output is "24 vehicles somewhere in frame", which cannot inform a decision
about which approach to prioritise.

**Camera calibration.** Speed, density and queue length in metres all require an
image-to-ground transform. Where it is absent, this stage must emit ``None`` for
those fields and raise
:attr:`~app.domain.enums.DataQualityFlag.UNCALIBRATED` -- a rule the
:class:`~app.domain.observation.JunctionObservation` validator enforces, so the
honest behaviour cannot be forgotten.

Counts survive without calibration; metric quantities do not. That asymmetry is
what makes an uncalibrated camera partially useful rather than useless.
"""

from __future__ import annotations

import abc
import datetime as dt
from typing import Annotated

from pydantic import Field, model_validator

from app.domain.base import DomainModel
from app.domain.observation import CameraCalibration, JunctionObservation
from app.vision.detector import DetectorInfo
from app.vision.tracker import Track, TrackerInfo, TrackerStats


class ImagePoint(DomainModel):
    """A point in image pixel coordinates."""

    x: float
    y: float


class CountingLine(DomainModel):
    """A line in image space; a track crossing it is counted.

    Directional: a crossing is only counted when it happens in the configured
    direction. Without that, a vehicle oscillating across the line in stop-start
    traffic is counted repeatedly, which inflates flow exactly when the junction
    is most congested and the number matters most.
    """

    line_id: Annotated[str, Field(min_length=1, max_length=64)]
    approach_id: Annotated[str, Field(min_length=1, max_length=64)]
    start: ImagePoint
    end: ImagePoint
    count_direction: Annotated[str, Field(pattern="^(inbound|outbound|both)$")] = "inbound"
    min_track_confirmations: Annotated[int, Field(ge=1)] = 3
    """Confirmed track observations required before a crossing counts. Filters
    detector flicker, at the cost of missing very fast-moving small objects."""


class QueueZone(DomainModel):
    """A polygon in image space covering an approach's queuing area.

    Queue length is measured by finding the furthest stationary vehicle from the
    stop line, which needs both the zone and the stop line. Reporting the count
    of stationary vehicles inside the polygon instead is a different quantity,
    and conflating the two makes a two-lane queue look twice as long as a
    single-lane one of the same physical extent.
    """

    zone_id: Annotated[str, Field(min_length=1, max_length=64)]
    approach_id: Annotated[str, Field(min_length=1, max_length=64)]
    polygon: tuple[ImagePoint, ...]
    stop_line: tuple[ImagePoint, ImagePoint] | None = None
    stationary_speed_threshold_mps: Annotated[float, Field(ge=0)] = 0.5
    """Speed below which a vehicle counts as queued. Site-specific: this default
    is a common convention and must be validated against the footage, not
    trusted."""

    @model_validator(mode="after")
    def _polygon_is_closed_area(self) -> QueueZone:
        if len(self.polygon) < 3:
            msg = "a queue zone polygon needs at least three points"
            raise ValueError(msg)
        return self


class ZoneConfiguration(DomainModel):
    """All image-space geometry binding one camera to one junction's approaches.

    Per camera, because it is expressed in that camera's pixel coordinates and is
    invalidated the moment the camera moves -- which is why
    :attr:`~app.domain.enums.DataQualityFlag.CAMERA_MOTION` exists and must
    invalidate this configuration rather than being logged and ignored.
    """

    config_id: Annotated[str, Field(min_length=1, max_length=64)]
    source_id: Annotated[str, Field(min_length=1, max_length=64)]
    junction_id: Annotated[str, Field(min_length=1, max_length=64)]
    frame_width: Annotated[int, Field(gt=0)]
    frame_height: Annotated[int, Field(gt=0)]
    counting_lines: tuple[CountingLine, ...] = ()
    queue_zones: tuple[QueueZone, ...] = ()
    calibration: CameraCalibration | None = None
    configured_at: dt.datetime | None = None

    @model_validator(mode="after")
    def _unique_ids(self) -> ZoneConfiguration:
        line_ids = [item.line_id for item in self.counting_lines]
        if len(set(line_ids)) != len(line_ids):
            msg = "duplicate line_id in zone configuration"
            raise ValueError(msg)
        zone_ids = [item.zone_id for item in self.queue_zones]
        if len(set(zone_ids)) != len(zone_ids):
            msg = "duplicate zone_id in zone configuration"
            raise ValueError(msg)
        return self

    @property
    def covered_approach_ids(self) -> frozenset[str]:
        """Approaches this camera can report on.

        An approach absent here yields *no* observation, not an empty one. The
        difference is the difference between "we did not look" and "there was
        nothing there", and only one of those justifies routing traffic into it.
        """
        return frozenset(
            [line.approach_id for line in self.counting_lines]
            + [zone.approach_id for zone in self.queue_zones]
        )


class ExtractionContext(DomainModel):
    """Everything the extractor needs besides the tracks themselves."""

    junction_id: Annotated[str, Field(min_length=1, max_length=64)]
    source_id: Annotated[str, Field(min_length=1, max_length=64)]
    zone_config: ZoneConfiguration
    analysis_window_s: Annotated[float, Field(gt=0)]
    """Aggregation interval. A raw count is not a flow until the window is known,
    and the window is a deliberate parameter: too short and the estimate is
    dominated by signal phase, too long and it lags a developing incident."""

    detector_info: DetectorInfo | None = None
    tracker_info: TrackerInfo | None = None
    tracker_stats: TrackerStats | None = None
    schema_version: Annotated[str, Field(min_length=1, max_length=20)]


class TrafficStateExtractor(abc.ABC):
    """Converts tracks into a :class:`JunctionObservation`.

    The last stage of the CV pipeline and the first stage of traffic engineering.
    Everything downstream sees only its output, which is what allows a dataset
    source to bypass detection and tracking entirely and still feed an identical
    decision pipeline.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Identifier of this extractor, recorded in provenance."""

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Version of the extraction logic, recorded in provenance."""

    @abc.abstractmethod
    def extract(
        self,
        tracks: tuple[Track, ...],
        *,
        context: ExtractionContext,
        captured_at: dt.datetime,
        ingested_at: dt.datetime,
    ) -> JunctionObservation:
        """Produce one observation from the current tracks.

        Implementations must:

        1. attribute tracks to approaches using the zone configuration, and omit
           approaches the camera does not cover rather than reporting zero;
        2. emit ``None`` for every metric quantity when calibration is absent,
           and set :attr:`~app.domain.enums.DataQualityFlag.UNCALIBRATED`;
        3. attach provenance naming the detector, tracker and extractor versions
           to every measurement;
        4. propagate tracker instability into the observation's quality flags.
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear accumulated windowed state, e.g. after a seek or reconnect."""
