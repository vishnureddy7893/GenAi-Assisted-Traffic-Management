"""Traffic analytics result schemas.

The central type here is :class:`MetricValue`. Every derived quantity travels in
one, and a metric that could not be computed says so **with a reason** rather
than arriving as ``0``.

That distinction is the whole point of this module. A queue length of zero and a
queue length that could not be measured are opposite claims: the first says the
road is clear, the second says we do not know. Returning zero for the second is
how a dashboard ends up reporting free-flowing traffic on a camera that is not
calibrated.

Image space versus the world
----------------------------
Measurements are separated into two families that are never mixed:

* **Image-space** — pixels, frame diagonals, fractions of frame area. Always
  computable from detections alone, and never presented as a physical quantity.
* **Physical** — metres, m/s, km/h, vehicles per km. Computable *only* with a
  camera calibration that maps image points onto the road plane. Without one
  they are reported unavailable, never estimated.

A pixel displacement is not a distance: the same object moving the same real
distance produces wildly different pixel motion depending on how far it is from
the camera. Converting without a homography would be inventing a scale.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Annotated, Generic, TypeVar

from pydantic import Field, model_validator

from app.domain.base import DomainModel
from app.domain.enums import VehicleClass
from app.vision.models import ModelIdentity

T = TypeVar("T")

#: Exact wording required when a physical speed is requested without calibration.
NO_CALIBRATION_MESSAGE = "Physical speed unavailable — camera calibration not configured."


class MetricAvailability(StrEnum):
    """Why a metric is or is not present.

    Deliberately granular: "we cannot measure this without a calibrated camera"
    and "this clip was too short" call for completely different operator
    responses, and collapsing them into a single ``null`` loses that.
    """

    AVAILABLE = "available"

    UNAVAILABLE_NO_CALIBRATION = "unavailable_no_calibration"
    """Requires an image-to-ground transform that is not configured."""

    UNAVAILABLE_NO_GEOMETRY = "unavailable_no_geometry"
    """Requires surveyed road geometry -- lane count, segment length, detection
    zone area -- that has not been provided."""

    UNAVAILABLE_INSUFFICIENT_DATA = "unavailable_insufficient_data"
    """Not enough frames or track points to compute it."""

    UNAVAILABLE_NOT_APPLICABLE = "unavailable_not_applicable"
    """Meaningless for this input, e.g. speed from a single still image."""


class MetricValue(DomainModel, Generic[T]):
    """A derived quantity, or an explicit statement that it is unavailable."""

    availability: MetricAvailability
    value: T | None = None
    unit: Annotated[str, Field(max_length=40)] | None = None

    reason: Annotated[str, Field(max_length=300)] | None = None
    """Required when unavailable. An unexplained absence is indistinguishable
    from a bug, and an operator cannot act on it."""

    method: Annotated[str, Field(max_length=120)] | None = None
    """How it was computed, so a number can be argued with rather than trusted."""

    confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None

    is_image_space: bool = False
    """True for pixel/frame-relative quantities. Presentation must never render
    an image-space value as if it were physical."""

    @model_validator(mode="after")
    def _availability_matches_content(self) -> MetricValue[T]:
        if self.availability is MetricAvailability.AVAILABLE:
            if self.value is None:
                msg = "A metric marked AVAILABLE must carry a value."
                raise ValueError(msg)
        else:
            if self.value is not None:
                msg = (
                    "An unavailable metric must not carry a value; a value with an "
                    "unavailable flag is exactly the ambiguity this type exists to "
                    "prevent."
                )
                raise ValueError(msg)
            if not self.reason:
                msg = "An unavailable metric must state why."
                raise ValueError(msg)
        return self

    @property
    def is_available(self) -> bool:
        """Whether this metric has a usable value."""
        return self.availability is MetricAvailability.AVAILABLE


def available(
    value: T,
    *,
    unit: str | None = None,
    method: str | None = None,
    confidence: float | None = None,
    image_space: bool = False,
) -> MetricValue[T]:
    """Build an available metric."""
    return MetricValue[T](
        availability=MetricAvailability.AVAILABLE,
        value=value,
        unit=unit,
        method=method,
        confidence=confidence,
        is_image_space=image_space,
    )


def unavailable(
    availability: MetricAvailability,
    reason: str,
    *,
    unit: str | None = None,
    method: str | None = None,
) -> MetricValue[T]:
    """Build an unavailable metric with a stated reason."""
    return MetricValue[T](
        availability=availability,
        value=None,
        unit=unit,
        reason=reason,
        method=method,
    )


def no_calibration(unit: str, *, method: str | None = None) -> MetricValue[T]:
    """Standard 'requires calibration' metric."""
    return unavailable(
        MetricAvailability.UNAVAILABLE_NO_CALIBRATION,
        NO_CALIBRATION_MESSAGE,
        unit=unit,
        method=method,
    )


# ---------------------------------------------------------------------------
# Provenance and quality
# ---------------------------------------------------------------------------
class AnalyticsProvenance(DomainModel):
    """Where a set of measurements came from, and how it was produced."""

    media_id: Annotated[str, Field(min_length=1, max_length=64)]
    source_id: Annotated[str, Field(min_length=1, max_length=120)]

    frame_index: Annotated[int, Field(ge=0)] | None = None
    """Single frame, for a still image."""

    frame_range: tuple[int, int] | None = None
    """Inclusive first and last analysed frame index, for video."""

    timestamp: dt.datetime | None = None
    time_range: tuple[dt.datetime, dt.datetime] | None = None

    model: ModelIdentity
    tracker: Annotated[str, Field(max_length=80)] | None = None
    """``None`` for a still image: there is nothing to track across."""

    method: Annotated[str, Field(min_length=1, max_length=120)]
    method_version: Annotated[str, Field(min_length=1, max_length=40)]
    analytics_schema_version: Annotated[str, Field(min_length=1, max_length=40)]
    computed_at: dt.datetime

    @model_validator(mode="after")
    def _tz_aware(self) -> AnalyticsProvenance:
        if self.computed_at.tzinfo is None:
            msg = "computed_at must be timezone-aware"
            raise ValueError(msg)
        return self


class QualityGrade(StrEnum):
    """Overall trust in a measurement set."""

    GOOD = "good"
    DEGRADED = "degraded"
    POOR = "poor"


class MeasurementQuality(DomainModel):
    """Conditions affecting how far these measurements can be trusted."""

    grade: QualityGrade
    frames_analysed: Annotated[int, Field(ge=0)]
    mean_detection_confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    low_confidence_fraction: Annotated[float, Field(ge=0.0, le=1.0)] | None = None

    tracks_total: Annotated[int, Field(ge=0)] = 0
    tracks_confirmed: Annotated[int, Field(ge=0)] = 0
    suspected_id_switches: Annotated[int, Field(ge=0)] = 0
    """Each switch corrupts one trajectory, so speed and direction from an
    affected track are unreliable even though the counts remain sound."""

    calibration_available: bool = False
    road_geometry_available: bool = False
    sampling_interval_s: float | None = None
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Object-level results
# ---------------------------------------------------------------------------
class ImagePoint2D(DomainModel):
    """A point in image pixel coordinates, with its normalised form."""

    x: float
    y: float
    x_norm: Annotated[float, Field(ge=0.0, le=1.0)]
    y_norm: Annotated[float, Field(ge=0.0, le=1.0)]


class ObjectPosition(DomainModel):
    """Where one detected object is in the frame."""

    road_user_class: VehicleClass
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    track_id: int | None = None

    centre: ImagePoint2D
    ground_contact: ImagePoint2D
    """Bottom-centre of the box: the best available proxy for where the object
    meets the road. Used for all motion measurement, because a box centroid sits
    at mid-height and drifts as the object's apparent size changes."""

    box_width_px: float
    box_height_px: float
    box_area_fraction: Annotated[float, Field(ge=0.0, le=1.0)]


class DirectionLabel(StrEnum):
    """Coarse image-space heading.

    Image-space, **not compass**. Mapping these to north/south would require
    knowing the camera's orientation and mounting, which nothing here has.
    """

    RIGHT = "right"
    DOWN_RIGHT = "down_right"
    DOWN = "down"
    DOWN_LEFT = "down_left"
    LEFT = "left"
    UP_LEFT = "up_left"
    UP = "up"
    UP_RIGHT = "up_right"


class TrajectoryMetrics(DomainModel):
    """Motion measurements for one tracked object."""

    track_id: int
    road_user_class: VehicleClass
    frames_tracked: Annotated[int, Field(ge=1)]
    first_frame_index: Annotated[int, Field(ge=0)]
    last_frame_index: Annotated[int, Field(ge=0)]

    duration_s: MetricValue[float]
    total_displacement_px: MetricValue[float]
    net_displacement_px: MetricValue[float]
    """Straight-line first-to-last distance. Compared against the path length it
    reveals circling or jitter: a stationary object with detection noise
    accumulates path length while its net displacement stays near zero."""

    mean_step_displacement_px: MetricValue[float]
    image_speed_px_per_s: MetricValue[float]
    image_speed_diagonals_per_s: MetricValue[float]
    """Resolution-independent: fractions of the frame diagonal per second."""

    direction_deg: MetricValue[float]
    """Clockwise from image-right, image-space."""

    direction_label: MetricValue[DirectionLabel]
    is_stationary: bool
    stationary_frames: Annotated[int, Field(ge=0)] = 0
    id_switch_suspected: bool = False

    physical_speed_mps: MetricValue[float]
    physical_speed_kph: MetricValue[float]
    """Both unavailable unless a camera calibration is configured."""


class QueueGroup(DomainModel):
    """A spatial cluster of stationary vehicles.

    Queue *evidence*, not a queue length. Membership requires low movement,
    persistence across frames, and spatial proximity -- never merely being
    present in the frame. The physical extent of the queue in metres remains
    unavailable without calibration, which is why ``length_m`` is a metric with
    an availability rather than a number.
    """

    group_id: int
    member_track_ids: tuple[int, ...]
    member_count: Annotated[int, Field(ge=1)]
    road_user_classes: dict[VehicleClass, int] = Field(default_factory=dict)
    extent_px: MetricValue[float]
    """Longest span across the cluster, in pixels. Image-space."""

    longest_stationary_duration_s: MetricValue[float]
    length_m: MetricValue[float]
    mean_confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None


class FlowBucket(DomainModel):
    """Activity within one time window."""

    window_start_s: float
    window_end_s: float
    arrivals: Annotated[int, Field(ge=0)]
    """Distinct tracks first seen in this window."""

    active: Annotated[int, Field(ge=0)]
    """Distinct tracks present at any point in this window."""


class SpatialDistribution(DomainModel):
    """Where objects sit in the frame. Image-space throughout."""

    centroid: ImagePoint2D | None = None
    spread_x_norm: MetricValue[float]
    spread_y_norm: MetricValue[float]
    grid_rows: Annotated[int, Field(ge=1)]
    grid_cols: Annotated[int, Field(ge=1)]
    grid_counts: tuple[tuple[int, ...], ...]
    """Object counts per grid cell, row-major. A coarse read of where activity
    is concentrated without claiming anything about lanes or approaches."""

    image_occupancy_ratio: MetricValue[float]
    """Fraction of frame area covered by road-user boxes, computed as a union so
    overlapping boxes are not double-counted.

    An image-space quantity. It is **not** traffic density and not a road
    occupancy percentage: both of those need road geometry, and this measures
    pixels."""


class ClassTallyOut(DomainModel):
    """Counts by road-user class, with vehicles and non-vehicles separated."""

    by_class: dict[VehicleClass, int] = Field(default_factory=dict)
    total: Annotated[int, Field(ge=0)] = 0
    vehicles: Annotated[int, Field(ge=0)] = 0
    pedestrians: Annotated[int, Field(ge=0)] = 0
    cyclists: Annotated[int, Field(ge=0)] = 0


# ---------------------------------------------------------------------------
# Top-level results
# ---------------------------------------------------------------------------
class ObservationMode(StrEnum):
    """Whether measurements came from one frame or a sequence."""

    SNAPSHOT = "snapshot"
    SEQUENCE = "sequence"


class TrafficAnalytics(DomainModel):
    """Measurable traffic behaviour derived from perception output.

    Describes **one observation area** -- the field of view of one camera. It
    makes no claim about junctions, approaches or lanes, because a single video
    carries no information about how its view maps onto a road network. That
    mapping is supplied separately, and only then can this be projected onto a
    :class:`~app.domain.observation.JunctionObservation`.
    """

    analytics_id: Annotated[str, Field(min_length=1, max_length=64)]
    mode: ObservationMode
    observation_area_id: Annotated[str, Field(min_length=1, max_length=120)]
    """Identifies the camera's field of view, not a surveyed junction."""

    provenance: AnalyticsProvenance
    quality: MeasurementQuality

    frame_width: Annotated[int, Field(gt=0)]
    frame_height: Annotated[int, Field(gt=0)]

    # --- counts -----------------------------------------------------------
    detection_tally: ClassTallyOut
    """Detection boxes. For a sequence this counts one object once per frame."""

    tracked_object_tally: ClassTallyOut
    """Distinct tracked objects. Absent classes simply do not appear."""

    active_tracked_objects: MetricValue[int]

    # --- positions and distribution ---------------------------------------
    object_positions: tuple[ObjectPosition, ...] = ()
    spatial_distribution: SpatialDistribution

    # --- motion -----------------------------------------------------------
    trajectories: tuple[TrajectoryMetrics, ...] = ()
    mean_image_speed_diagonals_per_s: MetricValue[float]
    direction_histogram: dict[DirectionLabel, int] = Field(default_factory=dict)
    mean_trajectory_duration_s: MetricValue[float]

    # --- flow -------------------------------------------------------------
    flow_series: tuple[FlowBucket, ...] = ()
    arrival_rate_per_minute: MetricValue[float]
    """Distinct objects first observed per minute, over the whole observation
    area. Not a cross-section flow: that needs a counting line, and without one
    an object entering and leaving the field of view is counted once regardless
    of which way it went."""

    # --- queue ------------------------------------------------------------
    stationary_vehicle_count: MetricValue[int]
    queue_groups: tuple[QueueGroup, ...] = ()
    total_queue_length_m: MetricValue[float]

    # --- physical ---------------------------------------------------------
    traffic_density_veh_per_km_per_lane: MetricValue[float]
    mean_speed_kph: MetricValue[float]

    @property
    def has_physical_measurements(self) -> bool:
        """Whether any metric is expressed in real-world units."""
        return (
            self.mean_speed_kph.is_available
            or self.traffic_density_veh_per_km_per_lane.is_available
        )
