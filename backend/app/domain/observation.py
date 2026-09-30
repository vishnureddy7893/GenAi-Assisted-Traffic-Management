"""What one source observed at one junction at one instant.

Scope discipline
----------------
A :class:`JunctionObservation` is deliberately narrow: **one source, one
junction, one timestamp**. It is a record of evidence, not a view of the world.
It knows nothing about neighbouring junctions, nothing about the route an
ambulance is taking, and nothing about whether it is still current.

That narrowness is the point. Observations, network state and emergency events
have genuinely different lifetimes -- an observation is immutable the moment it
is made and is never updated; a network state is rebuilt continuously; an
emergency event persists across many of both. Collapsing them into one object
means the shortest-lived thing dictates the lifetime of everything else.

Fusing observations into a network-wide view is the job of
:mod:`app.domain.state`, which applies an explicit staleness budget while doing
it. That is where multi-camera time alignment is handled, and it is handled
there precisely because an observation cannot know how old it will be when used.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated

from pydantic import Field, model_validator

from app.domain.base import DomainModel
from app.domain.enums import (
    DataQualityFlag,
    SignalIndication,
    VehicleClass,
    VisibilityLevel,
    WeatherCondition,
)
from app.domain.network import ApproachId, JunctionId, MovementId, SegmentId
from app.domain.provenance import FloatMeasurement, IntMeasurement

ObservationId = Annotated[str, Field(min_length=1, max_length=64)]
SourceId = Annotated[str, Field(min_length=1, max_length=64)]


class CameraCalibration(DomainModel):
    """Image-to-ground calibration for one camera.

    Without this, a tracker measures pixels per second. Pixels per second is not
    a speed and cannot be converted into one, because the pixel-to-metre scale
    varies across the image with perspective.

    The consequence is enforced downstream: when ``is_calibrated`` is ``False``,
    every metric speed and density field on the observation must be ``None`` and
    :attr:`DataQualityFlag.UNCALIBRATED` must be raised. Reporting an
    uncalibrated pixel rate as km/h would be a fabricated measurement, and it
    would propagate straight into an emergency ETA.
    """

    is_calibrated: bool
    method: Annotated[str, Field(max_length=100)] | None = None
    """e.g. ``homography_from_4_ground_points``, ``known_lane_width``."""

    homography: tuple[float, ...] | None = None
    """Row-major 3x3 image-to-ground homography, when available."""

    ground_scale_m_per_px: float | None = None
    """Only meaningful for a near-orthographic view; a single scalar cannot
    represent a perspective view and must not be used as if it could."""

    reprojection_error_px: float | None = None
    """Calibration residual. A large residual means metric outputs carry wide
    error bars, which downstream confidence should reflect."""

    calibrated_at: dt.datetime | None = None

    @model_validator(mode="after")
    def _calibrated_requires_evidence(self) -> CameraCalibration:
        if self.is_calibrated and self.homography is None and self.ground_scale_m_per_px is None:
            msg = (
                "is_calibrated=True requires either a homography or a ground scale; "
                "a calibration flag with no calibration data is a fabricated claim"
            )
            raise ValueError(msg)
        if self.homography is not None and len(self.homography) != 9:
            msg = "homography must be 9 values (row-major 3x3)"
            raise ValueError(msg)
        return self


class ObservationQuality(DomainModel):
    """Trustworthiness of one observation.

    Carried alongside the numbers rather than folded into them. A count of 40
    vehicles under heavy occlusion and a count of 40 in clear conditions are
    different evidence, and the decision layer is entitled to know which it has.
    """

    flags: frozenset[DataQualityFlag] = frozenset()
    mean_detection_confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    frames_analysed: Annotated[int, Field(ge=0)] | None = None
    frames_dropped: Annotated[int, Field(ge=0)] | None = None
    analysis_window_s: Annotated[float, Field(gt=0)] | None = None
    """Aggregation window. A count is meaningless without the interval it spans:
    "12 vehicles" is a flow only once you know it was over 60 seconds."""

    @property
    def is_degraded(self) -> bool:
        """Whether any condition reduces trust in this observation."""
        return bool(self.flags)


class PedestrianObservation(DomainModel):
    """Pedestrian presence at a crossing on this approach.

    Pedestrians are modelled explicitly rather than as an afterthought because
    they are the constraint that most often makes a fast emergency corridor
    unsafe. Whether that constraint is hard or soft is a policy decision made in
    :mod:`app.domain.safety`, not a property of the count itself.
    """

    waiting_count: IntMeasurement | None = None
    crossing_count: IntMeasurement | None = None
    density_person_per_m2: FloatMeasurement | None = None
    includes_vulnerable_users: bool | None = None
    """Wheelchair users, children, elderly -- longer clearance times. ``None``
    means not assessed, which must not be read as ``False``."""


class ApproachObservation(DomainModel):
    """Measurements for one incoming approach at one junction.

    Every quantity is optional and every present quantity carries provenance.
    Optionality is not laziness: a detector with no queue-detection zone
    configured genuinely cannot report a queue, and ``None`` is the honest
    representation. A zero would be read downstream as "no queue", which is a
    different and potentially dangerous claim.
    """

    approach_id: ApproachId
    incoming_segment_id: SegmentId

    vehicle_count_total: IntMeasurement | None = None
    vehicle_count_by_class: dict[VehicleClass, IntMeasurement] = Field(default_factory=dict)

    flow_veh_per_hour: FloatMeasurement | None = None
    """Requires ``ObservationQuality.analysis_window_s`` to be meaningful."""

    mean_speed_kph: FloatMeasurement | None = None
    """Space-mean speed. Requires calibration; otherwise must be ``None``."""

    density_veh_per_km_per_lane: FloatMeasurement | None = None

    queue_length_m: FloatMeasurement | None = None
    queue_length_veh: IntMeasurement | None = None
    stopped_vehicle_count: IntMeasurement | None = None

    mean_waiting_time_s: FloatMeasurement | None = None
    """Cumulative stopped delay per vehicle. The input to level-of-service."""

    occupancy_ratio: FloatMeasurement | None = None

    observed_signal_indication: SignalIndication = SignalIndication.UNKNOWN
    """What the camera could actually see. ``UNKNOWN`` when the signal head is
    out of frame -- never inferred from traffic behaviour, because inferring
    green from movement is exactly backwards during a violation."""

    observed_signal_movement_id: MovementId | None = None
    """Which movement the observed indication governs, when identifiable."""

    pedestrians: PedestrianObservation | None = None

    emergency_vehicle_detected: bool | None = None
    """A *visual* detection only. Never sufficient on its own to trigger
    emergency handling -- see :mod:`app.domain.emergency`, where an emergency is
    a declared event with a destination and a priority."""


class JunctionObservation(DomainModel):
    """One source's view of one junction at one timestamp.

    The primary output of the computer-vision pipeline (Mode A and Mode B alike)
    and the single point at which recorded and live inputs converge. Everything
    downstream consumes this type and is agnostic to how it was produced.
    """

    observation_id: ObservationId
    junction_id: JunctionId
    source_id: SourceId
    """Which data source produced this. Two cameras on the same junction produce
    two observations, reconciled at fusion time rather than silently averaged."""

    captured_at: dt.datetime
    """Timestamp of the underlying evidence on the source's clock. For recorded
    input this is media time mapped through the source epoch, never wall-clock
    ingest time -- otherwise replaying old video makes stale data look fresh."""

    ingested_at: dt.datetime
    """When this system received it. ``ingested_at - captured_at`` is pipeline
    latency, which matters for live sources and is meaningless for recorded ones."""

    schema_version: Annotated[str, Field(min_length=1, max_length=20)]

    approaches: tuple[ApproachObservation, ...] = ()
    quality: ObservationQuality = ObservationQuality()
    calibration: CameraCalibration | None = None

    observed_weather: WeatherCondition | None = None
    observed_visibility: VisibilityLevel | None = None
    """What this camera could infer about conditions. Network-wide environment
    is assembled separately; a single camera under a bridge is not evidence that
    it is not raining."""

    @model_validator(mode="after")
    def _validate_timestamps_and_calibration(self) -> JunctionObservation:
        if self.captured_at.tzinfo is None or self.ingested_at.tzinfo is None:
            msg = "observation timestamps must be timezone-aware"
            raise ValueError(msg)

        duplicates = [a.approach_id for a in self.approaches]
        if len(set(duplicates)) != len(duplicates):
            msg = f"duplicate approach_id in observation {self.observation_id!r}"
            raise ValueError(msg)

        # Metric outputs require calibration. This is the enforcement point for
        # the rule described on CameraCalibration.
        calibrated = self.calibration is not None and self.calibration.is_calibrated
        if not calibrated:
            offenders = [
                a.approach_id
                for a in self.approaches
                if a.mean_speed_kph is not None
                or a.density_veh_per_km_per_lane is not None
                or a.queue_length_m is not None
            ]
            if offenders:
                msg = (
                    "metric speed/density/queue-length reported without camera "
                    f"calibration for approaches {offenders}; pixel measurements "
                    "cannot be converted to metric units, so these must be None "
                    "and DataQualityFlag.UNCALIBRATED must be set"
                )
                raise ValueError(msg)

        return self

    @property
    def latency_s(self) -> float:
        """Pipeline latency between capture and ingest, in seconds."""
        return (self.ingested_at - self.captured_at).total_seconds()

    def approach(self, approach_id: str) -> ApproachObservation:
        """Return one approach's measurements, raising ``KeyError`` if absent."""
        for ap in self.approaches:
            if ap.approach_id == approach_id:
                return ap
        raise KeyError(approach_id)
