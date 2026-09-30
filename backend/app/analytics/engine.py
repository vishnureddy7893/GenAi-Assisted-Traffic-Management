"""Traffic analytics engine.

Answers *"what measurable traffic behaviour is occurring?"* — and stops there.
It computes no congestion level, no travel time, no route and no recommendation,
because those are interpretations rather than measurements.

Input is deliberately neutral
-----------------------------
The engine consumes :class:`ObservedFrame` records, not detector or tracker
objects. A still image is one frame with no track ids; a video is many frames
with them. Everything below branches on what the data actually contains rather
than on which pipeline produced it, so a dataset source that never ran a
detector can feed the same analytics unchanged.

One camera, one observation area
--------------------------------
The result describes the field of view of one camera. It asserts nothing about
junctions, approaches or lanes: a video carries no information about how its
view maps onto a road network. Splitting a frame into approaches requires a zone
configuration, and inventing one would attach real numbers to imaginary
geography.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import pairwise

from app.analytics import geometry
from app.analytics.calibration import GroundProjector, mps_to_kph
from app.analytics.schemas import (
    AnalyticsProvenance,
    ClassTallyOut,
    DirectionLabel,
    FlowBucket,
    ImagePoint2D,
    MeasurementQuality,
    MetricAvailability,
    MetricValue,
    ObjectPosition,
    ObservationMode,
    QualityGrade,
    QueueGroup,
    SpatialDistribution,
    TrafficAnalytics,
    TrajectoryMetrics,
    available,
    no_calibration,
    unavailable,
)
from app.core.config import Settings
from app.domain.enums import VEHICLE_CLASSES, VehicleClass
from app.domain.observation import CameraCalibration
from app.vision.models import ModelIdentity

METHOD = "image-space-traffic-analytics"
METHOD_VERSION = "1.0"
ANALYTICS_SCHEMA_VERSION = "0.1.0"

#: Detection confidence below which an object is counted toward the
#: low-confidence fraction that downgrades measurement quality.
LOW_CONFIDENCE_THRESHOLD = 0.4


@dataclass(frozen=True, slots=True)
class ObservedObject:
    """One detected object within a frame."""

    road_user_class: VehicleClass
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int | None = None

    @property
    def centre(self) -> tuple[float, float]:
        """Box centre in pixels."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def ground_contact(self) -> tuple[float, float]:
        """Bottom-centre: the road-plane proxy used for all motion measurement."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.y2 - self.y1

    @property
    def diagonal(self) -> float:
        """Box diagonal, the scale used for proximity thresholds."""
        return geometry.frame_diagonal(self.width, self.height)


@dataclass(frozen=True, slots=True)
class ObservedFrame:
    """One analysed frame."""

    frame_index: int
    timestamp: dt.datetime
    width: int
    height: int
    objects: tuple[ObservedObject, ...] = ()
    media_time_s: float | None = None


@dataclass(slots=True)
class AnalyticsContext:
    """Everything the engine needs besides the frames."""

    media_id: str
    source_id: str
    observation_area_id: str
    model: ModelIdentity
    tracker: str | None = None
    calibration: CameraCalibration | None = None
    lane_count: int | None = None
    """Surveyed lane count. Required for per-lane density; absent by default."""

    segment_length_m: float | None = None
    """Surveyed length of road in view. Required for density."""

    settings: Settings = field(default_factory=Settings)


@dataclass(slots=True)
class _TrackObservation:
    """One track's appearance in one frame."""

    frame_index: int
    timestamp: dt.datetime
    obj: ObservedObject


def _tally(classes: list[VehicleClass]) -> ClassTallyOut:
    """Count classes, separating vehicles from vulnerable road users."""
    counts = Counter(classes)
    return ClassTallyOut(
        by_class=dict(counts),
        total=len(classes),
        vehicles=sum(n for cls, n in counts.items() if cls in VEHICLE_CLASSES),
        pedestrians=counts.get(VehicleClass.PERSON, 0),
        cyclists=counts.get(VehicleClass.BICYCLE, 0),
    )


def _point(x: float, y: float, width: int, height: int) -> ImagePoint2D:
    """Build a point carrying both pixel and normalised coordinates."""
    return ImagePoint2D(
        x=x,
        y=y,
        x_norm=min(max(x / width, 0.0), 1.0) if width else 0.0,
        y_norm=min(max(y / height, 0.0), 1.0) if height else 0.0,
    )


class TrafficAnalyticsEngine:
    """Derives traffic measurements from observed frames."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()

    @property
    def method(self) -> str:
        """Calculation method name, recorded in provenance."""
        return METHOD

    @property
    def version(self) -> str:
        """Calculation method version, recorded in provenance."""
        return METHOD_VERSION

    # -----------------------------------------------------------------
    def analyse(
        self,
        frames: list[ObservedFrame],
        context: AnalyticsContext,
        *,
        computed_at: dt.datetime | None = None,
    ) -> TrafficAnalytics:
        """Compute traffic measurements over one or more frames."""
        if not frames:
            msg = "Analytics requires at least one analysed frame."
            raise ValueError(msg)

        settings = context.settings or self._settings
        ordered = sorted(frames, key=lambda f: f.frame_index)
        mode = ObservationMode.SNAPSHOT if len(ordered) == 1 else ObservationMode.SEQUENCE

        width = ordered[0].width
        height = ordered[0].height
        diagonal = geometry.frame_diagonal(width, height)
        projector = GroundProjector.from_calibration(context.calibration)

        all_objects = [obj for frame in ordered for obj in frame.objects]
        detection_tally = _tally([o.road_user_class for o in all_objects])

        trajectories, tracked_tally, active_count = self._motion(
            ordered, mode, diagonal, projector, settings
        )
        positions = self._positions(ordered[-1], width, height)
        distribution = self._distribution(ordered[-1], width, height)
        flow, arrival_rate = self._flow(ordered, mode, settings)
        stationary_count, queue_groups = self._queues(
            ordered, trajectories, mode, projector, settings
        )
        quality = self._quality(ordered, trajectories, all_objects, projector, context)

        return TrafficAnalytics(
            analytics_id=uuid.uuid4().hex,
            mode=mode,
            observation_area_id=context.observation_area_id,
            provenance=self._provenance(ordered, context, mode, computed_at),
            quality=quality,
            frame_width=width,
            frame_height=height,
            detection_tally=detection_tally,
            tracked_object_tally=tracked_tally,
            active_tracked_objects=active_count,
            object_positions=positions,
            spatial_distribution=distribution,
            trajectories=trajectories,
            mean_image_speed_diagonals_per_s=self._mean_speed(trajectories),
            direction_histogram=self._directions(trajectories),
            mean_trajectory_duration_s=self._mean_duration(trajectories),
            flow_series=flow,
            arrival_rate_per_minute=arrival_rate,
            stationary_vehicle_count=stationary_count,
            queue_groups=queue_groups,
            total_queue_length_m=self._queue_length(queue_groups, projector),
            traffic_density_veh_per_km_per_lane=self._density(
                tracked_tally, detection_tally, context, mode
            ),
            mean_speed_kph=self._mean_physical_speed(trajectories, projector),
        )

    # --- provenance ---------------------------------------------------
    def _provenance(
        self,
        frames: list[ObservedFrame],
        context: AnalyticsContext,
        mode: ObservationMode,
        computed_at: dt.datetime | None,
    ) -> AnalyticsProvenance:
        first, last = frames[0], frames[-1]
        snapshot = mode is ObservationMode.SNAPSHOT
        return AnalyticsProvenance(
            media_id=context.media_id,
            source_id=context.source_id,
            frame_index=first.frame_index if snapshot else None,
            frame_range=None if snapshot else (first.frame_index, last.frame_index),
            timestamp=first.timestamp if snapshot else None,
            time_range=None if snapshot else (first.timestamp, last.timestamp),
            model=context.model,
            tracker=context.tracker,
            method=METHOD,
            method_version=METHOD_VERSION,
            analytics_schema_version=ANALYTICS_SCHEMA_VERSION,
            computed_at=computed_at or dt.datetime.now(tz=dt.UTC),
        )

    # --- positions ----------------------------------------------------
    def _positions(
        self, frame: ObservedFrame, width: int, height: int
    ) -> tuple[ObjectPosition, ...]:
        """Positions in the most recent analysed frame."""
        frame_area = float(width * height) or 1.0
        return tuple(
            ObjectPosition(
                road_user_class=obj.road_user_class,
                confidence=obj.confidence,
                track_id=obj.track_id,
                centre=_point(*obj.centre, width, height),
                ground_contact=_point(*obj.ground_contact, width, height),
                box_width_px=obj.width,
                box_height_px=obj.height,
                box_area_fraction=min(obj.width * obj.height / frame_area, 1.0),
            )
            for obj in frame.objects
        )

    def _distribution(
        self, frame: ObservedFrame, width: int, height: int
    ) -> SpatialDistribution:
        """Where objects sit in the frame."""
        contacts = [obj.ground_contact for obj in frame.objects]
        boxes = [(o.x1, o.y1, o.x2, o.y2) for o in frame.objects]

        centroid = None
        if contacts:
            mean_x = sum(p[0] for p in contacts) / len(contacts)
            mean_y = sum(p[1] for p in contacts) / len(contacts)
            centroid = _point(mean_x, mean_y, width, height)

        spread_x = geometry.standard_deviation([p[0] / width for p in contacts]) if width else None
        spread_y = (
            geometry.standard_deviation([p[1] / height for p in contacts]) if height else None
        )

        insufficient = "At least two objects are needed to measure spatial spread."
        return SpatialDistribution(
            centroid=centroid,
            spread_x_norm=(
                available(
                    spread_x,
                    unit="fraction_of_width",
                    method="population_stddev",
                    image_space=True,
                )
                if spread_x is not None
                else unavailable(MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA, insufficient)
            ),
            spread_y_norm=(
                available(
                    spread_y,
                    unit="fraction_of_height",
                    method="population_stddev",
                    image_space=True,
                )
                if spread_y is not None
                else unavailable(
                    MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA, insufficient
                )
            ),
            grid_rows=3,
            grid_cols=3,
            grid_counts=geometry.grid_counts(contacts, width, height),
            image_occupancy_ratio=available(
                geometry.union_area_fraction(boxes, width, height),
                unit="fraction_of_frame_area",
                method="rasterised_box_union",
                image_space=True,
            ),
        )

    # --- motion -------------------------------------------------------
    def _motion(
        self,
        frames: list[ObservedFrame],
        mode: ObservationMode,
        diagonal: float,
        projector: GroundProjector | None,
        settings: Settings,
    ) -> tuple[tuple[TrajectoryMetrics, ...], ClassTallyOut, MetricValue[int]]:
        """Build per-track trajectory measurements."""
        if mode is ObservationMode.SNAPSHOT:
            reason = (
                "Tracking requires a sequence of frames; a single image has no "
                "temporal information."
            )
            return (
                (),
                ClassTallyOut(),
                unavailable(MetricAvailability.UNAVAILABLE_NOT_APPLICABLE, reason),
            )

        grouped: dict[int, list[_TrackObservation]] = defaultdict(list)
        for frame in frames:
            for obj in frame.objects:
                if obj.track_id is not None:
                    grouped[obj.track_id].append(
                        _TrackObservation(frame.frame_index, frame.timestamp, obj)
                    )

        if not grouped:
            reason = "No tracked objects: detections carry no track identity."
            return (
                (),
                ClassTallyOut(),
                unavailable(MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA, reason),
            )

        trajectories = [
            self._trajectory(
                track_id,
                sorted(obs, key=lambda o: o.frame_index),
                diagonal,
                projector,
                settings,
            )
            for track_id, obs in sorted(grouped.items())
        ]

        tracked_tally = _tally([t.road_user_class for t in trajectories])
        last_index = frames[-1].frame_index
        active = sum(1 for t in trajectories if t.last_frame_index == last_index)

        return (
            tuple(trajectories),
            tracked_tally,
            available(active, unit="objects", method="tracks_present_in_final_frame"),
        )

    def _trajectory(
        self,
        track_id: int,
        observations: list[_TrackObservation],
        diagonal: float,
        projector: GroundProjector | None,
        settings: Settings,
    ) -> TrajectoryMetrics:
        """Measure one track's motion."""
        points = [o.obj.ground_contact for o in observations]
        classes = Counter(o.obj.road_user_class for o in observations)
        # Majority class over the trajectory, matching how the tracker resolves
        # single-frame class flicker.
        road_class = classes.most_common(1)[0][0]

        duration = (observations[-1].timestamp - observations[0].timestamp).total_seconds()
        enough = len(observations) >= settings.analytics_min_track_frames and duration > 0
        short = (
            f"Needs at least {settings.analytics_min_track_frames} track points "
            "spanning a non-zero interval."
        )

        total_px = geometry.path_length(points)
        net_px = geometry.net_displacement(points)
        steps = max(len(points) - 1, 1)

        speed_px = total_px / duration if enough else None
        speed_diag = (speed_px / diagonal) if (speed_px is not None and diagonal > 0) else None
        heading = geometry.direction_degrees(points[0], points[-1]) if enough else None

        stationary_frames = self._stationary_frames(observations, diagonal, settings)
        is_stationary = (
            speed_diag is not None
            and speed_diag < settings.analytics_stationary_speed_frac
            and stationary_frames >= settings.analytics_min_stationary_frames
        )

        physical_mps = self._physical_speed(points, duration, projector) if enough else None

        return TrajectoryMetrics(
            track_id=track_id,
            road_user_class=road_class,
            frames_tracked=len(observations),
            first_frame_index=observations[0].frame_index,
            last_frame_index=observations[-1].frame_index,
            duration_s=(
                available(duration, unit="s", method="last_minus_first_timestamp")
                if duration > 0
                else unavailable(MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA, short, unit="s")
            ),
            total_displacement_px=available(
                total_px, unit="px", method="ground_contact_path_length", image_space=True
            ),
            net_displacement_px=available(
                net_px, unit="px", method="ground_contact_first_to_last", image_space=True
            ),
            mean_step_displacement_px=available(
                total_px / steps, unit="px", method="path_length_over_steps", image_space=True
            ),
            image_speed_px_per_s=(
                available(
                    speed_px,
                    unit="px/s",
                    method="path_length_over_duration",
                    image_space=True,
                )
                if speed_px is not None
                else unavailable(
                    MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA, short, unit="px/s"
                )
            ),
            image_speed_diagonals_per_s=(
                available(
                    speed_diag,
                    unit="frame_diagonals/s",
                    method="image_speed_normalised_by_frame_diagonal",
                    image_space=True,
                )
                if speed_diag is not None
                else unavailable(
                    MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA,
                    short,
                    unit="frame_diagonals/s",
                )
            ),
            direction_deg=(
                available(
                    heading,
                    unit="degrees_clockwise_from_image_right",
                    method="net_displacement_bearing",
                    image_space=True,
                )
                if heading is not None
                else unavailable(
                    MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA,
                    "No net displacement: a stationary object has no direction."
                    if enough
                    else short,
                    unit="degrees",
                )
            ),
            direction_label=(
                available(
                    geometry.direction_label(heading),
                    method="eight_sector_image_space",
                )
                if heading is not None
                else unavailable(
                    MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA,
                    "No net displacement to derive a heading from."
                    if enough
                    else short,
                )
            ),
            is_stationary=is_stationary,
            stationary_frames=stationary_frames,
            id_switch_suspected=False,
            physical_speed_mps=(
                available(
                    physical_mps,
                    unit="m/s",
                    method=f"ground_projection:{projector.method}",
                    confidence=projector.confidence,
                )
                if physical_mps is not None and projector is not None
                else no_calibration("m/s", method="ground_projection")
            ),
            physical_speed_kph=(
                available(
                    mps_to_kph(physical_mps),
                    unit="km/h",
                    method=f"ground_projection:{projector.method}",
                    confidence=projector.confidence,
                )
                if physical_mps is not None and projector is not None
                else no_calibration("km/h", method="ground_projection")
            ),
        )

    def _stationary_frames(
        self,
        observations: list[_TrackObservation],
        diagonal: float,
        settings: Settings,
    ) -> int:
        """Longest run of consecutive frames with near-zero movement."""
        if len(observations) < 2 or diagonal <= 0:
            return 0

        longest = 0
        current = 0
        for previous, nxt in pairwise(observations):
            gap = (nxt.timestamp - previous.timestamp).total_seconds()
            if gap <= 0:
                continue
            step = geometry.distance(previous.obj.ground_contact, nxt.obj.ground_contact)
            if (step / gap) / diagonal < settings.analytics_stationary_speed_frac:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest

    def _physical_speed(
        self,
        points: list[tuple[float, float]],
        duration: float,
        projector: GroundProjector | None,
    ) -> float | None:
        """Ground-plane speed in m/s, or ``None`` without a projector."""
        if projector is None or duration <= 0 or len(points) < 2:
            return None
        total = 0.0
        for a, b in pairwise(points):
            step = projector.ground_distance(a, b)
            if step is None:
                # A segment crossing the horizon makes the whole trajectory
                # unmeasurable rather than partially measurable.
                return None
            total += step
        return total / duration

    def _mean_speed(self, trajectories: tuple[TrajectoryMetrics, ...]) -> MetricValue[float]:
        """Mean image-space speed across moving objects."""
        values = [
            t.image_speed_diagonals_per_s.value
            for t in trajectories
            if t.image_speed_diagonals_per_s.is_available
            and t.image_speed_diagonals_per_s.value is not None
        ]
        if not values:
            return unavailable(
                MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA,
                "No trajectory had enough points to measure speed.",
                unit="frame_diagonals/s",
            )
        return available(
            sum(values) / len(values),
            unit="frame_diagonals/s",
            method="mean_over_trajectories",
            image_space=True,
        )

    def _mean_physical_speed(
        self, trajectories: tuple[TrajectoryMetrics, ...], projector: GroundProjector | None
    ) -> MetricValue[float]:
        """Mean physical speed, or an explicit unavailability."""
        if projector is None:
            return no_calibration("km/h", method="ground_projection")
        values = [
            t.physical_speed_kph.value
            for t in trajectories
            if t.physical_speed_kph.is_available and t.physical_speed_kph.value is not None
        ]
        if not values:
            return unavailable(
                MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA,
                "Calibration is configured but no trajectory was long enough to measure speed.",
                unit="km/h",
            )
        return available(
            sum(values) / len(values),
            unit="km/h",
            method=f"mean_over_trajectories:{projector.method}",
            confidence=projector.confidence,
        )

    def _directions(self, trajectories: tuple[TrajectoryMetrics, ...]) -> dict[DirectionLabel, int]:
        """Histogram of image-space headings."""
        counts: Counter[DirectionLabel] = Counter()
        for trajectory in trajectories:
            label = trajectory.direction_label
            if label.is_available and label.value is not None:
                counts[label.value] += 1
        return dict(counts)

    def _mean_duration(self, trajectories: tuple[TrajectoryMetrics, ...]) -> MetricValue[float]:
        """Mean trajectory duration."""
        values = [
            t.duration_s.value
            for t in trajectories
            if t.duration_s.is_available and t.duration_s.value is not None
        ]
        if not values:
            return unavailable(
                MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA,
                "No trajectory spanned a measurable interval.",
                unit="s",
            )
        return available(sum(values) / len(values), unit="s", method="mean_over_trajectories")

    # --- flow ---------------------------------------------------------
    def _flow(
        self, frames: list[ObservedFrame], mode: ObservationMode, settings: Settings
    ) -> tuple[tuple[FlowBucket, ...], MetricValue[float]]:
        """Arrival and presence rates over time."""
        if mode is ObservationMode.SNAPSHOT:
            reason = "Flow is a rate over time; a single image spans no interval."
            return (), unavailable(
                MetricAvailability.UNAVAILABLE_NOT_APPLICABLE, reason, unit="objects/min"
            )

        start = frames[0].timestamp
        span = (frames[-1].timestamp - start).total_seconds()
        if span <= 0:
            reason = "All analysed frames share one timestamp, so no rate can be derived."
            return (), unavailable(
                MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA, reason, unit="objects/min"
            )

        window = settings.analytics_flow_window_s
        first_seen: dict[int, float] = {}
        per_window_active: dict[int, set[int]] = defaultdict(set)
        per_window_arrivals: dict[int, set[int]] = defaultdict(set)

        for frame in frames:
            offset = (frame.timestamp - start).total_seconds()
            bucket = int(offset // window)
            for obj in frame.objects:
                if obj.track_id is None:
                    continue
                per_window_active[bucket].add(obj.track_id)
                if obj.track_id not in first_seen:
                    first_seen[obj.track_id] = offset
                    per_window_arrivals[bucket].add(obj.track_id)

        bucket_count = int(span // window) + 1
        series = tuple(
            FlowBucket(
                window_start_s=index * window,
                window_end_s=(index + 1) * window,
                arrivals=len(per_window_arrivals.get(index, set())),
                active=len(per_window_active.get(index, set())),
            )
            for index in range(bucket_count)
        )

        if not first_seen:
            return series, unavailable(
                MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA,
                "No tracked objects were observed, so no arrival rate exists.",
                unit="objects/min",
            )

        return series, available(
            len(first_seen) / span * 60.0,
            unit="objects/min",
            method="distinct_first_appearances_over_observed_span",
        )

    # --- queues -------------------------------------------------------
    def _queues(
        self,
        frames: list[ObservedFrame],
        trajectories: tuple[TrajectoryMetrics, ...],
        mode: ObservationMode,
        projector: GroundProjector | None,
        settings: Settings,
    ) -> tuple[MetricValue[int], tuple[QueueGroup, ...]]:
        """Estimate queues from movement, persistence and proximity.

        Never from a vehicle count. A queue is vehicles that are *stopped*,
        *stay* stopped, and are *adjacent*; a frame containing twenty
        free-flowing vehicles is busy, not queued, and reporting it as a queue of
        twenty would invert the meaning of the measurement.
        """
        if mode is ObservationMode.SNAPSHOT:
            reason = (
                "Queue estimation needs persistence across frames: a single image "
                "cannot distinguish a stopped vehicle from a moving one."
            )
            return (
                unavailable(
                    MetricAvailability.UNAVAILABLE_NOT_APPLICABLE, reason, unit="vehicles"
                ),
                (),
            )

        stationary = [
            t
            for t in trajectories
            if t.is_stationary and t.road_user_class in VEHICLE_CLASSES
        ]
        count = available(
            len(stationary),
            unit="vehicles",
            method="tracks_below_movement_threshold_for_min_frames",
        )
        if not stationary:
            return count, ()

        last_frame = frames[-1]
        by_track = {obj.track_id: obj for obj in last_frame.objects if obj.track_id is not None}
        # Fall back to each track's final observed position when it is not in the
        # last frame: a queued vehicle may have dropped out for a frame or two.
        positions: dict[int, ObservedObject] = {}
        for frame in frames:
            for obj in frame.objects:
                if obj.track_id is not None:
                    positions[obj.track_id] = obj
        positions.update(by_track)

        items = [
            (t.track_id, positions[t.track_id].centre, positions[t.track_id].diagonal)
            for t in stationary
            if t.track_id in positions
        ]
        clusters = geometry.cluster_by_proximity(
            items, proximity_factor=settings.analytics_queue_proximity_factor
        )

        by_id = {t.track_id: t for t in stationary}
        groups: list[QueueGroup] = []
        for index, member_ids in enumerate(clusters, start=1):
            members = [by_id[i] for i in member_ids if i in by_id]
            if not members:
                continue
            points = [positions[i].centre for i in member_ids if i in positions]
            durations = [
                m.duration_s.value
                for m in members
                if m.duration_s.is_available and m.duration_s.value is not None
            ]
            confidences = [
                positions[i].confidence for i in member_ids if i in positions
            ]
            groups.append(
                QueueGroup(
                    group_id=index,
                    member_track_ids=tuple(member_ids),
                    member_count=len(members),
                    road_user_classes=dict(Counter(m.road_user_class for m in members)),
                    extent_px=available(
                        geometry.cluster_extent(points),
                        unit="px",
                        method="max_pairwise_centre_distance",
                        image_space=True,
                    ),
                    longest_stationary_duration_s=(
                        available(max(durations), unit="s", method="max_member_track_duration")
                        if durations
                        else unavailable(
                            MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA,
                            "No member track spanned a measurable interval.",
                            unit="s",
                        )
                    ),
                    length_m=self._group_length(points, projector),
                    mean_confidence=(
                        sum(confidences) / len(confidences) if confidences else None
                    ),
                )
            )
        return count, tuple(groups)

    def _group_length(
        self, points: list[tuple[float, float]], projector: GroundProjector | None
    ) -> MetricValue[float]:
        """Physical extent of a queue cluster, when calibration allows it."""
        if projector is None:
            return unavailable(
                MetricAvailability.UNAVAILABLE_NO_CALIBRATION,
                "Queue length in metres requires camera calibration; the cluster "
                "extent is reported in pixels instead.",
                unit="m",
            )
        if len(points) < 2:
            return unavailable(
                MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA,
                "A single stopped vehicle has no measurable queue extent.",
                unit="m",
            )
        projected = [projector.project(p) for p in points]
        if any(p is None for p in projected):
            return unavailable(
                MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA,
                "Part of the cluster projects beyond the horizon, so its ground "
                "extent cannot be measured.",
                unit="m",
            )
        valid = [p for p in projected if p is not None]
        return available(
            geometry.cluster_extent(valid),
            unit="m",
            method=f"ground_projection:{projector.method}",
            confidence=projector.confidence,
        )

    def _queue_length(
        self, groups: tuple[QueueGroup, ...], projector: GroundProjector | None
    ) -> MetricValue[float]:
        """Total physical queue length across all clusters."""
        if projector is None:
            return unavailable(
                MetricAvailability.UNAVAILABLE_NO_CALIBRATION,
                "Total queue length in metres requires camera calibration.",
                unit="m",
            )
        lengths = [
            g.length_m.value
            for g in groups
            if g.length_m.is_available and g.length_m.value is not None
        ]
        if not lengths:
            return unavailable(
                MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA,
                "No queue cluster had a measurable ground extent.",
                unit="m",
            )
        return available(sum(lengths), unit="m", method="sum_of_cluster_extents")

    # --- density ------------------------------------------------------
    def _density(
        self,
        tracked: ClassTallyOut,
        detections: ClassTallyOut,
        context: AnalyticsContext,
        mode: ObservationMode,
    ) -> MetricValue[float]:
        """Vehicles per km per lane, when road geometry is surveyed.

        Requires both the length of road in view and its lane count. Neither can
        be inferred from an image: the same frame could show 30 m of a two-lane
        street or 300 m of a six-lane arterial, and the resulting densities
        differ by an order of magnitude.
        """
        if context.segment_length_m is None or context.lane_count is None:
            return unavailable(
                MetricAvailability.UNAVAILABLE_NO_GEOMETRY,
                "Traffic density requires surveyed road geometry (length of road "
                "in view and lane count), which is not configured for this "
                "observation area.",
                unit="veh/km/lane",
            )
        if context.segment_length_m <= 0 or context.lane_count <= 0:
            return unavailable(
                MetricAvailability.UNAVAILABLE_NO_GEOMETRY,
                "Configured road geometry is not usable (non-positive length or lane count).",
                unit="veh/km/lane",
            )

        vehicles = tracked.vehicles if mode is ObservationMode.SEQUENCE else detections.vehicles
        km = context.segment_length_m / 1000.0
        return available(
            vehicles / km / context.lane_count,
            unit="veh/km/lane",
            method="vehicles_over_surveyed_length_and_lanes",
        )

    # --- quality ------------------------------------------------------
    def _quality(
        self,
        frames: list[ObservedFrame],
        trajectories: tuple[TrajectoryMetrics, ...],
        objects: list[ObservedObject],
        projector: GroundProjector | None,
        context: AnalyticsContext,
    ) -> MeasurementQuality:
        """Assess how far these measurements can be trusted."""
        warnings: list[str] = []

        confidences = [o.confidence for o in objects]
        mean_conf = sum(confidences) / len(confidences) if confidences else None
        low_fraction = (
            sum(1 for c in confidences if c < LOW_CONFIDENCE_THRESHOLD) / len(confidences)
            if confidences
            else None
        )

        interval = None
        if len(frames) >= 2:
            span = (frames[-1].timestamp - frames[0].timestamp).total_seconds()
            if span > 0:
                interval = span / (len(frames) - 1)

        if projector is None:
            warnings.append(
                "No camera calibration: all measurements are image-space only."
            )
        if context.segment_length_m is None or context.lane_count is None:
            warnings.append(
                "No surveyed road geometry: traffic density is unavailable."
            )
        if low_fraction is not None and low_fraction > 0.3:
            warnings.append(
                f"{low_fraction:.0%} of detections are below "
                f"{LOW_CONFIDENCE_THRESHOLD:.0%} confidence."
            )
        if len(frames) < 3 and len(frames) > 1:
            warnings.append("Very short sequence: motion measurements are weakly supported.")

        switches = sum(1 for t in trajectories if t.id_switch_suspected)
        if switches:
            warnings.append(
                f"{switches} trajectory(ies) show a suspected identity switch; their "
                "speed and direction are unreliable."
            )

        if switches or (low_fraction is not None and low_fraction > 0.5) or len(frames) == 1:
            grade = QualityGrade.DEGRADED
        else:
            grade = QualityGrade.GOOD
        if (mean_conf is not None and mean_conf < 0.35) or (
            len(frames) > 1 and not trajectories
        ):
            grade = QualityGrade.POOR

        return MeasurementQuality(
            grade=grade,
            frames_analysed=len(frames),
            mean_detection_confidence=mean_conf,
            low_confidence_fraction=low_fraction,
            tracks_total=len(trajectories),
            tracks_confirmed=len(trajectories),
            suspected_id_switches=switches,
            calibration_available=projector is not None,
            road_geometry_available=(
                context.segment_length_m is not None and context.lane_count is not None
            ),
            sampling_interval_s=interval,
            warnings=tuple(warnings),
        )
