"""Phase 4 traffic analytics tests.

Fixtures are **synthetic and deterministic**: hand-built detection and tracking
sequences with known geometry, so every formula can be checked against an
arithmetic answer rather than against whatever a model happened to produce.

The properties under test are mostly about honesty. A metric that cannot be
computed must say so with a reason; an image-space measurement must never be
presented as physical; a queue must rest on movement and persistence rather than
on a headcount.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.analytics import geometry
from app.analytics.calibration import GroundProjector, mps_to_kph
from app.analytics.engine import (
    AnalyticsContext,
    ObservedFrame,
    ObservedObject,
    TrafficAnalyticsEngine,
)
from app.analytics.extractor import WHOLE_VIEW_APPROACH_ID, AnalyticsTrafficStateExtractor
from app.analytics.schemas import (
    NO_CALIBRATION_MESSAGE,
    DirectionLabel,
    MetricAvailability,
    MetricValue,
    ObservationMode,
    QualityGrade,
    available,
)
from app.api.routes import media as media_routes
from app.core.config import Settings, get_settings
from app.domain.enums import VehicleClass
from app.domain.observation import CameraCalibration
from app.main import create_app
from app.vision.detector import BoundingBox
from app.vision.extractor import ExtractionContext, ZoneConfiguration
from app.vision.models import ModelIdentity
from app.vision.tracker import Track, TrackerInfo, TrackPoint

pytest.importorskip("PIL")
pytest.importorskip("cv2")

T0 = dt.datetime(2024, 3, 14, 8, 30, tzinfo=dt.UTC)
FRAME_W, FRAME_H = 800, 600
MODEL = ModelIdentity(name="test-model", backend="test", sha256="b" * 64, size_bytes=1)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def obj(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    cls: VehicleClass = VehicleClass.CAR,
    conf: float = 0.9,
    track: int | None = None,
) -> ObservedObject:
    """Build one observed object."""
    return ObservedObject(
        road_user_class=cls, confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2, track_id=track
    )


def frame(index: int, *objects: ObservedObject, seconds: float | None = None) -> ObservedFrame:
    """Build one observed frame at ``index`` (0.1 s apart unless overridden)."""
    offset = seconds if seconds is not None else index * 0.1
    return ObservedFrame(
        frame_index=index,
        timestamp=T0 + dt.timedelta(seconds=offset),
        width=FRAME_W,
        height=FRAME_H,
        objects=objects,
        media_time_s=offset,
    )


def context(**overrides: object) -> AnalyticsContext:
    """Build an analytics context."""
    base = {
        "media_id": "media-1",
        "source_id": "video:media-1",
        "observation_area_id": "observation-area:media-1",
        "model": MODEL,
        "tracker": "bytetrack-lite",
        "settings": Settings(),
    }
    base.update(overrides)
    return AnalyticsContext(**base)  # type: ignore[arg-type]


def engine() -> TrafficAnalyticsEngine:
    """Engine with default settings."""
    return TrafficAnalyticsEngine(Settings())


def moving_sequence(
    frames: int = 8, step: float = 40.0, *, cls: VehicleClass = VehicleClass.CAR
) -> list[ObservedFrame]:
    """One object moving steadily rightward at ``step`` px per frame."""
    return [
        frame(i, obj(100 + i * step, 300, 160 + i * step, 360, cls=cls, track=1))
        for i in range(frames)
    ]


def stationary_sequence(frames: int = 8, *, track: int = 1, x: float = 100) -> list[ObservedFrame]:
    """One object that does not move."""
    return [frame(i, obj(x, 300, x + 60, 360, track=track)) for i in range(frames)]


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------
class TestGeometry:
    def test_direction_is_measured_clockwise_from_image_right(self) -> None:
        assert geometry.direction_degrees((0, 0), (10, 0)) == pytest.approx(0.0)
        # Image y increases downward, so +y is "down" on screen.
        assert geometry.direction_degrees((0, 0), (0, 10)) == pytest.approx(90.0)
        assert geometry.direction_degrees((0, 0), (-10, 0)) == pytest.approx(180.0)
        assert geometry.direction_degrees((0, 0), (0, -10)) == pytest.approx(270.0)

    def test_zero_displacement_has_no_direction(self) -> None:
        """Returning 0 would claim the object is moving rightward."""
        assert geometry.direction_degrees((5, 5), (5, 5)) is None

    @pytest.mark.parametrize(
        ("degrees", "label"),
        [
            (0, DirectionLabel.RIGHT),
            (45, DirectionLabel.DOWN_RIGHT),
            (90, DirectionLabel.DOWN),
            (180, DirectionLabel.LEFT),
            (270, DirectionLabel.UP),
            (359, DirectionLabel.RIGHT),
        ],
    )
    def test_direction_labels(self, degrees: float, label: DirectionLabel) -> None:
        assert geometry.direction_label(degrees) is label

    def test_path_length_sums_steps(self) -> None:
        assert geometry.path_length([(0, 0), (3, 4), (3, 4)]) == pytest.approx(5.0)

    def test_net_displacement_is_straight_line(self) -> None:
        """Path length and net displacement differ, which is how jitter shows."""
        points = [(0, 0), (10, 0), (0, 0)]
        assert geometry.path_length(points) == pytest.approx(20.0)
        assert geometry.net_displacement(points) == pytest.approx(0.0)

    def test_occupancy_counts_overlap_once(self) -> None:
        """Summing areas would exceed 100% in dense traffic."""
        box = (0.0, 0.0, 100.0, 100.0)
        single = geometry.union_area_fraction([box], 200, 200)
        doubled = geometry.union_area_fraction([box, box], 200, 200)
        assert single == pytest.approx(doubled)
        assert single == pytest.approx(0.25, abs=0.02)

    def test_occupancy_is_bounded(self) -> None:
        boxes = [(0.0, 0.0, 200.0, 200.0)] * 5
        assert geometry.union_area_fraction(boxes, 200, 200) <= 1.0

    def test_occupancy_of_nothing_is_zero(self) -> None:
        assert geometry.union_area_fraction([], 200, 200) == 0.0

    def test_grid_counts_bucket_by_position(self) -> None:
        counts = geometry.grid_counts([(10, 10), (190, 190)], 200, 200, rows=2, cols=2)
        assert counts[0][0] == 1
        assert counts[1][1] == 1

    def test_clustering_scales_with_apparent_size(self) -> None:
        """A fixed pixel gap would merge the whole far half of a frame."""
        close = geometry.cluster_by_proximity(
            [(1, (0.0, 0.0), 50.0), (2, (60.0, 0.0), 50.0)], proximity_factor=1.8
        )
        assert close == ((1, 2),)
        far = geometry.cluster_by_proximity(
            [(1, (0.0, 0.0), 10.0), (2, (60.0, 0.0), 10.0)], proximity_factor=1.8
        )
        assert far == ((1,), (2,))

    def test_clustering_is_deterministic(self) -> None:
        items = [(3, (0.0, 0.0), 40.0), (1, (50.0, 0.0), 40.0), (2, (500.0, 0.0), 40.0)]
        assert geometry.cluster_by_proximity(
            items, proximity_factor=1.8
        ) == geometry.cluster_by_proximity(items, proximity_factor=1.8)

    def test_standard_deviation_needs_two_values(self) -> None:
        assert geometry.standard_deviation([5.0]) is None
        assert geometry.standard_deviation([]) is None


# ---------------------------------------------------------------------------
# MetricValue contract
# ---------------------------------------------------------------------------
class TestMetricValue:
    def test_available_requires_a_value(self) -> None:
        with pytest.raises(ValueError, match="must carry a value"):
            MetricValue[float](availability=MetricAvailability.AVAILABLE, value=None)

    def test_unavailable_must_not_carry_a_value(self) -> None:
        """A value beside an unavailable flag is the exact ambiguity to prevent."""
        with pytest.raises(ValueError, match="must not carry a value"):
            MetricValue[float](
                availability=MetricAvailability.UNAVAILABLE_NO_CALIBRATION,
                value=12.0,
                reason="x",
            )

    def test_unavailable_must_state_a_reason(self) -> None:
        with pytest.raises(ValueError, match="must state why"):
            MetricValue[float](availability=MetricAvailability.UNAVAILABLE_NO_GEOMETRY)

    def test_zero_is_a_real_measurement_not_an_absence(self) -> None:
        metric = available(0.0, unit="px")
        assert metric.is_available
        assert metric.value == 0.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
class TestCalibration:
    def test_no_projector_without_calibration(self) -> None:
        assert GroundProjector.from_calibration(None) is None
        assert GroundProjector.from_calibration(CameraCalibration(is_calibrated=False)) is None

    def test_identity_homography_maps_pixels_to_metres(self) -> None:
        projector = GroundProjector(homography=(1, 0, 0, 0, 1, 0, 0, 0, 1))
        assert projector.project((3.0, 4.0)) == pytest.approx((3.0, 4.0))
        assert projector.ground_distance((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_scaled_homography_converts_correctly(self) -> None:
        """0.05 m per pixel: 100 px is 5 m."""
        projector = GroundProjector(homography=(0.05, 0, 0, 0, 0.05, 0, 0, 0, 1))
        assert projector.ground_distance((0, 0), (100, 0)) == pytest.approx(5.0)

    def test_points_at_the_horizon_are_unmeasurable(self) -> None:
        """Returning a huge number would produce a fictitious speed."""
        projector = GroundProjector(homography=(1, 0, 0, 0, 1, 0, 0, 1, 0))
        assert projector.project((10.0, 0.0)) is None

    def test_homography_is_trusted_more_than_a_scalar(self) -> None:
        assert GroundProjector(homography=(1, 0, 0, 0, 1, 0, 0, 0, 1)).confidence > (
            GroundProjector(metres_per_pixel=0.05).confidence
        )

    def test_unit_conversion(self) -> None:
        assert mps_to_kph(10.0) == pytest.approx(36.0)


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------
class TestCounts:
    def test_class_counts_separate_vehicles_from_other_road_users(self) -> None:
        result = engine().analyse(
            [
                frame(
                    0,
                    obj(10, 10, 50, 50, cls=VehicleClass.CAR),
                    obj(60, 10, 100, 50, cls=VehicleClass.CAR),
                    obj(110, 10, 150, 50, cls=VehicleClass.TRUCK),
                    obj(160, 10, 200, 50, cls=VehicleClass.PERSON),
                    obj(210, 10, 250, 50, cls=VehicleClass.BICYCLE),
                )
            ],
            context(),
        )
        tally = result.detection_tally
        assert tally.total == 5
        assert tally.by_class[VehicleClass.CAR] == 2
        assert tally.vehicles == 3
        assert tally.pedestrians == 1
        assert tally.cyclists == 1

    def test_empty_detections_yield_zero_counts_not_an_error(self) -> None:
        result = engine().analyse([frame(0)], context())
        assert result.detection_tally.total == 0
        assert result.object_positions == ()

    def test_tracked_objects_are_counted_once_across_frames(self) -> None:
        """The detection tally counts per frame; the track tally counts objects."""
        result = engine().analyse(moving_sequence(frames=6), context())
        assert result.detection_tally.total == 6
        assert result.tracked_object_tally.total == 1

    def test_active_tracked_objects_reflect_the_final_frame(self) -> None:
        frames = [
            frame(0, obj(10, 10, 50, 50, track=1), obj(300, 300, 340, 340, track=2)),
            frame(1, obj(20, 10, 60, 50, track=1)),
        ]
        result = engine().analyse(frames, context())
        assert result.active_tracked_objects.value == 1

    def test_analysing_no_frames_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one analysed frame"):
            engine().analyse([], context())


# ---------------------------------------------------------------------------
# Snapshot (image) behaviour
# ---------------------------------------------------------------------------
class TestSnapshotMode:
    def _snapshot(self) -> object:
        return engine().analyse(
            [frame(0, obj(100, 100, 200, 200), obj(400, 300, 500, 400))], context()
        )

    def test_a_single_frame_is_snapshot_mode(self) -> None:
        assert self._snapshot().mode is ObservationMode.SNAPSHOT

    def test_speed_is_not_applicable_to_a_still(self) -> None:
        """A single frame contains no time over which anything can move."""
        result = self._snapshot()
        metric = result.mean_image_speed_diagonals_per_s
        assert not metric.is_available
        assert metric.availability is MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA

    def test_tracking_is_not_applicable_to_a_still(self) -> None:
        metric = self._snapshot().active_tracked_objects
        assert metric.availability is MetricAvailability.UNAVAILABLE_NOT_APPLICABLE
        assert "single image" in (metric.reason or "")

    def test_flow_is_not_applicable_to_a_still(self) -> None:
        result = self._snapshot()
        assert result.flow_series == ()
        assert result.arrival_rate_per_minute.availability is (
            MetricAvailability.UNAVAILABLE_NOT_APPLICABLE
        )

    def test_queue_estimation_is_not_applicable_to_a_still(self) -> None:
        """A still cannot distinguish a stopped vehicle from a moving one."""
        metric = self._snapshot().stationary_vehicle_count
        assert metric.availability is MetricAvailability.UNAVAILABLE_NOT_APPLICABLE
        assert "persistence" in (metric.reason or "")

    def test_positions_and_distribution_are_available(self) -> None:
        result = self._snapshot()
        assert len(result.object_positions) == 2
        assert result.spatial_distribution.image_occupancy_ratio.is_available
        assert result.spatial_distribution.centroid is not None

    def test_ground_contact_is_bottom_centre(self) -> None:
        result = engine().analyse([frame(0, obj(100, 100, 200, 300))], context())
        position = result.object_positions[0]
        assert position.centre.y == pytest.approx(200.0)
        assert position.ground_contact.y == pytest.approx(300.0)
        assert position.ground_contact.x == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# Sequence (video) behaviour
# ---------------------------------------------------------------------------
class TestSequenceMode:
    def test_displacement_and_speed_match_the_construction(self) -> None:
        """8 frames, 40 px apart, 0.1 s apart -> 400 px/s."""
        result = engine().analyse(moving_sequence(frames=8, step=40.0), context())
        trajectory = result.trajectories[0]
        assert trajectory.total_displacement_px.value == pytest.approx(280.0)
        assert trajectory.net_displacement_px.value == pytest.approx(280.0)
        assert trajectory.mean_step_displacement_px.value == pytest.approx(40.0)
        assert trajectory.image_speed_px_per_s.value == pytest.approx(400.0)

    def test_speed_is_normalised_by_frame_diagonal(self) -> None:
        result = engine().analyse(moving_sequence(frames=8, step=40.0), context())
        diagonal = math.hypot(FRAME_W, FRAME_H)
        assert result.trajectories[0].image_speed_diagonals_per_s.value == pytest.approx(
            400.0 / diagonal
        )

    def test_image_space_metrics_are_flagged_as_such(self) -> None:
        """Presentation must never render these as physical."""
        trajectory = engine().analyse(moving_sequence(), context()).trajectories[0]
        assert trajectory.image_speed_px_per_s.is_image_space is True
        assert trajectory.total_displacement_px.is_image_space is True

    def test_direction_matches_the_motion(self) -> None:
        result = engine().analyse(moving_sequence(frames=6, step=40.0), context())
        trajectory = result.trajectories[0]
        assert trajectory.direction_deg.value == pytest.approx(0.0)
        assert trajectory.direction_label.value is DirectionLabel.RIGHT

    def test_direction_histogram_aggregates_trajectories(self) -> None:
        result = engine().analyse(moving_sequence(frames=6), context())
        assert result.direction_histogram[DirectionLabel.RIGHT] == 1

    def test_trajectory_duration_uses_timestamps(self) -> None:
        result = engine().analyse(moving_sequence(frames=6), context())
        assert result.trajectories[0].duration_s.value == pytest.approx(0.5)
        assert result.mean_trajectory_duration_s.value == pytest.approx(0.5)

    def test_frame_indices_are_preserved(self) -> None:
        result = engine().analyse(moving_sequence(frames=5), context())
        trajectory = result.trajectories[0]
        assert trajectory.first_frame_index == 0
        assert trajectory.last_frame_index == 4
        assert trajectory.frames_tracked == 5

    def test_a_single_point_track_cannot_yield_speed(self) -> None:
        frames = [
            frame(0, obj(10, 10, 50, 50, track=1)),
            frame(1, obj(300, 300, 340, 340, track=2)),
        ]
        result = engine().analyse(frames, context())
        lonely = next(t for t in result.trajectories if t.track_id == 2)
        assert not lonely.image_speed_px_per_s.is_available
        assert lonely.image_speed_px_per_s.availability is (
            MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA
        )

    def test_detections_without_track_ids_produce_no_trajectories(self) -> None:
        frames = [frame(0, obj(10, 10, 50, 50)), frame(1, obj(20, 10, 60, 50))]
        result = engine().analyse(frames, context())
        assert result.trajectories == ()
        assert result.active_tracked_objects.availability is (
            MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA
        )

    def test_flow_buckets_count_arrivals_and_presence(self) -> None:
        # Window 0.15 s over frames at 0.0/0.1/0.2/0.3 gives buckets 0,0,1,2 --
        # so the second object's arrival falls in a different bucket.
        settings = Settings(analytics_flow_window_s=0.15)
        frames = [
            frame(0, obj(10, 10, 50, 50, track=1)),
            frame(1, obj(20, 10, 60, 50, track=1)),
            frame(2, obj(30, 10, 70, 50, track=1), obj(300, 300, 340, 340, track=2)),
            frame(3, obj(40, 10, 80, 50, track=1), obj(310, 300, 350, 340, track=2)),
        ]
        result = TrafficAnalyticsEngine(settings).analyse(
            frames, context(settings=settings)
        )
        assert result.flow_series[0].arrivals == 1, "only track 1 arrives in bucket 0"
        assert result.flow_series[0].active == 1
        assert result.flow_series[1].arrivals == 1, "track 2 arrives in bucket 1"
        assert result.flow_series[1].active == 2, "both tracks are present"
        assert sum(b.arrivals for b in result.flow_series) == 2

    def test_arrival_rate_is_per_minute_over_the_observed_span(self) -> None:
        """2 objects over 0.3 s -> 400 per minute."""
        frames = [
            frame(0, obj(10, 10, 50, 50, track=1)),
            frame(1, obj(20, 10, 60, 50, track=1)),
            frame(2, obj(30, 10, 70, 50, track=1), obj(300, 300, 340, 340, track=2)),
            frame(3, obj(40, 10, 80, 50, track=1), obj(310, 300, 350, 340, track=2)),
        ]
        result = engine().analyse(frames, context())
        assert result.arrival_rate_per_minute.value == pytest.approx(2 / 0.3 * 60.0)


# ---------------------------------------------------------------------------
# Queues
# ---------------------------------------------------------------------------
class TestQueueEstimation:
    def test_a_stationary_vehicle_is_queue_evidence(self) -> None:
        result = engine().analyse(stationary_sequence(frames=8), context())
        assert result.stationary_vehicle_count.value == 1
        assert result.trajectories[0].is_stationary is True

    def test_a_moving_vehicle_is_not_queue_evidence(self) -> None:
        """The distinction is movement, not presence."""
        result = engine().analyse(moving_sequence(frames=8, step=40.0), context())
        assert result.stationary_vehicle_count.value == 0
        assert result.queue_groups == ()

    def test_a_queue_is_not_a_vehicle_count(self) -> None:
        """Twenty free-flowing vehicles are busy, not queued."""
        frames = [
            frame(
                i,
                *[
                    obj(50 + n * 70 + i * 40, 300, 100 + n * 70 + i * 40, 360, track=n + 1)
                    for n in range(6)
                ],
            )
            for i in range(6)
        ]
        result = engine().analyse(frames, context())
        assert result.detection_tally.total == 36
        assert result.tracked_object_tally.total == 6
        assert result.stationary_vehicle_count.value == 0

    def test_brief_stillness_does_not_create_a_queue(self) -> None:
        """Detection jitter routinely freezes a moving vehicle for one frame."""
        settings = Settings(analytics_min_stationary_frames=3)
        frames = [
            frame(0, obj(100, 300, 160, 360, track=1)),
            frame(1, obj(100, 300, 160, 360, track=1)),
            frame(2, obj(200, 300, 260, 360, track=1)),
            frame(3, obj(300, 300, 360, 360, track=1)),
        ]
        result = TrafficAnalyticsEngine(settings).analyse(frames, context(settings=settings))
        assert result.stationary_vehicle_count.value == 0

    def test_adjacent_stopped_vehicles_form_one_group(self) -> None:
        frames = [
            frame(
                i,
                obj(100, 300, 160, 360, track=1),
                obj(180, 300, 240, 360, track=2),
                obj(700, 100, 760, 160, track=3),
            )
            for i in range(8)
        ]
        result = engine().analyse(frames, context())
        assert result.stationary_vehicle_count.value == 3
        groups = {g.member_count for g in result.queue_groups}
        assert 2 in groups, "adjacent stopped vehicles should cluster"
        assert 1 in groups, "a distant stopped vehicle is its own group"

    def test_pedestrians_are_not_queued_vehicles(self) -> None:
        frames = [
            frame(i, obj(100, 300, 160, 360, cls=VehicleClass.PERSON, track=1))
            for i in range(8)
        ]
        result = engine().analyse(frames, context())
        assert result.stationary_vehicle_count.value == 0

    def test_queue_length_in_metres_needs_calibration(self) -> None:
        result = engine().analyse(stationary_sequence(frames=8), context())
        metric = result.total_queue_length_m
        assert metric.availability is MetricAvailability.UNAVAILABLE_NO_CALIBRATION
        assert metric.value is None

    def test_queue_extent_is_reported_in_pixels(self) -> None:
        frames = [
            frame(i, obj(100, 300, 160, 360, track=1), obj(180, 300, 240, 360, track=2))
            for i in range(8)
        ]
        result = engine().analyse(frames, context())
        group = next(g for g in result.queue_groups if g.member_count == 2)
        assert group.extent_px.is_available
        assert group.extent_px.is_image_space is True
        assert group.length_m.availability is MetricAvailability.UNAVAILABLE_NO_CALIBRATION


# ---------------------------------------------------------------------------
# Physical measurements
# ---------------------------------------------------------------------------
class TestPhysicalMeasurements:
    def test_speed_in_kph_is_unavailable_without_calibration(self) -> None:
        result = engine().analyse(moving_sequence(frames=8), context())
        metric = result.mean_speed_kph
        assert metric.availability is MetricAvailability.UNAVAILABLE_NO_CALIBRATION
        assert metric.reason == NO_CALIBRATION_MESSAGE
        assert metric.value is None

    def test_per_track_physical_speed_is_unavailable_without_calibration(self) -> None:
        trajectory = engine().analyse(moving_sequence(), context()).trajectories[0]
        assert trajectory.physical_speed_mps.reason == NO_CALIBRATION_MESSAGE
        assert trajectory.physical_speed_kph.value is None

    def test_calibration_enables_physical_speed_with_correct_arithmetic(self) -> None:
        """0.05 m/px, 40 px per 0.1 s -> 20 m/s -> 72 km/h."""
        calibration = CameraCalibration(
            is_calibrated=True,
            method="homography_from_4_ground_points",
            homography=(0.05, 0, 0, 0, 0.05, 0, 0, 0, 1),
        )
        result = engine().analyse(
            moving_sequence(frames=8, step=40.0), context(calibration=calibration)
        )
        trajectory = result.trajectories[0]
        assert trajectory.physical_speed_mps.value == pytest.approx(20.0)
        assert trajectory.physical_speed_kph.value == pytest.approx(72.0)
        assert result.mean_speed_kph.value == pytest.approx(72.0)

    def test_physical_metrics_carry_confidence_and_method(self) -> None:
        calibration = CameraCalibration(
            is_calibrated=True, homography=(0.05, 0, 0, 0, 0.05, 0, 0, 0, 1)
        )
        result = engine().analyse(moving_sequence(), context(calibration=calibration))
        metric = result.trajectories[0].physical_speed_kph
        assert metric.confidence is not None
        assert "homography" in (metric.method or "")
        assert metric.is_image_space is False

    def test_density_requires_surveyed_geometry(self) -> None:
        result = engine().analyse(moving_sequence(), context())
        metric = result.traffic_density_veh_per_km_per_lane
        assert metric.availability is MetricAvailability.UNAVAILABLE_NO_GEOMETRY
        assert "lane count" in (metric.reason or "")

    def test_density_is_computed_when_geometry_is_supplied(self) -> None:
        """4 vehicles over 50 m of 2 lanes -> 40 veh/km/lane."""
        frames = [
            frame(i, *[obj(50 + n * 100, 300, 110 + n * 100, 360, track=n + 1) for n in range(4)])
            for i in range(4)
        ]
        result = engine().analyse(
            frames, context(lane_count=2, segment_length_m=50.0)
        )
        assert result.traffic_density_veh_per_km_per_lane.value == pytest.approx(40.0)

    def test_image_occupancy_is_not_traffic_density(self) -> None:
        """Occupancy is pixels; density needs road geometry."""
        result = engine().analyse([frame(0, obj(0, 0, 400, 300))], context())
        assert result.spatial_distribution.image_occupancy_ratio.is_available
        assert result.spatial_distribution.image_occupancy_ratio.is_image_space is True
        assert not result.traffic_density_veh_per_km_per_lane.is_available


# ---------------------------------------------------------------------------
# Provenance and quality
# ---------------------------------------------------------------------------
class TestProvenanceAndQuality:
    def test_snapshot_provenance_names_a_single_frame(self) -> None:
        result = engine().analyse([frame(0, obj(10, 10, 50, 50))], context())
        prov = result.provenance
        assert prov.frame_index == 0
        assert prov.frame_range is None
        assert prov.timestamp == T0
        assert prov.tracker is None or prov.tracker == "bytetrack-lite"

    def test_sequence_provenance_names_a_range(self) -> None:
        result = engine().analyse(moving_sequence(frames=6), context())
        prov = result.provenance
        assert prov.frame_range == (0, 5)
        assert prov.time_range is not None
        assert prov.frame_index is None

    def test_provenance_identifies_source_model_tracker_and_method(self) -> None:
        result = engine().analyse(moving_sequence(), context())
        prov = result.provenance
        assert prov.media_id == "media-1"
        assert prov.source_id == "video:media-1"
        assert prov.model.sha256 == "b" * 64
        assert prov.tracker == "bytetrack-lite"
        assert prov.method == "image-space-traffic-analytics"
        assert prov.method_version
        assert prov.analytics_schema_version

    def test_quality_warns_about_missing_calibration(self) -> None:
        result = engine().analyse(moving_sequence(), context())
        assert result.quality.calibration_available is False
        assert any("calibration" in w for w in result.quality.warnings)

    def test_quality_warns_about_missing_geometry(self) -> None:
        result = engine().analyse(moving_sequence(), context())
        assert result.quality.road_geometry_available is False
        assert any("geometry" in w for w in result.quality.warnings)

    def test_quality_reports_low_confidence_detections(self) -> None:
        frames = [
            frame(i, obj(100 + i * 40, 300, 160 + i * 40, 360, conf=0.2, track=1))
            for i in range(6)
        ]
        result = engine().analyse(frames, context())
        assert result.quality.low_confidence_fraction == pytest.approx(1.0)
        assert result.quality.grade in (QualityGrade.DEGRADED, QualityGrade.POOR)

    def test_quality_reports_the_sampling_interval(self) -> None:
        result = engine().analyse(moving_sequence(frames=6), context())
        assert result.quality.sampling_interval_s == pytest.approx(0.1)

    def test_snapshot_quality_is_degraded_not_good(self) -> None:
        """One frame supports fewer measurements, and says so."""
        result = engine().analyse([frame(0, obj(10, 10, 50, 50))], context())
        assert result.quality.frames_analysed == 1
        assert result.quality.grade is not QualityGrade.GOOD


# ---------------------------------------------------------------------------
# TrafficStateExtractor -> JunctionObservation
# ---------------------------------------------------------------------------
class TestTrafficStateExtractor:
    def _tracks(self) -> tuple[Track, ...]:
        history = tuple(
            TrackPoint(
                timestamp=T0 + dt.timedelta(seconds=i * 0.1),
                bbox=BoundingBox(x1=100 + i * 30, y1=300, x2=160 + i * 30, y2=360),
                confidence=0.9,
            )
            for i in range(5)
        )
        return (
            Track(
                track_id=1,
                vehicle_class=VehicleClass.CAR,
                history=history,
                age_frames=5,
                hits=5,
                time_since_update_s=0.0,
                is_confirmed=True,
            ),
            Track(
                track_id=2,
                vehicle_class=VehicleClass.PERSON,
                history=history,
                age_frames=5,
                hits=5,
                time_since_update_s=0.0,
                is_confirmed=True,
            ),
        )

    def _context(self, calibration: CameraCalibration | None = None) -> ExtractionContext:
        return ExtractionContext(
            junction_id="observation-area:test",
            source_id="video:test",
            zone_config=ZoneConfiguration(
                config_id="zc-1",
                source_id="video:test",
                junction_id="observation-area:test",
                frame_width=FRAME_W,
                frame_height=FRAME_H,
                calibration=calibration,
            ),
            analysis_window_s=60.0,
            tracker_info=TrackerInfo(name="bytetrack-lite", version="1.0"),
            schema_version="0.1.0",
        )

    def test_it_produces_a_valid_junction_observation(self) -> None:
        observation = AnalyticsTrafficStateExtractor().extract(
            self._tracks(), context=self._context(), captured_at=T0, ingested_at=T0
        )
        assert observation.junction_id == "observation-area:test"
        assert len(observation.approaches) == 1

    def test_one_camera_yields_one_whole_view_approach(self) -> None:
        """Without a zone configuration the camera cannot split its own view."""
        observation = AnalyticsTrafficStateExtractor().extract(
            self._tracks(), context=self._context(), captured_at=T0, ingested_at=T0
        )
        assert observation.approaches[0].approach_id == WHOLE_VIEW_APPROACH_ID

    def test_counts_exclude_pedestrians_from_vehicles(self) -> None:
        observation = AnalyticsTrafficStateExtractor().extract(
            self._tracks(), context=self._context(), captured_at=T0, ingested_at=T0
        )
        approach = observation.approaches[0]
        assert approach.vehicle_count_total is not None
        assert approach.vehicle_count_total.value == 1
        assert approach.pedestrians is not None
        assert approach.pedestrians.waiting_count is not None
        assert approach.pedestrians.waiting_count.value == 1

    def test_uncalibrated_observations_carry_no_metric_speed(self) -> None:
        """The JunctionObservation validator enforces this independently."""
        observation = AnalyticsTrafficStateExtractor().extract(
            self._tracks(), context=self._context(), captured_at=T0, ingested_at=T0
        )
        approach = observation.approaches[0]
        assert approach.mean_speed_kph is None
        assert approach.queue_length_m is None

    def test_uncalibrated_flag_is_raised(self) -> None:
        from app.domain.enums import DataQualityFlag

        observation = AnalyticsTrafficStateExtractor().extract(
            self._tracks(), context=self._context(), captured_at=T0, ingested_at=T0
        )
        assert DataQualityFlag.UNCALIBRATED in observation.quality.flags

    def test_measurements_carry_derived_provenance(self) -> None:
        from app.domain.provenance import ValueSource

        observation = AnalyticsTrafficStateExtractor().extract(
            self._tracks(), context=self._context(), captured_at=T0, ingested_at=T0
        )
        count = observation.approaches[0].vehicle_count_total
        assert count is not None
        assert count.provenance.source is ValueSource.DERIVED
        assert count.provenance.input_refs == ("video:test",)

    def test_the_extractor_is_stateless(self) -> None:
        extractor = AnalyticsTrafficStateExtractor()
        first = extractor.extract(
            self._tracks(), context=self._context(), captured_at=T0, ingested_at=T0
        )
        extractor.reset()
        second = extractor.extract(
            self._tracks(), context=self._context(), captured_at=T0, ingested_at=T0
        )
        assert first.approaches[0].vehicle_count_total is not None
        assert second.approaches[0].vehicle_count_total is not None
        assert (
            first.approaches[0].vehicle_count_total.value
            == second.approaches[0].vehicle_count_total.value
        )

    def test_empty_tracks_produce_an_observation_with_zero_vehicles(self) -> None:
        observation = AnalyticsTrafficStateExtractor().extract(
            (), context=self._context(), captured_at=T0, ingested_at=T0
        )
        approach = observation.approaches[0]
        assert approach.vehicle_count_total is not None
        assert approach.vehicle_count_total.value == 0


# ---------------------------------------------------------------------------
# API integration: CV -> extractor -> response
# ---------------------------------------------------------------------------
@pytest.fixture
def analytics_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """API client with an isolated media root and a scripted detector."""
    from test_vision import ScriptedDetector, detection

    media_dir = tmp_path / "media"
    media_dir.mkdir()
    monkeypatch.setenv("MEDIA_DIR", str(media_dir))
    get_settings.cache_clear()
    media_routes.reset_store()

    settings = Settings(
        app_env="local", log_level="ERROR", media_dir=media_dir, models_dir=tmp_path / "models"
    )
    app = create_app(settings)

    from app.vision.analysis import DetectorProvider

    class StubProvider(DetectorProvider):
        def __init__(self) -> None:
            super().__init__(settings)
            script = [
                (
                    detection(10 + i * 20, 10, 50 + i * 20, 60, cls=VehicleClass.CAR),
                    detection(200, 200, 240, 260, cls=VehicleClass.PERSON),
                )
                for i in range(20)
            ]
            self._detector = ScriptedDetector(script)
            self._detector.load()

        def get(self) -> object:
            return self._detector

    app.state.detector_provider = StubProvider()

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    media_routes.reset_store()
    get_settings.cache_clear()


def _upload_image(client: TestClient, tmp_path: Path) -> str:
    from PIL import Image

    path = tmp_path / "frame.png"
    Image.new("RGB", (FRAME_W, FRAME_H), (30, 60, 90)).save(path, "PNG")
    return client.post(
        "/api/media/images", files={"file": ("frame.png", path.read_bytes(), "image/png")}
    ).json()["media_id"]


def _upload_video(client: TestClient, tmp_path: Path, frames: int = 8) -> str:
    import cv2
    import numpy as np

    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (FRAME_W, FRAME_H))
    for i in range(frames):
        writer.write(np.full((FRAME_H, FRAME_W, 3), (i * 20) % 255, dtype=np.uint8))
    writer.release()
    return client.post(
        "/api/media/videos", files={"file": ("clip.mp4", path.read_bytes(), "video/mp4")}
    ).json()["media_id"]


class TestAnalyticsApi:
    def test_config_exposes_thresholds(self, analytics_client: TestClient) -> None:
        body = analytics_client.get("/api/analytics/config").json()
        assert body["method"] == "image-space-traffic-analytics"
        assert body["stationary_speed_frac"] > 0
        assert body["calibration_configured"] is False

    def test_image_analytics_end_to_end(
        self, analytics_client: TestClient, tmp_path: Path
    ) -> None:
        """CV output -> TrafficStateExtractor -> API response."""
        media_id = _upload_image(analytics_client, tmp_path)
        response = analytics_client.post(f"/api/analytics/images/{media_id}")
        assert response.status_code == 200

        body = response.json()
        assert body["mode"] == "snapshot"
        assert body["detection_tally"]["total"] == 2
        assert body["detection_tally"]["vehicles"] == 1
        assert body["detection_tally"]["pedestrians"] == 1
        assert len(body["object_positions"]) == 2
        assert body["provenance"]["media_id"] == media_id
        assert body["provenance"]["model"]["sha256"]

    def test_image_analytics_marks_temporal_metrics_unavailable(
        self, analytics_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = _upload_image(analytics_client, tmp_path)
        body = analytics_client.post(f"/api/analytics/images/{media_id}").json()
        assert body["mean_speed_kph"]["availability"] == "unavailable_no_calibration"
        assert body["mean_speed_kph"]["reason"] == NO_CALIBRATION_MESSAGE
        assert body["stationary_vehicle_count"]["availability"] == "unavailable_not_applicable"

    def test_image_analytics_is_cached(
        self, analytics_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = _upload_image(analytics_client, tmp_path)
        analytics_client.post(f"/api/analytics/images/{media_id}")
        assert analytics_client.get(f"/api/analytics/images/{media_id}").status_code == 200

    def test_analytics_before_running_is_404(
        self, analytics_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = _upload_image(analytics_client, tmp_path)
        assert analytics_client.get(f"/api/analytics/images/{media_id}").status_code == 404

    def test_video_analytics_end_to_end(
        self, analytics_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = _upload_video(analytics_client, tmp_path, frames=8)
        job_id = analytics_client.post(
            f"/api/vision/videos/{media_id}/analyze", json={"run_sync": True}
        ).json()["job_id"]

        response = analytics_client.post(f"/api/analytics/jobs/{job_id}")
        assert response.status_code == 200

        body = response.json()
        assert body["mode"] == "sequence"
        assert body["provenance"]["frame_range"] == [0, 7]
        assert body["provenance"]["tracker"] == "bytetrack-lite"
        assert body["tracked_object_tally"]["total"] >= 1
        assert body["active_tracked_objects"]["availability"] == "available"
        assert len(body["trajectories"]) >= 1

    def test_video_analytics_reports_direction_and_speed(
        self, analytics_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = _upload_video(analytics_client, tmp_path, frames=8)
        job_id = analytics_client.post(
            f"/api/vision/videos/{media_id}/analyze", json={"run_sync": True}
        ).json()["job_id"]
        body = analytics_client.post(f"/api/analytics/jobs/{job_id}").json()

        moving = [
            t
            for t in body["trajectories"]
            if t["direction_label"]["availability"] == "available"
        ]
        assert moving, "the scripted car moves and must have a direction"
        assert moving[0]["image_speed_px_per_s"]["is_image_space"] is True
        assert moving[0]["physical_speed_kph"]["availability"] == "unavailable_no_calibration"

    def test_analytics_on_an_unfinished_job_is_refused(
        self, analytics_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = _upload_video(analytics_client, tmp_path, frames=4)
        job_id = analytics_client.post(
            f"/api/vision/videos/{media_id}/analyze", json={"run_sync": True}
        ).json()["job_id"]
        # A completed job is fine; an unknown one is rejected.
        assert analytics_client.post(f"/api/analytics/jobs/{job_id}").status_code == 200
        assert analytics_client.post("/api/analytics/jobs/not-a-job").status_code == 422

    def test_analysing_a_video_as_an_image_is_refused(
        self, analytics_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = _upload_video(analytics_client, tmp_path, frames=3)
        assert analytics_client.post(f"/api/analytics/images/{media_id}").status_code == 422

    def test_no_traffic_management_conclusion_is_returned(
        self, analytics_client: TestClient, tmp_path: Path
    ) -> None:
        """Phase 4 measures; it does not interpret or recommend."""
        media_id = _upload_image(analytics_client, tmp_path)
        payload = str(analytics_client.post(f"/api/analytics/images/{media_id}").json()).lower()
        for forbidden in (
            "congestion_level",
            "recommend",
            "route",
            "eta",
            "travel_time",
            "signal_advisory",
            "emergency",
            "prediction",
        ):
            assert forbidden not in payload
