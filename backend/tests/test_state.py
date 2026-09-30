"""Phase 5 traffic state tests: congestion interpretation and fusion.

Analytics inputs are **built directly** rather than produced by running the
Phase 4 engine. Interpretation is a pure function of a measurement set, so
feeding it a hand-written set with known values is what makes a band assertion
an arithmetic check rather than a restatement of whatever the detector happened
to find.

The properties under test are mostly about refusal. A verdict must be refused
when the evidence does not support it, an unavailable signal must contribute
nothing rather than zero, and an ``UNKNOWN`` must never be orderable against a
real band.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.analytics.schemas import (
    AnalyticsProvenance,
    ClassTallyOut,
    MeasurementQuality,
    MetricAvailability,
    MetricValue,
    ObservationMode,
    QualityGrade,
    SpatialDistribution,
    TrafficAnalytics,
    available,
    unavailable,
)
from app.core.config import Settings
from app.domain.enums import CongestionLevel, FreshnessStatus, VehicleClass
from app.state.assembler import assemble, freshness_for, interpret, observed_at, policy_from
from app.state.classifier import band_for, classify, thresholds_from, vehicle_count_for
from app.state.schemas import (
    CongestionAssessment,
    CongestionSignal,
    CongestionThresholds,
    EvidenceBasis,
    IntersectionTrafficState,
    SignalName,
)
from app.vision.models import ModelIdentity

T0 = dt.datetime(2024, 3, 14, 8, 30, tzinfo=dt.UTC)
MODEL = ModelIdentity(name="test-model", backend="test", sha256="c" * 64, size_bytes=1)

NO_TIME = "A single frame carries no time."
NO_CALIB = "Physical speed unavailable - camera calibration not configured."


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def make_provenance(
    *,
    area: str = "observation-area:test",
    timestamp: dt.datetime | None = T0,
    time_range: tuple[dt.datetime, dt.datetime] | None = None,
) -> AnalyticsProvenance:
    """Provenance for a synthetic measurement set."""
    return AnalyticsProvenance(
        media_id="media-1",
        source_id=area,
        timestamp=timestamp,
        time_range=time_range,
        model=MODEL,
        tracker="test-tracker",
        method="test-method",
        method_version="1.0",
        analytics_schema_version="1.0",
        computed_at=T0,
    )


def make_analytics(
    *,
    area: str = "observation-area:test",
    mode: ObservationMode = ObservationMode.SEQUENCE,
    grade: QualityGrade = QualityGrade.GOOD,
    tracked_vehicles: int = 10,
    detected_vehicles: int = 10,
    occupancy: MetricValue[float] | None = None,
    image_speed: MetricValue[float] | None = None,
    stationary: MetricValue[int] | None = None,
    density: MetricValue[float] | None = None,
    mean_speed_kph: MetricValue[float] | None = None,
    timestamp: dt.datetime | None = T0,
    time_range: tuple[dt.datetime, dt.datetime] | None = None,
) -> TrafficAnalytics:
    """A measurement set with exactly the signals a test wants to control."""
    unavailable_speed: MetricValue[float] = unavailable(
        MetricAvailability.UNAVAILABLE_NO_CALIBRATION, NO_CALIB, unit="km/h"
    )
    unavailable_density: MetricValue[float] = unavailable(
        MetricAvailability.UNAVAILABLE_NO_GEOMETRY, "No lane count supplied.", unit="veh/km/lane"
    )
    unavailable_float: MetricValue[float] = unavailable(
        MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA, "Not enough data."
    )
    unavailable_int: MetricValue[int] = unavailable(
        MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA, "Not enough data."
    )

    return TrafficAnalytics(
        analytics_id="analytics-1",
        mode=mode,
        observation_area_id=area,
        provenance=make_provenance(area=area, timestamp=timestamp, time_range=time_range),
        quality=MeasurementQuality(
            grade=grade,
            frames_analysed=30,
            calibration_available=density is not None or mean_speed_kph is not None,
        ),
        frame_width=800,
        frame_height=600,
        detection_tally=ClassTallyOut(
            by_class={VehicleClass.CAR: detected_vehicles},
            total=detected_vehicles,
            vehicles=detected_vehicles,
        ),
        tracked_object_tally=ClassTallyOut(
            by_class={VehicleClass.CAR: tracked_vehicles},
            total=tracked_vehicles,
            vehicles=tracked_vehicles,
        ),
        active_tracked_objects=available(tracked_vehicles, unit="vehicles"),
        spatial_distribution=SpatialDistribution(
            spread_x_norm=unavailable_float,
            spread_y_norm=unavailable_float,
            grid_rows=3,
            grid_cols=3,
            grid_counts=((0, 0, 0), (0, 0, 0), (0, 0, 0)),
            image_occupancy_ratio=(
                occupancy
                if occupancy is not None
                else unavailable(
                    MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA, "No boxes."
                )
            ),
        ),
        mean_image_speed_diagonals_per_s=(
            image_speed if image_speed is not None else unavailable_float
        ),
        mean_trajectory_duration_s=unavailable_float,
        arrival_rate_per_minute=unavailable_float,
        stationary_vehicle_count=(stationary if stationary is not None else unavailable_int),
        total_queue_length_m=unavailable_float,
        traffic_density_veh_per_km_per_lane=(
            density if density is not None else unavailable_density
        ),
        mean_speed_kph=(mean_speed_kph if mean_speed_kph is not None else unavailable_speed),
    )


@pytest.fixture
def state_settings() -> Settings:
    """Deterministic settings with the default congestion policy."""
    return Settings(
        app_env="local",
        log_level="WARNING",
        max_observation_staleness_s=5.0,
        drop_observation_after_s=30.0,
    )


# ---------------------------------------------------------------------------
# Band boundaries
# ---------------------------------------------------------------------------
class TestBandBoundaries:
    """Cut points are lower bounds, so a value exactly on one takes the higher band."""

    @pytest.mark.parametrize(
        ("index", "expected"),
        [
            (0.0, CongestionLevel.FREE_FLOW),
            (0.1999, CongestionLevel.FREE_FLOW),
            (0.20, CongestionLevel.LIGHT),
            (0.3999, CongestionLevel.LIGHT),
            (0.40, CongestionLevel.MODERATE),
            (0.5999, CongestionLevel.MODERATE),
            (0.60, CongestionLevel.HEAVY),
            (0.7799, CongestionLevel.HEAVY),
            (0.78, CongestionLevel.SEVERE),
            (0.9199, CongestionLevel.SEVERE),
            (0.92, CongestionLevel.GRIDLOCK),
            (1.0, CongestionLevel.GRIDLOCK),
        ],
    )
    def test_each_cut_point_and_the_value_below_it(
        self, index: float, expected: CongestionLevel, state_settings: Settings
    ) -> None:
        assert band_for(index, thresholds_from(state_settings)) is expected

    def test_a_band_is_never_unknown(self, state_settings: Settings) -> None:
        """UNKNOWN is a refusal, not a band. Banding must never produce it."""
        thresholds = thresholds_from(state_settings)
        for step in range(0, 101):
            assert band_for(step / 100, thresholds) is not CongestionLevel.UNKNOWN

    def test_reconfigured_cut_points_move_the_band(self) -> None:
        """Thresholds are policy: changing configuration must change the verdict."""
        strict = Settings(
            app_env="local",
            congestion_index_light=0.05,
            congestion_index_moderate=0.10,
            congestion_index_heavy=0.15,
            congestion_index_severe=0.20,
            congestion_index_gridlock=0.25,
        )
        assert band_for(0.30, thresholds_from(strict)) is CongestionLevel.GRIDLOCK
        assert band_for(0.30, thresholds_from(Settings(app_env="local"))) is (
            CongestionLevel.LIGHT
        )


# ---------------------------------------------------------------------------
# Index computation
# ---------------------------------------------------------------------------
class TestIndexComputation:
    def test_a_single_signal_gives_its_own_score_as_the_index(
        self, state_settings: Settings
    ) -> None:
        """One available signal, weighted against itself, is that signal's score."""
        analytics = make_analytics(
            mode=ObservationMode.SNAPSHOT,
            detected_vehicles=8,
            occupancy=available(0.175, unit="fraction", image_space=True),
        )
        assessment = classify(analytics, state_settings)

        # 0.175 / 0.35 reference = 0.5
        assert assessment.index == pytest.approx(0.5)
        assert assessment.level is CongestionLevel.MODERATE
        assert assessment.basis is EvidenceBasis.IMAGE_SPACE

    def test_a_score_above_its_reference_clamps_rather_than_exceeding_one(
        self, state_settings: Settings
    ) -> None:
        analytics = make_analytics(
            mode=ObservationMode.SNAPSHOT,
            detected_vehicles=40,
            occupancy=available(0.95, unit="fraction", image_space=True),
        )
        assessment = classify(analytics, state_settings)
        assert assessment.index == pytest.approx(1.0)
        assert assessment.level is CongestionLevel.GRIDLOCK

    def test_two_signals_combine_as_a_weighted_mean(self, state_settings: Settings) -> None:
        analytics = make_analytics(
            tracked_vehicles=10,
            occupancy=available(0.35, unit="fraction", image_space=True),
            stationary=available(4, unit="vehicles"),
        )
        assessment = classify(analytics, state_settings)

        # occupancy score 1.0 at weight 0.40; stationary 0.4/0.8 = 0.5 at weight 0.35
        expected = (1.0 * 0.40 + 0.5 * 0.35) / (0.40 + 0.35)
        assert assessment.index == pytest.approx(expected, abs=1e-4)

    def test_an_unavailable_signal_contributes_nothing_rather_than_zero(
        self, state_settings: Settings
    ) -> None:
        """The property the whole design turns on.

        Scoring a missing measurement as 0.0 would drag every uncalibrated
        camera toward free flow, which is the one direction it is dangerous to
        be wrong in.
        """
        occupancy_only = classify(
            make_analytics(
                mode=ObservationMode.SNAPSHOT,
                detected_vehicles=8,
                occupancy=available(0.35, unit="fraction", image_space=True),
            ),
            state_settings,
        )
        assert occupancy_only.index == pytest.approx(1.0)
        assert occupancy_only.level is CongestionLevel.GRIDLOCK

    def test_a_missing_signal_lowers_confidence_without_moving_the_index(
        self, state_settings: Settings
    ) -> None:
        partial = classify(
            make_analytics(
                mode=ObservationMode.SNAPSHOT,
                detected_vehicles=8,
                occupancy=available(0.175, unit="fraction", image_space=True),
            ),
            state_settings,
        )
        complete = classify(
            make_analytics(
                tracked_vehicles=8,
                occupancy=available(0.175, unit="fraction", image_space=True),
                image_speed=available(0.04, unit="diagonals/s", image_space=True),
                stationary=available(4, unit="vehicles"),
            ),
            state_settings,
        )
        assert partial.confidence is not None
        assert complete.confidence is not None
        assert complete.confidence > partial.confidence

    def test_classification_is_deterministic(self, state_settings: Settings) -> None:
        analytics = make_analytics(
            tracked_vehicles=12,
            occupancy=available(0.2, unit="fraction", image_space=True),
            stationary=available(6, unit="vehicles"),
        )
        first = classify(analytics, state_settings)
        second = classify(analytics, state_settings)
        assert first.model_dump() == second.model_dump()

    def test_every_signal_is_listed_including_the_unavailable_ones(
        self, state_settings: Settings
    ) -> None:
        """An assessment resting on two signals out of three must show the third."""
        assessment = classify(
            make_analytics(
                tracked_vehicles=10,
                occupancy=available(0.1, unit="fraction", image_space=True),
                stationary=available(2, unit="vehicles"),
            ),
            state_settings,
        )
        names = {signal.name for signal in assessment.signals}
        assert names == {
            SignalName.IMAGE_OCCUPANCY,
            SignalName.IMAGE_SPEED,
            SignalName.STATIONARY_FRACTION,
        }
        missing = next(s for s in assessment.signals if s.name is SignalName.IMAGE_SPEED)
        assert missing.availability is not MetricAvailability.AVAILABLE
        assert missing.reason
        assert missing.weight == 0.0


# ---------------------------------------------------------------------------
# Physical versus image-space evidence
# ---------------------------------------------------------------------------
class TestEvidenceBasis:
    def test_calibrated_measurements_produce_a_physical_basis(
        self, state_settings: Settings
    ) -> None:
        assessment = classify(
            make_analytics(
                tracked_vehicles=20,
                density=available(45.0, unit="veh/km/lane"),
                mean_speed_kph=available(25.0, unit="km/h"),
            ),
            state_settings,
        )
        assert assessment.basis is EvidenceBasis.PHYSICAL
        # density 45/90 = 0.5 at 0.40; speed 1 - 25/50 = 0.5 at 0.35
        assert assessment.index == pytest.approx(0.5, abs=1e-4)

    def test_image_space_signals_are_excluded_from_a_physical_assessment(
        self, state_settings: Settings
    ) -> None:
        """Occupancy and density measure the same thing at different scales.

        Counting both would let a camera's mounting angle move a verdict that is
        supposed to rest on surveyed quantities.
        """
        assessment = classify(
            make_analytics(
                tracked_vehicles=20,
                density=available(45.0, unit="veh/km/lane"),
                mean_speed_kph=available(25.0, unit="km/h"),
                occupancy=available(0.9, unit="fraction", image_space=True),
                image_speed=available(0.001, unit="diagonals/s", image_space=True),
            ),
            state_settings,
        )
        assert assessment.basis is EvidenceBasis.PHYSICAL
        names = {signal.name for signal in assessment.signals}
        assert SignalName.IMAGE_OCCUPANCY not in names
        assert SignalName.IMAGE_SPEED not in names

    def test_physical_evidence_is_more_confident_than_image_space(
        self, state_settings: Settings
    ) -> None:
        physical = classify(
            make_analytics(
                tracked_vehicles=10,
                density=available(45.0, unit="veh/km/lane"),
                mean_speed_kph=available(25.0, unit="km/h"),
                stationary=available(4, unit="vehicles"),
            ),
            state_settings,
        )
        image = classify(
            make_analytics(
                tracked_vehicles=10,
                occupancy=available(0.175, unit="fraction", image_space=True),
                image_speed=available(0.04, unit="diagonals/s", image_space=True),
                stationary=available(4, unit="vehicles"),
            ),
            state_settings,
        )
        assert physical.confidence is not None
        assert image.confidence is not None
        assert physical.confidence > image.confidence

    def test_queue_evidence_alone_does_not_claim_a_physical_basis(
        self, state_settings: Settings
    ) -> None:
        """The stationary fraction needs no calibration, so it cannot certify one.

        It is derived from counts and appears in both signal sets. Letting it
        select the physical basis would label an uncalibrated camera's verdict
        "physical" and discard the image-space signals that were the only real
        evidence available.
        """
        assessment = classify(
            make_analytics(tracked_vehicles=10, stationary=available(8, unit="vehicles")),
            state_settings,
        )
        assert assessment.basis is EvidenceBasis.IMAGE_SPACE
        assert assessment.level is not CongestionLevel.UNKNOWN

    def test_image_signals_are_kept_when_only_queue_evidence_accompanies_them(
        self, state_settings: Settings
    ) -> None:
        assessment = classify(
            make_analytics(
                tracked_vehicles=10,
                occupancy=available(0.35, unit="fraction", image_space=True),
                stationary=available(4, unit="vehicles"),
            ),
            state_settings,
        )
        names = {
            signal.name
            for signal in assessment.signals
            if signal.availability is MetricAvailability.AVAILABLE
        }
        assert names == {SignalName.IMAGE_OCCUPANCY, SignalName.STATIONARY_FRACTION}

    def test_a_physical_signal_is_never_flagged_image_space(
        self, state_settings: Settings
    ) -> None:
        assessment = classify(
            make_analytics(tracked_vehicles=10, density=available(45.0, unit="veh/km/lane")),
            state_settings,
        )
        for signal in assessment.signals:
            assert signal.is_image_space is False


# ---------------------------------------------------------------------------
# Refusals: invalid and missing input
# ---------------------------------------------------------------------------
class TestRefusals:
    def test_poor_quality_refuses_to_classify(self, state_settings: Settings) -> None:
        assessment = classify(
            make_analytics(
                grade=QualityGrade.POOR,
                tracked_vehicles=10,
                occupancy=available(0.3, unit="fraction", image_space=True),
            ),
            state_settings,
        )
        assert assessment.level is CongestionLevel.UNKNOWN
        assert assessment.index is None
        assert assessment.basis is EvidenceBasis.INSUFFICIENT
        assert "quality" in (assessment.reason or "").lower()

    def test_detections_without_confirmed_tracks_refuse_to_classify(
        self, state_settings: Settings
    ) -> None:
        """Tracking failed, so the vehicle count cannot be trusted.

        Reporting free flow here would announce an empty road on footage that
        may show a standing queue.
        """
        assessment = classify(
            make_analytics(
                mode=ObservationMode.SEQUENCE,
                tracked_vehicles=0,
                detected_vehicles=25,
                occupancy=available(0.3, unit="fraction", image_space=True),
            ),
            state_settings,
        )
        assert assessment.level is CongestionLevel.UNKNOWN
        assert assessment.index is None
        assert "track" in (assessment.reason or "").lower()

    def test_no_usable_signal_refuses_to_classify(self, state_settings: Settings) -> None:
        assessment = classify(make_analytics(tracked_vehicles=10), state_settings)
        assert assessment.level is CongestionLevel.UNKNOWN
        assert assessment.index is None
        assert assessment.reason

    def test_a_refusal_still_reports_which_signals_were_missing(
        self, state_settings: Settings
    ) -> None:
        assessment = classify(make_analytics(tracked_vehicles=10), state_settings)
        assert assessment.signals
        assert all(
            signal.availability is not MetricAvailability.AVAILABLE
            for signal in assessment.signals
        )
        assert all(signal.reason for signal in assessment.signals)

    def test_quality_is_gated_before_an_empty_scene_is_called_free_flow(
        self, state_settings: Settings
    ) -> None:
        """Zero vehicles from an untrustworthy analysis is not evidence of a clear road."""
        assessment = classify(
            make_analytics(grade=QualityGrade.POOR, tracked_vehicles=0, detected_vehicles=0),
            state_settings,
        )
        assert assessment.level is CongestionLevel.UNKNOWN


class TestEmptyScene:
    def test_no_vehicles_is_free_flow_not_unknown(self, state_settings: Settings) -> None:
        """A zero count from a working detector is a real observation.

        Counts survive without calibration where metric quantities do not, which
        is exactly what makes an uncalibrated camera partially useful.
        """
        assessment = classify(
            make_analytics(tracked_vehicles=0, detected_vehicles=0), state_settings
        )
        assert assessment.level is CongestionLevel.FREE_FLOW
        assert assessment.index == 0.0

    def test_a_stationary_fraction_needs_a_denominator(self, state_settings: Settings) -> None:
        analytics = make_analytics(
            tracked_vehicles=10,
            occupancy=available(0.1, unit="fraction", image_space=True),
            stationary=available(0, unit="vehicles"),
        )
        assessment = classify(analytics, state_settings)
        stationary = next(
            s for s in assessment.signals if s.name is SignalName.STATIONARY_FRACTION
        )
        assert stationary.availability is MetricAvailability.AVAILABLE
        assert stationary.score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------
class TestSchemaInvariants:
    def test_an_unknown_verdict_cannot_carry_an_index(
        self, state_settings: Settings
    ) -> None:
        with pytest.raises(ValidationError, match="must not carry an index"):
            CongestionAssessment(
                level=CongestionLevel.UNKNOWN,
                index=0.5,
                basis=EvidenceBasis.INSUFFICIENT,
                thresholds=thresholds_from(state_settings),
                reason="stated",
                method="m",
                method_version="1",
            )

    def test_an_unknown_verdict_must_state_why(self, state_settings: Settings) -> None:
        with pytest.raises(ValidationError, match="must state why"):
            CongestionAssessment(
                level=CongestionLevel.UNKNOWN,
                basis=EvidenceBasis.INSUFFICIENT,
                thresholds=thresholds_from(state_settings),
                method="m",
                method_version="1",
            )

    def test_a_band_requires_the_index_that_produced_it(
        self, state_settings: Settings
    ) -> None:
        with pytest.raises(ValidationError, match="requires the index"):
            CongestionAssessment(
                level=CongestionLevel.HEAVY,
                basis=EvidenceBasis.IMAGE_SPACE,
                thresholds=thresholds_from(state_settings),
                method="m",
                method_version="1",
            )

    def test_an_unavailable_signal_cannot_carry_a_value(self) -> None:
        with pytest.raises(ValidationError, match="must not contribute a number"):
            CongestionSignal(
                name=SignalName.DENSITY,
                availability=MetricAvailability.UNAVAILABLE_NO_CALIBRATION,
                raw_value=12.0,
                score=0.5,
                reason="stated",
            )

    def test_an_unavailable_signal_must_state_why(self) -> None:
        with pytest.raises(ValidationError, match="must state why"):
            CongestionSignal(
                name=SignalName.DENSITY,
                availability=MetricAvailability.UNAVAILABLE_NO_CALIBRATION,
            )

    def test_an_unavailable_signal_carries_no_weight(self) -> None:
        with pytest.raises(ValidationError, match="zero weight"):
            CongestionSignal(
                name=SignalName.DENSITY,
                availability=MetricAvailability.UNAVAILABLE_NO_CALIBRATION,
                reason="stated",
                weight=0.4,
            )

    def test_overlapping_bands_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="strictly increase"):
            CongestionThresholds(
                light=0.5,
                moderate=0.4,
                heavy=0.6,
                severe=0.7,
                gridlock=0.8,
                occupancy_reference=0.35,
                stationary_reference=0.8,
                image_speed_reference=0.08,
                density_reference_veh_per_km_per_lane=90.0,
                free_flow_speed_kph=50.0,
            )

    def test_misconfigured_bands_are_rejected_at_settings_construction(self) -> None:
        """Caught as a configuration fault, not as an argument about traffic."""
        with pytest.raises(ValidationError, match="strictly increase"):
            Settings(app_env="local", congestion_index_heavy=0.1)

    def test_level_of_service_is_not_assigned(self, state_settings: Settings) -> None:
        """LOS is a function of control delay, which nothing here measures."""
        assessment = classify(
            make_analytics(
                tracked_vehicles=10,
                occupancy=available(0.2, unit="fraction", image_space=True),
            ),
            state_settings,
        )
        assert not hasattr(assessment, "level_of_service")


# ---------------------------------------------------------------------------
# Interpretation of one area
# ---------------------------------------------------------------------------
class TestInterpret:
    def test_state_carries_the_analytics_provenance_chain(
        self, state_settings: Settings
    ) -> None:
        state = interpret(
            make_analytics(tracked_vehicles=5), state_settings, assessed_at=T0
        )
        assert state.analytics_provenance.media_id == "media-1"
        assert state.analytics_provenance.model.name == "test-model"

    def test_as_of_is_the_end_of_an_analysed_range(self, state_settings: Settings) -> None:
        """A verdict over a clip describes the traffic by the end of that clip."""
        end = T0 + dt.timedelta(seconds=30)
        state = interpret(
            make_analytics(timestamp=None, time_range=(T0, end)),
            state_settings,
            assessed_at=T0,
        )
        assert state.as_of == end

    def test_state_ids_are_deterministic(self, state_settings: Settings) -> None:
        analytics = make_analytics(tracked_vehicles=5)
        first = interpret(analytics, state_settings, assessed_at=T0)
        second = interpret(analytics, state_settings, assessed_at=T0)
        assert first.state_id == second.state_id

    def test_vehicle_count_uses_tracks_for_a_sequence(self, state_settings: Settings) -> None:
        """Detections would count one stopped car once per frame."""
        analytics = make_analytics(
            mode=ObservationMode.SEQUENCE, tracked_vehicles=4, detected_vehicles=120
        )
        assert vehicle_count_for(analytics) == 4
        assert interpret(analytics, state_settings, assessed_at=T0).vehicle_count == 4

    def test_vehicle_count_uses_detections_for_a_snapshot(
        self, state_settings: Settings
    ) -> None:
        analytics = make_analytics(
            mode=ObservationMode.SNAPSHOT, tracked_vehicles=0, detected_vehicles=7
        )
        assert vehicle_count_for(analytics) == 7

    def test_stationary_count_is_none_when_unmeasured(self, state_settings: Settings) -> None:
        state = interpret(make_analytics(tracked_vehicles=5), state_settings, assessed_at=T0)
        assert state.stationary_vehicle_count is None

    def test_observed_at_falls_back_to_computation_time(self) -> None:
        provenance = make_provenance(timestamp=None)
        assert observed_at(provenance) == T0


# ---------------------------------------------------------------------------
# Fusion across areas
# ---------------------------------------------------------------------------
class TestAssembly:
    def _state(
        self, settings: Settings, area: str, at: dt.datetime, vehicles: int = 5
    ) -> IntersectionTrafficState:
        return interpret(
            make_analytics(
                area=area,
                tracked_vehicles=vehicles,
                timestamp=at,
                occupancy=available(0.1, unit="fraction", image_space=True),
            ),
            settings,
            assessed_at=at,
        )

    def test_an_empty_snapshot_is_valid_and_not_degraded(
        self, state_settings: Settings
    ) -> None:
        snapshot = assemble((), state_settings, assembled_at=T0)
        assert snapshot.elements == ()
        assert snapshot.is_degraded is False
        assert snapshot.as_of == T0

    def test_ages_are_measured_against_the_newest_element(
        self, state_settings: Settings
    ) -> None:
        """Replaying old footage must not expire every element on arrival."""
        old = T0 - dt.timedelta(days=365)
        snapshot = assemble(
            (
                self._state(state_settings, "area-a", old),
                self._state(state_settings, "area-b", old - dt.timedelta(seconds=2)),
            ),
            state_settings,
            assembled_at=T0,
        )
        assert snapshot.as_of == old
        assert all(e.freshness is FreshnessStatus.FRESH for e in snapshot.elements)

    @pytest.mark.parametrize(
        ("age_s", "expected"),
        [
            (0.0, FreshnessStatus.FRESH),
            (5.0, FreshnessStatus.FRESH),
            (5.001, FreshnessStatus.STALE),
            (30.0, FreshnessStatus.STALE),
            (30.001, FreshnessStatus.EXPIRED),
        ],
    )
    def test_freshness_boundaries(
        self, age_s: float, expected: FreshnessStatus, state_settings: Settings
    ) -> None:
        assert freshness_for(age_s, policy_from(state_settings)) is expected

    def test_a_lagging_camera_is_marked_stale_not_dropped(
        self, state_settings: Settings
    ) -> None:
        snapshot = assemble(
            (
                self._state(state_settings, "area-a", T0),
                self._state(state_settings, "area-b", T0 - dt.timedelta(seconds=10)),
            ),
            state_settings,
            assembled_at=T0,
        )
        assert snapshot.stale_area_ids == {"area-b"}
        assert snapshot.is_degraded is True
        assert len(snapshot.elements) == 2

    def test_the_newest_analysis_of_an_area_wins(self, state_settings: Settings) -> None:
        """Two analyses of one camera are not two samples to average."""
        newer = T0
        older = T0 - dt.timedelta(seconds=60)
        snapshot = assemble(
            (
                self._state(state_settings, "area-a", older, vehicles=30),
                self._state(state_settings, "area-a", newer, vehicles=3),
            ),
            state_settings,
            assembled_at=T0,
        )
        assert len(snapshot.elements) == 1
        assert snapshot.element("area-a").state.vehicle_count == 3

    def test_unclassified_areas_are_surfaced_separately_from_stale_ones(
        self, state_settings: Settings
    ) -> None:
        unclassified = interpret(
            make_analytics(area="area-poor", grade=QualityGrade.POOR, timestamp=T0),
            state_settings,
            assessed_at=T0,
        )
        snapshot = assemble(
            (self._state(state_settings, "area-a", T0), unclassified),
            state_settings,
            assembled_at=T0,
        )
        assert snapshot.unclassified_area_ids == {"area-poor"}
        assert snapshot.stale_area_ids == frozenset()
        assert snapshot.is_degraded is True

    def test_the_policy_travels_inside_the_snapshot(self, state_settings: Settings) -> None:
        """A verdict rendered against a different budget would be unreproducible."""
        snapshot = assemble(
            (self._state(state_settings, "area-a", T0),), state_settings, assembled_at=T0
        )
        assert snapshot.policy.max_staleness_s == 5.0
        assert snapshot.policy.drop_after_s == 30.0
        assert snapshot.policy.fusion_strategy == "latest_per_observation_area"

    def test_lookup_of_an_absent_area_raises(self, state_settings: Settings) -> None:
        snapshot = assemble(
            (self._state(state_settings, "area-a", T0),), state_settings, assembled_at=T0
        )
        with pytest.raises(KeyError):
            snapshot.element("area-missing")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
class TestStateApi:
    def test_config_exposes_the_thresholds_in_force(self, client: TestClient) -> None:
        body = client.get("/api/state/config").json()
        assert body["method"] == "weighted-threshold-congestion"
        assert body["thresholds"]["light"] == 0.20
        assert body["thresholds"]["gridlock"] == 0.92
        assert body["fusion_strategy"] == "latest_per_observation_area"

    def test_interpreting_without_analytics_is_a_404_naming_the_next_step(
        self, client: TestClient
    ) -> None:
        response = client.post("/api/state/images/unknown-media")
        assert response.status_code == 404
        message = response.json()["error"]["message"]
        assert "/api/analytics/images/unknown-media" in message

    def test_reading_a_state_that_was_never_interpreted_is_a_404(
        self, client: TestClient
    ) -> None:
        assert client.get("/api/state/images/unknown-media").status_code == 404
        assert client.get("/api/state/jobs/unknown-job").status_code == 404

    def test_a_job_without_analytics_is_a_404(self, client: TestClient) -> None:
        response = client.post("/api/state/jobs/unknown-job")
        assert response.status_code == 404
        assert "/api/analytics/jobs/unknown-job" in response.json()["error"]["message"]

    def test_interpret_then_read_round_trip(self, client: TestClient) -> None:
        client.app.state.analytics_results = {  # type: ignore[attr-defined]
            "media-x": make_analytics(
                area="observation-area:media-x",
                tracked_vehicles=10,
                occupancy=available(0.35, unit="fraction", image_space=True),
                stationary=available(8, unit="vehicles"),
            )
        }

        posted = client.post("/api/state/images/media-x")
        assert posted.status_code == 200
        body = posted.json()
        assert body["congestion"]["level"] in {"heavy", "severe", "gridlock"}
        assert body["observation_area_id"] == "observation-area:media-x"

        fetched = client.get("/api/state/images/media-x")
        assert fetched.status_code == 200
        assert fetched.json()["state_id"] == body["state_id"]

    def test_an_unclassifiable_state_is_a_200_not_an_error(
        self, client: TestClient
    ) -> None:
        """A refusal is a true answer about the traffic, not a service fault."""
        client.app.state.analytics_results = {  # type: ignore[attr-defined]
            "media-poor": make_analytics(
                area="observation-area:media-poor", grade=QualityGrade.POOR
            )
        }
        response = client.post("/api/state/images/media-poor")
        assert response.status_code == 200
        congestion = response.json()["congestion"]
        assert congestion["level"] == "unknown"
        assert congestion["index"] is None
        assert congestion["reason"]

    def test_the_response_carries_the_signals_behind_the_verdict(
        self, client: TestClient
    ) -> None:
        client.app.state.analytics_results = {  # type: ignore[attr-defined]
            "media-y": make_analytics(
                area="observation-area:media-y",
                tracked_vehicles=10,
                occupancy=available(0.1, unit="fraction", image_space=True),
                stationary=available(2, unit="vehicles"),
            )
        }
        body = client.post("/api/state/images/media-y").json()["congestion"]
        assert body["thresholds"]["occupancy_reference"] == 0.35
        signals = {s["name"]: s for s in body["signals"]}
        assert signals["image_occupancy"]["availability"] == "available"
        assert signals["image_speed"]["availability"] != "available"
        assert signals["image_speed"]["reason"]

    def test_network_state_is_empty_but_successful_before_anything_is_interpreted(
        self, client: TestClient
    ) -> None:
        """Distinct from /api/network/state: this reports our own records."""
        response = client.get("/api/state/network")
        assert response.status_code == 200
        body = response.json()
        assert body["elements"] == []
        assert body["policy"]["fusion_strategy"] == "latest_per_observation_area"

    def test_network_state_lists_every_interpreted_area(self, client: TestClient) -> None:
        client.app.state.analytics_results = {  # type: ignore[attr-defined]
            "media-a": make_analytics(
                area="area-a",
                tracked_vehicles=4,
                occupancy=available(0.05, unit="fraction", image_space=True),
            ),
            "media-b": make_analytics(
                area="area-b",
                tracked_vehicles=30,
                occupancy=available(0.34, unit="fraction", image_space=True),
            ),
        }
        client.post("/api/state/images/media-a")
        client.post("/api/state/images/media-b")

        body = client.get("/api/state/network").json()
        areas = {e["state"]["observation_area_id"] for e in body["elements"]}
        assert areas == {"area-a", "area-b"}
        levels = {
            e["state"]["observation_area_id"]: e["state"]["congestion"]["level"]
            for e in body["elements"]
        }
        assert levels["area-a"] == "free_flow"
        assert levels["area-b"] in {"heavy", "severe", "gridlock"}

    def test_the_road_network_endpoints_refuse_until_a_map_is_loaded(
        self, client: TestClient
    ) -> None:
        """Interpreted state exists; a surveyed road network is not yet chosen.

        Conflating the two would let an operator read a camera's verdict as
        coverage of a network that has never been loaded.
        """
        for path in ("/api/network", "/api/network/state"):
            response = client.get(path)
            assert response.status_code == 404
            assert response.json()["error"]["code"] == "network_not_loaded"

    def test_state_endpoints_return_no_route_or_advisory(
        self, client: TestClient
    ) -> None:
        """Phase 5 interprets; it does not recommend."""
        client.app.state.analytics_results = {  # type: ignore[attr-defined]
            "media-z": make_analytics(
                area="area-z",
                tracked_vehicles=10,
                occupancy=available(0.2, unit="fraction", image_space=True),
            )
        }
        body = client.post("/api/state/images/media-z").json()
        forbidden = ("route", "eta", "travel_time", "advisory", "recommendation", "priority")
        rendered = repr(body).lower()
        for token in forbidden:
            assert token not in rendered, f"phase 5 must not emit {token!r}"
