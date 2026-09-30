"""Observation integrity and network-state assembly semantics."""

from __future__ import annotations

import datetime as dt

import pytest
from domain_fixtures import T0, measure_float, measure_int
from pydantic import ValidationError

from app.domain.enums import (
    DataQualityFlag,
    FreshnessStatus,
    IncidentSeverity,
    IncidentStatus,
    IncidentType,
    SignalIndication,
    VehicleClass,
)
from app.domain.incident import ExclusionZone, Incident, IncidentLocation
from app.domain.observation import (
    ApproachObservation,
    CameraCalibration,
    JunctionObservation,
    ObservationQuality,
)
from app.domain.provenance import Unit, ValueSource, make_provenance
from app.domain.state import (
    EmergencyTraversalAssumption,
    JunctionState,
    NetworkState,
    RoadSegmentState,
    StateAssemblyPolicy,
    TravelTimeEstimate,
)

CALIBRATED = CameraCalibration(
    is_calibrated=True,
    method="homography_from_4_ground_points",
    homography=(1, 0, 0, 0, 1, 0, 0, 0, 1),
)
UNCALIBRATED = CameraCalibration(is_calibrated=False)


def _observation(
    *,
    approaches: tuple[ApproachObservation, ...],
    calibration: CameraCalibration | None,
) -> JunctionObservation:
    return JunctionObservation(
        observation_id="obs-1",
        junction_id="J11",
        source_id="cam-J11-north",
        captured_at=T0,
        ingested_at=T0 + dt.timedelta(milliseconds=250),
        schema_version="0.1.0",
        approaches=approaches,
        calibration=calibration,
        quality=ObservationQuality(analysis_window_s=60.0),
    )


class TestCameraCalibration:
    def test_claiming_calibration_requires_calibration_data(self) -> None:
        with pytest.raises(ValidationError, match="fabricated claim"):
            CameraCalibration(is_calibrated=True)

    def test_homography_must_be_nine_values(self) -> None:
        with pytest.raises(ValidationError, match="9 values"):
            CameraCalibration(is_calibrated=True, homography=(1, 0, 0, 0))


class TestJunctionObservation:
    def test_counts_survive_without_calibration(self) -> None:
        """Counting does not need a ground transform; speed does."""
        observation = _observation(
            approaches=(
                ApproachObservation(
                    approach_id="J11-A1",
                    incoming_segment_id="S1",
                    vehicle_count_total=measure_int(24, Unit.VEHICLE),
                    vehicle_count_by_class={
                        VehicleClass.CAR: measure_int(20, Unit.VEHICLE),
                        VehicleClass.BUS: measure_int(4, Unit.VEHICLE),
                    },
                ),
            ),
            calibration=UNCALIBRATED,
        )
        assert observation.approach("J11-A1").vehicle_count_total is not None

    def test_metric_speed_without_calibration_is_rejected(self) -> None:
        """Pixels per second is not a speed and cannot be converted into one."""
        with pytest.raises(ValidationError, match="without camera"):
            _observation(
                approaches=(
                    ApproachObservation(
                        approach_id="J11-A1",
                        incoming_segment_id="S1",
                        mean_speed_kph=measure_float(42.0, Unit.KILOMETER_PER_HOUR),
                    ),
                ),
                calibration=UNCALIBRATED,
            )

    def test_metric_queue_length_without_calibration_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _observation(
                approaches=(
                    ApproachObservation(
                        approach_id="J11-A1",
                        incoming_segment_id="S1",
                        queue_length_m=measure_float(85.0, Unit.METER),
                    ),
                ),
                calibration=UNCALIBRATED,
            )

    def test_metric_values_are_accepted_when_calibrated(self) -> None:
        observation = _observation(
            approaches=(
                ApproachObservation(
                    approach_id="J11-A1",
                    incoming_segment_id="S1",
                    mean_speed_kph=measure_float(42.0, Unit.KILOMETER_PER_HOUR),
                    queue_length_m=measure_float(85.0, Unit.METER),
                ),
            ),
            calibration=CALIBRATED,
        )
        assert observation.approach("J11-A1").mean_speed_kph is not None

    def test_missing_calibration_object_is_treated_as_uncalibrated(self) -> None:
        with pytest.raises(ValidationError):
            _observation(
                approaches=(
                    ApproachObservation(
                        approach_id="J11-A1",
                        incoming_segment_id="S1",
                        mean_speed_kph=measure_float(42.0, Unit.KILOMETER_PER_HOUR),
                    ),
                ),
                calibration=None,
            )

    def test_signal_indication_defaults_to_unknown_not_to_a_guess(self) -> None:
        """A camera that cannot see the head reports UNKNOWN; that is fail-safe."""
        approach = ApproachObservation(approach_id="J11-A1", incoming_segment_id="S1")
        assert approach.observed_signal_indication is SignalIndication.UNKNOWN

    def test_duplicate_approaches_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate approach_id"):
            _observation(
                approaches=(
                    ApproachObservation(approach_id="A", incoming_segment_id="S1"),
                    ApproachObservation(approach_id="A", incoming_segment_id="S1"),
                ),
                calibration=UNCALIBRATED,
            )

    def test_latency_is_capture_to_ingest(self) -> None:
        observation = _observation(approaches=(), calibration=UNCALIBRATED)
        assert observation.latency_s == pytest.approx(0.25)

    def test_uncalibrated_flag_is_available_for_quality(self) -> None:
        quality = ObservationQuality(flags=frozenset({DataQualityFlag.UNCALIBRATED}))
        assert quality.is_degraded is True


class TestTravelTimeEstimate:
    def test_emergency_estimate_must_state_its_assumptions(self) -> None:
        """An emergency ETA is conditional on behaviour we can only recommend."""
        with pytest.raises(ValidationError, match="requires emergency_assumptions"):
            TravelTimeEstimate(
                emergency_travel_time_s=measure_float(120.0, Unit.SECOND)
            )

    def test_the_two_estimates_are_independent_fields(self) -> None:
        estimate = TravelTimeEstimate(
            nominal_travel_time_s=measure_float(240.0, Unit.SECOND),
            emergency_travel_time_s=measure_float(160.0, Unit.SECOND),
            emergency_assumptions=(EmergencyTraversalAssumption.SIGNAL_PRIORITY_GRANTED,),
        )
        assert estimate.emergency_advantage_ratio == pytest.approx(160 / 240)

    def test_no_function_converts_nominal_into_emergency_time(self) -> None:
        """The constant-ratio shortcut is wrong precisely where it matters most.

        Guarded structurally: this module exposes no such conversion, so the
        shortcut cannot be taken by accident.
        """
        import app.domain.state as state_module

        forbidden = [
            name
            for name in dir(state_module)
            if "to_emergency" in name.lower() or "emergency_from" in name.lower()
        ]
        assert forbidden == []

    def test_advantage_ratio_is_none_without_both_estimates(self) -> None:
        assert TravelTimeEstimate().emergency_advantage_ratio is None


class TestRoadSegmentState:
    def test_a_blockage_must_name_its_cause(self) -> None:
        """A blockage with no provenance cannot be explained or cleared."""
        with pytest.raises(ValidationError, match="no causing"):
            RoadSegmentState(segment_id="S1", is_blocked=True)

    def test_blocked_with_cause_is_accepted(self) -> None:
        state = RoadSegmentState(
            segment_id="S1",
            is_blocked=True,
            blocked_by_incident_ids=frozenset({"INC-1"}),
        )
        assert state.is_blocked is True

    def test_unknown_state_is_the_default_not_zero(self) -> None:
        """An unmonitored road has unknown queues, not empty ones."""
        state = RoadSegmentState(segment_id="S1")
        assert state.queue_length_m is None
        assert state.freshness is FreshnessStatus.UNKNOWN
        assert state.is_emergency_accessible is None


def _incident(
    *,
    incident_id: str = "INC-1",
    blocks: frozenset[str] = frozenset(),
    affected: frozenset[str] = frozenset(),
    zone: ExclusionZone | None = None,
    status: IncidentStatus = IncidentStatus.CONFIRMED,
) -> Incident:
    return Incident(
        incident_id=incident_id,
        incident_type=IncidentType.FIRE,
        severity=IncidentSeverity.MAJOR,
        status=status,
        location=IncidentLocation(segment_id="S9"),
        reported_at=T0,
        provenance=make_provenance(
            source=ValueSource.OBSERVED,
            producer="test.dispatch",
            producer_version="1",
            observed_at=T0,
        ),
        blocks_segment_ids=blocks,
        affected_segment_ids=affected,
        exclusion_zone=zone,
        cleared_at=T0 if status is IncidentStatus.CLEARED else None,
    )


class TestIncident:
    def test_a_segment_cannot_be_both_blocked_and_merely_affected(self) -> None:
        """The ambiguity would change whether it is pruned or priced."""
        with pytest.raises(ValidationError, match="both fully blocked"):
            _incident(blocks=frozenset({"S1"}), affected=frozenset({"S1"}))

    def test_exclusion_zone_segments_count_as_hard_blocked(self) -> None:
        incident = _incident(
            blocks=frozenset({"S1"}),
            zone=ExclusionZone(
                radius_m=measure_float(150.0, Unit.METER, source=ValueSource.DERIVED,
                                       input_refs=("INC-1",)),
                excluded_segment_ids=frozenset({"S2", "S3"}),
                reason="active fire",
            ),
        )
        assert incident.hard_blocked_segment_ids == {"S1", "S2", "S3"}

    def test_a_cleared_incident_constrains_nothing(self) -> None:
        incident = _incident(blocks=frozenset({"S1"}), status=IncidentStatus.CLEARED)
        assert incident.is_active is False
        assert incident.hard_blocked_segment_ids == frozenset()

    def test_cleared_status_requires_a_cleared_timestamp(self) -> None:
        with pytest.raises(ValidationError, match="must record cleared_at"):
            Incident(
                incident_id="INC-2",
                incident_type=IncidentType.ACCIDENT,
                severity=IncidentSeverity.MINOR,
                status=IncidentStatus.CLEARED,
                location=IncidentLocation(junction_id="J11"),
                reported_at=T0,
                provenance=make_provenance(
                    source=ValueSource.OBSERVED,
                    producer="t",
                    producer_version="1",
                    observed_at=T0,
                ),
            )

    def test_location_must_anchor_to_the_graph(self) -> None:
        """A bare coordinate cannot prune or penalise anything."""
        with pytest.raises(ValidationError, match="segment or a junction"):
            IncidentLocation(description="somewhere near the market")


class TestNetworkState:
    def _state(
        self,
        *,
        junctions: tuple[JunctionState, ...] = (),
        segments: tuple[RoadSegmentState, ...] = (),
        incidents: tuple[Incident, ...] = (),
    ) -> NetworkState:
        return NetworkState(
            state_id="ns-1",
            network_id="net-test",
            network_version="1.0",
            schema_version="0.1.0",
            as_of=T0,
            assembled_at=T0,
            policy=StateAssemblyPolicy(max_staleness_s=5.0, drop_after_s=30.0),
            junction_states=junctions,
            segment_states=segments,
            active_incidents=incidents,
        )

    def test_blocked_segments_union_state_and_incidents(self) -> None:
        """A blockage must not be missed because only one source recorded it."""
        state = self._state(
            segments=(
                RoadSegmentState(
                    segment_id="S1",
                    is_blocked=True,
                    blocked_by_incident_ids=frozenset({"INC-1"}),
                ),
            ),
            incidents=(_incident(blocks=frozenset({"S5"})),),
        )
        assert state.blocked_segment_ids == {"S1", "S5"}

    def test_stale_and_unknown_junctions_are_tracked_separately(self) -> None:
        state = self._state(
            junctions=(
                JunctionState(junction_id="J11", freshness=FreshnessStatus.FRESH),
                JunctionState(junction_id="J12", freshness=FreshnessStatus.STALE),
                JunctionState(junction_id="J13", freshness=FreshnessStatus.UNKNOWN),
                JunctionState(junction_id="J14", freshness=FreshnessStatus.EXPIRED),
            )
        )
        assert state.stale_junction_ids == {"J12"}
        assert state.unknown_junction_ids == {"J13", "J14"}
        assert state.is_degraded is True

    def test_a_fully_fresh_snapshot_is_not_degraded(self) -> None:
        state = self._state(
            junctions=(JunctionState(junction_id="J11", freshness=FreshnessStatus.FRESH),)
        )
        assert state.is_degraded is False

    def test_stale_evidence_is_still_usable(self) -> None:
        """Withholding a route because one camera lagged is its own failure."""
        stale = JunctionState(junction_id="J12", freshness=FreshnessStatus.STALE)
        unknown = JunctionState(junction_id="J13", freshness=FreshnessStatus.UNKNOWN)
        assert stale.is_usable_evidence is True
        assert unknown.is_usable_evidence is False

    def test_assembly_policy_travels_inside_the_snapshot(self) -> None:
        """A decision is audited against the freshness rules then in force."""
        state = self._state()
        assert state.policy.max_staleness_s == 5.0
        assert state.policy.fusion_strategy == "latest_per_junction"

    def test_drop_threshold_must_not_precede_staleness_threshold(self) -> None:
        with pytest.raises(ValidationError, match="must be >="):
            StateAssemblyPolicy(max_staleness_s=30.0, drop_after_s=5.0)

    def test_duplicate_junction_states_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate junction_id"):
            self._state(
                junctions=(
                    JunctionState(junction_id="J11"),
                    JunctionState(junction_id="J11"),
                )
            )

    def test_contributing_sources_are_retained_for_disagreement(self) -> None:
        """Two cameras disagreeing is worth surfacing, not silently averaging."""
        js = JunctionState(
            junction_id="J11",
            contributing_source_ids=("cam-north", "cam-south"),
            contributing_observation_ids=("obs-1", "obs-2"),
        )
        assert len(js.contributing_source_ids) == 2
