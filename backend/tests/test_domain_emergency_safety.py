"""Emergency response modelling and the two-tier safety system."""

from __future__ import annotations

import datetime as dt

import pytest
from domain_fixtures import T0
from pydantic import ValidationError

from app.domain.emergency import (
    EmergencyDestination,
    EmergencyState,
    EmergencyVehicle,
    NetworkPosition,
)
from app.domain.enums import (
    EmergencyPriority,
    EmergencyStatus,
    EmergencyVehicleType,
    FacilityType,
    IncidentSeverity,
)
from app.domain.provenance import ValueSource, make_provenance
from app.domain.safety import (
    PHASE1_SAFETY_CATALOG,
    ConstraintScope,
    ConstraintTier,
    SafetyConstraint,
    SafetyRule,
    SafetyRuleCatalog,
)

PROV = make_provenance(
    source=ValueSource.OBSERVED,
    producer="test.dispatch",
    producer_version="1",
    observed_at=T0,
)


def _emergency(
    *,
    vehicle_type: EmergencyVehicleType = EmergencyVehicleType.AMBULANCE,
    destination: EmergencyDestination | None = None,
    priority: EmergencyPriority = EmergencyPriority.P1_IMMEDIATE,
    status: EmergencyStatus = EmergencyStatus.EN_ROUTE,
    triggering_incident_id: str | None = None,
) -> EmergencyState:
    return EmergencyState(
        emergency_id="EMG-1",
        vehicle=EmergencyVehicle(vehicle_id="AMB-7", vehicle_type=vehicle_type),
        priority=priority,
        status=status,
        declared_at=T0,
        updated_at=T0 + dt.timedelta(seconds=30),
        current_position=NetworkPosition(junction_id="J11"),
        destination=destination
        or EmergencyDestination(facility_id="HOSP-A", facility_type=FacilityType.HOSPITAL),
        triggering_incident_id=triggering_incident_id,
        provenance=PROV,
    )


class TestEmergencyDestination:
    def test_exactly_one_target_is_required(self) -> None:
        """Facility, incident and coordinate are genuinely different targets."""
        with pytest.raises(ValidationError, match="exactly one"):
            EmergencyDestination(facility_id="HOSP-A", incident_id="INC-1")

    def test_a_destination_with_no_target_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exactly one"):
            EmergencyDestination()

    def test_facility_type_requires_a_facility(self) -> None:
        with pytest.raises(ValidationError, match="only meaningful together"):
            EmergencyDestination(incident_id="INC-1", facility_type=FacilityType.HOSPITAL)


class TestEmergencyState:
    def test_ambulance_to_hospital_is_valid(self) -> None:
        assert _emergency().is_active is True

    def test_ambulance_cannot_be_dispatched_to_a_fire_station(self) -> None:
        """Nonsensical dispatches are rejected before reaching the optimiser."""
        with pytest.raises(ValidationError, match="cannot be dispatched to a fire_station"):
            _emergency(
                destination=EmergencyDestination(
                    facility_id="FS-2", facility_type=FacilityType.FIRE_STATION
                )
            )

    def test_fire_engine_may_be_routed_to_an_incident(self) -> None:
        emergency = _emergency(
            vehicle_type=EmergencyVehicleType.FIRE_ENGINE,
            destination=EmergencyDestination(incident_id="INC-9"),
            triggering_incident_id="INC-9",
        )
        assert emergency.destination.incident_id == "INC-9"

    def test_destination_incident_must_match_the_triggering_incident(self) -> None:
        with pytest.raises(ValidationError, match="does not match the triggering incident"):
            _emergency(
                vehicle_type=EmergencyVehicleType.FIRE_ENGINE,
                destination=EmergencyDestination(incident_id="INC-9"),
                triggering_incident_id="INC-4",
            )

    def test_only_p1_warrants_corridor_preemption(self) -> None:
        """Pre-emption imposes real delay and risk on conflicting movements."""
        assert _emergency(priority=EmergencyPriority.P1_IMMEDIATE).warrants_corridor_preemption
        assert not _emergency(priority=EmergencyPriority.P2_URGENT).warrants_corridor_preemption
        assert not _emergency(priority=EmergencyPriority.P3_ROUTINE).warrants_corridor_preemption

    def test_a_completed_response_does_not_warrant_preemption(self) -> None:
        emergency = _emergency(status=EmergencyStatus.ARRIVED)
        assert emergency.is_active is False
        assert emergency.warrants_corridor_preemption is False

    def test_completed_status_requires_a_completion_timestamp(self) -> None:
        with pytest.raises(ValidationError, match="must record completed_at"):
            _emergency(status=EmergencyStatus.COMPLETED)

    def test_updated_at_cannot_precede_declaration(self) -> None:
        with pytest.raises(ValidationError, match="cannot precede declared_at"):
            EmergencyState(
                emergency_id="EMG-2",
                vehicle=EmergencyVehicle(
                    vehicle_id="A1", vehicle_type=EmergencyVehicleType.AMBULANCE
                ),
                priority=EmergencyPriority.P1_IMMEDIATE,
                status=EmergencyStatus.EN_ROUTE,
                declared_at=T0,
                updated_at=T0 - dt.timedelta(seconds=5),
                current_position=NetworkPosition(junction_id="J11"),
                destination=EmergencyDestination(facility_id="HOSP-A"),
                provenance=PROV,
            )

    def test_response_time_is_measured_on_the_callers_clock(self) -> None:
        """Under recorded replay the correct 'now' is media time."""
        emergency = _emergency()
        assert emergency.elapsed_response_time_s(T0 + dt.timedelta(minutes=4)) == 240.0

    def test_response_time_rejects_a_naive_instant(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            _emergency().elapsed_response_time_s(dt.datetime(2024, 3, 14))

    def test_position_must_anchor_to_the_graph(self) -> None:
        with pytest.raises(ValidationError, match="not routable"):
            NetworkPosition(heading_degrees=90.0)


def _constraint(
    *,
    rule_id: str,
    tier: ConstraintTier,
    scope: ConstraintScope = ConstraintScope.SEGMENT,
    penalty: float | None = None,
    targets: frozenset[str] = frozenset({"S1"}),
) -> SafetyConstraint:
    return SafetyConstraint(
        constraint_id=f"C-{rule_id}",
        rule_id=rule_id,
        tier=tier,
        scope=scope,
        target_segment_ids=targets if scope is ConstraintScope.SEGMENT else frozenset(),
        target_junction_ids=targets if scope is ConstraintScope.JUNCTION else frozenset(),
        active_from=T0,
        penalty_equivalent_s=penalty,
        message="test constraint",
        provenance=PROV,
        severity=IncidentSeverity.MAJOR,
    )


class TestSafetyTiers:
    def test_a_hard_constraint_may_not_carry_a_price(self) -> None:
        """Hard constraints prune; they are never traded off against travel time."""
        with pytest.raises(ValidationError, match="must not carry a"):
            _constraint(
                rule_id="hard.segment_blocked", tier=ConstraintTier.HARD, penalty=600.0
            )

    def test_a_soft_constraint_must_carry_a_price(self) -> None:
        """A soft constraint with no cost is a silently ignored safety signal."""
        with pytest.raises(ValidationError, match="must carry an explicit"):
            _constraint(rule_id="soft.adverse_weather", tier=ConstraintTier.SOFT)

    def test_a_soft_constraint_with_a_price_is_valid(self) -> None:
        constraint = _constraint(
            rule_id="soft.adverse_weather", tier=ConstraintTier.SOFT, penalty=45.0
        )
        assert constraint.penalty_equivalent_s == 45.0

    def test_an_unbound_constraint_is_rejected(self) -> None:
        """An unbound constraint silently affects nothing."""
        with pytest.raises(ValidationError, match="no targets"):
            _constraint(
                rule_id="hard.segment_blocked",
                tier=ConstraintTier.HARD,
                targets=frozenset(),
            )

    def test_activity_window_is_respected(self) -> None:
        constraint = SafetyConstraint(
            constraint_id="C1",
            rule_id="hard.segment_blocked",
            tier=ConstraintTier.HARD,
            scope=ConstraintScope.SEGMENT,
            target_segment_ids=frozenset({"S1"}),
            active_from=T0,
            active_until=T0 + dt.timedelta(minutes=30),
            message="closed",
            provenance=PROV,
        )
        assert constraint.is_active_at(T0 + dt.timedelta(minutes=10)) is True
        assert constraint.is_active_at(T0 - dt.timedelta(minutes=1)) is False
        assert constraint.is_active_at(T0 + dt.timedelta(hours=2)) is False

    def test_vehicle_scoping_narrows_never_widens(self) -> None:
        constraint = SafetyConstraint(
            constraint_id="C2",
            rule_id="hard.vehicle_physically_excluded",
            tier=ConstraintTier.HARD,
            scope=ConstraintScope.SEGMENT,
            target_segment_ids=frozenset({"S1"}),
            active_from=T0,
            message="low bridge",
            provenance=PROV,
            applies_to_vehicle_types=frozenset({EmergencyVehicleType.FIRE_ENGINE}),
        )
        assert constraint.applies_to(EmergencyVehicleType.FIRE_ENGINE) is True
        assert constraint.applies_to(EmergencyVehicleType.AMBULANCE) is False
        assert constraint.applies_to(None) is True


class TestSafetyCatalog:
    def test_the_phase_1_catalog_defines_both_tiers(self) -> None:
        hard = PHASE1_SAFETY_CATALOG.rules_of_tier(ConstraintTier.HARD)
        soft = PHASE1_SAFETY_CATALOG.rules_of_tier(ConstraintTier.SOFT)
        assert len(hard) >= 4
        assert len(soft) >= 5

    def test_blockage_and_fire_exclusion_are_hard(self) -> None:
        for rule_id in ("hard.segment_blocked", "hard.fire_exclusion_zone"):
            assert PHASE1_SAFETY_CATALOG.rule(rule_id).tier is ConstraintTier.HARD

    def test_pedestrian_density_and_weather_are_soft(self) -> None:
        for rule_id in ("soft.pedestrian_density", "soft.adverse_weather"):
            assert PHASE1_SAFETY_CATALOG.rule(rule_id).tier is ConstraintTier.SOFT

    def test_congestion_is_never_hard(self) -> None:
        """A jammed road is slow; a blocked road is impassable."""
        assert PHASE1_SAFETY_CATALOG.rule("soft.severe_congestion").tier is ConstraintTier.SOFT

    def test_stale_evidence_is_priced_not_pruned(self) -> None:
        """Pruning every unobserved element would disconnect a sparse network."""
        assert PHASE1_SAFETY_CATALOG.rule("soft.stale_evidence").tier is ConstraintTier.SOFT

    def test_every_rule_explains_itself(self) -> None:
        """Rules that cannot be reviewed accumulate until nobody knows why."""
        for rule in PHASE1_SAFETY_CATALOG.rules:
            assert len(rule.rationale) > 40

    def test_no_soft_penalty_weights_are_invented_in_phase_1(self) -> None:
        """Penalty weights are empirical and are calibrated in Phase 11."""
        for rule in PHASE1_SAFETY_CATALOG.rules:
            assert rule.default_penalty_equivalent_s is None

    def test_a_hard_rule_cannot_define_a_penalty(self) -> None:
        with pytest.raises(ValidationError, match="never priced"):
            SafetyRule(
                rule_id="hard.x",
                name="x",
                tier=ConstraintTier.HARD,
                scope=ConstraintScope.SEGMENT,
                description="d",
                rationale="r",
                default_penalty_equivalent_s=10.0,
            )

    def test_a_constraint_may_not_retier_its_rule(self) -> None:
        """Otherwise a congested alternative could outbid a fire exclusion zone."""
        downgraded = _constraint(
            rule_id="hard.fire_exclusion_zone", tier=ConstraintTier.SOFT, penalty=999.0
        )
        with pytest.raises(ValueError, match="may not re-tier"):
            PHASE1_SAFETY_CATALOG.validate_constraint(downgraded)

    def test_a_constraint_may_not_rescope_its_rule(self) -> None:
        mismatched = _constraint(
            rule_id="hard.segment_blocked",
            tier=ConstraintTier.HARD,
            scope=ConstraintScope.JUNCTION,
        )
        with pytest.raises(ValueError, match="defines scope"):
            PHASE1_SAFETY_CATALOG.validate_constraint(mismatched)

    def test_a_consistent_constraint_validates(self) -> None:
        ok = _constraint(rule_id="hard.segment_blocked", tier=ConstraintTier.HARD)
        PHASE1_SAFETY_CATALOG.validate_constraint(ok)

    def test_an_unknown_rule_is_rejected(self) -> None:
        stray = _constraint(rule_id="hard.invented", tier=ConstraintTier.HARD)
        with pytest.raises(KeyError):
            PHASE1_SAFETY_CATALOG.validate_constraint(stray)

    def test_duplicate_rule_ids_are_rejected(self) -> None:
        rule = PHASE1_SAFETY_CATALOG.rules[0]
        with pytest.raises(ValidationError, match="duplicate rule_id"):
            SafetyRuleCatalog(version="0.0.1", rules=(rule, rule))
