"""Provenance discipline and road network topology integrity."""

from __future__ import annotations

import datetime as dt

import pytest
from domain_fixtures import T0, measure_float
from pydantic import ValidationError

from app.domain.enums import (
    DrivingSide,
    EmergencyVehicleType,
    FacilityType,
    JunctionKind,
    MovementType,
)
from app.domain.network import (
    Approach,
    Facility,
    Junction,
    PhysicalRestrictions,
    RoadNetwork,
    RoadSegment,
    TurnMovement,
)
from app.domain.provenance import (
    Measurement,
    Provenance,
    Unit,
    ValueSource,
    make_provenance,
)


class TestProvenance:
    def test_there_is_no_mock_or_placeholder_value_source(self) -> None:
        """The absence of a 'placeholder' source is the point of the enum.

        A component that does not know a value must report ``None``, not a
        plausible-looking default.
        """
        names = {member.value for member in ValueSource}
        assert names == {
            "observed",
            "derived",
            "predicted",
            "simulated",
            "static",
            "assumed",
        }

    def test_assumed_values_must_justify_themselves(self) -> None:
        """An unjustified assumption is indistinguishable from a fabrication."""
        with pytest.raises(ValidationError, match="requires 'notes'"):
            Provenance(
                source=ValueSource.ASSUMED,
                producer="test",
                producer_version="1",
                observed_at=T0,
                computed_at=T0,
            )

    def test_derived_values_must_name_their_inputs(self) -> None:
        with pytest.raises(ValidationError, match="requires 'input_refs'"):
            Provenance(
                source=ValueSource.DERIVED,
                producer="test",
                producer_version="1",
                observed_at=T0,
                computed_at=T0,
            )

    def test_predicted_and_simulated_values_are_not_factual(self) -> None:
        """Modality must survive to the narrative layer."""
        for source in (ValueSource.PREDICTED, ValueSource.SIMULATED):
            prov = make_provenance(
                source=source, producer="m", producer_version="1", observed_at=T0
            )
            assert prov.is_factual is False
        for source in (ValueSource.OBSERVED, ValueSource.STATIC):
            prov = make_provenance(
                source=source, producer="m", producer_version="1", observed_at=T0
            )
            assert prov.is_factual is True

    def test_computed_at_does_not_default_to_wall_clock_now(self) -> None:
        """A recorded pipeline stamps values on the media timeline."""
        prov = make_provenance(
            source=ValueSource.OBSERVED, producer="p", producer_version="1", observed_at=T0
        )
        assert prov.computed_at == T0

    def test_naive_timestamps_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            Provenance(
                source=ValueSource.OBSERVED,
                producer="p",
                producer_version="1",
                observed_at=dt.datetime(2024, 3, 14),
                computed_at=T0,
            )

    def test_measurement_carries_unit_and_age(self) -> None:
        m = measure_float(48.5, Unit.KILOMETER_PER_HOUR)
        assert m.unit is Unit.KILOMETER_PER_HOUR
        assert m.age_seconds_at(T0 + dt.timedelta(seconds=12)) == 12.0

    def test_measurement_cannot_be_built_without_provenance(self) -> None:
        with pytest.raises(ValidationError):
            Measurement[float](value=10.0, unit=Unit.SECOND)  # type: ignore[call-arg]

    def test_domain_models_are_immutable(self) -> None:
        """Snapshots must mean what they say."""
        m = measure_float(1.0, Unit.SECOND)
        with pytest.raises(ValidationError):
            m.value = 2.0  # type: ignore[misc]

    def test_unknown_fields_are_rejected(self) -> None:
        """A typo in a safety-relevant field must not be silently ignored."""
        with pytest.raises(ValidationError):
            Provenance(
                source=ValueSource.OBSERVED,
                producer="p",
                producer_version="1",
                observed_at=T0,
                computed_at=T0,
                confidenc=0.9,  # type: ignore[call-arg]
            )


def _segment(seg_id: str, frm: str, to: str, length_m: float = 400.0) -> RoadSegment:
    return RoadSegment(
        segment_id=seg_id,
        from_junction_id=frm,
        to_junction_id=to,
        length_m=measure_float(length_m, Unit.METER, source=ValueSource.STATIC),
        lane_count=2,
    )


def _network(
    junctions: tuple[Junction, ...],
    segments: tuple[RoadSegment, ...],
    facilities: tuple[Facility, ...] = (),
) -> RoadNetwork:
    return RoadNetwork(
        network_id="net-test",
        version="1.0",
        driving_side=DrivingSide.LEFT,
        provenance=make_provenance(
            source=ValueSource.STATIC,
            producer="test.map",
            producer_version="1",
            observed_at=T0,
        ),
        junctions=junctions,
        segments=segments,
        facilities=facilities,
    )


class TestRoadNetwork:
    def test_a_valid_network_builds(self) -> None:
        net = _network(
            junctions=(Junction(junction_id="J11"), Junction(junction_id="J12")),
            segments=(_segment("S1", "J11", "J12"),),
        )
        assert net.build_index().segment("S1").length_m.value == 400.0

    def test_dangling_segment_reference_fails_at_load_not_at_routing(self) -> None:
        """Better to fail on import than inside an emergency route request."""
        with pytest.raises(ValidationError, match="unknown junction"):
            _network(
                junctions=(Junction(junction_id="J11"),),
                segments=(_segment("S1", "J11", "J99"),),
            )

    def test_self_loop_segments_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="self-loops"):
            _segment("S1", "J11", "J11")

    def test_duplicate_ids_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate junction_id"):
            _network(
                junctions=(Junction(junction_id="J11"), Junction(junction_id="J11")),
                segments=(),
            )

    def test_approach_must_use_a_segment_that_arrives_here(self) -> None:
        """An approach fed by a segment that terminates elsewhere is a map error."""
        approach = Approach(approach_id="J11-A1", incoming_segment_id="S1")
        with pytest.raises(ValidationError, match="does not terminate here"):
            _network(
                junctions=(
                    Junction(junction_id="J11", approaches=(approach,)),
                    Junction(junction_id="J12"),
                ),
                segments=(_segment("S1", "J11", "J12"),),
            )

    def test_movement_must_connect_segments_incident_to_its_junction(self) -> None:
        movement = TurnMovement(
            movement_id="M1",
            junction_id="J12",
            from_segment_id="S1",
            to_segment_id="S2",
            movement_type=MovementType.THROUGH,
        )
        with pytest.raises(ValidationError, match="does not depart from"):
            _network(
                junctions=(
                    Junction(junction_id="J11"),
                    Junction(junction_id="J12", movements=(movement,)),
                    Junction(junction_id="J13"),
                ),
                segments=(_segment("S1", "J11", "J12"), _segment("S2", "J13", "J11")),
            )

    def test_movements_cannot_belong_to_another_junction(self) -> None:
        movement = TurnMovement(
            movement_id="M1",
            junction_id="J99",
            from_segment_id="S1",
            to_segment_id="S2",
            movement_type=MovementType.LEFT,
        )
        with pytest.raises(ValidationError, match="belonging to another junction"):
            Junction(junction_id="J12", movements=(movement,))

    def test_prohibited_turns_are_recorded_rather_than_omitted(self) -> None:
        """Omission is indistinguishable from an incomplete map import."""
        movement = TurnMovement(
            movement_id="M1",
            junction_id="J12",
            from_segment_id="S1",
            to_segment_id="S2",
            movement_type=MovementType.RIGHT,
            is_permitted=False,
        )
        assert movement.is_permitted is False

    def test_facility_must_hang_off_a_real_junction(self) -> None:
        facility = Facility(
            facility_id="HOSP-A",
            name="Hospital A",
            facility_type=FacilityType.HOSPITAL,
            access_junction_id="J99",
        )
        with pytest.raises(ValidationError, match="unknown access junction"):
            _network(
                junctions=(Junction(junction_id="J11"),),
                segments=(),
                facilities=(facility,),
            )

    def test_physical_restrictions_are_vehicle_specific(self) -> None:
        """A low bridge excludes a ladder truck but not an ambulance."""
        segment = RoadSegment(
            segment_id="S1",
            from_junction_id="J11",
            to_junction_id="J12",
            length_m=measure_float(300.0, Unit.METER, source=ValueSource.STATIC),
            lane_count=1,
            physical_restrictions=PhysicalRestrictions(
                excluded_vehicle_types=frozenset({EmergencyVehicleType.FIRE_ENGINE})
            ),
        )
        assert segment.permits(EmergencyVehicleType.AMBULANCE) is True
        assert segment.permits(EmergencyVehicleType.FIRE_ENGINE) is False


class TestNetworkIndex:
    def test_adjacency_is_precomputed_for_search(self) -> None:
        net = _network(
            junctions=(
                Junction(junction_id="J11", kind=JunctionKind.SIGNALIZED),
                Junction(junction_id="J12"),
                Junction(junction_id="J13"),
            ),
            segments=(
                _segment("S1", "J11", "J12"),
                _segment("S2", "J11", "J13"),
                _segment("S3", "J12", "J13"),
            ),
        )
        index = net.build_index()
        assert {s.segment_id for s in index.outgoing_segments("J11")} == {"S1", "S2"}
        assert {s.segment_id for s in index.incoming_segments("J13")} == {"S2", "S3"}

    def test_index_is_not_stored_on_the_frozen_model(self) -> None:
        """A derived acceleration structure has a different lifetime to the schema."""
        net = _network(junctions=(Junction(junction_id="J11"),), segments=())
        assert net.build_index() is not net.build_index()
